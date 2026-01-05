import os
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Optional

try:
    from google.cloud import aiplatform
except Exception:
    aiplatform = None

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def build_run_id(prefix: str = "temporal-vit") -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{timestamp}"


def _resolve_project(project_id: Optional[str]) -> Optional[str]:
    return project_id or os.environ.get("AIP_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")


def _resolve_location(location: Optional[str]) -> Optional[str]:
    return location or os.environ.get("AIP_LOCATION")


def _resolve_experiment_name(name: Optional[str]) -> Optional[str]:
    return name or os.environ.get("AIP_EXPERIMENT_NAME") or "temporal-vit"


def _resolve_tb_log_dir(run_id: str, output_dir: Optional[str]) -> str:
    base = os.environ.get("AIP_TENSORBOARD_LOG_DIR")
    if base:
        return os.path.join(base, run_id)
    if output_dir and not output_dir.startswith("gs://"):
        return os.path.join(output_dir, "tb")
    return os.path.join("runs", run_id)


class ExperimentLogger:
    def __init__(
        self,
        run_id: str,
        *,
        output_dir: Optional[str] = None,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        experiment_name: Optional[str] = None,
        enable_vertex: bool = True,
        enable_tensorboard: bool = True,
    ) -> None:
        self.run_id = run_id
        self._vertex_active = False
        self._writer = None

        if enable_tensorboard and SummaryWriter is not None:
            log_dir = _resolve_tb_log_dir(run_id, output_dir)
            if not log_dir.startswith("gs://"):
                os.makedirs(log_dir, exist_ok=True)
            self._writer = SummaryWriter(log_dir)

        if enable_vertex and aiplatform is not None:
            project = _resolve_project(project_id)
            location_resolved = _resolve_location(location)
            experiment = _resolve_experiment_name(experiment_name)
            if project and location_resolved and experiment:
                aiplatform.init(project=project, location=location_resolved, experiment=experiment)
                aiplatform.start_run(run_id)
                self._vertex_active = True

    def log_params(self, params: Dict[str, Any]) -> None:
        if self._vertex_active:
            aiplatform.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self._vertex_active:
            aiplatform.log_metrics(metrics, step=step)
        if self._writer is not None:
            for name, value in metrics.items():
                self._writer.add_scalar(name, value, global_step=step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
        if self._vertex_active:
            aiplatform.end_run()


def log_config(logger: ExperimentLogger, config) -> None:
    logger.log_params(asdict(config))
