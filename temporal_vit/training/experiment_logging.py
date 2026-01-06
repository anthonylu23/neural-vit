import json
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


def _build_metrics_filename() -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return f"metrics_{timestamp}_{os.getpid()}.jsonl"


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


def _resolve_metrics_dir(run_id: str, output_dir: Optional[str]) -> str:
    if output_dir and not output_dir.startswith("gs://"):
        return os.path.join(output_dir, run_id, "metrics")
    return os.path.join("runs", run_id, "metrics")


def _coerce_param_value(value: Any) -> Optional[object]:
    if value is None:
        return None
    if isinstance(value, (str, int, float)):
        return value
    if isinstance(value, (list, tuple, set, dict)):
        return json.dumps(value, default=str)
    return str(value)


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
        self._metrics_path = None
        self._metrics_gcs_path = None

        if enable_tensorboard and SummaryWriter is not None:
            log_dir = _resolve_tb_log_dir(run_id, output_dir)
            if not log_dir.startswith("gs://"):
                os.makedirs(log_dir, exist_ok=True)
            self._writer = SummaryWriter(log_dir)

        metrics_dir = _resolve_metrics_dir(run_id, output_dir)
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_filename = _build_metrics_filename()
        self._metrics_path = os.path.join(metrics_dir, metrics_filename)
        if output_dir and output_dir.startswith("gs://"):
            self._metrics_gcs_path = f"{output_dir.rstrip('/')}/{run_id}/metrics/{metrics_filename}"

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
            sanitized: Dict[str, object] = {}
            for key, value in params.items():
                coerced = _coerce_param_value(value)
                if coerced is not None:
                    sanitized[key] = coerced
            if sanitized:
                aiplatform.log_params(sanitized)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self._vertex_active:
            try:
                aiplatform.log_metrics(metrics, step=step)
            except TypeError:
                aiplatform.log_metrics(metrics)
        if self._writer is not None:
            for name, value in metrics.items():
                self._writer.add_scalar(name, value, global_step=step)
        if self._metrics_path is not None:
            record = {"step": step}
            record.update(metrics)
            with open(self._metrics_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record))
                handle.write("\n")

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
        if self._metrics_gcs_path and self._metrics_path and os.path.exists(self._metrics_path):
            try:
                import gcsfs

                fs = gcsfs.GCSFileSystem()
                fs.put(self._metrics_path, self._metrics_gcs_path)
            except Exception:
                pass
        if self._vertex_active:
            aiplatform.end_run()


def log_config(logger: ExperimentLogger, config) -> None:
    logger.log_params(asdict(config))
