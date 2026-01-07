import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from google.cloud import aiplatform
except Exception:
    aiplatform = None

try:
    import gcsfs
except Exception:
    gcsfs = None


def _resolve_project(project: Optional[str]) -> Optional[str]:
    return project or os.environ.get("AIP_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")


def _resolve_location(location: Optional[str]) -> str:
    return location or os.environ.get("AIP_LOCATION") or "us-central1"


def _resolve_experiment(experiment: Optional[str]) -> str:
    return experiment or os.environ.get("AIP_EXPERIMENT_NAME") or "lfp-temporal-vit-experiments"


def _normalize_gcs_root(path: str) -> str:
    return path.replace("gs://", "", 1).rstrip("/")


def _list_metrics_files(metrics_root: str) -> List[str]:
    if metrics_root.startswith("gs://"):
        if gcsfs is None:
            raise RuntimeError("gcsfs is required to read metrics from GCS")
        fs = gcsfs.GCSFileSystem()
        root = _normalize_gcs_root(metrics_root)
        patterns = [f"{root}/**/metrics/*.jsonl", f"{root}/*/metrics/*.jsonl"]
        candidates: List[str] = []
        for pattern in patterns:
            try:
                candidates.extend(fs.glob(pattern))
            except Exception:
                continue
        if not candidates:
            try:
                candidates = fs.find(root)
            except Exception:
                candidates = []
        files = [
            f"gs://{path}" for path in candidates
            if "/metrics/" in path and path.endswith(".jsonl")
        ]
        return sorted(set(files))

    root = Path(metrics_root)
    if not root.exists():
        return []
    files = [str(path) for path in root.rglob("metrics/*.jsonl")]
    return sorted(files)


def _parse_run_id(path: str) -> Optional[str]:
    normalized = path.replace("gs://", "", 1)
    parts = normalized.split("/")
    metric_indices = [idx for idx, part in enumerate(parts) if part == "metrics"]
    if not metric_indices:
        return None
    idx = metric_indices[-1]
    if idx == 0:
        return None
    return parts[idx - 1]


def _open_metrics(path: str) -> List[Dict[str, Any]]:
    metrics: List[Dict[str, Any]] = []
    if path.startswith("gs://"):
        if gcsfs is None:
            raise RuntimeError("gcsfs is required to read metrics from GCS")
        fs = gcsfs.GCSFileSystem()
        with fs.open(path, "r") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                metrics.append(json.loads(line))
        return metrics

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            metrics.append(json.loads(line))
    return metrics


def _metric_value(record: Dict[str, Any], key: str) -> Optional[float]:
    value = record.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _summarize_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {}
    records_sorted = sorted(records, key=lambda r: r.get("step", 0))
    best_val_auc = None
    best_val_auc_step = None
    for record in records_sorted:
        val_auc = _metric_value(record, "val/auc")
        if val_auc is None:
            continue
        if best_val_auc is None or val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_auc_step = record.get("step")

    last = records_sorted[-1]
    summary = {
        "steps": len(records_sorted),
        "best_val_auc": best_val_auc,
        "best_val_auc_step": best_val_auc_step,
        "last_step": last.get("step"),
        "last_train_loss": _metric_value(last, "train/loss"),
        "last_train_acc": _metric_value(last, "train/acc"),
        "last_train_auc": _metric_value(last, "train/auc"),
        "last_val_loss": _metric_value(last, "val/loss"),
        "last_val_acc": _metric_value(last, "val/acc"),
        "last_val_auc": _metric_value(last, "val/auc"),
        "last_test_loss": _metric_value(last, "test/loss"),
        "last_test_acc": _metric_value(last, "test/acc"),
        "last_test_auc": _metric_value(last, "test/auc"),
    }
    return summary


def _resource_to_dict(resource: Any) -> Optional[Dict[str, Any]]:
    if resource is None:
        return None
    if hasattr(resource, "to_dict"):
        try:
            return resource.to_dict()
        except Exception:
            return None
    try:
        from google.protobuf.json_format import MessageToDict

        return MessageToDict(resource)
    except Exception:
        return None


def _parse_resource_name(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    parts = value.split("/")
    return parts[-1] if parts else value


def _unwrap_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "HasField"):
        for field in ("string_value", "number_value", "bool_value", "struct_value", "list_value"):
            try:
                if value.HasField(field):
                    field_value = getattr(value, field)
                    break
            except Exception:
                field_value = None
                break
        else:
            field_value = None
    else:
        field_value = None

    if field_value is None:
        if hasattr(value, "string_value") and value.string_value:
            return value.string_value
        if hasattr(value, "number_value"):
            return value.number_value
        if hasattr(value, "bool_value"):
            return value.bool_value
        if hasattr(value, "struct_value"):
            field_value = value.struct_value
        if hasattr(value, "list_value"):
            field_value = value.list_value

    if field_value is None:
        return value

    if hasattr(field_value, "fields"):
        return {k: _unwrap_value(v) for k, v in field_value.fields.items()}
    if hasattr(field_value, "values"):
        return [_unwrap_value(v) for v in field_value.values]
    return field_value


def _normalize_kv_collection(value: Any, name_keys: List[str], value_keys: List[str]) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        result: Dict[str, Any] = {}
        for item in value:
            name = None
            val = None
            if isinstance(item, dict):
                name = next((item.get(k) for k in name_keys if item.get(k) is not None), None)
                val = next((item.get(k) for k in value_keys if item.get(k) is not None), None)
            else:
                for key in name_keys:
                    if hasattr(item, key):
                        name = getattr(item, key)
                        break
                for key in value_keys:
                    if hasattr(item, key):
                        val = getattr(item, key)
                        break
            if name is not None:
                result[str(name)] = _unwrap_value(val)
        return result or value
    return value


def _extract_params(run_obj: Any) -> Any:
    params = None
    for name in ("get_params", "list_params", "parameters", "params", "hyperparameters"):
        if hasattr(run_obj, name):
            value = getattr(run_obj, name)
            if callable(value):
                try:
                    params = value()
                except Exception:
                    params = None
            else:
                params = value
        if params is not None:
            break

    params = _normalize_kv_collection(
        params,
        name_keys=["name", "parameter_id", "parameter", "param", "display_name"],
        value_keys=["value", "string_value", "double_value", "float_value", "int_value", "number_value", "bool_value"],
    )

    if params is None:
        resource_dict = _resource_to_dict(getattr(run_obj, "_gca_resource", None))
        if resource_dict:
            metadata = resource_dict.get("metadata") or {}
            for key in ("parameters", "params", "hyperparameters"):
                if key in metadata:
                    params = metadata[key]
                    break
    params = _normalize_kv_collection(
        params,
        name_keys=["name", "parameter_id", "parameter", "param", "display_name"],
        value_keys=["value", "string_value", "double_value", "float_value", "int_value", "number_value", "bool_value"],
    )
    return params


def _call_list_method(method: Any, experiment: str) -> Optional[List[Any]]:
    for kwargs in (
        {"experiment": experiment},
        {"experiment_name": experiment},
        {"experiment_id": experiment},
        {},
    ):
        try:
            result = method(**kwargs)
            if result is None:
                continue
            return list(result)
        except TypeError:
            continue
    return None


def _load_experiment_runs(project: str, location: str, experiment: str) -> Tuple[List[Any], List[str]]:
    if aiplatform is None:
        raise RuntimeError("google-cloud-aiplatform is required to read experiment runs")

    aiplatform.init(project=project, location=location, experiment=experiment)

    errors: List[str] = []
    runs: Optional[List[Any]] = None

    if hasattr(aiplatform, "ExperimentRun"):
        for name in ("list", "list_experiment_runs", "list_runs"):
            method = getattr(aiplatform.ExperimentRun, name, None)
            if method is None:
                continue
            try:
                runs = _call_list_method(method, experiment)
            except Exception as exc:
                errors.append(f"ExperimentRun.{name}: {exc}")
            if runs:
                break

    if not runs and hasattr(aiplatform, "Experiment"):
        try:
            exp = aiplatform.Experiment(experiment)
        except Exception as exc:
            errors.append(f"Experiment init: {exc}")
            exp = None
        if exp is not None:
            for name in ("list_runs", "get_experiment_runs", "list_experiment_runs"):
                method = getattr(exp, name, None)
                if method is None:
                    continue
                try:
                    runs = list(method())
                except Exception as exc:
                    errors.append(f"Experiment.{name}: {exc}")
                if runs:
                    break

    return runs or [], errors


def _match_run_params(run_id: Optional[str], runs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not run_id:
        return None
    for run in runs:
        if run_id == run.get("run_id") or run_id == run.get("display_name"):
            return run.get("params")
    for run in runs:
        for candidate in (
            run.get("resource_name"),
            run.get("name"),
            run.get("display_name"),
        ):
            if candidate and run_id in candidate:
                return run.get("params")
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect metrics JSONL from GCS and attach experiment parameters."
    )
    parser.add_argument(
        "--metrics-root",
        default="gs://lfp-temporal-vit/vertex-runs",
        help="Root path to search for metrics jsonl files.",
    )
    parser.add_argument("--project", default=None, help="GCP project id.")
    parser.add_argument("--location", default=None, help="GCP location (e.g., us-central1).")
    parser.add_argument("--experiment", default=None, help="Vertex experiment name.")
    parser.add_argument("--output", default="evals/run_details.json", help="Output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    project = _resolve_project(args.project)
    location = _resolve_location(args.location)
    experiment = _resolve_experiment(args.experiment)

    metrics_files = _list_metrics_files(args.metrics_root)
    metrics_by_run: Dict[str, Dict[str, Any]] = {}
    for path in metrics_files:
        run_id = _parse_run_id(path) or "unknown"
        record = metrics_by_run.setdefault(
            run_id,
            {"run_id": run_id, "metrics_files": [], "metrics": [], "summary": {}},
        )
        record["metrics_files"].append(path)
        record["metrics"].extend(_open_metrics(path))

    for record in metrics_by_run.values():
        record["summary"] = _summarize_metrics(record["metrics"])

    run_params = []
    if project and location and experiment:
        try:
            run_objects, errors = _load_experiment_runs(project, location, experiment)
            for run in run_objects:
                run_id = getattr(run, "display_name", None) or _parse_resource_name(
                    getattr(run, "resource_name", None)
                )
                run_params.append(
                    {
                        "run_id": run_id,
                        "name": getattr(run, "name", None),
                        "display_name": getattr(run, "display_name", None),
                        "resource_name": getattr(run, "resource_name", None),
                        "params": _extract_params(run),
                    }
                )
            if errors:
                run_params.append({"errors": errors})
        except Exception as exc:
            run_params = [{"error": str(exc)}]

    runs = []
    for record in metrics_by_run.values():
        params = _match_run_params(record.get("run_id"), run_params)
        runs.append({**record, "params": params})

    payload = {
        "metrics_root": args.metrics_root,
        "project": project,
        "location": location,
        "experiment": experiment,
        "runs": runs,
        "experiment_runs": run_params,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote summary to {output_path}")


if __name__ == "__main__":
    main()
