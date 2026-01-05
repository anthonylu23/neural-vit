import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional

try:
    from google.cloud import aiplatform
except Exception:
    aiplatform = None

try:
    import gcsfs
except Exception:
    gcsfs = None


def _to_float(value: str) -> Optional[float]:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def _read_csv_metrics(path: str) -> Dict[str, Any]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)

    train_rows = []
    test_row = None
    for row in rows:
        epoch = (row.get("epoch") or "").strip()
        if epoch.lower() == "test":
            test_row = row
            continue
        if epoch.isdigit():
            train_rows.append(row)

    summary: Dict[str, Any] = {"row_count": len(rows)}
    if train_rows:
        def _row_metric(row: Dict[str, str], key: str) -> float:
            return _to_float(row.get(key, "")) or float("-inf")

        best_val_auc_row = max(train_rows, key=lambda r: _row_metric(r, "val_auc"))
        best_val_acc_row = max(train_rows, key=lambda r: _row_metric(r, "val_acc"))
        last_row = train_rows[-1]
        summary.update(
            {
                "best_val_auc": _to_float(best_val_auc_row.get("val_auc", "")),
                "best_val_auc_epoch": best_val_auc_row.get("epoch"),
                "best_val_acc": _to_float(best_val_acc_row.get("val_acc", "")),
                "best_val_acc_epoch": best_val_acc_row.get("epoch"),
                "last_epoch": last_row.get("epoch"),
                "last_train_loss": _to_float(last_row.get("train_loss", "")),
                "last_train_acc": _to_float(last_row.get("train_acc", "")),
                "last_val_loss": _to_float(last_row.get("val_loss", "")),
                "last_val_acc": _to_float(last_row.get("val_acc", "")),
                "last_val_auc": _to_float(last_row.get("val_auc", "")),
            }
        )

    if test_row:
        summary["test_loss"] = _to_float(test_row.get("test_loss", ""))
        summary["test_acc"] = _to_float(test_row.get("test_acc", ""))
        summary["test_auc"] = _to_float(test_row.get("test_auc", ""))

    return summary


def _resolve_project(project: Optional[str]) -> Optional[str]:
    return project or os.environ.get("AIP_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")


def _resolve_location(location: Optional[str]) -> Optional[str]:
    return location or os.environ.get("AIP_LOCATION") or "us-central1"


def _resolve_experiment(experiment: Optional[str]) -> Optional[str]:
    return experiment or os.environ.get("AIP_EXPERIMENT_NAME")


def _fetch_experiment_run(
    project: Optional[str],
    location: Optional[str],
    experiment: Optional[str],
    run_id: Optional[str],
) -> Dict[str, Any]:
    if not run_id:
        return {"status": "skipped", "reason": "run_id_not_provided"}
    if aiplatform is None:
        return {"status": "unavailable", "reason": "google-cloud-aiplatform_not_installed"}
    if not project or not location or not experiment:
        return {"status": "skipped", "reason": "missing_project_location_or_experiment"}

    try:
        aiplatform.init(project=project, location=location, experiment=experiment)
    except Exception as exc:
        return {"status": "error", "error": f"init_failed: {exc}"}

    run_obj = None
    errors = []
    if hasattr(aiplatform, "Experiment"):
        try:
            exp = aiplatform.Experiment(experiment)
            runs = exp.list_runs()
            for candidate in runs:
                name = getattr(candidate, "name", "") or ""
                display = getattr(candidate, "display_name", "") or ""
                resource = getattr(candidate, "resource_name", "") or ""
                if run_id in (name, display) or run_id in resource:
                    run_obj = candidate
                    break
        except Exception as exc:
            errors.append(f"list_runs_failed: {exc}")

    if run_obj is None and hasattr(aiplatform, "ExperimentRun"):
        try:
            run_obj = aiplatform.ExperimentRun(run_id)
        except Exception as exc:
            errors.append(f"experiment_run_failed: {exc}")

    if run_obj is None:
        return {"status": "not_found", "errors": errors}

    data = {"status": "found", "run_id": run_id}
    for attr in (
        "name",
        "display_name",
        "resource_name",
        "state",
        "create_time",
        "update_time",
    ):
        if hasattr(run_obj, attr):
            data[attr] = getattr(run_obj, attr)
    return data


def _list_paths(base_path: str) -> Dict[str, Any]:
    base = base_path.rstrip("/")
    if base.startswith("gs://"):
        if gcsfs is None:
            return {"status": "unavailable", "reason": "gcsfs_not_installed"}
        fs = gcsfs.GCSFileSystem()
        try:
            entries = fs.ls(base)
        except Exception as exc:
            return {"status": "error", "error": str(exc)}
        return {"status": "ok", "entries": entries}

    if not os.path.exists(base):
        return {"status": "missing", "entries": []}
    try:
        entries = [os.path.join(base, name) for name in os.listdir(base)]
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
    return {"status": "ok", "entries": entries}


def _resolve_tb_log_dir(tb_log_dir: Optional[str]) -> Optional[str]:
    return tb_log_dir or os.environ.get("AIP_TENSORBOARD_LOG_DIR")


def _tensorboard_details(tb_log_dir: Optional[str], run_id: Optional[str]) -> Dict[str, Any]:
    if not tb_log_dir:
        return {"status": "skipped", "reason": "tensorboard_log_dir_not_set"}
    target = tb_log_dir.rstrip("/")
    if run_id:
        target = f"{target}/{run_id}"
    details = _list_paths(target)
    details["path"] = target
    return details


def _checkpoint_details(checkpoint_root: Optional[str], run_id: Optional[str]) -> Dict[str, Any]:
    if not checkpoint_root:
        return {"status": "skipped", "reason": "checkpoint_root_not_set"}
    target = checkpoint_root.rstrip("/")
    if run_id:
        target = f"{target}/{run_id}/checkpoints"
    details = _list_paths(target)
    details["path"] = target
    return details


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize CSV runs and fetch experiment/tensorboard/checkpoint details."
    )
    parser.add_argument(
        "--csvs",
        nargs="+",
        default=[
            "evals/no_class_weights.csv",
            "evals/class_weighted_incr_dropout.csv",
        ],
        help="CSV files to summarize.",
    )
    parser.add_argument(
        "--run-ids",
        nargs="*",
        default=[],
        help="Run IDs aligned with --csvs (optional).",
    )
    parser.add_argument("--project", default=None, help="GCP project id.")
    parser.add_argument("--location", default=None, help="GCP location (e.g., us-central1).")
    parser.add_argument("--experiment", default=None, help="Vertex experiment name.")
    parser.add_argument("--tensorboard-log-dir", default=None, help="TensorBoard log dir.")
    parser.add_argument(
        "--checkpoint-root",
        default="gs://lfp-temporal-vit/vertex-runs",
        help="Root path for checkpoints.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    project = _resolve_project(args.project)
    location = _resolve_location(args.location)
    experiment = _resolve_experiment(args.experiment) or "lfp-temporal-vit-experiments"
    tb_log_dir = _resolve_tb_log_dir(args.tensorboard_log_dir)

    results = []
    run_ids = list(args.run_ids)
    while len(run_ids) < len(args.csvs):
        run_ids.append(None)

    for csv_path, run_id in zip(args.csvs, run_ids):
        summary = _read_csv_metrics(csv_path)
        result = {
            "csv": csv_path,
            "run_id": run_id,
            "csv_summary": summary,
            "experiment": _fetch_experiment_run(project, location, experiment, run_id),
            "tensorboard": _tensorboard_details(tb_log_dir, run_id),
            "checkpoints": _checkpoint_details(args.checkpoint_root, run_id),
        }
        results.append(result)

    payload = {"project": project, "location": location, "experiment": experiment, "runs": results}
    output = json.dumps(payload, indent=2, sort_keys=True)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(output)
        print(f"Wrote summary to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
