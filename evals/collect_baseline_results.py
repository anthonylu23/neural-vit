"""Collect baseline model results from GCS bucket.

This script pulls metrics from baseline models (logistic regression, XGBoost, etc.)
stored in gs://lfp-baselines and consolidates them into a JSON file for comparison
with ViT model runs.

Usage:
    python evals/collect_baseline_results.py
    python evals/collect_baseline_results.py --bucket gs://lfp-baselines --output evals/baseline_results.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import gcsfs
except ImportError:
    gcsfs = None


def _normalize_gcs_path(path: str) -> str:
    """Remove gs:// prefix for gcsfs operations."""
    return path.replace("gs://", "", 1).rstrip("/")


def _list_json_files(bucket_path: str) -> List[str]:
    """List all JSON files in the GCS bucket recursively."""
    if gcsfs is None:
        raise RuntimeError("gcsfs is required. Install with: pip install gcsfs")
    
    fs = gcsfs.GCSFileSystem()
    root = _normalize_gcs_path(bucket_path)
    
    try:
        all_files = fs.find(root)
    except Exception as e:
        print(f"Error listing files in {bucket_path}: {e}")
        return []
    
    json_files = [f"gs://{f}" for f in all_files if f.endswith(".json")]
    return sorted(json_files)


def _read_json_from_gcs(path: str) -> Optional[Dict[str, Any]]:
    """Read a JSON file from GCS."""
    if gcsfs is None:
        raise RuntimeError("gcsfs is required")
    
    fs = gcsfs.GCSFileSystem()
    gcs_path = _normalize_gcs_path(path)
    
    try:
        with fs.open(gcs_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None


def _extract_model_type(path: str) -> str:
    """Extract model type from file path (e.g., log_reg, xgboost)."""
    # Path like gs://lfp-baselines/log_reg/log_reg_20260107-072719.json
    parts = path.split("/")
    if len(parts) >= 2:
        # Get the directory name (e.g., log_reg)
        for part in parts:
            if part in ("log_reg", "xgboost", "random_forest", "svm"):
                return part
    
    # Fallback: extract from filename
    filename = parts[-1] if parts else ""
    for model in ("log_reg", "xgboost", "random_forest", "svm"):
        if model in filename:
            return model
    
    return "unknown"


def _generate_baseline_name(result: Dict[str, Any], path: str) -> str:
    """Generate a descriptive name for a baseline result."""
    model = result.get("model") or _extract_model_type(path)
    timestamp = result.get("timestamp") or ""
    
    # Model name mapping
    model_names = {
        "logistic_regression": "LogReg",
        "log_reg": "LogReg",
        "xgboost": "XGBoost",
        "random_forest": "RandomForest",
        "svm": "SVM",
    }
    display_name = model_names.get(model, str(model))
    
    # Add regularization info if present
    reg_c = result.get("regularization_C")
    if reg_c is not None:
        display_name = f"{display_name} (C={reg_c})"
    else:
        # Check if this is the older run without regularization
        if model in ("logistic_regression", "log_reg"):
            display_name = f"{display_name} (no reg)"
    
    # Add short timestamp
    if timestamp:
        short_ts = timestamp[-6:-2] if len(timestamp) >= 6 else timestamp
        display_name = f"{display_name} [{short_ts}]"
    
    return display_name


def collect_baselines(bucket_path: str) -> List[Dict[str, Any]]:
    """Collect all baseline results from GCS bucket."""
    json_files = _list_json_files(bucket_path)
    print(f"Found {len(json_files)} baseline result files in {bucket_path}")
    
    baselines = []
    for path in json_files:
        result = _read_json_from_gcs(path)
        if result is None:
            continue
        
        model_type = _extract_model_type(path)
        name = _generate_baseline_name(result, path)
        
        # Extract key metrics
        metrics = result.get("metrics", {})
        
        baseline = {
            "name": name,
            "model_type": model_type,
            "source_path": path,
            "timestamp": result.get("timestamp"),
            "feature_mode": result.get("feature_mode"),
            "n_trials": result.get("n_trials"),
            "stride": result.get("stride"),
            "regularization_C": result.get("regularization_C"),
            "feature_dim": result.get("feature_dim"),
            "train": metrics.get("train", {}),
            "val": metrics.get("val", {}),
            "test": metrics.get("test", {}),
            "timing": result.get("timing", {}),
        }
        baselines.append(baseline)
        test_auc = baseline["test"].get("auc")
        if isinstance(test_auc, float):
            test_auc_str = f"{test_auc:.4f}"
        else:
            test_auc_str = "N/A"
        print(f"  - {name}: test_auc={test_auc_str}")
    
    return baselines


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect baseline model results from GCS bucket."
    )
    parser.add_argument(
        "--bucket",
        default="gs://lfp-baselines",
        help="GCS bucket path containing baseline results.",
    )
    parser.add_argument(
        "--output",
        default="evals/baseline_results.json",
        help="Output JSON file path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    
    baselines = collect_baselines(args.bucket)
    
    payload = {
        "source_bucket": args.bucket,
        "baselines": baselines,
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {len(baselines)} baseline results to {output_path}")


if __name__ == "__main__":
    main()
