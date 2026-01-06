import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as pafs

try:
    import gcsfs
except Exception:
    gcsfs = None


DEFAULT_BUCKET = "lfp_spec_datasets"
DEFAULT_PREFIX = "neural/v2"


def _normalize_paths(paths: Iterable[str]) -> List[str]:
    normalized = []
    for path in paths:
        if path.startswith("gs://"):
            normalized.append(path.replace("gs://", "", 1))
        else:
            normalized.append(path)
    return normalized


def _filesystem_for_paths(paths: Iterable[str]) -> pafs.FileSystem:
    if any(path.startswith("gs://") for path in paths):
        return pafs.GcsFileSystem()
    return pafs.LocalFileSystem()


def _to_numpy_spec(value) -> np.ndarray:
    if value is None:
        return np.array([], dtype=np.float32)
    if isinstance(value, np.ndarray) and value.dtype == object:
        value = value.tolist()
    return np.asarray(value, dtype=np.float32)


def _write_json(path: str, payload: dict) -> None:
    content = json.dumps(payload, indent=2).encode("utf-8")
    if path.startswith("gs://"):
        if gcsfs is None:
            raise RuntimeError("gcsfs is required to write to GCS")
        fs = gcsfs.GCSFileSystem()
        with fs.open(path, "wb") as handle:
            handle.write(content)
        return
    Path(path).write_bytes(content)


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def load_parquet(paths: Iterable[str]) -> Tuple[pd.DataFrame, List[np.ndarray]]:
    paths = list(paths)
    if not paths:
        raise ValueError("paths must contain at least one parquet file")
    filesystem = _filesystem_for_paths(paths)
    dataset = ds.dataset(_normalize_paths(paths), format="parquet", filesystem=filesystem)
    table = dataset.to_table(columns=["session", "condition", "trial_num", "spectrogram"])
    df = table.to_pandas().reset_index(drop=True)
    spectrograms = [_to_numpy_spec(value) for value in df["spectrogram"]]
    return df, spectrograms


def _sequence_feature(specs: np.ndarray, mode: str) -> np.ndarray:
    if mode == "trial_time_stats":
        # specs: (trials, freq, time) -> mean over time -> (trials, freq)
        reduced = specs.mean(axis=2)
        mean = reduced.mean(axis=0)
        std = reduced.std(axis=0)
        return np.concatenate([mean.ravel(), std.ravel()]).astype(np.float32)
    if mode == "trial_stats":
        mean = specs.mean(axis=0)
        std = specs.std(axis=0)
        return np.concatenate([mean.ravel(), std.ravel()]).astype(np.float32)
    raise ValueError(f"Unknown feature mode: {mode}")


def build_sequence_features(
    df: pd.DataFrame,
    spectrograms: List[np.ndarray],
    *,
    n_trials: int,
    stride: int,
    label_map: Optional[Dict[str, int]] = None,
    feature_mode: str = "trial_time_stats",
) -> Tuple[np.ndarray, np.ndarray]:
    label_map = label_map or {"FMR1": 1}
    features: List[np.ndarray] = []
    labels: List[int] = []

    for session_id, group in df.groupby("session"):
        group = group.sort_values("trial_num")
        if group["condition"].nunique() != 1:
            raise ValueError(f"Session {session_id} has mixed conditions.")
        condition = group["condition"].iloc[0]
        label = label_map.get(condition, 0)
        indices = group.index.tolist()

        if len(indices) < n_trials:
            continue

        for i in range(0, len(indices) - n_trials + 1, stride):
            seq_indices = indices[i : i + n_trials]
            if any(spectrograms[idx].size == 0 for idx in seq_indices):
                continue
            seq_specs = np.stack([spectrograms[idx] for idx in seq_indices], axis=0)
            features.append(_sequence_feature(seq_specs, feature_mode))
            labels.append(label)

    if not features:
        raise ValueError("No valid sequences generated. Check n_trials/stride and spectrograms.")

    return np.stack(features), np.array(labels)


def class_balance(labels: np.ndarray) -> Dict[str, float]:
    unique, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    return {str(label): float(count) / float(total) for label, count in zip(unique, counts)}


def default_paths(split: str) -> str:
    return f"gs://{DEFAULT_BUCKET}/{DEFAULT_PREFIX}/{split}_preprocessed.parquet"


def build_run_metadata(
    model_name: str,
    train_paths: List[str],
    val_paths: List[str],
    test_paths: List[str],
    feature_mode: str,
    n_trials: int,
    stride: int,
) -> Dict[str, object]:
    return {
        "model": model_name,
        "timestamp": _timestamp(),
        "train_paths": train_paths,
        "val_paths": val_paths,
        "test_paths": test_paths,
        "feature_mode": feature_mode,
        "n_trials": n_trials,
        "stride": stride,
        "cwd": os.getcwd(),
    }


def write_metrics(
    output_dir: str,
    model_name: str,
    payload: Dict[str, object],
) -> str:
    output_dir = output_dir.rstrip("/")
    filename = f"{model_name}_{payload['timestamp']}.json"
    output_path = f"{output_dir}/{filename}"
    _write_json(output_path, payload)
    return output_path
