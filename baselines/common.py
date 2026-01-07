import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as pafs
from joblib import Parallel, delayed

try:
    import gcsfs
except Exception:
    gcsfs = None


def _get_n_jobs() -> int:
    """Get optimal number of parallel jobs."""
    n_cpus = os.cpu_count() or 1
    return max(1, n_cpus - 1)


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


def gpu_available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        pass
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        pass
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    return bool(env) and env not in ("-1", "")


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


def _process_session(
    group: pd.DataFrame,
    session_spectrograms: List[np.ndarray],
    n_trials: int,
    stride: int,
    label_map: Dict[str, int],
    feature_mode: str,
) -> Tuple[List[np.ndarray], List[int]]:
    """Process a single session to extract features (for parallel execution).
    
    Args:
        group: DataFrame for this session (with reset index starting at 0)
        session_spectrograms: Spectrograms for this session only (not full list)
        n_trials: Number of trials per sequence
        stride: Stride for sliding window
        label_map: Mapping from condition to label
        feature_mode: Feature extraction mode
    """
    features: List[np.ndarray] = []
    labels: List[int] = []

    group = group.sort_values("trial_num").reset_index(drop=True)
    if group["condition"].nunique() != 1:
        session_id = group["session"].iloc[0] if len(group) > 0 else "unknown"
        raise ValueError(f"Session {session_id} has mixed conditions.")

    condition = group["condition"].iloc[0]
    label = label_map.get(condition, 0)
    n_rows = len(group)

    if n_rows < n_trials:
        return features, labels

    for i in range(0, n_rows - n_trials + 1, stride):
        seq_indices = list(range(i, i + n_trials))
        if any(session_spectrograms[idx].size == 0 for idx in seq_indices):
            continue
        seq_specs = np.stack([session_spectrograms[idx] for idx in seq_indices], axis=0)
        features.append(_sequence_feature(seq_specs, feature_mode))
        labels.append(label)

    return features, labels


def build_sequence_features(
    df: pd.DataFrame,
    spectrograms: List[np.ndarray],
    *,
    n_trials: int,
    stride: int,
    label_map: Optional[Dict[str, int]] = None,
    feature_mode: str = "trial_time_stats",
    n_jobs: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build sequence features with optional parallelization.
    
    Args:
        df: DataFrame with session, condition, trial_num columns
        spectrograms: List of spectrogram arrays corresponding to df rows
        n_trials: Number of trials per sequence
        stride: Stride for sliding window
        label_map: Mapping from condition to label (default: {"FMR1": 1})
        feature_mode: Feature extraction mode ("trial_stats" or "trial_time_stats")
        n_jobs: Number of parallel jobs (default: n_cpus - 1, use 1 for serial)
    
    Returns:
        Tuple of (features array, labels array)
    """
    label_map = label_map or {"FMR1": 1}
    n_jobs = n_jobs if n_jobs is not None else _get_n_jobs()

    # Pre-extract session groups with their spectrograms to avoid passing full list
    session_data: List[Tuple[pd.DataFrame, List[np.ndarray]]] = []
    for _, group in df.groupby("session"):
        indices = group.index.tolist()
        session_specs = [spectrograms[idx] for idx in indices]
        session_data.append((group.copy(), session_specs))

    n_sessions = len(session_data)

    if n_jobs == 1 or n_sessions <= 2:
        # Serial execution for small workloads
        all_features: List[np.ndarray] = []
        all_labels: List[int] = []
        for group, session_specs in session_data:
            feats, labs = _process_session(
                group, session_specs, n_trials, stride, label_map, feature_mode
            )
            all_features.extend(feats)
            all_labels.extend(labs)
    else:
        # Parallel execution - each worker gets only its session's spectrograms
        results: List[Tuple[List[np.ndarray], List[int]]] = Parallel(
            n_jobs=n_jobs, backend="loky", verbose=0
        )(
            delayed(_process_session)(
                group, session_specs, n_trials, stride, label_map, feature_mode
            )
            for group, session_specs in session_data
        )  # type: ignore[assignment]
        all_features = []
        all_labels = []
        for feats, labs in results:
            all_features.extend(feats)
            all_labels.extend(labs)

    if not all_features:
        raise ValueError("No valid sequences generated. Check n_trials/stride and spectrograms.")

    return np.stack(all_features), np.array(all_labels)


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
