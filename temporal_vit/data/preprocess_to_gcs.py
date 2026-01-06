import json
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow.fs as pafs
import concurrent.futures
import multiprocessing as mp

from temporal_vit.data.preprocessing_core import (
    process_trace_column,
    baseline_correction,
    time_windowing,
    compute_spectrogram_single,
)


def _normalize_gs_path(path: str) -> str:
    return path.replace("gs://", "", 1) if path.startswith("gs://") else path


def _filesystem_for_path(path: str) -> pafs.FileSystem:
    if path.startswith("gs://"):
        return pafs.GcsFileSystem()
    return pafs.LocalFileSystem()


def _ensure_local_parent(path: str):
    if path.startswith("gs://"):
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _has_min_len(trace, min_len: int) -> bool:
    try:
        return len(trace) >= min_len
    except TypeError:
        return False


def _preprocess_frame(df, fs, baseline_end, apply_time_window, start_time, end_time):
    df = df.copy()
    df["trace"] = process_trace_column(df["trace"])
    df["trace"] = baseline_correction(df["trace"], fs, baseline_end)
    if apply_time_window:
        df["trace"] = time_windowing(df["trace"], fs, start_time, end_time)
        min_len = int((end_time - start_time) * fs)
        df = df[df["trace"].apply(lambda x: _has_min_len(x, min_len))].reset_index(drop=True)
    return df


def _coerce_paths(paths: Iterable[str]) -> List[str]:
    if isinstance(paths, str):
        return [paths]
    return list(paths)


def _write_json(path: str, payload: dict):
    content = json.dumps(payload, indent=2).encode("utf-8")
    if path.startswith("gs://"):
        fs = pafs.GcsFileSystem()
        normalized = _normalize_gs_path(path)
        with fs.open_output_stream(normalized) as stream:
            stream.write(content)
        return
    Path(path).write_bytes(content)


def compute_spectrogram_stats(
    input_paths: Iterable[str],
    fs: int = 1000,
    baseline_end: float = 2.0,
    apply_time_window: bool = True,
    start_time: float = 0.0,
    end_time: float = 5.0,
    batch_size: int = 2048,
    spectrogram_config: Optional[dict] = None,
):
    spectrogram_config = spectrogram_config or {}
    input_paths = _coerce_paths(input_paths)
    input_fs = _filesystem_for_path(input_paths[0])
    normalized_inputs = [_normalize_gs_path(p) for p in input_paths]

    dataset = ds.dataset(normalized_inputs, format="parquet", filesystem=input_fs)
    scanner = dataset.scanner(batch_size=batch_size)

    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    for batch in scanner.to_batches():
        df = batch.to_pandas()
        processed = _preprocess_frame(
            df,
            fs=fs,
            baseline_end=baseline_end,
            apply_time_window=apply_time_window,
            start_time=start_time,
            end_time=end_time,
        )
        for trace in processed["trace"]:
            if isinstance(trace, np.ndarray):
                arr = trace
            else:
                arr = np.array(trace)
            if arr.size == 0:
                continue
            spec, _, _ = compute_spectrogram_single(arr, **spectrogram_config)
            total_sum += float(np.sum(spec))
            total_sq_sum += float(np.sum(spec ** 2))
            total_count += int(spec.size)

    if total_count == 0:
        stats = {"mean": 0.0, "std": 1.0, "count": 0}
        return stats

    mean = total_sum / total_count
    variance = (total_sq_sum / total_count) - (mean ** 2)
    std = float(np.sqrt(variance)) if variance > 0 else 0.0
    stats = {"mean": float(mean), "std": float(std), "count": int(total_count)}
    return stats


def preprocess_parquet_to_gcs(
    input_paths: Iterable[str],
    output_path: str,
    fs: int = 1000,
    baseline_end: float = 2.0,
    apply_time_window: bool = True,
    start_time: float = 0.0,
    end_time: float = 5.0,
    batch_size: int = 2048,
    normalize_stats: Optional[dict] = None,
    spectrogram_config: Optional[dict] = None,
    keep_trace: bool = True,
):
    """
    Stream a raw parquet dataset, apply preprocessing, and write to a single parquet.

    Args:
        input_paths: Iterable of parquet paths (local or gs://).
        output_path: Output parquet path (local or gs://). Must be a single file path.
        fs: Sampling frequency (Hz).
        baseline_end: End of baseline period (seconds).
        apply_time_window: Whether to crop traces to a time window.
        start_time, end_time: Time window bounds (seconds).
        batch_size: Record batch size to process at a time.
    """
    spectrogram_config = spectrogram_config or {}
    input_paths = _coerce_paths(input_paths)
    if not input_paths:
        raise ValueError("input_paths must contain at least one path.")

    input_fs = _filesystem_for_path(input_paths[0])
    output_fs = _filesystem_for_path(output_path)
    normalized_inputs = [_normalize_gs_path(p) for p in input_paths]
    normalized_output = _normalize_gs_path(output_path)

    _ensure_local_parent(output_path)

    dataset = ds.dataset(normalized_inputs, format="parquet", filesystem=input_fs)
    scanner = dataset.scanner(batch_size=batch_size)

    writer = None
    rows_written = 0
    for idx, batch in enumerate(scanner.to_batches()):
        df = batch.to_pandas()
        processed = _preprocess_frame(
            df,
            fs=fs,
            baseline_end=baseline_end,
            apply_time_window=apply_time_window,
            start_time=start_time,
            end_time=end_time,
        )
        spectrograms = []
        for trace in processed["trace"]:
            if isinstance(trace, np.ndarray):
                arr = trace
            else:
                arr = np.array(trace)
            if arr.size == 0:
                spectrograms.append(np.array([]))
                continue
            spec, _, _ = compute_spectrogram_single(arr, **spectrogram_config)
            spectrograms.append(spec)

        if normalize_stats is not None:
            mean = normalize_stats["mean"]
            std = normalize_stats["std"] + 1e-8
            spectrograms = [(spec - mean) / std for spec in spectrograms]

        processed["spectrogram"] = [
            spec.tolist() if isinstance(spec, np.ndarray) else spec
            for spec in spectrograms
        ]

        if keep_trace:
            processed["trace"] = processed["trace"].apply(
                lambda value: value.tolist() if isinstance(value, np.ndarray) else value
            )
        else:
            processed = processed.drop(columns=["trace"])
        if writer is None:
            table = pa.Table.from_pandas(processed, preserve_index=False)
            writer = pq.ParquetWriter(normalized_output, table.schema, filesystem=output_fs)
        else:
            table = pa.Table.from_pandas(processed, schema=writer.schema, preserve_index=False)

        writer.write_table(table)
        rows_written += len(processed)
        print(f"Processed batch {idx + 1}, rows written: {rows_written}")

    if writer is not None:
        writer.close()
        print(f"Saved preprocessed parquet to {output_path}")
    else:
        raise ValueError("No rows were read from input paths.")


def _run_split_job(name, inputs, output, kwargs):
    print(f"Processing {name} split...")
    preprocess_parquet_to_gcs(inputs, output, **kwargs)
    return name, output


def preprocess_splits_to_gcs(
    train_inputs: Iterable[str],
    val_inputs: Iterable[str],
    test_inputs: Iterable[str],
    train_output: str,
    val_output: str,
    test_output: str,
    fs: int = 1000,
    baseline_end: float = 2.0,
    apply_time_window: bool = True,
    start_time: float = 0.0,
    end_time: float = 5.0,
    batch_size: int = 2048,
    normalize: bool = True,
    stats_output_path: Optional[str] = None,
    spectrogram_config: Optional[dict] = None,
    keep_trace: bool = True,
    parallel: bool = False,
    parallel_workers: Optional[int] = None,
):
    stats = None
    if normalize:
        print("Computing normalization stats from training data...")
        stats = compute_spectrogram_stats(
            train_inputs,
            fs=fs,
            baseline_end=baseline_end,
            apply_time_window=apply_time_window,
            start_time=start_time,
            end_time=end_time,
            batch_size=batch_size,
            spectrogram_config=spectrogram_config,
        )
        print(f"Stats ready. Mean={stats['mean']:.4f} Std={stats['std']:.4f}")
        if stats_output_path:
            _write_json(stats_output_path, stats)

    common_kwargs = dict(
        fs=fs,
        baseline_end=baseline_end,
        apply_time_window=apply_time_window,
        start_time=start_time,
        end_time=end_time,
        batch_size=batch_size,
        normalize_stats=stats,
        spectrogram_config=spectrogram_config,
        keep_trace=keep_trace,
    )

    splits = [
        ("train", train_inputs, train_output),
        ("val", val_inputs, val_output),
        ("test", test_inputs, test_output),
    ]

    if not parallel:
        for name, inputs, output in splits:
            print(f"Processing {name} split...")
            preprocess_parquet_to_gcs(inputs, output, **common_kwargs)
        return

    max_workers = parallel_workers or len(splits)
    context = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=context
    ) as executor:
        futures = [
            executor.submit(_run_split_job, name, inputs, output, common_kwargs)
            for name, inputs, output in splits
        ]
        for future in concurrent.futures.as_completed(futures):
            name, output = future.result()
            print(f"{name} split complete: {output}")


def main():
    bucket_name = "lfp_spec_datasets"
    prefix = "neural/v2"

    train_input = f"gs://{bucket_name}/{prefix}/train.parquet"
    val_input = f"gs://{bucket_name}/{prefix}/val.parquet"
    test_input = f"gs://{bucket_name}/{prefix}/test.parquet"
    train_output = f"gs://{bucket_name}/{prefix}/train_preprocessed.parquet"
    val_output = f"gs://{bucket_name}/{prefix}/val_preprocessed.parquet"
    test_output = f"gs://{bucket_name}/{prefix}/test_preprocessed.parquet"
    stats_output = f"gs://{bucket_name}/{prefix}/spectrogram_norm_stats.json"

    preprocess_splits_to_gcs(
        train_input,
        val_input,
        test_input,
        train_output,
        val_output,
        test_output,
        fs=1000,
        baseline_end=2.0,
        apply_time_window=True,
        start_time=0.0,
        end_time=5.0,
        stats_output_path=stats_output,
        spectrogram_config={
            "fs": 1000,
            "nperseg": 126,
            "noverlap": 116,
            "freq_max": None,
            "log_scale": True,
        },
        parallel=True,
    )


if __name__ == "__main__":
    main()
