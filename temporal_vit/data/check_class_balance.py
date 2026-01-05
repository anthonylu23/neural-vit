import argparse
from collections import Counter
from typing import Iterable, List, Optional, Tuple

import pyarrow.dataset as ds
import pyarrow.fs as pafs


def _parse_paths(values: Optional[List[str]]) -> List[str]:
    if not values:
        return []
    paths: List[str] = []
    for value in values:
        parts = [part.strip() for part in value.split(",") if part.strip()]
        paths.extend(parts)
    return paths


def _filesystem_for_paths(paths: Iterable[str]) -> Tuple[pafs.FileSystem, List[str]]:
    paths = list(paths)
    if any(path.startswith("gs://") for path in paths):
        if not all(path.startswith("gs://") for path in paths):
            raise ValueError("Mixing gs:// and local paths is not supported.")
        return pafs.GcsFileSystem(), [path.replace("gs://", "", 1) for path in paths]
    return pafs.LocalFileSystem(), paths


def _count_labels(paths: List[str], label_column: str, batch_size: int) -> Counter:
    filesystem, normalized = _filesystem_for_paths(paths)
    dataset = ds.dataset(normalized, format="parquet", filesystem=filesystem)
    if label_column not in dataset.schema.names:
        raise ValueError(f"Column '{label_column}' not found in dataset.")
    counts: Counter = Counter()
    scanner = dataset.scanner(columns=[label_column], batch_size=batch_size)
    for batch in scanner.to_batches():
        labels = batch.column(0).to_pylist()
        for label in labels:
            if label is None:
                continue
            counts[label] += 1
    return counts


def _print_counts(name: str, counts: Counter) -> None:
    total = sum(counts.values())
    print(f"{name} (total={total})")
    for label, count in counts.most_common():
        pct = (count / total * 100.0) if total else 0.0
        print(f"  {label}: {count} ({pct:.2f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check class balance for train/val/test parquet datasets."
    )
    parser.add_argument(
        "--train",
        action="append",
        help="Train parquet path(s). Repeat or provide comma-separated paths.",
    )
    parser.add_argument(
        "--val",
        action="append",
        help="Validation parquet path(s). Repeat or provide comma-separated paths.",
    )
    parser.add_argument(
        "--test",
        action="append",
        help="Test parquet path(s). Repeat or provide comma-separated paths.",
    )
    parser.add_argument(
        "--label-column",
        default="condition",
        help="Label column name (default: condition).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Record batch size for scanning.",
    )
    args = parser.parse_args()

    splits = [
        ("train", _parse_paths(args.train)),
        ("val", _parse_paths(args.val)),
        ("test", _parse_paths(args.test)),
    ]
    for name, paths in splits:
        if not paths:
            continue
        counts = _count_labels(paths, args.label_column, args.batch_size)
        _print_counts(name, counts)


if __name__ == "__main__":
    main()
