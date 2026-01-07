import argparse
from typing import Dict, List, Tuple

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as pafs


def _normalize_paths(paths: List[str]) -> List[str]:
    normalized = []
    for path in paths:
        if path.startswith("gs://"):
            normalized.append(path.replace("gs://", "", 1))
        else:
            normalized.append(path)
    return normalized


def _filesystem_for_paths(paths: List[str]) -> pafs.FileSystem:
    if any(path.startswith("gs://") for path in paths):
        return pafs.GcsFileSystem()
    return pafs.LocalFileSystem()


def _load_df(paths: List[str]) -> pd.DataFrame:
    filesystem = _filesystem_for_paths(paths)
    dataset = ds.dataset(_normalize_paths(paths), format="parquet", filesystem=filesystem)
    columns = ["session", "trial_num", "condition"]
    return dataset.to_table(columns=columns).to_pandas()


def _overlap(a: pd.Series, b: pd.Series) -> int:
    return len(set(a.unique()).intersection(set(b.unique())))


def _key_overlap(df_a: pd.DataFrame, df_b: pd.DataFrame) -> int:
    keys_a = set(zip(df_a["session"], df_a["trial_num"]))
    keys_b = set(zip(df_b["session"], df_b["trial_num"]))
    return len(keys_a.intersection(keys_b))


def _mixed_session_conditions(df: pd.DataFrame) -> int:
    counts = df.groupby("session")["condition"].nunique()
    return int((counts > 1).sum())


def _summarize(name: str, df: pd.DataFrame) -> Dict[str, object]:
    return {
        "split": name,
        "rows": int(len(df)),
        "sessions": int(df["session"].nunique()),
        "mixed_label_sessions": _mixed_session_conditions(df),
        "class_counts": df["condition"].value_counts().to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check split integrity for session leakage and duplicates.")
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--val", nargs="+", required=True)
    parser.add_argument("--test", nargs="+", required=True)
    args = parser.parse_args()

    train_df = _load_df(args.train)
    val_df = _load_df(args.val)
    test_df = _load_df(args.test)

    summaries = [
        _summarize("train", train_df),
        _summarize("val", val_df),
        _summarize("test", test_df),
    ]

    overlaps = {
        "session_overlap_train_val": _overlap(train_df["session"], val_df["session"]),
        "session_overlap_train_test": _overlap(train_df["session"], test_df["session"]),
        "session_overlap_val_test": _overlap(val_df["session"], test_df["session"]),
        "trial_overlap_train_val": _key_overlap(train_df, val_df),
        "trial_overlap_train_test": _key_overlap(train_df, test_df),
        "trial_overlap_val_test": _key_overlap(val_df, test_df),
    }

    print("Split summaries:")
    for summary in summaries:
        print(summary)

    print("Overlap checks:")
    print(overlaps)


if __name__ == "__main__":
    main()
