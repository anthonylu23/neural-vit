import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from baselines.common import (
    build_run_metadata,
    build_sequence_features,
    class_balance,
    default_paths,
    load_parquet,
    write_metrics,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Logistic regression baseline on spectrogram sequences.")
    parser.add_argument("--train", nargs="+", default=[default_paths("train")])
    parser.add_argument("--val", nargs="+", default=[default_paths("val")])
    parser.add_argument("--test", nargs="+", default=[default_paths("test")])
    parser.add_argument("--n-trials", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument(
        "--feature-mode",
        choices=["trial_stats", "trial_time_stats"],
        default="trial_stats",
    )
    parser.add_argument(
        "--output-dir",
        default="gs://lfp-baselines/log_reg",
        help="GCS or local output directory for metrics JSON.",
    )
    return parser.parse_args()


def _evaluate(model, X, y) -> dict:
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan")
    return {"acc": float(acc), "auc": float(auc)}


def main() -> None:
    args = _parse_args()

    train_df, train_specs = load_parquet(args.train)
    val_df, val_specs = load_parquet(args.val)
    test_df, test_specs = load_parquet(args.test)

    X_train, y_train = build_sequence_features(
        train_df,
        train_specs,
        n_trials=args.n_trials,
        stride=args.stride,
        feature_mode=args.feature_mode,
    )
    X_val, y_val = build_sequence_features(
        val_df,
        val_specs,
        n_trials=args.n_trials,
        stride=args.stride,
        feature_mode=args.feature_mode,
    )
    X_test, y_test = build_sequence_features(
        test_df,
        test_specs,
        n_trials=args.n_trials,
        stride=args.stride,
        feature_mode=args.feature_mode,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="saga",
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    metrics = {
        "train": _evaluate(model, X_train_scaled, y_train),
        "val": _evaluate(model, X_val_scaled, y_val),
        "test": _evaluate(model, X_test_scaled, y_test),
    }

    payload = build_run_metadata(
        model_name="logistic_regression",
        train_paths=args.train,
        val_paths=args.val,
        test_paths=args.test,
        feature_mode=args.feature_mode,
        n_trials=args.n_trials,
        stride=args.stride,
    )
    payload.update(
        {
            "metrics": metrics,
            "train_class_balance": class_balance(y_train),
            "val_class_balance": class_balance(y_val),
            "test_class_balance": class_balance(y_test),
            "feature_dim": int(X_train.shape[1]),
        }
    )

    output_path = write_metrics(args.output_dir, "log_reg", payload)
    print(f"Metrics written to {output_path}")


if __name__ == "__main__":
    main()
