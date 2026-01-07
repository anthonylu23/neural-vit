import argparse
from pathlib import Path
import sys
import time

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
    gpu_available,
    load_parquet,
    write_metrics,
)

try:
    import cupy as cp
except Exception:
    cp = None

try:
    from cuml.linear_model import LogisticRegression as CuMLLogisticRegression
except Exception:
    CuMLLogisticRegression = None


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


def _to_numpy(values):
    if hasattr(values, "get"):
        return values.get()
    return np.asarray(values)


def _evaluate(model, X, y) -> dict:
    probs = model.predict_proba(X)[:, 1]
    probs = _to_numpy(probs)
    preds = (probs >= 0.5).astype(int)
    y = _to_numpy(y)
    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan")
    return {"acc": float(acc), "auc": float(auc)}


def main() -> None:
    args = _parse_args()
    total_start = time.perf_counter()

    print("Loading train/val/test parquets...")
    t0 = time.perf_counter()
    train_df, train_specs = load_parquet(args.train)
    val_df, val_specs = load_parquet(args.val)
    test_df, test_specs = load_parquet(args.test)
    load_time = time.perf_counter() - t0
    print(f"  Load time: {load_time:.1f}s")

    print("Building sequence features...")
    t0 = time.perf_counter()
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
    feature_time = time.perf_counter() - t0
    print(f"  Feature build time: {feature_time:.1f}s")

    print(f"Feature dim: {X_train.shape[1]} | Train sequences: {len(y_train)}")
    print("Scaling features...")
    t0 = time.perf_counter()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    scale_time = time.perf_counter() - t0
    print(f"  Scale time: {scale_time:.1f}s")

    print("Training logistic regression...")
    t0 = time.perf_counter()
    use_gpu = gpu_available() and CuMLLogisticRegression is not None and cp is not None
    print(f"  GPU available: {gpu_available()}, cuML available: {CuMLLogisticRegression is not None}")
    
    if use_gpu:
        print("  Using GPU via cuML LogisticRegression")
        model = CuMLLogisticRegression(
            max_iter=500,
            tol=1e-3,
            class_weight="balanced",
        )
        model.fit(X_train_scaled, y_train)
    else:
        print("  Using CPU LogisticRegression")
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        
        # First attempt with fast settings
        model = LogisticRegression(
            max_iter=500,
            tol=1e-3,
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=-1,
        )
        
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X_train_scaled, y_train)
            
            # Check if we hit convergence warning
            convergence_failed = any(
                issubclass(w.category, ConvergenceWarning) for w in caught_warnings
            )
        
        if convergence_failed:
            print("  Warning: Model did not converge. Retrying with max_iter=2000...")
            model = LogisticRegression(
                max_iter=2000,
                tol=1e-4,
                class_weight="balanced",
                solver="lbfgs",
                n_jobs=-1,
            )
            model.fit(X_train_scaled, y_train)
            print("  Retry completed.")
    
    train_time = time.perf_counter() - t0
    print(f"  Train time: {train_time:.1f}s")

    print("Evaluating...")
    t0 = time.perf_counter()
    metrics = {
        "train": _evaluate(model, X_train_scaled, y_train),
        "val": _evaluate(model, X_val_scaled, y_val),
        "test": _evaluate(model, X_test_scaled, y_test),
    }
    eval_time = time.perf_counter() - t0
    print(f"  Eval time: {eval_time:.1f}s")

    total_time = time.perf_counter() - total_start
    timing = {
        "load_time_s": round(load_time, 2),
        "feature_time_s": round(feature_time, 2),
        "scale_time_s": round(scale_time, 2),
        "train_time_s": round(train_time, 2),
        "eval_time_s": round(eval_time, 2),
        "total_time_s": round(total_time, 2),
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
            "timing": timing,
            "train_class_balance": class_balance(y_train),
            "val_class_balance": class_balance(y_val),
            "test_class_balance": class_balance(y_test),
            "feature_dim": int(X_train.shape[1]),
        }
    )

    print("Saving metrics...")
    output_path = write_metrics(args.output_dir, "log_reg", payload)
    print(f"Metrics written to {output_path}")
    print(f"\nTotal time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
