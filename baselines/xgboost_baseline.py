import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

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
    import xgboost as xgb
except Exception as exc:
    xgb = None
    _XGB_IMPORT_ERROR = exc
else:
    _XGB_IMPORT_ERROR = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XGBoost baseline on spectrogram sequences.")
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
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument(
        "--output-dir",
        default="gs://lfp-baselines/xgboost",
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
    if xgb is None:
        raise RuntimeError(f"xgboost is required: {_XGB_IMPORT_ERROR}")

    args = _parse_args()

    print("Loading train/val/test parquets...")
    train_df, train_specs = load_parquet(args.train)
    val_df, val_specs = load_parquet(args.val)
    test_df, test_specs = load_parquet(args.test)

    print("Building sequence features...")
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

    print(f"Feature dim: {X_train.shape[1]} | Train sequences: {len(y_train)}")
    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    scale_pos_weight = neg / max(pos, 1.0)

    print("Training XGBoost...")
    use_gpu = gpu_available()
    tree_method = "gpu_hist" if use_gpu else "hist"
    predictor = "gpu_predictor" if use_gpu else "auto"

    def _build_model(method, pred):
        return xgb.XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            objective="binary:logistic",
            eval_metric="auc",
            scale_pos_weight=scale_pos_weight,
            tree_method=method,
            predictor=pred,
        )

    def _fit_model(model):
        try:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
                early_stopping_rounds=args.early_stopping_rounds,
            )
        except TypeError as exc:
            if "early_stopping_rounds" not in str(exc):
                raise
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

    model = _build_model(tree_method, predictor)
    try:
        if use_gpu:
            print("Using GPU for XGBoost")
        else:
            print("Using CPU for XGBoost")
        _fit_model(model)
    except Exception as exc:
        if use_gpu:
            print(f"GPU training failed ({exc}), falling back to CPU.")
            model = _build_model("hist", "auto")
            _fit_model(model)
        else:
            raise

    print("Evaluating...")
    metrics = {
        "train": _evaluate(model, X_train, y_train),
        "val": _evaluate(model, X_val, y_val),
        "test": _evaluate(model, X_test, y_test),
    }

    payload = build_run_metadata(
        model_name="xgboost",
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
            "xgb_params": {
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "learning_rate": args.learning_rate,
                "subsample": args.subsample,
                "colsample_bytree": args.colsample_bytree,
                "early_stopping_rounds": args.early_stopping_rounds,
                "scale_pos_weight": scale_pos_weight,
            },
        }
    )

    print("Saving metrics...")
    output_path = write_metrics(args.output_dir, "xgboost", payload)
    print(f"Metrics written to {output_path}")


if __name__ == "__main__":
    main()
