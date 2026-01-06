from collections import Counter
import os
from dataclasses import asdict
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score

from temporal_vit.data.data_loader import (
    ParquetSequenceDataset,
    build_parquet_dataloaders,
)
from temporal_vit.models.model import CONFIGS, Temporal3DViT, Temporal3DViTConfig
from temporal_vit.training.config import TrainConfig
from temporal_vit.training.experiment_logging import (
    ExperimentLogger,
    build_run_id,
    log_config,
)


def _is_gcs_path(path: str) -> bool:
    return path.startswith("gs://")


def _checkpoint_path(output_dir: str, name: str) -> str:
    if output_dir.endswith("/"):
        return f"{output_dir}{name}"
    return f"{output_dir}/{name}"


def _checkpoint_dir(base_dir: str, run_id: str) -> str:
    base = base_dir.rstrip("/")
    return f"{base}/{run_id}/checkpoints"


def _save_checkpoint(ckpt: dict, path: str) -> None:
    if _is_gcs_path(path):
        import gcsfs

        fs = gcsfs.GCSFileSystem()
        with fs.open(path, "wb") as handle:
            torch.save(ckpt, handle)
    else:
        torch.save(ckpt, path)


def infer_input_dims(dataset: ParquetSequenceDataset):
    specs, _ = dataset[0]
    return specs.shape[1], specs.shape[2]


def build_model(cfg, freq_size, time_size):
    base_config = CONFIGS[cfg.model_size]
    config_dict = asdict(base_config)
    config_dict.update(
        {
            "n_trials": cfg.n_trials,
            "freq_size": freq_size,
            "time_size": time_size,
            "patch_trial": cfg.patch_trial or base_config.patch_trial,
            "patch_freq": cfg.patch_freq or base_config.patch_freq,
            "patch_time": cfg.patch_time or base_config.patch_time,
            "embed_dim": cfg.embed_dim or base_config.embed_dim,
            "n_heads": cfg.n_heads or base_config.n_heads,
            "n_layers": cfg.n_layers or base_config.n_layers,
            "mlp_ratio": cfg.mlp_ratio or base_config.mlp_ratio,
            "dropout": cfg.dropout,
            "attention_dropout": cfg.attention_dropout,
            "drop_path": cfg.drop_path,
        }
    )
    config = Temporal3DViTConfig(**config_dict)
    return Temporal3DViT(config)


def evaluate(model, loader, device, criterion=None):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for specs, labels in loader:
            specs = specs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(specs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())
    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")
    return avg_loss, acc, auc


def train(cfg: TrainConfig):
    if not cfg.train_paths or not cfg.val_paths or not cfg.test_paths:
        raise ValueError("train_paths, val_paths, and test_paths must be provided.")

    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available, running on CPU.")

    if not cfg.use_preprocessed:
        raise ValueError(
            "Training expects preprocessed spectrograms. Set use_preprocessed=True."
        )
    spectrogram_column = cfg.spectrogram_column
    if not spectrogram_column:
        raise ValueError("spectrogram_column must be set for preprocessed datasets.")
    if cfg.use_preprocessed and cfg.stats_path:
        print("Using preprocessed spectrograms; stats_path will be ignored.")

    print("Initializing training/validation/test datasets...")
    train_loader, val_loader, test_loader, (train_ds, val_ds, test_ds) = (
        build_parquet_dataloaders(
            train_paths=cfg.train_paths,
            val_paths=cfg.val_paths,
            test_paths=cfg.test_paths,
            n_trials=cfg.n_trials,
            stride=cfg.stride,
            spectrogram_column=spectrogram_column,
            loader_cfg=cfg.loader,
            device=cfg.device,
        )
    )
    print(f"Training dataset ready. Sequences: {len(train_ds)}")
    print(f"Validation dataset ready. Sequences: {len(val_ds)}")
    print(f"Test dataset ready. Sequences: {len(test_ds)}")

    device = torch.device(cfg.device)

    if cfg.freq_size and cfg.time_size:
        freq_size, time_size = cfg.freq_size, cfg.time_size
    else:
        freq_size, time_size = infer_input_dims(train_ds)

    model = build_model(cfg, freq_size, time_size)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    label_counts = Counter(train_ds.sequence_labels)
    num_classes = max(label_counts.keys(), default=-1) + 1
    if num_classes <= 0:
        raise ValueError("Training dataset has no labels.")
    counts = torch.tensor(
        [label_counts.get(i, 0) for i in range(num_classes)],
        dtype=torch.float32,
    )
    weights = counts.sum() / (counts * num_classes)
    weights = torch.where(counts > 0, weights, torch.zeros_like(weights))
    criterion = torch.nn.CrossEntropyLoss(
        weight=weights.to(device),
        label_smoothing=cfg.label_smoothing,
    )

    best_val_acc = 0.0
    output_dir = (
        cfg.output_dir
        or os.environ.get("AIP_MODEL_DIR")
        or os.environ.get("AIP_CHECKPOINT_DIR")
    )
    run_id = cfg.run_name or build_run_id()
    checkpoint_dir = None
    checkpoint_dir_gcs = None
    if output_dir:
        if _is_gcs_path(output_dir):
            checkpoint_dir_gcs = _checkpoint_dir(output_dir, run_id)
            checkpoint_dir = os.path.join("runs", run_id, "checkpoints")
        else:
            checkpoint_dir = _checkpoint_dir(output_dir, run_id)
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    logger = ExperimentLogger(
        run_id=run_id,
        output_dir=output_dir,
        project_id=cfg.project_id,
        location=cfg.location,
        experiment_name=cfg.experiment_name,
    )
    log_config(logger, cfg)
    logger.log_params(
        {
            "train_sequences": len(train_ds),
            "val_sequences": len(val_ds),
            "test_sequences": len(test_ds),
            "class_0_count": int(label_counts.get(0, 0)),
            "class_1_count": int(label_counts.get(1, 0)),
        }
    )

    try:
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            train_probs = []
            train_labels = []

            batch_started = False
            for specs, labels in train_loader:
                if not batch_started:
                    print(f"Epoch {epoch}/{cfg.epochs} first train batch start")
                    batch_started = True
                specs = specs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()
                logits = model(specs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * labels.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                probs = torch.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy()
                train_probs.extend(probs.tolist())
                train_labels.extend(labels.detach().cpu().numpy().tolist())

            train_loss = running_loss / max(total, 1)
            train_acc = correct / max(total, 1)
            try:
                train_auc = roc_auc_score(train_labels, train_probs)
            except ValueError:
                train_auc = float("nan")

            print(f"Epoch {epoch}/{cfg.epochs} validation start")
            val_loss, val_acc, val_auc = evaluate(model, val_loader, device, criterion)
            print(f"Epoch {epoch}/{cfg.epochs} logging metrics")
            logger.log_metrics(
                {
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "train/auc": train_auc,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "val/auc": val_auc,
                },
                step=epoch,
            )
            print(f"Epoch {epoch}/{cfg.epochs} metrics logged")
            print(
                f"Epoch {epoch}/{cfg.epochs} | "
                f"train loss {train_loss:.4f}, acc {train_acc:.4f}, auc {train_auc:.4f} | "
                f"val loss {val_loss:.4f}, acc {val_acc:.4f}, auc {val_auc:.4f}"
            )

            if checkpoint_dir and val_acc > best_val_acc:
                print(f"Epoch {epoch}/{cfg.epochs} checkpoint save start")
                best_val_acc = val_acc
                ckpt = {
                    "model_state": model.state_dict(),
                    "config": asdict(model.config),
                }
                _save_checkpoint(
                    ckpt, _checkpoint_path(checkpoint_dir, f"best_epoch_{epoch}.pt")
                )
                print(f"Epoch {epoch}/{cfg.epochs} checkpoint save complete")

        test_loss, test_acc, test_auc = evaluate(model, test_loader, device, criterion)
        logger.log_metrics(
            {
                "test/loss": test_loss,
                "test/acc": test_acc,
                "test/auc": test_auc,
            },
            step=cfg.epochs + 1,
        )
        print(f"Test loss {test_loss:.4f}, acc {test_acc:.4f}, auc {test_auc:.4f}")
    finally:
        logger.close()

    if checkpoint_dir:
        ckpt = {
            "model_state": model.state_dict(),
            "config": asdict(model.config),
        }
        _save_checkpoint(ckpt, _checkpoint_path(checkpoint_dir, "final.pt"))
        if checkpoint_dir_gcs:
            print("Uploading checkpoints to GCS...")
            try:
                import gcsfs

                fs = gcsfs.GCSFileSystem()
                fs.put(checkpoint_dir, checkpoint_dir_gcs, recursive=True)
                print("Checkpoint upload complete.")
            except Exception as exc:
                print(f"Checkpoint upload failed: {exc}")


def main():
    bucket_name = "lfp_spec_datasets"
    prefix = "neural/v2"
    output_dir = os.environ.get("AIP_MODEL_DIR") or os.environ.get("AIP_CHECKPOINT_DIR") or "runs/run1"
    config = TrainConfig(
        train_paths=[f"gs://{bucket_name}/{prefix}/train_preprocessed.parquet"],
        val_paths=[f"gs://{bucket_name}/{prefix}/val_preprocessed.parquet"],
        test_paths=[f"gs://{bucket_name}/{prefix}/test_preprocessed.parquet"],
        use_preprocessed=True,
        stats_path=None,
        output_dir=output_dir,
        model_size="small",
        experiment_name="lfp-temporal-vit-experiments",
        project_id="lfp-temporal-vit",
        location="us-central1",
    )
    train(config)


if __name__ == "__main__":
    main()
