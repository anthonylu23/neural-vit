import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from temporal_vit.data.gcs_dataset import GCSTrialSequenceDataset, build_global_normalizer
from temporal_vit.models.model import Temporal3DViT, Temporal3DViTConfig, CONFIGS
from temporal_vit.training.config import TrainConfig


def load_stats(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_stats(path: Path, stats: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2))


def make_normalize_fn(stats: dict):
    def _normalize(x):
        return (x - stats["mean"]) / (stats["std"] + 1e-8)
    return _normalize


def infer_input_dims(dataset: GCSTrialSequenceDataset):
    specs, _ = dataset[0]
    return specs.shape[1], specs.shape[2]


def build_model(cfg, freq_size, time_size):
    base_config = CONFIGS[cfg.model_size]
    config_dict = asdict(base_config)
    config_dict.update({
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
    })
    config = Temporal3DViTConfig(**config_dict)
    return Temporal3DViT(config)


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
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
    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def train(cfg: TrainConfig):
    if not cfg.train_paths or not cfg.val_paths or not cfg.test_paths:
        raise ValueError("train_paths, val_paths, and test_paths must be provided.")

    stats_path = Path(cfg.stats_path) if cfg.stats_path else None

    spec_config = {
        "nperseg": cfg.nperseg,
        "noverlap": cfg.noverlap,
        "freq_max": cfg.freq_max,
        "log_scale": cfg.log_scale,
    }
    train_ds = GCSTrialSequenceDataset(
        cfg.train_paths,
        n_trials=cfg.n_trials,
        stride=cfg.stride,
        spectrogram_config=spec_config,
        baseline_end=cfg.baseline_end,
        fs=cfg.fs,
    )

    stats = load_stats(stats_path) if stats_path else None
    if stats is None:
        stats, _ = build_global_normalizer(train_ds)
        if stats_path:
            save_stats(stats_path, stats)

    normalize_fn = make_normalize_fn(stats)

    train_ds.transform = normalize_fn
    val_ds = GCSTrialSequenceDataset(
        cfg.val_paths,
        n_trials=cfg.n_trials,
        stride=cfg.stride,
        spectrogram_config=spec_config,
        baseline_end=cfg.baseline_end,
        fs=cfg.fs,
        transform=normalize_fn,
    )
    test_ds = GCSTrialSequenceDataset(
        cfg.test_paths,
        n_trials=cfg.n_trials,
        stride=cfg.stride,
        spectrogram_config=spec_config,
        baseline_end=cfg.baseline_end,
        fs=cfg.fs,
        transform=normalize_fn,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    if cfg.freq_size and cfg.time_size:
        freq_size, time_size = cfg.freq_size, cfg.time_size
    else:
        freq_size, time_size = infer_input_dims(train_ds)

    model = build_model(cfg, freq_size, time_size)
    device = torch.device(cfg.device)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    output_dir = Path(cfg.output_dir) if cfg.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for specs, labels in train_loader:
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

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch}/{cfg.epochs} | "
            f"train loss {train_loss:.4f}, acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f}, acc {val_acc:.4f}"
        )

        if output_dir and val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = {
                "model_state": model.state_dict(),
                "config": asdict(model.config),
                "stats": stats,
            }
            torch.save(ckpt, output_dir / "best.pt")

    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Test loss {test_loss:.4f}, acc {test_acc:.4f}")

    if output_dir:
        ckpt = {
            "model_state": model.state_dict(),
            "config": asdict(model.config),
            "stats": stats,
        }
        torch.save(ckpt, output_dir / "last.pt")


def main():
    config = TrainConfig(
        train_paths=["gs://your-bucket/train.parquet"],
        val_paths=["gs://your-bucket/val.parquet"],
        test_paths=["gs://your-bucket/test.parquet"],
        stats_path="stats.json",
        output_dir="runs/run1",
    )
    train(config)


if __name__ == "__main__":
    main()
