"""Training script for Vertex AI Hyperparameter Tuning.

This script wraps the standard training logic with:
1. CLI argument parsing for hyperparameters
2. Metric reporting to Vertex AI HP Tuning via cloudml-hypertune
"""

import argparse
import os
from collections import Counter
from dataclasses import asdict

import torch
from sklearn.metrics import roc_auc_score

try:
    import hypertune
except ImportError:
    hypertune = None

from temporal_vit.data.data_loader import (
    DataLoaderConfig,
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


def parse_args() -> argparse.Namespace:
    """Parse hyperparameters from command line arguments."""
    parser = argparse.ArgumentParser(
        description="Temporal 3D ViT training with hyperparameter tuning support."
    )
    
    # Tunable hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate")
    parser.add_argument("--attention_dropout", type=float, default=0.1,
                        help="Attention dropout rate")
    parser.add_argument("--drop_path", type=float, default=0.1,
                        help="Stochastic depth rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW")
    parser.add_argument("--label_smoothing", type=float, default=0.05,
                        help="Label smoothing factor")
    parser.add_argument("--n_layers", type=int, default=None,
                        help="Number of transformer layers (overrides model_size)")
    parser.add_argument("--n_heads", type=int, default=None,
                        help="Number of attention heads (overrides model_size)")
    parser.add_argument("--embed_dim", type=int, default=None,
                        help="Embedding dimension (overrides model_size)")
    parser.add_argument("--model_size", type=str, default="small",
                        choices=["tiny", "small", "base"],
                        help="Base model size configuration")
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--n_trials", type=int, default=8,
                        help="Number of trials per sequence")
    parser.add_argument("--stride", type=int, default=4,
                        help="Stride for sequence generation")
    
    # Paths (typically from environment in Vertex AI)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for checkpoints")
    parser.add_argument("--train_path", type=str, default=None,
                        help="Training data path")
    parser.add_argument("--val_path", type=str, default=None,
                        help="Validation data path")
    parser.add_argument("--test_path", type=str, default=None,
                        help="Test data path")
    
    # Experiment tracking
    parser.add_argument("--experiment_name", type=str, 
                        default="lfp-temporal-vit-hptune",
                        help="Vertex AI experiment name")
    parser.add_argument("--project_id", type=str, default="lfp-temporal-vit",
                        help="GCP project ID")
    parser.add_argument("--location", type=str, default="us-central1",
                        help="GCP location")
    
    return parser.parse_args()


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
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, path)


def infer_input_dims(dataset: ParquetSequenceDataset):
    specs, _ = dataset[0]
    return specs.shape[1], specs.shape[2]


def build_model(args, freq_size: int, time_size: int) -> Temporal3DViT:
    """Build model from CLI args, using model_size as base config."""
    base_config = CONFIGS[args.model_size]
    config_dict = asdict(base_config)
    config_dict.update({
        "n_trials": args.n_trials,
        "freq_size": freq_size,
        "time_size": time_size,
        "embed_dim": args.embed_dim or base_config.embed_dim,
        "n_heads": args.n_heads or base_config.n_heads,
        "n_layers": args.n_layers or base_config.n_layers,
        "dropout": args.dropout,
        "attention_dropout": args.attention_dropout,
        "drop_path": args.drop_path,
    })
    config = Temporal3DViTConfig(**config_dict)
    return Temporal3DViT(config)


def evaluate(model, loader, device, criterion=None):
    """Evaluate model on a data loader."""
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


def report_metric(hpt, metric_tag: str, metric_value: float, global_step: int):
    """Report metric to Vertex AI HP Tuning."""
    if hpt is not None:
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag=metric_tag,
            metric_value=metric_value,
            global_step=global_step,
        )


def train_with_hptune(args: argparse.Namespace):
    """Main training loop with hyperparameter tuning support."""
    
    # Initialize hypertune reporter
    hpt = None
    if hypertune is not None:
        try:
            hpt = hypertune.HyperTune()
            print("Hypertune initialized successfully.")
        except Exception as e:
            print(f"Hypertune initialization failed: {e}")
            hpt = None
    else:
        print("Hypertune not available, metrics will only be logged locally.")
    
    # Resolve data paths
    bucket_name = "lfp_spec_datasets"
    prefix = "neural/v2"
    train_paths = [args.train_path] if args.train_path else [
        f"gs://{bucket_name}/{prefix}/train_preprocessed.parquet"
    ]
    val_paths = [args.val_path] if args.val_path else [
        f"gs://{bucket_name}/{prefix}/val_preprocessed.parquet"
    ]
    test_paths = [args.test_path] if args.test_path else [
        f"gs://{bucket_name}/{prefix}/test_preprocessed.parquet"
    ]
    
    output_dir = (
        args.output_dir
        or os.environ.get("AIP_MODEL_DIR")
        or os.environ.get("AIP_CHECKPOINT_DIR")
        or "gs://lfp-temporal-vit/hptune-runs"
    )
    
    # Print configuration
    print("=" * 60)
    print("Hyperparameter Tuning Configuration")
    print("=" * 60)
    print(f"  lr: {args.lr}")
    print(f"  dropout: {args.dropout}")
    print(f"  attention_dropout: {args.attention_dropout}")
    print(f"  drop_path: {args.drop_path}")
    print(f"  weight_decay: {args.weight_decay}")
    print(f"  label_smoothing: {args.label_smoothing}")
    print(f"  model_size: {args.model_size}")
    print(f"  n_layers: {args.n_layers}")
    print(f"  n_heads: {args.n_heads}")
    print(f"  epochs: {args.epochs}")
    print(f"  batch_size: {args.batch_size}")
    print("=" * 60)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available, running on CPU.")
    
    # Build data loaders
    print("Initializing datasets...")
    loader_cfg = DataLoaderConfig(
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
    )
    train_loader, val_loader, test_loader, (train_ds, val_ds, test_ds) = (
        build_parquet_dataloaders(
            train_paths=train_paths,
            val_paths=val_paths,
            test_paths=test_paths,
            n_trials=args.n_trials,
            stride=args.stride,
            spectrogram_column="spectrogram",
            loader_cfg=loader_cfg,
            device=str(device),
        )
    )
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)} sequences")
    
    # Build model
    freq_size, time_size = infer_input_dims(train_ds)
    model = build_model(args, freq_size, time_size)
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    label_counts = Counter(train_ds.sequence_labels)
    num_classes = max(label_counts.keys(), default=-1) + 1
    counts = torch.tensor(
        [label_counts.get(i, 0) for i in range(num_classes)],
        dtype=torch.float32,
    )
    weights = counts.sum() / (counts * num_classes)
    weights = torch.where(counts > 0, weights, torch.zeros_like(weights))
    criterion = torch.nn.CrossEntropyLoss(
        weight=weights.to(device),
        label_smoothing=args.label_smoothing,
    )
    
    # Experiment logging
    run_id = build_run_id()
    checkpoint_dir = None
    checkpoint_dir_gcs = None
    if output_dir:
        if _is_gcs_path(output_dir):
            checkpoint_dir_gcs = _checkpoint_dir(output_dir, run_id)
            checkpoint_dir = os.path.join("runs", run_id, "checkpoints")
        else:
            checkpoint_dir = _checkpoint_dir(output_dir, run_id)
        from pathlib import Path
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    logger = ExperimentLogger(
        run_id=run_id,
        output_dir=output_dir,
        project_id=args.project_id,
        location=args.location,
        experiment_name=args.experiment_name,
    )
    
    # Log hyperparameters
    logger.log_params({
        "lr": args.lr,
        "dropout": args.dropout,
        "attention_dropout": args.attention_dropout,
        "drop_path": args.drop_path,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "model_size": args.model_size,
        "n_layers": args.n_layers or CONFIGS[args.model_size].n_layers,
        "n_heads": args.n_heads or CONFIGS[args.model_size].n_heads,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "n_trials": args.n_trials,
        "stride": args.stride,
        "train_sequences": len(train_ds),
        "val_sequences": len(val_ds),
        "test_sequences": len(test_ds),
    })
    
    # Training loop
    best_val_auc = 0.0
    
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            train_probs = []
            train_labels = []
            
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
                probs = torch.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy()
                train_probs.extend(probs.tolist())
                train_labels.extend(labels.detach().cpu().numpy().tolist())
            
            train_loss = running_loss / max(total, 1)
            train_acc = correct / max(total, 1)
            try:
                train_auc = roc_auc_score(train_labels, train_probs)
            except ValueError:
                train_auc = float("nan")
            
            # Validation
            val_loss, val_acc, val_auc = evaluate(model, val_loader, device, criterion)
            
            # Log metrics
            metrics = {
                "train/loss": train_loss,
                "train/acc": train_acc,
                "train/auc": train_auc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "val/auc": val_auc,
            }
            logger.log_metrics(metrics, step=epoch)
            
            # Report to Vertex AI HP Tuning
            report_metric(hpt, "val_auc", val_auc, epoch)
            
            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train loss {train_loss:.4f}, acc {train_acc:.4f}, auc {train_auc:.4f} | "
                f"val loss {val_loss:.4f}, acc {val_acc:.4f}, auc {val_auc:.4f}"
            )
            
            # Checkpoint on best val AUC
            if checkpoint_dir and val_auc > best_val_auc:
                best_val_auc = val_auc
                ckpt = {
                    "model_state": model.state_dict(),
                    "config": asdict(model.config),
                    "epoch": epoch,
                    "val_auc": val_auc,
                }
                _save_checkpoint(
                    ckpt, _checkpoint_path(checkpoint_dir, f"best_epoch_{epoch}.pt")
                )
        
        # Final test evaluation
        test_loss, test_acc, test_auc = evaluate(model, test_loader, device, criterion)
        logger.log_metrics({
            "test/loss": test_loss,
            "test/acc": test_acc,
            "test/auc": test_auc,
        }, step=args.epochs + 1)
        print(f"Test loss {test_loss:.4f}, acc {test_acc:.4f}, auc {test_auc:.4f}")
        
        # Report final test AUC
        report_metric(hpt, "test_auc", test_auc, args.epochs + 1)
        
    finally:
        logger.close()
    
    # Save final checkpoint
    if checkpoint_dir:
        ckpt = {
            "model_state": model.state_dict(),
            "config": asdict(model.config),
            "final_val_auc": best_val_auc,
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
    
    print(f"Training complete. Best val AUC: {best_val_auc:.4f}")


def main():
    args = parse_args()
    train_with_hptune(args)


if __name__ == "__main__":
    main()
