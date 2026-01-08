# Temporal 3D Neural ViT

End-to-end pipeline for classifying WT vs FMR1 knockout mice from multi-trial LFP (Local Field Potential) spectrogram sequences using a Temporal 3D Vision Transformer. The model treats each sample as a 3D token volume (trial x frequency x time) to capture cross-trial dynamics that single-trial models miss.

## Overview

This project pulls raw LFP traces from BigQuery, exports session-based splits to GCS, and preprocesses each trial into normalized spectrogram parquets. Training runs on Vertex AI with GPU support, checkpointing, and experiment tracking. EDA and evaluation notebooks focus on data quality checks, class balance, and generalization diagnostics across runs and hyperparameter tuning trials.

## Pipeline Snapshot

- **Source data**: BigQuery tables of per-trial LFP traces.
- **Export**: Session-stratified train/val/test splits written to GCS.
- **Preprocessing**: Spectrogram computation, train-set normalization, and parquet emission.
- **Training**: Temporal 3D ViT with configurable depth/width and regularization.
- **Tracking**: Metrics logged to Vertex Experiments + TensorBoard, checkpoints saved to GCS.
- **Evaluation**: Run aggregation, baselines, and HP tuning comparisons in `evals/`.

## Repo Highlights

- `temporal_vit/models/`: Temporal 3D ViT architecture and configs.
- `temporal_vit/data/`: preprocessing, dataloaders, and data audit utilities.
- `temporal_vit/training/`: training loop, config, and experiment logging.
- `temporal_vit/cloud/`: BigQuery/GCS export helpers.
- `baselines/`: logistic regression + XGBoost baselines on sequence features.
- `evals/`: run aggregation, plots, and integrity checks.
- `notebooks/eda.ipynb`: EDA and data quality checks.
