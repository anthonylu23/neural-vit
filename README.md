# Temporal 3D Neural ViT

Temporal 3D Vision Transformer for multi-trial LFP (Local Field Potential) spectrogram sequences, built to classify WT vs FMR1 knockout mice and capture trial-to-trial dynamics.

## Project Summary

This project models sequences of LFP trials as a 3D token volume (trial x frequency x time) and trains a Temporal 3D ViT to learn cross-trial patterns that single-trial models miss. The data originates in BigQuery, is exported to GCS, preprocessed into normalized spectrogram parquets, and trained on Vertex AI with experiment tracking.

## Pipeline Overview

- Raw LFP traces live in BigQuery and are exported to GCS as train/val/test splits.
- `preprocess_to_gcs.py` computes spectrograms, applies train-set normalization, and writes preprocessed parquets plus normalization stats.
- `data_loader.py` builds PyTorch datasets/dataloaders directly from preprocessed parquets.
- Training runs on Vertex AI with GPU support, checkpointing to GCS, and metrics logged to Vertex Experiments + TensorBoard.

## Repo Highlights

- `temporal_vit/models/`: Temporal 3D ViT architecture and configs.
- `temporal_vit/data/`: preprocessing, dataloaders, and data audit utilities.
- `temporal_vit/training/`: training loop, config, and experiment logging.
- `temporal_vit/cloud/`: BigQuery/GCS export helpers.
- `notebooks/eda.ipynb`: EDA and data quality checks.
