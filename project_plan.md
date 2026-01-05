# 📋 Full Project Plans

---

# Project A: Temporal 3D Vision Transformer for Multi-Trial LFP Analysis

## Executive Summary

**Goal**: Build a Vision Transformer that processes sequences of LFP spectrograms across trials to classify WT vs FMR1 knockout mice, capturing trial-to-trial dynamics that single-trial models miss.

**Dataset**: Mouse auditory cortex LFP from `lab6/8` (WT vs FMR1, ~32 sessions, ~77k trials)

**Status**: Phase 2 (Cloud Training) Active. Preprocessed spectrogram parquets are in GCS and Vertex training runs are live with experiment logging.

---

## Cloud Infrastructure Overview

**Selected Workflow:** Vertex AI + GCS (Managed)

### Architecture

```
┌─────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│  BigQuery   │────▶│    GCS (Raw)        │────▶│ Preprocess to GCS    │
│ (raw data)  │     │ (train/val/test)    │     │ (spectrograms)      │
└─────────────┘     └──────────┬──────────┘     └──────────┬──────────┘
                               │                           │
                               │                   ┌───────▼────────┐
                               │                   │ spectrogram    │
                               │                   │ norm stats     │
                               │                   └───────┬────────┘
                               │                           │
                       ┌───────▼────────┐          ┌───────▼────────┐
                       │ GCS (Preproc)  │────────▶│ Vertex Training │
                       │ train/val/test │          │ + Experiments   │
                       └────────────────┘          └────────────────┘
```

---

## Phase 1: Data Preparation & Exploration (Completed)

| Task | Status | Output |
| --- | --- | --- |
| **Data Inventory** | ✅ | `temporal_vit/data/data_audit.py` reports dataset stats. |
| **Preprocessing Logic** | ✅ | `temporal_vit/data/preprocessing_local.py` verified (Baseline -> Window). |
| **Local Pipeline** | ✅ | `temporal_vit/local/test_pipeline.py` verifies end-to-end flow. |
| **Normalization** | ✅ | Spectrogram normalization computed in `preprocess_to_gcs.py` (train-only stats). |

---

## Phase 2: Cloud Migration & Pipeline (Current Focus)

### Step 1: GCS Setup (once)
- [x] **Buckets**:
  - `gs://lfp_spec_datasets` for raw + preprocessed data
  - `gs://lfp-temporal-vit` for outputs (checkpoints, logs)
- [x] **Permissions**: Vertex service account `vertex-runner@lfp-temporal-vit.iam.gserviceaccount.com` has
  - `storage.objects.get` on `gs://lfp_spec_datasets`
  - `storage.objects.create` on `gs://lfp-temporal-vit`

### Step 2: Export Full Dataset → GCS (`temporal_vit/cloud/export_to_gcs.py`)
- [x] **Query BigQuery**: Fetch full raw traces.
- [x] **Split**: Use `create_session_splits_df` to get train/val/test by session.
- [x] **Upload**:
  - `gs://lfp_spec_datasets/neural/v1/train.parquet`
  - `gs://lfp_spec_datasets/neural/v1/val.parquet`
  - `gs://lfp_spec_datasets/neural/v1/test.parquet`

**Brief run (Vertex/Workbench):**
```python
from temporal_vit.cloud.export_to_gcs import export_full_dataset_to_gcs
export_full_dataset_to_gcs(project_id, dataset_id, table_id, bucket_name, prefix="neural/v1")
```

### Step 3: Preprocess + Spectrogram Normalization (Current Path)
- [x] **Script**: `temporal_vit/data/preprocess_to_gcs.py`
- [x] **Train Only Stats**: Compute global spectrogram mean/std from train.
- [x] **Write Outputs**:
  - `gs://lfp_spec_datasets/neural/v1/train_preprocessed.parquet`
  - `gs://lfp_spec_datasets/neural/v1/val_preprocessed.parquet`
  - `gs://lfp_spec_datasets/neural/v1/test_preprocessed.parquet`
  - `gs://lfp_spec_datasets/neural/v1/spectrogram_norm_stats.json`

### Step 4: Data Loader (Preprocessed Parquets)
- [x] **Dataset**: `ParquetSequenceDataset` in `temporal_vit/data/data_loader.py`.
- [x] **Input**: Preprocessed parquets with `spectrogram` column.
- [x] **Loader**: `build_parquet_dataloaders` builds train/val/test loaders.
- [x] **Normalization**: Applied during preprocessing; no runtime normalization.

### Step 5: Training Script + Docker
- [x] **Train Script**: `temporal_vit/training/train.py` uses `TrainConfig` in code.
- [x] **Data**: `ParquetSequenceDataset` + `build_parquet_dataloaders`.
- [x] **Metrics**: Accuracy + AUC in eval; class-weighted loss + label smoothing.
- [x] **Outputs**: Checkpoints written to `gs://lfp-temporal-vit/vertex-runs/<run_id>/checkpoints/`.
- [x] **Tracking**: Vertex Experiments + TensorBoard logging enabled.

**Dockerfile** (root):
```bash
docker build -t temporal-vit:latest .
```

**Artifact Registry Image**:
`us-central1-docker.pkg.dev/lfp-temporal-vit/vertex-job1/temporal-vit:latest`

**Brief custom job (CPU example):**
```bash
gcloud ai custom-jobs create --region=us-central1 --display-name=temporal-vit-train-cpu \
  --worker-pool-spec=machine-type=n1-standard-8,replica-count=1,container-image-uri=us-central1-docker.pkg.dev/lfp-temporal-vit/vertex-job1/temporal-vit:latest
```

**Current job config**: `vertex_custom_job_a100_tensorboard.yaml` (A100 + TensorBoard + baseOutputDirectory).

### Step 6: Vertex Eval Job
- [ ] **Load Best Checkpoint** from GCS.
- [ ] **Run Test Eval** and export metrics (JSON + confusion matrix).

---

## Phase 3: Model Development & Experiments (Upcoming)

### Architecture Implementation
- [x] **3D Embeddings**: Factorized 3D positional embeddings in `temporal_vit/models/model.py`.
- [x] **Transformer**: `Temporal3DViT` with LayerScale/Stochastic Depth.

### Experiment Matrix
| Exp ID | Model | n_trials | Status |
| --- | --- | --- | --- |
| E1 | 2D ViT (Single Trial) | 1 | Pending |
| E2 | 3D ViT | 4 | Pending |
| E3 | 3D ViT | 8 | Pending |
| E4 | 3D ViT + Augmentation | 8 | Pending |

---

## Phase 4: Evaluation

- [ ] **Test Set Eval**: Run best model on held-out Test split.
- [ ] **Metrics**: Accuracy, AUC, F1, Confusion Matrix.
- [ ] **Interpretability**: Visualize attention maps (Time vs. Trial).
