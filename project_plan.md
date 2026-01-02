# 📋 Full Project Plans

---

# Project A: Temporal 3D Vision Transformer for Multi-Trial LFP Analysis

## Executive Summary

**Goal**: Build a Vision Transformer that processes sequences of LFP spectrograms across trials to classify WT vs FMR1 knockout mice, capturing trial-to-trial dynamics that single-trial models miss.

**Dataset**: Mouse auditory cortex LFP from `lab6/8` (WT vs FMR1, ~32 sessions, ~77k trials)

**Status**: Phase 1 (Data Prep) Complete. Moving to Cloud Migration & Model Training.

---

## Cloud Infrastructure Overview

**Selected Workflow:** Vertex AI + GCS (Managed)

### Architecture

```
┌─────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│  BigQuery   │────▶│    GCS (Raw)        │────▶│  Vertex AI Training │
│ (raw data)  │     │ (train/val/test)    │     │ (Lazy Loading)      │
└─────────────┘     └──────────┬──────────┘     └─────────────────────┘
                               │
                       ┌───────▼────────┐
                       │ stats.json     │
                       │ (spec stats)   │
                       └────────────────┘
```

---

## Phase 1: Data Preparation & Exploration (Completed)

| Task | Status | Output |
| --- | --- | --- |
| **Data Inventory** | ✅ | `temporal_vit/data/data_audit.py` reports dataset stats. |
| **Preprocessing Logic** | ✅ | `temporal_vit/data/preprocessing_local.py` verified (Baseline -> Window). |
| **Local Pipeline** | ✅ | `temporal_vit/local/test_pipeline.py` verifies end-to-end flow. |
| **Normalization** | ✅ | Global normalization implemented (data loader + training). |

---

## Phase 2: Cloud Migration & Pipeline (Current Focus)

### Step 1: GCS Setup (once)
- [ ] **Bucket**: Create GCS bucket for data and outputs.
- [ ] **Permissions**: Vertex service account needs BigQuery read + GCS write.

### Step 2: Export Full Dataset → GCS (`temporal_vit/cloud/export_to_gcs.py`)
- [ ] **Query BigQuery**: Fetch full raw traces.
- [ ] **Split**: Use `create_session_splits_df` to get train/val/test by session.
- [ ] **Upload**:
  - `gs://.../v1/train.parquet`
  - `gs://.../v1/val.parquet`
  - `gs://.../v1/test.parquet`

**Brief run (Vertex/Workbench):**
```python
from temporal_vit.cloud.export_to_gcs import export_full_dataset_to_gcs
export_full_dataset_to_gcs(project_id, dataset_id, table_id, bucket_name, prefix="neural/v1")
```

### Step 3: Preprocess + Spectrogram Normalization (Optional)
- [x] **Script**: `temporal_vit/data/preprocess_to_gcs.py`
- [x] **Train Only Stats**: Compute global spectrogram mean/std from train.
- [x] **Write Outputs**:
  - `gs://.../v1/train_preprocessed.parquet`
  - `gs://.../v1/val_preprocessed.parquet`
  - `gs://.../v1/test_preprocessed.parquet`
  - `gs://.../v1/trace_norm_stats.json` (spectrogram stats)

### Step 4: Cloud Dataset (`temporal_vit/data/gcs_dataset.py`)
- [x] **Streaming**: Metadata-only index; traces loaded per row group.
- [x] **On-the-fly Specs**: Baseline → spectrogram in `__getitem__`.
- [x] **GCS Paths**: Accepts `gs://...` and normalizes for PyArrow.
- [ ] **Normalization**: Apply stats via `transform` (not inside dataset).

### Step 5: Training Script + Docker
- [x] **Train Script**: `temporal_vit/training/train.py` uses `TrainConfig` in code.
- [x] **Data**: `GCSTrialSequenceDataset` with `transform=normalize_fn`.
- [x] **Metrics**: Accuracy + AUC in eval.
- [ ] **Outputs**: Save checkpoints and stats to GCS (currently local paths).

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
