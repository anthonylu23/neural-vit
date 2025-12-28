# 📋 Full Project Plans

---

# Project A: Temporal 3D Vision Transformer for Multi-Trial LFP Analysis

## Executive Summary

**Goal**: Build a Vision Transformer that processes sequences of LFP spectrograms across trials to classify WT vs FMR1 knockout mice, capturing trial-to-trial dynamics that single-trial models miss.

**Dataset**: Mouse auditory cortex LFP from `lab6/8` (WT vs FMR1, ~40 sessions)

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
                       │ metadata.json  │
                       │ (norm stats)   │
                       └────────────────┘
```

---

## Phase 1: Data Preparation & Exploration (Completed)

| Task | Status | Output |
| --- | --- | --- |
| **Data Inventory** | ✅ | `data_audit.py` reports dataset stats. |
| **Preprocessing Logic** | ✅ | `preprocessing.py` verified (Baseline -> Window -> Spec). |
| **Local Pipeline** | ✅ | `test_pipeline.py` verifies end-to-end flow. |
| **Normalization** | ✅ | Iterative Global Normalization implemented (prevented OOM). |

---

## Phase 2: Cloud Migration & Pipeline (Current Focus)

### Step 1: GCS Setup (once)
- [ ] **Bucket**: Create GCS bucket for data and outputs.
- [ ] **Permissions**: Vertex service account needs BigQuery read + GCS write.

### Step 2: Export Full Dataset → GCS (`export_to_gcs.py`)
- [ ] **Query BigQuery**: Fetch full raw traces.
- [ ] **Split**: Use `create_session_splits_df` to get train/val/test by session.
- [ ] **Upload**:
  - `gs://.../v1/train.parquet`
  - `gs://.../v1/val.parquet`
  - `gs://.../v1/test.parquet`

**Brief run (Vertex/Workbench):**
```python
from export_to_gcs import export_full_dataset_to_gcs
export_full_dataset_to_gcs(project_id, dataset_id, table_id, bucket_name, prefix="neural/v1")
```

### Step 3: Compute Normalization Stats
- [ ] **Train Only**: Use `build_global_normalizer()` on the train dataset.
- [ ] **Save**: Persist `stats.json` alongside GCS data.

### Step 4: Cloud Dataset (`gcs_dataset.py`)
- [x] **Streaming**: Metadata-only index; traces loaded per row group.
- [x] **On-the-fly Specs**: Baseline → spectrogram in `__getitem__`.
- [ ] **Normalization**: Apply stats via `transform` (not inside dataset).

### Step 5: Vertex Training Job
- [ ] **Train Script**: Accept `--train/--val/--test` GCS paths + `--stats`.
- [ ] **Data**: `GCSTrialSequenceDataset` with `transform=normalize_fn`.
- [ ] **Outputs**: Save checkpoints and metrics to GCS.

**Brief custom job (example):**
```bash
gcloud ai custom-jobs create --region=us-central1 --display-name=vit-train \
  --worker-pool-spec=machine-type=n1-standard-8,replica-count=1,container-image-uri=IMAGE_URI,command=python,args=train.py,--train,gs://.../train.parquet,--val,gs://.../val.parquet,--test,gs://.../test.parquet,--stats,gs://.../stats.json
```

### Step 6: Vertex Eval Job
- [ ] **Load Best Checkpoint** from GCS.
- [ ] **Run Test Eval** and export metrics (JSON + confusion matrix).

---

## Phase 3: Model Development & Experiments (Upcoming)

### Architecture Implementation
- [ ] **3D Embeddings**: Implement Factorized 3D Positional Embeddings.
- [ ] **Transformer**: Implement `Temporal3DViT` with LayerScale/Stochastic Depth.

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
- [ ] **Metrics**: Accuracy, F1, Confusion Matrix.
- [ ] **Interpretability**: Visualize attention maps (Time vs. Trial).
