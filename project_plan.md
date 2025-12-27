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

### Step 1: Split & Export Script (`export_to_gcs.py`)
- [ ] **Query BigQuery**: Fetch all raw traces.
- [ ] **Split**: Use `session_splits.py` to create Train/Val/Test DataFrames.
- [ ] **Compute Stats**: Calculate Global Mean/Std from a sample of the Train split.
- [ ] **Upload**:
    - `gs://.../v1/train.parquet`
    - `gs://.../v1/val.parquet`
    - `gs://.../v1/test.parquet`
    - `gs://.../v1/stats.json`

### Step 2: Cloud Dataset (`gcs_dataset.py`)
- [x] **Implementation**: `GCSTrialSequenceDataset` created.
- [x] **Lazy Loading**: Implemented to parse traces and compute spectrograms on-the-fly.
- [x] **Normalization**: Integrated `normalization_stats` into `__getitem__`.

### Step 3: Training Script (`train.py`)
- [ ] **Argument Parsing**: Update to accept GCS paths and hyperparams.
- [ ] **Integration**: Use `GCSTrialSequenceDataset` with `stats.json`.
- [ ] **Model**: Finalize `Temporal3DViT` implementation (currently a placeholder).
- [ ] **Logging**: Integrate `wandb` or TensorBoard.

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