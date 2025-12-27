# 📝 Project Checklist & Status

## 🟢 Phase 1: Data Preparation & Exploration (Weeks 1-2)

### Week 1: Data Audit & Preprocessing
- [x] **1.1 Data Inventory** (`data_audit.py`)
    - Count sessions, trials, conditions.
    - [x] Script created.
- [x] **1.2 Quality Control** (`data_audit.py`)
    - Identify corrupted/missing trials, outliers.
    - [x] Script created.
- [x] **1.3 Preprocessing Pipeline** (`preprocessing.py`)
    - [x] Initial implementation of `baseline_correction`, `time_windowing`, `spectrogram`.
    - [x] **CRITICAL FIX:** Swap order of operations (Baseline Correction -> Time Windowing) to prevent data loss.
    - [x] Sequence generation (`build_trial_sequences`).
- [x] **1.4 Spectrogram Parameters**
    - [x] Defaults selected (`nperseg=128`, `noverlap=120`).
    - [x] Configurable in pipeline.

### Week 2: Exploratory Data Analysis
- [x] **2.1 EDA Setup** (`eda.ipynb`)
    - [x] Notebook created.
    - [x] Local sample data generation (`sequences.pkl`).
- [ ] **2.2 Visual Analysis**
    - [ ] Visualize WT vs FMR1 spectrograms.
    - [ ] Compare average power spectra.
- [ ] **2.3 Variability Analysis**
    - [ ] Quantify within-session trial variability.
    - [ ] Calculate coefficient of variation (CV) for sequences.

---

## 🟡 Phase 2: Model Development (Weeks 3-5)

### Week 3: Architecture Implementation
- [ ] **3.1 Core Architecture**
    - [ ] Implement `Temporal3DViT` class (PyTorch).
    - [ ] Implement 3D Patch Embeddings.
- [ ] **3.2 Positional Embeddings**
    - [ ] Implement factorized 3D encodings (Time, Frequency, Trial).
- [ ] **3.3 Unit Tests**
    - [ ] Test forward pass shapes.
    - [ ] Test gradient flow.

### Week 4: Data Pipeline (Production)
- [x] **4.1 BigQuery Split & Export** (`get_data.py`)
    - [x] Query logic implemented.
    - [ ] Finalize "Production" export to GCS/Parquet (currently using local sample).
- [ ] **4.2 PyTorch Dataset**
    - [ ] Create `GCSTrialSequenceDataset` (or local Parquet loader).
    - [ ] Integrate on-the-fly spectrogram computation.
- [ ] **4.3 Data Augmentation**
    - [ ] Implement Time Masking / Frequency Masking.
    - [ ] Implement Mixup/Cutmix (optional).

### Week 5: Infrastructure (Vertex AI / SSH)
- [ ] **5.1 Training Scripts**
    - [ ] Create `train.py` with `argparse`.
    - [ ] Integrate W&B logging.
- [ ] **5.2 Environment Setup**
    - [ ] Dockerfile (if using Vertex AI).
    - [ ] Conda environment setup (if using SSH).

---

## 🔴 Phase 3: Experiments & Evaluation (Weeks 6-8)

### Week 6: Baselines & Tuning
- [ ] **E1:** 2D ViT (Single Trial Baseline).
- [ ] **E2:** 3D ViT (4 Trials).
- [ ] **E3:** 3D ViT (8 Trials).
- [ ] **E4:** 3D ViT (8 Trials + Augmentation).

### Week 7: Evaluation
- [ ] **7.1 Metrics**
    - [ ] Test Accuracy, F1-Score.
    - [ ] Confusion Matrix.
- [ ] **7.2 Interpretability**
    - [ ] Attention Map visualization.
