# Temporal 3D Neural ViT

Temporal 3D Vision Transformer for Multi-Trial LFP (Local Field Potential) Analysis to classify WT vs FMR1 knockout mice.

## 🚀 Quick Start

1. **Get Data**: Pull sample data from BigQuery.
   ```bash
   python get_data.py
   ```
2. **Preprocess**: Parse traces and generate trial sequences (spectrograms computed on-the-fly).
   ```bash
   python preprocessing.py
   ```
3. **Analyze**: Explore sequences in the notebook.
   ```bash
   jupyter notebook eda.ipynb
   ```

## 📂 Project Structure

- `get_data.py`: BigQuery extraction and stratified sampling.
- `preprocessing.py`: Baseline correction, windowing, and 3D sequence generation from raw traces.
- `gcs_dataset.py`: Streaming dataset for GCS/local parquet with on-the-fly spectrograms.
- `data_loader.py`: Normalization utilities and local dataloader helpers.
- `data_audit.py`: Dataset inventory and quality control reporting.
- `eda.ipynb`: Exploratory data analysis and visualization.
- `project_plan.md`: Detailed architecture and infrastructure roadmap.
- `checklist.md`: Feature tracking and implementation status.

## 🛠 Tech Stack

- **Core**: Python, PyTorch
- **Data**: Google BigQuery, Pandas, Parquet
- **Signal**: SciPy (Spectrograms)
- **Infrastructure**: Vertex AI / GCS
