# Temporal 3D Neural ViT

Temporal 3D Vision Transformer for Multi-Trial LFP (Local Field Potential) Analysis to classify WT vs FMR1 knockout mice.

## 🚀 Quick Start

1. **Get Data**: Pull sample data from BigQuery.
   ```bash
   python -m temporal_vit.cloud.get_data
   ```
2. **Preprocess**: Parse traces and generate trial sequences (spectrograms computed on-the-fly).
   ```bash
   python -m temporal_vit.data.preprocessing_local
   ```
3. **Analyze**: Explore sequences in the notebook.
   ```bash
   jupyter notebook notebooks/eda.ipynb
   ```

## 📂 Project Structure

- `temporal_vit/cloud/get_data.py`: BigQuery extraction and stratified sampling.
- `temporal_vit/data/preprocessing_local.py`: Baseline correction, windowing, and 3D sequence generation from raw traces.
- `temporal_vit/data/gcs_dataset.py`: Streaming dataset for GCS/local parquet with on-the-fly spectrograms.
- `temporal_vit/data/data_loader.py`: Normalization utilities and local dataloader helpers.
- `temporal_vit/data/data_audit.py`: Dataset inventory and quality control reporting.
- `temporal_vit/local/`: Local experiments and pipeline checks.
- `temporal_vit/cloud/`: Cloud/Vertex utilities (BigQuery, GCS export).
- `notebooks/eda.ipynb`: Exploratory data analysis and visualization.
- `data/`: Local parquet samples and generated sequence artifacts.
- `project_plan.md`: Detailed architecture and infrastructure roadmap.
- `checklist.md`: Feature tracking and implementation status.

## 🛠 Tech Stack

- **Core**: Python, PyTorch
- **Data**: Google BigQuery, Pandas, Parquet
- **Signal**: SciPy (Spectrograms)
- **Infrastructure**: Vertex AI / GCS
