import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Callable
import json
import io
import pyarrow.parquet as pq

# Import core preprocessing logic
from preprocessing import compute_spectrogram_single, parse_trace

class GCSTrialSequenceDataset(Dataset):
    """
    PyTorch Dataset that streams trial data from GCS Parquet files (or local files)
    and computes spectrograms on-the-fly (Lazy Loading).
    
    Features:
    - Reads raw traces from Parquet.
    - Groups by session to build sequences.
    - Applies baseline correction and spectrogram computation in __getitem__.
    """

    def __init__(
        self,
        data_paths: List[str],  # List of GCS URIs or local paths to parquet files
        n_trials: int = 8,
        stride: int = 4,
        spectrogram_config: dict = None,
        normalization_stats: Optional[dict] = None, # {'mean': float, 'std': float} (unused)
        baseline_end: float = 2.0,
        fs: int = 1000,
        transform: Optional[Callable] = None,
        cache_dir: str = '/tmp/neural_cache',
        use_gcs: bool = False
    ):
        """
        Args:
            data_paths: List of file paths (gs://... or local).
            n_trials: Number of trials per sequence.
            stride: Stride for sliding window sequences.
            spectrogram_config: Dict with keys 'nperseg', 'noverlap', 'freq_max'.
            normalization_stats: Unused; compute stats via build_global_normalizer().
            baseline_end: Time (s) to use for baseline correction.
            fs: Sampling frequency.
            transform: Optional transform to apply to the final tensor (e.g., Augmentation).
            use_gcs: Whether to use Google Cloud Storage client.
        """
        self.n_trials = n_trials
        self.stride = stride
        self.fs = fs
        self.baseline_end = baseline_end
        self.normalization_stats = normalization_stats
        self.transform = transform
        self.use_gcs = use_gcs or any(path.startswith("gs://") for path in data_paths)
        
        self.spec_config = spectrogram_config or {
            'nperseg': 128,
            'noverlap': 120,
            'freq_max': None,
            'log_scale': True
        }

        # 1. Load Data (Metadata + Traces)
        # For efficiency, we load the whole dataframe into memory. 
        # (As discussed, ~100MB-1GB is fine for cloud RAM).
        # If this grows >10GB, we would switch to loading just metadata and random-seeking files.
        self.data = self._load_data(data_paths)

        # 2. Build Sequence Index
        # We don't store the 3D tensors, just the INDICES of the trials that form a sequence.
        self.sequence_indices = self._build_sequence_indices()
        
        print(f"Dataset initialized with {len(self.sequence_indices)} sequences from {len(self.data)} trials.")

    def _load_data(self, paths: List[str]) -> pd.DataFrame:
        """Load and concatenate all parquet files."""
        dfs = []
        for path in paths:
            if path.startswith("gs://") and self.use_gcs:
                df = self._read_gcs_parquet(path)
            else:
                df = pd.read_parquet(path)
            dfs.append(df)
        
        full_df = pd.concat(dfs, ignore_index=True)
        return full_df

    def _read_gcs_parquet(self, gcs_path: str) -> pd.DataFrame:
        """Read a single parquet file from GCS."""
        try:
            from google.cloud import storage
        except ImportError as exc:
            raise ImportError(
                "google-cloud-storage is required for gs:// paths. "
                "Install it or use local parquet paths."
            ) from exc

        bucket_name = gcs_path.replace('gs://', '').split('/')[0]
        blob_name = '/'.join(gcs_path.replace('gs://', '').split('/')[1:])
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        content = blob.download_as_bytes()
        return pd.read_parquet(io.BytesIO(content))

    def _build_sequence_indices(self) -> List[Tuple[str, int, List[int]]]:
        """
        Group by session and create sliding windows.
        Returns list of tuples: (session_id, label, [list_of_global_dataframe_indices])
        """
        sequences = []
        
        # Group by session (this preserves the original index in 'data')
        # We need the global index to fetch the row later in __getitem__
        for session_id, group in self.data.groupby('session'):
            # Sort by trial number to ensure temporal order
            group = group.sort_values('trial_num')
            
            # Get condition label (assumes consistent per session)
            condition = group['condition'].iloc[0]
            label = 1 if condition == 'FMR1' else 0
            
            # Get global indices of these sorted trials
            global_indices = group.index.tolist()
            
            # Create sliding windows
            if len(global_indices) < self.n_trials:
                continue
                
            for i in range(0, len(global_indices) - self.n_trials + 1, self.stride):
                seq_indices = global_indices[i : i + self.n_trials]
                sequences.append((session_id, label, seq_indices))
                
        return sequences

    def _parse_trace(self, trace_entry) -> np.ndarray:
        """Parse trace from string or array."""
        return parse_trace(trace_entry)

    def _process_single_trial(self, idx: int) -> np.ndarray:
        """Fetch raw trace, apply baseline correction, compute spectrogram."""
        row = self.data.loc[idx]
        
        # 1. Parse Raw Trace
        trace = self._parse_trace(row['trace'])
        
        # 2. Baseline Correction
        # Calculate mean of pre-stimulus period
        baseline_samples = int(self.baseline_end * self.fs)
        baseline_mean = trace[:baseline_samples].mean()
        trace = trace - baseline_mean
        
        # 3. Compute Spectrogram
        spec, _, _ = compute_spectrogram_single(
            trace,
            fs=self.fs,
            nperseg=self.spec_config['nperseg'],
            noverlap=self.spec_config['noverlap'],
            freq_max=self.spec_config['freq_max'],
            log_scale=self.spec_config['log_scale']
        )
        
        return spec

    def __len__(self):
        return len(self.sequence_indices)

    def __getitem__(self, idx):
        session_id, label, trial_indices = self.sequence_indices[idx]
        
        # Compute spectrograms for all trials in this sequence
        # Shape: (n_trials, freq, time)
        specs = [self._process_single_trial(i) for i in trial_indices]
        specs = np.stack(specs).astype(np.float32)
        
        # Convert to Tensor
        specs_tensor = torch.from_numpy(specs)
        
        # Apply Augmentations (if any)
        if self.transform:
            specs_tensor = self.transform(specs_tensor)
            
        return specs_tensor, torch.tensor(label, dtype=torch.long)


def build_global_normalizer(dataset: GCSTrialSequenceDataset):
    """
    Compute global mean/std across spectrogram values and return a normalizer.

    Mirrors the global normalization logic in data_loader.normalize.
    Returns:
        stats: {'mean': float, 'std': float}
        normalize_fn: callable that applies (x - mean) / (std + 1e-8)
    """
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    # Use a set to track unique trial indices to avoid redundant computation
    unique_trial_indices = set()
    for _, _, trial_indices in dataset.sequence_indices:
        unique_trial_indices.update(trial_indices)

    for idx in unique_trial_indices:
        spec = dataset._process_single_trial(idx)
        total_sum += np.sum(spec)
        total_sq_sum += np.sum(spec ** 2)
        total_count += spec.size

    if total_count == 0:
        stats = {'mean': 0.0, 'std': 1.0}
    else:
        mean = total_sum / total_count
        variance = (total_sq_sum / total_count) - (mean ** 2)
        std = np.sqrt(variance)
        stats = {'mean': mean, 'std': std}

    def normalize_fn(x):
        return (x - stats['mean']) / (stats['std'] + 1e-8)

    return stats, normalize_fn
