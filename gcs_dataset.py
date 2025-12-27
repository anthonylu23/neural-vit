import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Callable
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow.fs as pafs

from preprocessing import compute_spectrogram_single, parse_trace


class GCSTrialSequenceDataset(Dataset):
    """
    PyTorch Dataset that streams trial data from Parquet (GCS or local)
    and computes spectrograms on-the-fly.

    Features:
    - Loads metadata only (session/condition/trial_num) into memory.
    - Builds sequence indices by session.
    - Reads trace rows on-demand from Parquet row groups.
    - Applies baseline correction and spectrogram computation in __getitem__.
    """

    def __init__(
        self,
        data_paths: List[str],
        n_trials: int = 8,
        stride: int = 4,
        spectrogram_config: dict = None,
        normalization_stats: Optional[dict] = None,  # unused (apply via transform)
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
            spectrogram_config: Dict with keys 'nperseg', 'noverlap', 'freq_max', 'log_scale'.
            normalization_stats: Unused; compute stats via build_global_normalizer().
            baseline_end: Time (s) to use for baseline correction.
            fs: Sampling frequency.
            transform: Optional transform to apply to the final tensor (e.g., normalization/augmentation).
            use_gcs: Whether to use Google Cloud Storage filesystem.
        """
        self.n_trials = n_trials
        self.stride = stride
        self.fs = fs
        self.baseline_end = baseline_end
        self.normalization_stats = normalization_stats
        self.transform = transform
        self.use_gcs = use_gcs or any(path.startswith("gs://") for path in data_paths)
        self.filesystem = self._init_filesystem(data_paths)
        self._parquet_files = {}
        self.cache_dir = cache_dir

        self.spec_config = spectrogram_config or {
            'nperseg': 128,
            'noverlap': 120,
            'freq_max': None,
            'log_scale': True
        }

        self.index_df, self.fragment_paths = self._build_index(data_paths)
        self.sequence_indices = self._build_sequence_indices()

        print(f"Dataset initialized with {len(self.sequence_indices)} sequences from {len(self.index_df)} trials.")

    def _init_filesystem(self, paths: List[str]):
        if any(path.startswith("gs://") for path in paths):
            try:
                return pafs.GcsFileSystem()
            except Exception as exc:
                raise RuntimeError(
                    "Failed to initialize GCS filesystem. Ensure gcloud auth is set and "
                    "pyarrow has GCS support."
                ) from exc
        return pafs.LocalFileSystem()

    def _build_index(self, paths: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Load metadata only and build row-group index."""
        dataset = ds.dataset(paths, format="parquet", filesystem=self.filesystem)
        fragments = list(dataset.get_fragments())
        index_frames = []
        fragment_paths = []

        for frag_id, fragment in enumerate(fragments):
            table = fragment.to_table(columns=["session", "condition", "trial_num"])
            df = table.to_pandas()

            fragment_path = fragment.path
            fragment_paths.append(fragment_path)

            pf = pq.ParquetFile(fragment_path, filesystem=self.filesystem)
            num_groups = pf.metadata.num_row_groups
            if num_groups == 0:
                continue
            row_counts = [pf.metadata.row_group(i).num_rows for i in range(num_groups)]
            rg_bounds = np.cumsum(row_counts)

            row_indices = np.arange(len(df))
            rg_ids = np.searchsorted(rg_bounds, row_indices, side='right')
            rg_starts = np.concatenate(([0], rg_bounds[:-1]))
            row_in_group = row_indices - rg_starts[rg_ids]

            df['frag_id'] = frag_id
            df['row_group'] = rg_ids
            df['row_in_group'] = row_in_group
            index_frames.append(df)

        if index_frames:
            index_df = pd.concat(index_frames, ignore_index=True)
        else:
            index_df = pd.DataFrame(
                columns=["session", "condition", "trial_num", "frag_id", "row_group", "row_in_group"]
            )

        return index_df, fragment_paths

    def _get_parquet_file(self, frag_id: int) -> pq.ParquetFile:
        pf = self._parquet_files.get(frag_id)
        if pf is None:
            pf = pq.ParquetFile(self.fragment_paths[frag_id], filesystem=self.filesystem)
            self._parquet_files[frag_id] = pf
        return pf

    def _build_sequence_indices(self) -> List[Tuple[str, int, List[int]]]:
        """
        Group by session and create sliding windows.
        Returns list of tuples: (session_id, label, [list_of_index_df_indices])
        """
        sequences = []

        for session_id, group in self.index_df.groupby('session'):
            group = group.sort_values('trial_num')

            if group['condition'].nunique() != 1:
                raise ValueError(f"Session {session_id} has mixed conditions.")
            condition = group['condition'].iloc[0]
            label = 1 if condition == 'FMR1' else 0

            global_indices = group.index.tolist()
            if len(global_indices) < self.n_trials:
                continue

            for i in range(0, len(global_indices) - self.n_trials + 1, self.stride):
                seq_indices = global_indices[i:i + self.n_trials]
                sequences.append((session_id, label, seq_indices))

        return sequences

    def _parse_trace(self, trace_entry) -> np.ndarray:
        return parse_trace(trace_entry)

    def _process_single_trial(self, idx: int) -> np.ndarray:
        """Fetch raw trace, apply baseline correction, compute spectrogram."""
        row = self.index_df.iloc[idx]
        pf = self._get_parquet_file(int(row['frag_id']))
        rg_id = int(row['row_group'])
        row_in_group = int(row['row_in_group'])

        table = pf.read_row_group(rg_id, columns=['trace'])
        trace_entry = table.column('trace')[row_in_group].as_py()

        trace = self._parse_trace(trace_entry)

        baseline_samples = int(self.baseline_end * self.fs)
        baseline_mean = trace[:baseline_samples].mean()
        trace = trace - baseline_mean

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

        specs = [self._process_single_trial(i) for i in trial_indices]
        specs = np.stack(specs).astype(np.float32)
        specs_tensor = torch.from_numpy(specs)

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

    for _, _, trial_indices in dataset.sequence_indices:
        for idx in trial_indices:
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
