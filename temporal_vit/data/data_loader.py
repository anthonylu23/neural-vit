from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from .preprocessing_core import compute_spectrogram_single


def normalize(sequences, global_normalization=True, stats=None, spectrogram_params=None):
    """
    Normalize spectrograms in sequences using Z-score normalization.

    Args:
        sequences: List of dicts with 'spectrograms' or 'traces'
        global_normalization: If True, compute mean/std across all spectrograms.
                              If False, normalize each spectrogram independently.
        stats: Optional dict with 'mean' and 'std' to apply (for val/test normalization).
        spectrogram_params: Optional dict of params for compute_spectrogram_single when traces are provided.

    Returns:
        sequences: List of dicts (spectrograms normalized in place if present)
        stats: Dict with 'mean' and 'std' (only meaningful for global normalization)
    """
    if not sequences:
        return sequences, stats or {'mean': 0.0, 'std': 1.0}

    spectrogram_params = spectrogram_params or {}
    has_spectrograms = 'spectrograms' in sequences[0]
    has_traces = 'traces' in sequences[0]

    if global_normalization:
        if stats is None:
            total_sum = 0.0
            total_sq_sum = 0.0
            total_count = 0

            if has_spectrograms:
                for seq in sequences:
                    specs = seq['spectrograms']
                    total_sum += np.sum(specs)
                    total_sq_sum += np.sum(specs ** 2)
                    total_count += specs.size
            elif has_traces:
                for seq in sequences:
                    for trace in seq['traces']:
                        spec, _, _ = compute_spectrogram_single(trace, **spectrogram_params)
                        total_sum += np.sum(spec)
                        total_sq_sum += np.sum(spec ** 2)
                        total_count += spec.size

            mean = total_sum / total_count
            variance = (total_sq_sum / total_count) - (mean ** 2)
            std = np.sqrt(variance)
            stats = {'mean': mean, 'std': std}
        else:
            mean = stats['mean']
            std = stats['std']

        if has_spectrograms:
            for seq in sequences:
                seq['spectrograms'] = (seq['spectrograms'] - mean) / (std + 1e-8)

    else:
        if has_spectrograms:
            for seq in sequences:
                specs = seq['spectrograms']
                mean = specs.mean(axis=(1, 2), keepdims=True)
                std = specs.std(axis=(1, 2), keepdims=True)
                seq['spectrograms'] = (specs - mean) / (std + 1e-8)

        stats = {'mean': None, 'std': None}

    return sequences, stats


class LFPSequenceDataset(Dataset):
    def __init__(self, sequences, spectrogram_params=None, normalization_stats=None, global_normalization=True):
        self.sequences = sequences
        self.spectrogram_params = spectrogram_params or {}
        self.normalization_stats = normalization_stats
        self.global_normalization = global_normalization

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        if 'spectrograms' in sequence:
            spectrograms = sequence['spectrograms']
        else:
            specs = []
            for trace in sequence['traces']:
                spec, _, _ = compute_spectrogram_single(trace, **self.spectrogram_params)
                specs.append(spec)
            spectrograms = np.stack(specs)
            if self.global_normalization:
                if self.normalization_stats is not None:
                    mean = self.normalization_stats['mean']
                    std = self.normalization_stats['std']
                    spectrograms = (spectrograms - mean) / (std + 1e-8)
            else:
                mean = spectrograms.mean(axis=(1, 2), keepdims=True)
                std = spectrograms.std(axis=(1, 2), keepdims=True)
                spectrograms = (spectrograms - mean) / (std + 1e-8)

        spectrograms = spectrograms.astype(np.float32)
        label = sequence['label']
        return torch.from_numpy(spectrograms), torch.tensor(label, dtype=torch.long)


def create_dataloaders(
    train_seqs,
    val_seqs,
    test_seqs,
    batch_size=16,
    spectrogram_params=None,
    normalization_stats=None,
    global_normalization=True
):
    """
    Create dataloaders for training, validation, and test sets.

    Note: If sequences already contain spectrograms, normalize them before
    calling this function. If sequences contain traces, normalization can
    be applied on-the-fly using normalization_stats.

    Input:
        train_seqs: list of dicts, each containing 'spectrograms' or 'traces' and 'label'
        val_seqs: list of dicts, each containing 'spectrograms' or 'traces' and 'label'
        test_seqs: list of dicts, each containing 'spectrograms' or 'traces' and 'label'
        batch_size: int, batch size
        spectrogram_params: Optional dict for spectrogram computation
        normalization_stats: Optional dict with 'mean' and 'std' to apply
        global_normalization: If True, apply global normalization in dataset
    """
    train_dataset = LFPSequenceDataset(
        train_seqs,
        spectrogram_params=spectrogram_params,
        normalization_stats=normalization_stats,
        global_normalization=global_normalization
    )
    val_dataset = LFPSequenceDataset(
        val_seqs,
        spectrogram_params=spectrogram_params,
        normalization_stats=normalization_stats,
        global_normalization=global_normalization
    )
    test_dataset = LFPSequenceDataset(
        test_seqs,
        spectrogram_params=spectrogram_params,
        normalization_stats=normalization_stats,
        global_normalization=global_normalization
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
