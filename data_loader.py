from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


def normalize(sequences, global_normalization=True, stats=None):
    """
    Normalize spectrograms in sequences using Z-score normalization.
    
    Args:
        sequences: List of dicts, each containing 'spectrograms' key with 3D array (n_trials, freq, time)
        global_normalization: If True, compute mean/std across all spectrograms.
                              If False, normalize each spectrogram independently.
        stats: Optional dict with 'mean' and 'std' to apply (for val/test normalization using training stats).
               If provided, these stats are used instead of computing from sequences.
    
    Returns:
        sequences: List of dicts with normalized spectrograms (modified in place)
        stats: Dict with 'mean' and 'std' (only meaningful for global normalization)
    """
    if global_normalization:
        if stats is None:
            # Compute global statistics iteratively to save memory
            total_sum = 0.0
            total_sq_sum = 0.0
            total_count = 0
            
            # First pass: compute sums
            for seq in sequences:
                specs = seq['spectrograms']
                total_sum += np.sum(specs)
                total_sq_sum += np.sum(specs ** 2)
                total_count += specs.size
            
            mean = total_sum / total_count
            variance = (total_sq_sum / total_count) - (mean ** 2)
            std = np.sqrt(variance)
            stats = {'mean': mean, 'std': std}
        else:
            # Use provided stats (for val/test sets)
            mean = stats['mean']
            std = stats['std']
        
        # Second pass: Apply global normalization
        for seq in sequences:
            # Perform operation in-place to save memory
            seq['spectrograms'] = (seq['spectrograms'] - mean) / (std + 1e-8)
            
    else:
        # Normalize each spectrogram independently
        for seq in sequences:
            specs = seq['spectrograms']  # (n_trials, freq, time)
            # Vectorized version for the whole sequence of trials at once
            mean = specs.mean(axis=(1, 2), keepdims=True)
            std = specs.std(axis=(1, 2), keepdims=True)
            seq['spectrograms'] = (specs - mean) / (std + 1e-8)
        
        stats = {'mean': None, 'std': None}
    
    return sequences, stats


class LFPSequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        spectrograms = sequence['spectrograms'].astype(np.float32)
        label = sequence['label']
        return torch.from_numpy(spectrograms), torch.tensor(label, dtype=torch.long)

def create_dataloaders(train_seqs, val_seqs, test_seqs, batch_size=16):
    '''
    Create dataloaders for training, validation, and test sets.
    
    Note: Sequences should be normalized BEFORE calling this function
    using normalize() from this module.
    
    Input: 
        train_seqs: list of dicts, each containing 'spectrograms' and 'label'
        val_seqs: list of dicts, each containing 'spectrograms' and 'label'
        test_seqs: list of dicts, each containing 'spectrograms' and 'label'
        batch_size: int, batch size
    Output:
        train_loader: DataLoader
        val_loader: DataLoader
        test_loader: DataLoader
    '''
    # Create datasets
    train_dataset = LFPSequenceDataset(train_seqs)
    val_dataset = LFPSequenceDataset(val_seqs)
    test_dataset = LFPSequenceDataset(test_seqs)
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # Shuffle training data
        num_workers=4,          # Parallel loading
        pin_memory=True         # Faster GPU transfer
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