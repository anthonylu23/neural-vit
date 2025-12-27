from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

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
    using preprocessing.normalize().
    
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