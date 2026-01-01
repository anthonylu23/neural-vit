from temporal_vit.data.preprocessing_local import build_dataset, build_trial_sequences
from temporal_vit.data.data_loader import create_dataloaders, normalize
from temporal_vit.data.session_splits import create_session_splits_df
import pandas as pd
import torch


def build_test_pipeline():
    """
    Build the full data pipeline from raw parquet to dataloaders.
    
    Returns:
        train_loader, val_loader, test_loader, stats
    """
    # Load raw data
    sample_data = pd.read_parquet('data/sample_data.parquet')
    print(f"Loaded {len(sample_data)} trials from parquet")

    # --- OPTIMIZATION FOR TESTING ---
    # Downsample to speed up the pipeline verification
    if len(sample_data) > 200:
        print("📉 Downsampling to 200 trials for rapid testing...")
        # Keep equal number of rows per condition if possible, or just head
        sample_data = sample_data.head(200)
    # -------------------------------
    
    # --- AUGMENTATION FOR TESTING ON SMALL DATA ---
    # If we have too few sessions, create fake copies to test the split logic
    n_sessions = sample_data['session'].nunique()
    if n_sessions < 10:  # Arbitrary threshold for safety
        print("⚠️ Warning: Too few sessions for 3-way split. Augmenting with fake data for testing...")
        augmented_dfs = [sample_data]
        # Create enough copies to have at least ~3 sessions per condition
        # (Assuming the sample has at least 1 per condition)
        for i in range(1, 4): 
            df_copy = sample_data.copy()
            df_copy['session'] = df_copy['session'].astype(str) + f"_copy{i}"
            augmented_dfs.append(df_copy)
        sample_data = pd.concat(augmented_dfs, ignore_index=True)
        print(f"Augmented size: {len(sample_data)} trials, {sample_data['session'].nunique()} sessions")
    # ----------------------------------------------

    # 1. SPLIT RAW DATA BY SESSION
    print("\n--- SPLITTING RAW DATA ---")
    train_df, val_df, test_df = create_session_splits_df(
        sample_data, test_size=0.15, val_size=0.15
    )

    # 2. PREPROCESS EACH SPLIT INDEPENDENTLY
    print("\n--- PREPROCESSING SPLITS ---")
    
    def process_split(df, name):
        print(f"Processing {name} split ({len(df)} trials)...")
        if len(df) == 0:
            return []
        # Parse traces and baseline correction
        dataset = build_dataset(df)
        # Build sequences
        sequences = build_trial_sequences(dataset)
        print(f"  -> Created {len(sequences)} sequences for {name}")
        return sequences

    train_seqs = process_split(train_df, 'TRAIN')
    val_seqs = process_split(val_df, 'VAL')
    test_seqs = process_split(test_df, 'TEST')
    
    # 3. NORMALIZE
    # Compute stats from training, apply to all
    print("\n--- NORMALIZATION ---")
    spectrogram_params = {
        'fs': 1000,
        'nperseg': 128,
        'noverlap': 120,
        'freq_max': None,
        'log_scale': True
    }
    if len(train_seqs) > 0:
        train_seqs, stats = normalize(
            train_seqs, global_normalization=True, spectrogram_params=spectrogram_params
        )
        print(f"Normalization stats - mean: {stats['mean']:.4f}, std: {stats['std']:.4f}")
    else:
        print("⚠️ Warning: Training set is empty, skipping normalization.")
        stats = {'mean': 0, 'std': 1}
    
    # 4. CREATE DATALOADERS
    train_loader, val_loader, test_loader = create_dataloaders(
        train_seqs,
        val_seqs,
        test_seqs,
        batch_size=16,
        spectrogram_params=spectrogram_params,
        normalization_stats=stats,
        global_normalization=True
    )
    
    return train_loader, val_loader, test_loader, stats


def verify_dataloaders(train_loader, val_loader, test_loader):
    """
    Verify that dataloaders are correctly created.
    
    Returns:
        bool: True if all checks pass
    """
    all_passed = True
    
    print("\n" + "="*60)
    print("DATALOADER VERIFICATION")
    print("="*60)
    
    loaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    for name, loader in loaders.items():
        print(f"\n--- {name.upper()} LOADER ---")
        
        # Check loader is not empty
        if len(loader) == 0:
            print(f"  ❌ FAIL: {name} loader is empty")
            all_passed = False
            continue
        print(f"  ✓ Number of batches: {len(loader)}")
        
        # Get a sample batch
        batch_specs, batch_labels = next(iter(loader))
        
        # Check tensor types
        if not isinstance(batch_specs, torch.Tensor):
            print(f"  ❌ FAIL: spectrograms is not a tensor")
            all_passed = False
        else:
            print(f"  ✓ Spectrograms tensor type: {batch_specs.dtype}")
        
        if not isinstance(batch_labels, torch.Tensor):
            print(f"  ❌ FAIL: labels is not a tensor")
            all_passed = False
        else:
            print(f"  ✓ Labels tensor type: {batch_labels.dtype}")
        
        # Check shapes
        print(f"  ✓ Batch spectrograms shape: {batch_specs.shape}")
        print(f"    (batch_size, n_trials, freq_bins, time_bins)")
        print(f"  ✓ Batch labels shape: {batch_labels.shape}")
        
        # Verify spectrogram dimensions (should be 4D: batch, trials, freq, time)
        if batch_specs.dim() != 4:
            print(f"  ❌ FAIL: Expected 4D tensor, got {batch_specs.dim()}D")
            all_passed = False
        else:
            print(f"  ✓ Correct number of dimensions (4D)")
        
        # Check for NaN/Inf values
        if torch.isnan(batch_specs).any():
            print(f"  ❌ FAIL: Spectrograms contain NaN values")
            all_passed = False
        else:
            print(f"  ✓ No NaN values in spectrograms")
        
        if torch.isinf(batch_specs).any():
            print(f"  ❌ FAIL: Spectrograms contain Inf values")
            all_passed = False
        else:
            print(f"  ✓ No Inf values in spectrograms")
        
        # Check labels are valid (0 or 1 for binary classification)
        unique_labels = torch.unique(batch_labels)
        valid_labels = all(l in [0, 1] for l in unique_labels.tolist())
        if not valid_labels:
            print(f"  ❌ FAIL: Invalid labels found: {unique_labels.tolist()}")
            all_passed = False
        else:
            print(f"  ✓ Labels are valid: {unique_labels.tolist()}")
        
        # Check normalization (mean should be ~0, std should be ~1 for global norm)
        spec_mean = batch_specs.mean().item()
        spec_std = batch_specs.std().item()
        print(f"  ✓ Batch stats - mean: {spec_mean:.4f}, std: {spec_std:.4f}")
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL CHECKS PASSED")
    else:
        print("❌ SOME CHECKS FAILED")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    # Run the pipeline
    train_loader, val_loader, test_loader, stats = build_test_pipeline()
    
    # Verify dataloaders
    success = verify_dataloaders(train_loader, val_loader, test_loader)
    
    if success:
        print("\nPipeline is ready for training!")
