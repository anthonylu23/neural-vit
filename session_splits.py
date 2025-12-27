from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedGroupKFold

def create_kfold_splits(sequences, n_splits=5):
    """K-fold CV at session level."""
    sessions = [s['session'] for s in sequences]
    labels = [s['label'] for s in sequences]
    
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    folds = []
    for train_idx, test_idx in sgkf.split(sequences, labels, sessions):
        train_seqs = [sequences[i] for i in train_idx]
        test_seqs = [sequences[i] for i in test_idx]
        folds.append((train_seqs, test_seqs))
    
    return folds

def create_session_splits(sequences, test_size, val_size, random_state=42):
    """
    Split sequences by session, stratified by condition.
    Input:
        sequences: list of dicts, each containing 'session' and 'label'
        test_size: float, test size
        val_size: float, validation size
        random_state: int, random state
    Output:
        train_seqs: list of dicts, each containing 'session' and 'label'
        val_seqs: list of dicts, each containing 'session' and 'label'
        test_seqs: list of dicts, each containing 'session' and 'label'
    """
    # Get unique sessions with their labels
    session_to_label = {}
    for seq in sequences:
        session_to_label[seq['session']] = seq['label']
    
    sessions = list(session_to_label.keys())
    labels = [session_to_label[s] for s in sessions]
    
    # First split: separate test set
    train_val_sessions, test_sessions = train_test_split(
        sessions,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )
    
    # Second split: separate validation from training
    train_val_labels = [session_to_label[s] for s in train_val_sessions]
    val_ratio = val_size / (1 - test_size)  # Adjust ratio for remaining data
    
    train_sessions, val_sessions = train_test_split(
        train_val_sessions,
        test_size=val_ratio,
        stratify=train_val_labels,
        random_state=random_state
    )
    
    # Convert to sets for fast lookup
    train_sessions = set(train_sessions)
    val_sessions = set(val_sessions)
    test_sessions = set(test_sessions)
    
    # Assign sequences to splits
    train_seqs = [s for s in sequences if s['session'] in train_sessions]
    val_seqs = [s for s in sequences if s['session'] in val_sessions]
    test_seqs = [s for s in sequences if s['session'] in test_sessions]
    
    # Print summary
    print(f"Sessions - Train: {len(train_sessions)}, Val: {len(val_sessions)}, Test: {len(test_sessions)}")
    print(f"Sequences - Train: {len(train_seqs)}, Val: {len(val_seqs)}, Test: {len(test_seqs)}")
    
    # Verify no overlap
    assert train_sessions.isdisjoint(val_sessions)
    assert train_sessions.isdisjoint(test_sessions)
    assert val_sessions.isdisjoint(test_sessions)
    
    return train_seqs, val_seqs, test_seqs