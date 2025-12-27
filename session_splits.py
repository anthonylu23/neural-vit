from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedGroupKFold

def create_session_splits(df, test_size, val_size, random_state=42):
    """
    Split DataFrame by session, stratified by condition.
    Input:
        df: pandas DataFrame containing 'session' and 'condition' columns
        test_size: float, test size
        val_size: float, validation size
        random_state: int, random state
    Output:
        train_df: pandas DataFrame
        val_df: pandas DataFrame
        test_df: pandas DataFrame
    """
    # Get unique sessions with their labels
    session_info = df[['session', 'condition']].drop_duplicates()
    sessions = session_info['session'].values
    labels = session_info['condition'].values
    
    # First split: separate test set
    train_val_sessions, test_sessions = train_test_split(
        sessions,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )
    
    # Identify labels for the remaining train_val set for the second split
    # Note: np.isin returns a boolean mask matching the shape of 'sessions' (the first argument).
    # So 'train_val_mask' aligns with 'sessions' and 'labels', not 'train_val_sessions'.
    train_val_mask = np.isin(sessions, train_val_sessions)
    train_val_labels = labels[train_val_mask]
    
    # Second split: separate validation from training
    # We want val_size relative to the TOTAL dataset.
    # Since we already removed test_size, the remaining data is (1 - test_size).
    # So the new test_size for this second split needs to be: val_size / (1 - test_size).
    val_ratio = val_size / (1 - test_size)  # Adjust ratio for remaining data
    
    train_sessions, val_sessions = train_test_split(
        train_val_sessions,
        test_size=val_ratio,
        stratify=train_val_labels,
        random_state=random_state
    )
    
    # Filter original DataFrame
    train_df = df[df['session'].isin(train_sessions)].copy()
    val_df = df[df['session'].isin(val_sessions)].copy()
    test_df = df[df['session'].isin(test_sessions)].copy()
    
    # Print summary 
    print(f"Sessions - Train: {len(train_sessions)}, Val: {len(val_sessions)}, Test: {len(test_sessions)}")
    print(f"Trials - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Verify no overlap
    train_s = set(train_sessions)
    val_s = set(val_sessions)
    test_s = set(test_sessions)
    
    assert train_s.isdisjoint(val_s)
    assert train_s.isdisjoint(test_s)
    assert val_s.isdisjoint(test_s)
    
    return train_df, val_df, test_df
