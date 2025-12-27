import numpy as np
from sklearn.model_selection import train_test_split, StratifiedGroupKFold


def create_session_splits(sequences, test_size, val_size, random_state=42):
    """
    Split sequences by session, stratified by condition label.

    Args:
        sequences: list of dicts, each containing 'session' and 'label'
        test_size: float, test size
        val_size: float, validation size
        random_state: int, random state

    Returns:
        train_seqs, val_seqs, test_seqs
    """
    session_to_label = {}
    for seq in sequences:
        session_to_label[seq['session']] = seq['label']

    sessions = list(session_to_label.keys())
    labels = [session_to_label[s] for s in sessions]

    train_val_sessions, test_sessions = train_test_split(
        sessions,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )

    train_val_labels = [session_to_label[s] for s in train_val_sessions]
    val_ratio = val_size / (1 - test_size)

    train_sessions, val_sessions = train_test_split(
        train_val_sessions,
        test_size=val_ratio,
        stratify=train_val_labels,
        random_state=random_state
    )

    train_sessions = set(train_sessions)
    val_sessions = set(val_sessions)
    test_sessions = set(test_sessions)

    train_seqs = [s for s in sequences if s['session'] in train_sessions]
    val_seqs = [s for s in sequences if s['session'] in val_sessions]
    test_seqs = [s for s in sequences if s['session'] in test_sessions]

    print(f"Sessions - Train: {len(train_sessions)}, Val: {len(val_sessions)}, Test: {len(test_sessions)}")
    print(f"Sequences - Train: {len(train_seqs)}, Val: {len(val_seqs)}, Test: {len(test_seqs)}")

    assert train_sessions.isdisjoint(val_sessions)
    assert train_sessions.isdisjoint(test_sessions)
    assert val_sessions.isdisjoint(test_sessions)

    return train_seqs, val_seqs, test_seqs


def create_session_splits_df(df, test_size, val_size, random_state=42):
    """
    Split DataFrame by session, stratified by condition.

    Args:
        df: pandas DataFrame containing 'session' and 'condition' columns
        test_size: float, test size
        val_size: float, validation size
        random_state: int, random state

    Returns:
        train_df, val_df, test_df
    """
    session_info = df[['session', 'condition']].drop_duplicates()
    sessions = session_info['session'].values
    labels = session_info['condition'].values

    train_val_sessions, test_sessions = train_test_split(
        sessions,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )

    train_val_mask = np.isin(sessions, train_val_sessions)
    train_val_labels = labels[train_val_mask]
    val_ratio = val_size / (1 - test_size)

    train_sessions, val_sessions = train_test_split(
        train_val_sessions,
        test_size=val_ratio,
        stratify=train_val_labels,
        random_state=random_state
    )

    train_df = df[df['session'].isin(train_sessions)].copy()
    val_df = df[df['session'].isin(val_sessions)].copy()
    test_df = df[df['session'].isin(test_sessions)].copy()

    print(f"Sessions - Train: {len(train_sessions)}, Val: {len(val_sessions)}, Test: {len(test_sessions)}")
    print(f"Trials - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    train_s = set(train_sessions)
    val_s = set(val_sessions)
    test_s = set(test_sessions)

    assert train_s.isdisjoint(val_s)
    assert train_s.isdisjoint(test_s)
    assert val_s.isdisjoint(test_s)

    return train_df, val_df, test_df


def create_kfold_splits(sequences, n_splits=5, random_state=42):
    """K-fold CV at session level for sequence lists."""
    sessions = [s['session'] for s in sequences]
    labels = [s['label'] for s in sequences]

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    folds = []
    for train_idx, test_idx in sgkf.split(sequences, labels, sessions):
        train_seqs = [sequences[i] for i in train_idx]
        test_seqs = [sequences[i] for i in test_idx]
        folds.append((train_seqs, test_seqs))

    return folds
