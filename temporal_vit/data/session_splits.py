import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def _stratified_group_split(labels, groups, test_size, random_state=42):
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    groups = np.asarray(groups)
    labels = np.asarray(labels)
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise ValueError("Need at least 2 groups to split.")

    n_splits = max(2, int(round(1 / test_size)))
    n_splits = min(n_splits, unique_groups.size)
    if n_splits < 2:
        raise ValueError("Not enough groups to create a split.")

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best = None
    for train_idx, test_idx in sgkf.split(np.zeros(labels.shape[0]), labels, groups):
        ratio = len(test_idx) / len(labels)
        diff = abs(ratio - test_size)
        if best is None or diff < best[0]:
            best = (diff, train_idx, test_idx)

    if best is None:
        raise ValueError("Unable to create a stratified group split.")

    return best[1], best[2]


def create_session_splits(sequences, test_size, val_size, random_state=42):
    """
    Split sequences by session with trial-level stratification.

    Args:
        sequences: list of dicts, each containing 'session' and 'label'
        test_size: float, test size
        val_size: float, validation size
        random_state: int, random state

    Returns:
        train_seqs, val_seqs, test_seqs
    """
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1.")
    val_ratio = val_size / (1 - test_size)
    if not 0 < val_ratio < 1:
        raise ValueError("val_size is too large relative to test_size.")

    labels = np.array([seq["label"] for seq in sequences])
    groups = np.array([seq["session"] for seq in sequences])

    train_val_idx, test_idx = _stratified_group_split(
        labels, groups, test_size=test_size, random_state=random_state
    )
    train_val = [sequences[i] for i in train_val_idx]
    test_seqs = [sequences[i] for i in test_idx]

    train_val_labels = labels[train_val_idx]
    train_val_groups = groups[train_val_idx]
    train_idx, val_idx = _stratified_group_split(
        train_val_labels, train_val_groups, test_size=val_ratio, random_state=random_state
    )

    train_seqs = [train_val[i] for i in train_idx]
    val_seqs = [train_val[i] for i in val_idx]

    train_sessions = {s["session"] for s in train_seqs}
    val_sessions = {s["session"] for s in val_seqs}
    test_sessions = {s["session"] for s in test_seqs}

    print(
        f"Sessions - Train: {len(train_sessions)}, Val: {len(val_sessions)}, Test: {len(test_sessions)}"
    )
    print(
        f"Sequences - Train: {len(train_seqs)}, Val: {len(val_seqs)}, Test: {len(test_seqs)}"
    )

    assert train_sessions.isdisjoint(val_sessions)
    assert train_sessions.isdisjoint(test_sessions)
    assert val_sessions.isdisjoint(test_sessions)

    return train_seqs, val_seqs, test_seqs


def create_session_splits_df(df, test_size, val_size, random_state=42):
    """
    Split DataFrame by session with trial-level stratification.

    Args:
        df: pandas DataFrame containing 'session' and 'condition' columns
        test_size: float, test size
        val_size: float, validation size
        random_state: int, random state

    Returns:
        train_df, val_df, test_df
    """
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1.")
    val_ratio = val_size / (1 - test_size)
    if not 0 < val_ratio < 1:
        raise ValueError("val_size is too large relative to test_size.")

    labels = df["condition"].values
    groups = df["session"].values

    train_val_idx, test_idx = _stratified_group_split(
        labels, groups, test_size=test_size, random_state=random_state
    )
    train_val_df = df.iloc[train_val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    train_idx, val_idx = _stratified_group_split(
        train_val_df["condition"].values,
        train_val_df["session"].values,
        test_size=val_ratio,
        random_state=random_state,
    )

    train_df = train_val_df.iloc[train_idx].copy()
    val_df = train_val_df.iloc[val_idx].copy()

    train_sessions = set(train_df["session"].unique())
    val_sessions = set(val_df["session"].unique())
    test_sessions = set(test_df["session"].unique())

    print(
        f"Sessions - Train: {len(train_sessions)}, Val: {len(val_sessions)}, Test: {len(test_sessions)}"
    )
    print(f"Trials - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    assert train_sessions.isdisjoint(val_sessions)
    assert train_sessions.isdisjoint(test_sessions)
    assert val_sessions.isdisjoint(test_sessions)

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
