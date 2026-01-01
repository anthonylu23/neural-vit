import numpy as np
import pandas as pd

from .preprocessing_core import (
    process_trace_column,
    time_windowing,
    baseline_correction,
)


def build_dataset(
    raw_data,
    fs=1000,
    apply_time_window=False,
    start_time=0.0,
    end_time=5.0,
    baseline_end=2.0,
    nperseg=128,
    noverlap=120,
    freq_max=None,
    log_scale=True
):
    """
    Build preprocessed dataset from raw data.

    Args:
        raw_data: DataFrame with 'trace' column (string arrays)
        fs: Sampling frequency (Hz)
        apply_time_window: Whether to crop traces to time window
        start_time, end_time: Time window bounds (seconds)
        baseline_end: End of baseline period for correction (seconds)
        nperseg, noverlap: Unused (spectrograms computed during dataset access)
        freq_max: Unused (spectrograms computed during dataset access)
        log_scale: Unused (spectrograms computed during dataset access)

    Returns:
        DataFrame with parsed and baseline-corrected traces
    """
    dataset = raw_data.copy()

    # Parse trace strings to numpy arrays
    dataset['trace'] = process_trace_column(dataset['trace'])

    # Baseline correction
    dataset['trace'] = baseline_correction(dataset['trace'], fs, baseline_end)

    # Optional time windowing
    if apply_time_window:
        dataset['trace'] = time_windowing(dataset['trace'], fs, start_time, end_time)

    return dataset


def build_trial_sequences(
    df,
    n_trials=8,
    stride=4,
    min_trials=8
):
    """
    Build sequences of consecutive trials per session.

    Args:
        df: DataFrame with 'session', 'condition', 'trial_num', 'trace'
        n_trials: Number of trials per sequence
        stride: Step size between sequences (overlap = n_trials - stride)
        min_trials: Minimum trials required in session

    Returns:
        List of dicts: {'traces': 2D array, 'label': int, 'session': id}
    """
    sequences = []

    for session_id, session_df in df.groupby('session'):
        # Sort by trial number
        session_df = session_df.sort_values('trial_num')

        if len(session_df) < min_trials:
            continue

        # Get condition label (0=WT, 1=FMR1)
        if session_df['condition'].nunique() != 1:
            raise ValueError(f"Session {session_id} has mixed conditions.")
        condition = session_df['condition'].iloc[0]
        label = 1 if condition == 'FMR1' else 0

        # Stack traces
        traces = np.stack(session_df['trace'].values)  # (n_trials_in_session, n_samples)

        # Create sliding window sequences
        for start_idx in range(0, len(traces) - n_trials + 1, stride):
            seq_traces = traces[start_idx:start_idx + n_trials]

            sequences.append({
                'traces': seq_traces,  # (n_trials, n_samples)
                'label': label,
                'session': session_id,
                'start_trial': start_idx
            })

    return sequences


def main():
    import pickle

    df = pd.read_parquet("data/sample_data.parquet")
    print(f"Loaded {len(df)} trials")

    dataset = build_dataset(df)
    print(f"Processed dataset shape: {dataset.shape}")
    print(f"Trace length: {len(dataset['trace'].iloc[0])}")

    sequences = build_trial_sequences(dataset)
    print(f"Created {len(sequences)} sequences")

    # Save as pickle (sequences contain numpy arrays)
    with open("data/sequences.pkl", "wb") as f:
        pickle.dump(sequences, f)
    print("Saved to data/sequences.pkl")


if __name__ == "__main__":
    main()
