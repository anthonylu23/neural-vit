import pandas as pd
import numpy as np
from scipy.signal import spectrogram, windows
from typing import List, Dict

def parse_trace(trace_array):
    parsed_trace = np.fromstring(trace_array.strip('[]'), sep=',')
    return parsed_trace
    
def process_trace_column(trace_column):
    processed_trace = trace_column.apply(parse_trace)
    return processed_trace

def time_windowing(trace_column, fs = 1000, start_time = 0.0, end_time = 5.0):
    start_sample = int(start_time * fs)
    end_sample = int(end_time * fs)
    return trace_column.apply(lambda x: x[start_sample:end_sample])

def baseline_correction(trace_column, fs=1000, baseline_end=2.0):
    """
    Subtract baseline mean from each trace.
    
    Args:
        trace_column: pandas Series of numpy arrays
        fs: Sampling frequency (Hz)
        baseline_end: End of baseline period (seconds)
    
    Returns:
        pandas Series of baseline-corrected traces
    """
    baseline_sample = int(baseline_end * fs)
    
    def correct_baseline(trace):
        baseline_mean = trace[:baseline_sample].mean()
        return trace - baseline_mean
    
    return trace_column.apply(correct_baseline)

def compute_spectrogram_single(
    trace,
    fs=1000,
    nperseg=128,
    noverlap=120,
    freq_max=None,
    log_scale=True
):
    """
    Compute spectrogram from a single LFP trace.
    
    Args:
        trace: 1D numpy array of LFP samples
        fs: Sampling frequency (Hz)
        nperseg: Samples per FFT segment (frequency resolution = fs/nperseg)
        noverlap: Overlapping samples between segments
        freq_max: Maximum frequency to include (Hz), None for no filtering
        log_scale: Apply log10 transform
        
    Returns:
        spec: 2D array (freq_bins, time_bins)
        freqs: Frequency axis
        times: Time axis
    """
    window = windows.hann(nperseg)
    freqs, times, Sxx = spectrogram(
        trace,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density'
    )
    
    # Crop to relevant frequency range if specified
    if freq_max is not None:
        freq_mask = freqs <= freq_max
        freqs = freqs[freq_mask]
        Sxx = Sxx[freq_mask, :]
    
    # Log transform (converts to dB-like scale)
    if log_scale:
        Sxx = np.log10(Sxx + 1e-10)
    
    return Sxx, freqs, times


def compute_spectrogram_column(
    trace_column,
    fs=1000,
    nperseg=128,
    noverlap=120,
    freq_max=None,
    log_scale=True
):
    """
    Apply spectrogram computation to a pandas Series of traces.
    
    Args:
        trace_column: pandas Series where each element is a 1D numpy array
        fs: Sampling frequency (Hz)
        nperseg: Samples per FFT segment
        noverlap: Overlapping samples between segments
        freq_max: Maximum frequency to include (Hz), None for no filtering
        log_scale: Apply log10 transform
        
    Returns:
        pandas Series of 2D spectrograms
    """
    def _compute(trace):
        spec, _, _ = compute_spectrogram_single(
            trace, 
            fs=fs, 
            nperseg=nperseg, 
            noverlap=noverlap,
            freq_max=freq_max, 
            log_scale=log_scale
        )
        return spec
    
    return trace_column.apply(_compute)

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
        nperseg, noverlap: Spectrogram parameters
        freq_max: Max frequency to keep (Hz)
        log_scale: Apply log transform to spectrograms
    
    Returns:
        DataFrame with 'spectrograms' column added
    """
    dataset = raw_data.copy()
    
    # Parse trace strings to numpy arrays
    dataset['trace'] = process_trace_column(dataset['trace'])

    # Baseline correction
    dataset['trace'] = baseline_correction(dataset['trace'], fs, baseline_end)
    
    # Optional time windowing
    if apply_time_window:
        dataset['trace'] = time_windowing(dataset['trace'], fs, start_time, end_time)
    
    # Compute spectrograms
    dataset['spectrograms'] = compute_spectrogram_column(
        dataset['trace'], fs, nperseg, noverlap, freq_max, log_scale
    )
    
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
        df: DataFrame with 'session', 'condition', 'trial_num', 'spectrograms'
        n_trials: Number of trials per sequence
        stride: Step size between sequences (overlap = n_trials - stride)
        min_trials: Minimum trials required in session
    
    Returns:
        List of dicts: {'spectrograms': 3D array, 'label': int, 'session': id}
    """
    sequences = []
    
    for session_id, session_df in df.groupby('session'):
        # Sort by trial number
        session_df = session_df.sort_values('trial_num')
        
        if len(session_df) < min_trials:
            continue
        
        # Get condition label (0=WT, 1=FMR1)
        condition = session_df['condition'].iloc[0]
        label = 1 if condition == 'FMR1' else 0
        
        # Stack spectrograms
        specs = np.stack(session_df['spectrograms'].values)  # (n_trials_in_session, freq, time)
        
        # Create sliding window sequences
        for start_idx in range(0, len(specs) - n_trials + 1, stride):
            seq_specs = specs[start_idx:start_idx + n_trials]
            
            sequences.append({
                'spectrograms': seq_specs,  # (n_trials, freq, time)
                'label': label,
                'session': session_id,
                'start_trial': start_idx
            })
    
    return sequences

if __name__ == "__main__":
    import pickle
    
    df = pd.read_parquet("sample_data.parquet")
    print(f"Loaded {len(df)} trials")
    
    dataset = build_dataset(df)
    print(f"Processed dataset shape: {dataset.shape}")
    print(f"Spectrogram shape: {dataset['spectrograms'].iloc[0].shape}")
    
    sequences = build_trial_sequences(dataset)
    print(f"Created {len(sequences)} sequences")
    
    # Save as pickle (sequences contain numpy arrays)
    with open("sequences.pkl", "wb") as f:
        pickle.dump(sequences, f)
    print("Saved to sequences.pkl")