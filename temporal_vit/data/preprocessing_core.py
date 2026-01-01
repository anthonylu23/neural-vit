import pandas as pd
import numpy as np
from scipy.signal import spectrogram, windows


def parse_trace(trace_array):
    if isinstance(trace_array, np.ndarray):
        return trace_array
    if isinstance(trace_array, list):
        return np.array(trace_array)
    if isinstance(trace_array, str):
        cleaned = trace_array.strip()
        if cleaned.startswith('[') and cleaned.endswith(']'):
            cleaned = cleaned[1:-1]
        cleaned = cleaned.replace(',', ' ').strip()
        if not cleaned:
            return np.array([])
        return np.fromstring(cleaned, sep=' ')
    return np.array(trace_array)


def process_trace_column(trace_column):
    processed_trace = trace_column.apply(parse_trace)
    return processed_trace


def time_windowing(trace_column, fs=1000, start_time=0.0, end_time=5.0):
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
