import pandas as pd
import numpy as np


def process_trace_column(trace_column):
    def parse_trace(trace_array):
        parsed_trace = np.fromstring(trace_array.strip('[]'), sep=',')
        return parsed_trace
    processed_trace = trace_column.apply(parse_trace)
    return processed_trace

def time_windowing(trace_column, fs = 1000, start_time = 0.0, end_time = 5.0):
    start_sample = int(start_time * fs)
    end_sample = int(end_time * fs)
    return trace_column.apply(lambda x: x[start_sample:end_sample])

def baseline_correction(trace_column, fs = 1000, baseline_end = 2.0):
    baseline_sample = int(baseline_end * fs)
    baseline_mean = trace_column[:baseline_sample].mean(axis=1)
    return trace_column.apply(lambda x: x - baseline_mean)

def reject_artifacts():
    pass

def compute_spectrogram():
    pass

def export_splits():
    pass
