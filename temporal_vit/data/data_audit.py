import numpy as np
import pandas as pd
import google.cloud.bigquery as bigquery

def audit_lfp_dataset(auditory_cortex_df):
    """
    Comprehensive audit of LFP dataset.
    """
    report = {}

    # Basic counts
    report['total_trials'] = len(auditory_cortex_df)
    report['total_sessions'] = auditory_cortex_df['session'].nunique()

    # Per-condition session counts
    session_level = (
        auditory_cortex_df.groupby(['session', 'condition'])
        .size()
        .reset_index(name='trials_in_session')
    )
    report['wt_sessions'] = session_level.loc[
        session_level['condition'] == 'WT', 'session'
    ].nunique()
    report['fmr1_sessions'] = session_level.loc[
        session_level['condition'] == 'FMR1', 'session'
    ].nunique()

    # Trials per session distribution
    trials_per_session = auditory_cortex_df.groupby('session')['trial_num'].count()
    report['mean_trials_per_session'] = trials_per_session.mean()
    report['std_trials_per_session'] = trials_per_session.std()
    report['min_trials_per_session'] = trials_per_session.min()
    report['max_trials_per_session'] = trials_per_session.max()

    # Stimulus coverage
    report['n_frequencies'] = auditory_cortex_df['frequency'].nunique()
    report['n_amplitudes'] = auditory_cortex_df['amplitude'].nunique()
    report['n_freq_amp_combos'] = auditory_cortex_df[['frequency', 'amplitude']].drop_duplicates().shape[0]

    # Unique values (for coverage check)
    report['all_frequencies'] = sorted(auditory_cortex_df['frequency'].dropna().unique())
    report['all_amplitudes'] = sorted(auditory_cortex_df['amplitude'].dropna().unique())

    # Trace count quality checks
    def trace_count(value):
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            return len(value), False, False
        if isinstance(value, str):
            truncated = "..." in value
            cleaned = value.strip()
            if cleaned.startswith("[") and cleaned.endswith("]"):
                cleaned = cleaned[1:-1]
            cleaned = cleaned.replace(",", " ").strip()
            if not cleaned:
                return 0, truncated, False
            values = np.fromstring(cleaned, sep=" ")
            if values.size == 0:
                return np.nan, truncated, True
            return int(values.size), truncated, False
        return np.nan, False, True

    counts = []
    truncated_count = 0
    parse_errors = 0
    parsed_count = 0
    valid_counts = []

    for value in auditory_cortex_df.get('trace', []):
        count, truncated, parse_error = trace_count(value)
        counts.append(count)
        if truncated:
            truncated_count += 1
        if parse_error:
            parse_errors += 1
        else:
            parsed_count += 1
            if not truncated and not (isinstance(count, float) and np.isnan(count)):
                valid_counts.append(count)

    expected = None
    if valid_counts:
        count_series = pd.Series(valid_counts)
        expected = int(count_series.mode().iloc[0])

    trace_stats = {
        'total_traces': len(auditory_cortex_df),
        'parsed_traces': parsed_count,
        'expected': expected,
        'min': int(np.min(valid_counts)) if valid_counts else None,
        'max': int(np.max(valid_counts)) if valid_counts else None,
        'mean': float(np.mean(valid_counts)) if valid_counts else None,
        'std': float(np.std(valid_counts)) if valid_counts else None,
        'unique_counts': sorted(set(valid_counts)) if valid_counts else [],
        'n_mismatched': int(sum(1 for c in valid_counts if c != expected)) if expected is not None else None,
        'n_truncated': truncated_count,
        'n_parse_errors': parse_errors
    }
    report['trace_count_quality'] = trace_stats

    return report


def print_audit_report(report, dataset_stats_df=None):
    """Pretty print the audit report and compare sample vs full dataset stats."""
    stats = None
    if isinstance(dataset_stats_df, pd.DataFrame) and not dataset_stats_df.empty:
        stats = dataset_stats_df.iloc[0].to_dict()

    def format_count(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "n/a"
        if isinstance(value, (int, np.integer)):
            return f"{value}"
        if isinstance(value, float) and value.is_integer():
            return f"{int(value)}"
        return f"{value}"

    def format_float(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "n/a"
        return f"{value:.1f}"

    def to_list(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return []
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            return list(value)
        return [value]

    def format_ratio(sample_value, full_value):
        if full_value in (0, None) or (isinstance(full_value, float) and np.isnan(full_value)):
            return "n/a"
        return f"{(sample_value / full_value) * 100:.1f}%"

    def format_list(values, max_items=10):
        if not values:
            return "[]"
        if len(values) <= max_items:
            return str(values)
        return f"{values[:max_items]} +{len(values) - max_items} more"

    print("=" * 60)
    print("LFP SAMPLE DATASET AUDIT REPORT")
    print("=" * 60)

    print("\nSAMPLE DATASET STATS")
    print(f"   Total trials: {format_count(report.get('total_trials'))}")
    print(f"   Total sessions: {format_count(report.get('total_sessions'))}")
    print(f"   WT sessions: {format_count(report.get('wt_sessions'))}")
    print(f"   FMR1 sessions: {format_count(report.get('fmr1_sessions'))}")
    print(f"   Mean trials/session: {format_float(report.get('mean_trials_per_session'))}")
    print(f"   Std trials/session: {format_float(report.get('std_trials_per_session'))}")
    print(f"   Min trials/session: {format_count(report.get('min_trials_per_session'))}")
    print(f"   Max trials/session: {format_count(report.get('max_trials_per_session'))}")
    print(f"   # frequencies: {format_count(report.get('n_frequencies'))}")
    print(f"   # amplitudes: {format_count(report.get('n_amplitudes'))}")
    print(f"   # freq-amp combos: {format_count(report.get('n_freq_amp_combos'))}")

    trace_quality = report.get('trace_count_quality')
    if trace_quality:
        print("\nTRACE COUNT QUALITY")
        print(f"   Expected length: {format_count(trace_quality.get('expected'))}")
        print(f"   Parsed traces: {format_count(trace_quality.get('parsed_traces'))} of "
              f"{format_count(trace_quality.get('total_traces'))}")
        print(f"   Min length: {format_count(trace_quality.get('min'))}")
        print(f"   Max length: {format_count(trace_quality.get('max'))}")
        print(f"   Mean length: {format_float(trace_quality.get('mean'))}")
        print(f"   Std length: {format_float(trace_quality.get('std'))}")
        print(f"   Mismatched lengths: {format_count(trace_quality.get('n_mismatched'))}")
        print(f"   Truncated traces: {format_count(trace_quality.get('n_truncated'))}")
        print(f"   Parse errors: {format_count(trace_quality.get('n_parse_errors'))}")
        print(f"   Unique lengths: {format_list(trace_quality.get('unique_counts', []))}")
    print("=" * 60)

    if stats is None:
        return

    print("\nFULL DATASET STATS (BQ)")
    print(f"   Total trials: {format_count(stats.get('total_trials'))}")
    print(f"   Total sessions: {format_count(stats.get('total_sessions'))}")
    print(f"   WT sessions: {format_count(stats.get('wt_sessions'))}")
    print(f"   FMR1 sessions: {format_count(stats.get('fmr1_sessions'))}")
    print(f"   Mean trials/session: {format_float(stats.get('mean_trials_per_session'))}")
    print(f"   Std trials/session: {format_float(stats.get('std_trials_per_session'))}")
    print(f"   Min trials/session: {format_count(stats.get('min_trials_per_session'))}")
    print(f"   Max trials/session: {format_count(stats.get('max_trials_per_session'))}")
    print(f"   # frequencies: {format_count(stats.get('n_frequencies'))}")
    print(f"   # amplitudes: {format_count(stats.get('n_amplitudes'))}")
    print(f"   # freq-amp combos: {format_count(stats.get('n_freq_amp_combos'))}")

    print("\nCOMPARISON (SAMPLE vs FULL)")
    print(f"   Total trials: {report['total_trials']} vs {format_count(stats.get('total_trials'))}"
          f" ({format_ratio(report['total_trials'], stats.get('total_trials'))} of full)")
    print(f"   Total sessions: {report['total_sessions']} vs {format_count(stats.get('total_sessions'))}"
          f" ({format_ratio(report['total_sessions'], stats.get('total_sessions'))} of full)")

    wt_sample_sessions = report.get('wt_sessions', 0) or 0
    fmr1_sample_sessions = report.get('fmr1_sessions', 0) or 0
    print(f"   WT sessions: {format_count(wt_sample_sessions)} vs {format_count(stats.get('wt_sessions'))}"
          f" ({format_ratio(wt_sample_sessions, stats.get('wt_sessions'))} of full)")
    print(f"   FMR1 sessions: {format_count(fmr1_sample_sessions)} vs {format_count(stats.get('fmr1_sessions'))}"
          f" ({format_ratio(fmr1_sample_sessions, stats.get('fmr1_sessions'))} of full)")

    print("   Trials/session mean: "
          f"{format_float(report.get('mean_trials_per_session'))} vs {format_float(stats.get('mean_trials_per_session'))}")
    print("   Trials/session std:  "
          f"{format_float(report.get('std_trials_per_session'))} vs {format_float(stats.get('std_trials_per_session'))}")
    print("   Trials/session min:  "
          f"{format_float(report.get('min_trials_per_session'))} vs {format_float(stats.get('min_trials_per_session'))}")
    print("   Trials/session max:  "
          f"{format_float(report.get('max_trials_per_session'))} vs {format_float(stats.get('max_trials_per_session'))}")

    sample_freqs = sorted(to_list(report.get('all_frequencies')))
    full_freqs = sorted(to_list(stats.get('all_frequencies')))
    sample_amps = sorted(to_list(report.get('all_amplitudes')))
    full_amps = sorted(to_list(stats.get('all_amplitudes')))

    missing_freqs = sorted(set(full_freqs) - set(sample_freqs))
    extra_freqs = sorted(set(sample_freqs) - set(full_freqs))
    missing_amps = sorted(set(full_amps) - set(sample_amps))
    extra_amps = sorted(set(sample_amps) - set(full_amps))

    print(f"   Frequencies covered: {len(sample_freqs)} of {len(full_freqs)}")
    if missing_freqs:
        print(f"   Missing frequencies in sample: {missing_freqs}")
    if extra_freqs:
        print(f"   Extra frequencies in sample: {extra_freqs}")

    print(f"   Amplitudes covered: {len(sample_amps)} of {len(full_amps)}")
    if missing_amps:
        print(f"   Missing amplitudes in sample: {missing_amps}")
    if extra_amps:
        print(f"   Extra amplitudes in sample: {extra_amps}")

def main():
    from temporal_vit.cloud.get_data import dataset_stats
    dataset_stats_df = dataset_stats(bigquery.Client(project='neural-ds-fe73'))
    sample_data = pd.read_parquet('data/sample_data.parquet')
    report = audit_lfp_dataset(sample_data)
    print_audit_report(report, dataset_stats_df)

if __name__ == '__main__':
    main()
