import numpy as np
import pandas as pd
from collections import defaultdict
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
    from get_data import dataset_stats
    dataset_stats_df = dataset_stats(bigquery.Client(project='neural-ds-fe73'))
    sample_data = pd.read_parquet('sample_data.parquet')
    report = audit_lfp_dataset(sample_data)
    print_audit_report(report, dataset_stats_df)

if __name__ == '__main__':
    main()
