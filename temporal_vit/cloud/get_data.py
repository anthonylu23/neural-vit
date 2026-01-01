from google.cloud import bigquery
# For local auth, run: gcloud auth application-default login
print("Using Application Default Credentials for authentication...")

def dataset_stats(client):
    """
    Get comprehensive statistics about the dataset including size and sample statistics.
    Returns a pandas DataFrame with dataset metrics.
    """
    stats_query = """
    WITH session_level AS (
        SELECT 
            session,
            condition,
            COUNT(*) as trials_in_session
        FROM `neural-ds-fe73.lab6_mouse_lfp.auditory_cortex`
        GROUP BY session, condition
    ),
    stimulus_combos AS (
        SELECT DISTINCT frequency, amplitude
        FROM `neural-ds-fe73.lab6_mouse_lfp.auditory_cortex`
    )
    SELECT 
        -- Counts
        (SELECT COUNT(*) FROM `neural-ds-fe73.lab6_mouse_lfp.auditory_cortex`) as total_trials,
        (SELECT COUNT(DISTINCT session) FROM `neural-ds-fe73.lab6_mouse_lfp.auditory_cortex`) as total_sessions,
        
        -- Condition breakdown
        (SELECT COUNT(*) FROM session_level WHERE condition = 'WT') as wt_sessions,
        (SELECT COUNT(*) FROM session_level WHERE condition = 'FMR1') as fmr1_sessions,
        
        -- Trials per session distribution
        (SELECT AVG(trials_in_session) FROM session_level) as mean_trials_per_session,
        (SELECT STDDEV(trials_in_session) FROM session_level) as std_trials_per_session,
        (SELECT MIN(trials_in_session) FROM session_level) as min_trials_per_session,
        (SELECT MAX(trials_in_session) FROM session_level) as max_trials_per_session,
        
        -- Stimulus coverage
        (SELECT COUNT(DISTINCT frequency) FROM `neural-ds-fe73.lab6_mouse_lfp.auditory_cortex`) as n_frequencies,
        (SELECT COUNT(DISTINCT amplitude) FROM `neural-ds-fe73.lab6_mouse_lfp.auditory_cortex`) as n_amplitudes,
        (SELECT COUNT(*) FROM stimulus_combos) as n_freq_amp_combos,
        
        -- Unique values (for coverage check)
        (SELECT ARRAY_AGG(DISTINCT frequency ORDER BY frequency) FROM `neural-ds-fe73.lab6_mouse_lfp.auditory_cortex`) as all_frequencies,
        (SELECT ARRAY_AGG(DISTINCT amplitude ORDER BY amplitude) FROM `neural-ds-fe73.lab6_mouse_lfp.auditory_cortex`) as all_amplitudes
    """
    details_df = client.query(stats_query).to_dataframe(create_bqstorage_client=False)
    return details_df

def get_stratified_sample(client, sample_fraction=0.01):
    """
    Get stratified sample maintaining session integrity and condition balance.
    """
    query = f"""
    WITH session_info AS (
        SELECT 
            session,
            condition,
            ROW_NUMBER() OVER (
                PARTITION BY condition 
                ORDER BY FARM_FINGERPRINT(CAST(session AS STRING))
            ) as rank_in_condition,
            COUNT(*) OVER (PARTITION BY condition) as total_in_condition
        FROM `neural-ds-fe73.lab6_mouse_lfp.auditory_cortex`
        GROUP BY session, condition
    ),
    sampled_sessions AS (
        SELECT session
        FROM session_info
        WHERE rank_in_condition <= CEIL(total_in_condition * {sample_fraction})
    )
    SELECT 
        ac.session,
        ac.condition,
        ac.frequency,
        ac.amplitude,
        ac.trial_num,
        ac.trace
    FROM `neural-ds-fe73.lab6_mouse_lfp.auditory_cortex` ac
    INNER JOIN sampled_sessions ss ON ac.session = ss.session
    """
    
    return client.query(query).to_dataframe(
        create_bqstorage_client=False,
        progress_bar_type='tqdm'
    )

def main():
    client = bigquery.Client(project='neural-ds-fe73')
    # sample_data = get_stratified_sample(client, sample_fraction=0.01)
    # sample_data.to_parquet('data/sample_data.parquet')
    dataset_stats_df = dataset_stats(client)
    print(dataset_stats_df)

if __name__ == "__main__":
    main()   
