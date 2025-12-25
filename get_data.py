from google.cloud import bigquery
from tqdm import tqdm

# For local auth, run: gcloud auth application-default login
print("Using Application Default Credentials for authentication...")

def data_size(client):
    size_query = """
    SELECT 
        COUNT(*) as row_count,
        SUM(LENGTH(trace)) as total_trace_chars
    FROM `neural-ds-fe73.lab6_mouse_lfp.auditory_cortex`
    """
    size_df = client.query(size_query).to_dataframe(create_bqstorage_client=False)
    return size_df

def get_data(client):
    query = """
    select
      session,
      condition,
      frequency,
      amplitude,
      trial_num,
      array(select safe_cast(x as float64) from unnest(split(trace, ',')) as x) as trace
    from `neural-ds-fe73.lab6_mouse_lfp.auditory_cortex`
    """
    
    auditory_cortex = client.query(query).to_dataframe(
        create_bqstorage_client=False,
        progress_bar_type='tqdm'
    )
    return auditory_cortex

def main():
    client = bigquery.Client(project='neural-ds-fe73')
    data_df = get_data(client)
    # Save to CSV
    data_df.to_csv('auditory_cortex_data.csv', index=False)
    print(f"Data saved to auditory_cortex_data.csv ({len(data_df)} rows)")
    print(data_df.head())

if __name__ == "__main__":
    main()
