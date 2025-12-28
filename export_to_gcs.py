from google.cloud import bigquery
import pandas as pd

from session_splits import create_session_splits_df


def export_full_dataset_to_gcs(
    project_id: str,
    dataset_id: str,
    table_id: str,
    bucket_name: str,
    prefix: str = "neural/v1",
    test_size: float = 0.15,
    val_size: float = 0.15,
):
    """
    Query the full dataset from BigQuery, split by session, and export to GCS.

    Note: Requires gcsfs (pandas to_parquet with gs:// paths).
    """
    client = bigquery.Client(project=project_id)
    table = f"`{project_id}.{dataset_id}.{table_id}`"

    query = f"""
    SELECT
        session,
        condition,
        frequency,
        amplitude,
        trial_num,
        trace
    FROM {table}
    """
    df = client.query(query).to_dataframe(create_bqstorage_client=False)

    train_df, val_df, test_df = create_session_splits_df(
        df, test_size=test_size, val_size=val_size
    )

    base = f"gs://{bucket_name}/{prefix}"
    train_df.to_parquet(f"{base}/train.parquet", index=False)
    val_df.to_parquet(f"{base}/val.parquet", index=False)
    test_df.to_parquet(f"{base}/test.parquet", index=False)

    return {
        "train": f"{base}/train.parquet",
        "val": f"{base}/val.parquet",
        "test": f"{base}/test.parquet",
    }
