from google.cloud import bigquery
import pandas as pd
import os
from session_splits import create_session_splits_df
from dotenv import load_dotenv


def export_full_dataset_to_gcs(
    bq_project_id: str,
    bq_dataset_id: str,
    bq_table_id: str,
    gcs_bucket: str,
    gcs_prefix: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
):
    """
    Query the full dataset from BigQuery, split by session, and export to GCS.


    Note: Requires gcsfs (pandas to_parquet with gs:// paths).
    """
    client = bigquery.Client(project=bq_project_id)
    table = f"`{bq_project_id}.{bq_dataset_id}.{bq_table_id}`"

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

    base = f"gs://{gcs_bucket}/{gcs_prefix}"
    train_df.to_parquet(f"{base}/train.parquet", index=False)
    val_df.to_parquet(f"{base}/val.parquet", index=False)
    test_df.to_parquet(f"{base}/test.parquet", index=False)

    return {
        "train": f"{base}/train.parquet",
        "val": f"{base}/val.parquet",
        "test": f"{base}/test.parquet",
    }

if __name__ == "__main__":
    export_full_dataset_to_gcs(
        bq_project_id="neural-ds-fe73",
        bq_dataset_id="lab6_mouse_lfp",
        bq_table_id="auditory_cortex",
        gcs_bucket="lfp_spec_datasets",
        gcs_prefix="neural/v1",
    )