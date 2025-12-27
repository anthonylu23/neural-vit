from gcs_dataset import GCSTrialSequenceDataset


def main():
    dataset = GCSTrialSequenceDataset(
        ["sample_data.parquet"],
        n_trials=4,
        stride=2,
        use_gcs=False
    )
    if len(dataset) == 0:
        print("No sequences found.")
        return
    specs, label = dataset[0]
    print(f"Sequences: {len(dataset)}")
    print(f"Spec shape: {specs.shape}, dtype: {specs.dtype}, label: {label}")


if __name__ == "__main__":
    main()
