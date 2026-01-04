from temporal_vit.data.data_loader import ParquetSequenceDataset


def main():
    dataset = ParquetSequenceDataset.from_parquet(
        ["data/sample_data.parquet"],
        n_trials=4,
        stride=2,
        spectrogram_column="spectrogram",
    )
    if len(dataset) == 0:
        print("No sequences found.")
        return
    specs, label = dataset[0]
    print(f"Sequences: {len(dataset)}")
    print(f"Spec shape: {specs.shape}, dtype: {specs.dtype}, label: {label}")


if __name__ == "__main__":
    main()
