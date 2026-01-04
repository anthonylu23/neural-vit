from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class DataLoaderConfig:
    batch_size: int = 4
    num_workers: int = 0
    pin_memory: Optional[bool] = None
    persistent_workers: bool = False
    prefetch_factor: int = 2
    drop_last: bool = False
    shuffle_train: bool = True


def _resolve_pin_memory(loader_cfg: DataLoaderConfig, device: str) -> bool:
    if loader_cfg.pin_memory is not None:
        return loader_cfg.pin_memory
    return torch.cuda.is_available() and device.startswith("cuda")


def _loader_kwargs(
    loader_cfg: DataLoaderConfig,
    device: str,
    shuffle: bool,
) -> dict:
    kwargs = {
        "batch_size": loader_cfg.batch_size,
        "shuffle": shuffle,
        "num_workers": loader_cfg.num_workers,
        "pin_memory": _resolve_pin_memory(loader_cfg, device),
        "drop_last": loader_cfg.drop_last,
    }
    if loader_cfg.num_workers > 0:
        kwargs["persistent_workers"] = loader_cfg.persistent_workers
        kwargs["prefetch_factor"] = loader_cfg.prefetch_factor
    return kwargs


def _filesystem_for_paths(paths: Iterable[str]) -> pafs.FileSystem:
    if any(path.startswith("gs://") for path in paths):
        return pafs.GcsFileSystem()
    return pafs.LocalFileSystem()


def _normalize_paths(paths: Iterable[str]) -> List[str]:
    normalized = []
    for path in paths:
        if path.startswith("gs://"):
            normalized.append(path.replace("gs://", "", 1))
        else:
            normalized.append(path)
    return normalized


def _open_dataset(paths: Iterable[str]) -> ds.Dataset:
    paths = list(paths)
    if not paths:
        raise ValueError("paths must contain at least one parquet file.")
    filesystem = _filesystem_for_paths(paths)
    return ds.dataset(_normalize_paths(paths), format="parquet", filesystem=filesystem)


def _to_numpy_spec(value) -> np.ndarray:
    if value is None:
        return np.array([], dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    return arr


class ParquetSequenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        spectrograms: List[np.ndarray],
        *,
        n_trials: int,
        stride: int,
        label_map: Optional[Dict[str, int]] = None,
    ):
        self.df = df
        self.spectrograms = spectrograms
        self.n_trials = n_trials
        self.stride = stride
        self.label_map = label_map or {"FMR1": 1}
        self.sequence_indices, self.sequence_labels = self._build_sequences()

    @classmethod
    def from_parquet(
        cls,
        paths: Iterable[str],
        *,
        n_trials: int,
        stride: int,
        spectrogram_column: str = "spectrogram",
        label_map: Optional[Dict[str, int]] = None,
    ) -> "ParquetSequenceDataset":
        if not spectrogram_column:
            raise ValueError("spectrogram_column must be provided for preprocessed datasets.")
        dataset = _open_dataset(paths)
        available_columns = set(dataset.schema.names)
        columns = ["session", "condition", "trial_num"]
        if spectrogram_column not in available_columns:
            raise ValueError("Preprocessed parquet is missing the spectrogram column.")
        columns.append(spectrogram_column)

        table = dataset.to_table(columns=columns)
        df = table.to_pandas().reset_index(drop=True)
        spectrograms = [_to_numpy_spec(value) for value in df[spectrogram_column]]

        return cls(
            df=df,
            spectrograms=spectrograms,
            n_trials=n_trials,
            stride=stride,
            label_map=label_map,
        )

    def _build_sequences(self) -> Tuple[List[List[int]], List[int]]:
        sequences: List[List[int]] = []
        labels: List[int] = []

        for session_id, group in self.df.groupby("session"):
            group = group.sort_values("trial_num")
            if group["condition"].nunique() != 1:
                raise ValueError(f"Session {session_id} has mixed conditions.")
            condition = group["condition"].iloc[0]
            label = self.label_map.get(condition, 0)
            indices = group.index.tolist()

            if len(indices) < self.n_trials:
                continue

            for i in range(0, len(indices) - self.n_trials + 1, self.stride):
                seq_indices = indices[i:i + self.n_trials]
                if any(self.spectrograms[idx].size == 0 for idx in seq_indices):
                    continue
                sequences.append(seq_indices)
                labels.append(label)

        return sequences, labels

    def __len__(self) -> int:
        return len(self.sequence_indices)

    def __getitem__(self, idx: int):
        trial_indices = self.sequence_indices[idx]
        specs = [self.spectrograms[i] for i in trial_indices]
        specs = np.stack(specs).astype(np.float32)
        label = self.sequence_labels[idx]
        return torch.from_numpy(specs), torch.tensor(label, dtype=torch.long)


class InMemorySequenceDataset(Dataset):
    def __init__(self, sequences: List[dict]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        sequence = self.sequences[idx]
        if "spectrograms" not in sequence:
            raise ValueError("Sequence missing precomputed spectrograms.")
        spectrograms = np.asarray(sequence["spectrograms"], dtype=np.float32)

        label = sequence["label"]
        return torch.from_numpy(spectrograms), torch.tensor(label, dtype=torch.long)


def create_dataloaders(
    train_seqs: List[dict],
    val_seqs: List[dict],
    test_seqs: List[dict],
    *,
    loader_cfg: Optional[DataLoaderConfig] = None,
    device: str = "cpu",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    loader_cfg = loader_cfg or DataLoaderConfig()
    train_dataset = InMemorySequenceDataset(train_seqs)
    val_dataset = InMemorySequenceDataset(val_seqs)
    test_dataset = InMemorySequenceDataset(test_seqs)

    train_loader = DataLoader(train_dataset, **_loader_kwargs(loader_cfg, device, loader_cfg.shuffle_train))
    val_loader = DataLoader(val_dataset, **_loader_kwargs(loader_cfg, device, False))
    test_loader = DataLoader(test_dataset, **_loader_kwargs(loader_cfg, device, False))

    return train_loader, val_loader, test_loader


def build_parquet_dataloaders(
    train_paths: List[str],
    val_paths: List[str],
    test_paths: List[str],
    *,
    n_trials: int,
    stride: int,
    spectrogram_column: str,
    loader_cfg: DataLoaderConfig,
    device: str,
    label_map: Optional[Dict[str, int]] = None,
) -> Tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    Tuple[ParquetSequenceDataset, ParquetSequenceDataset, ParquetSequenceDataset],
]:
    train_ds = ParquetSequenceDataset.from_parquet(
        train_paths,
        n_trials=n_trials,
        stride=stride,
        spectrogram_column=spectrogram_column,
        label_map=label_map,
    )
    val_ds = ParquetSequenceDataset.from_parquet(
        val_paths,
        n_trials=n_trials,
        stride=stride,
        spectrogram_column=spectrogram_column,
        label_map=label_map,
    )
    test_ds = ParquetSequenceDataset.from_parquet(
        test_paths,
        n_trials=n_trials,
        stride=stride,
        spectrogram_column=spectrogram_column,
        label_map=label_map,
    )

    train_loader = DataLoader(train_ds, **_loader_kwargs(loader_cfg, device, loader_cfg.shuffle_train))
    val_loader = DataLoader(val_ds, **_loader_kwargs(loader_cfg, device, False))
    test_loader = DataLoader(test_ds, **_loader_kwargs(loader_cfg, device, False))

    return train_loader, val_loader, test_loader, (train_ds, val_ds, test_ds)
