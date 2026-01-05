from dataclasses import dataclass, field
from typing import List, Optional

import torch

from temporal_vit.data.data_loader import DataLoaderConfig


@dataclass
class TrainConfig:
    train_paths: List[str]
    val_paths: List[str]
    test_paths: List[str]
    stats_path: Optional[str] = None
    output_dir: Optional[str] = None
    use_preprocessed: bool = False
    spectrogram_column: str = "spectrogram"

    epochs: int = 20
    loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    lr: float = 3e-4
    weight_decay: float = 0.01
    label_smoothing: float = 0.05
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    n_trials: int = 8
    stride: int = 4

    model_size: str = "small"
    freq_size: Optional[int] = None
    time_size: Optional[int] = None
    patch_trial: Optional[int] = None
    patch_freq: Optional[int] = None
    patch_time: Optional[int] = None
    embed_dim: Optional[int] = None
    n_heads: Optional[int] = None
    n_layers: Optional[int] = None
    mlp_ratio: Optional[float] = None
    dropout: float = 0.2
    attention_dropout: float = 0.1
    drop_path: float = 0.1
