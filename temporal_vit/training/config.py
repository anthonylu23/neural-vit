from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class TrainConfig:
    train_paths: List[str]
    val_paths: List[str]
    test_paths: List[str]
    stats_path: Optional[str] = None
    output_dir: Optional[str] = None
    use_preprocessed: bool = False
    spectrogram_column: str = "spectrogram"

    epochs: int = 10
    batch_size: int = 16
    num_workers: int = 4
    lr: float = 3e-4
    weight_decay: float = 0.01
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    n_trials: int = 8
    stride: int = 4
    baseline_end: float = 2.0
    fs: int = 1000

    nperseg: int = 128
    noverlap: int = 120
    freq_max: Optional[float] = None
    log_scale: bool = True

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
    dropout: float = 0.1
    attention_dropout: float = 0.1
    drop_path: float = 0.1
