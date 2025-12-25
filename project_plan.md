# 📋 Full Project Plans

---

# Project A: Temporal 3D Vision Transformer for Multi-Trial LFP Analysis

## Executive Summary

**Goal**: Build a Vision Transformer that processes sequences of LFP spectrograms across trials to classify WT vs FMR1 knockout mice, capturing trial-to-trial dynamics that single-trial models miss.

**Dataset**: Mouse auditory cortex LFP from `lab6/8` (WT vs FMR1, ~40 sessions)

**Timeline**: 8-10 weeks

**Key Innovation**: 3D patch embedding that captures both within-trial spectral patterns and cross-trial variability signatures characteristic of Fragile X syndrome.

---

## Cloud Infrastructure Overview

Two workflow options are supported. Choose based on your needs:

| Aspect           | Option A: Vertex AI + GCS        | Option B: BigQuery + SSH (Hyperbolic) |
| ---------------- | -------------------------------- | ------------------------------------- |
| **Data storage** | Pre-exported Parquet in GCS      | Query BigQuery directly               |
| **Compute**      | Managed containers               | Bare-metal SSH access                 |
| **Setup**        | Dockerfile + job submission      | SSH + conda/pip environment           |
| **Cost**         | Higher (GCP markup)              | Often 50-70% cheaper                  |
| **Flexibility**  | Constrained to job API           | Full control, interactive             |
| **Best for**     | Production, parallel experiments | Iteration, debugging, cost savings    |

---

### Option A: Vertex AI + GCS (Managed)

#### Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐
│  BigQuery   │────▶│    GCS      │────▶│  Vertex AI Training │
│ (raw data)  │     │ (parquet)   │     │  (PyTorch container)│
└─────────────┘     └─────────────┘     └──────────┬──────────┘
                                                   │
                    ┌─────────────┐                │
                    │    GCS      │◀───────────────┘
                    │ (checkpoints,│   writes model artifacts
                    │  metrics)    │
                    └─────────────┘
```

#### GCS Data Layout

```
gs://bucket/neural/
├── v1/
│   ├── metadata.json          # Session→split mapping, version info
│   ├── train-*.parquet        # Sharded by session
│   ├── val-*.parquet
│   ├── test-*.parquet
│   └── eda-sample.parquet     # Small subset for local EDA
├── checkpoints/
│   └── {experiment_id}/
│       ├── best.pt
│       └── last.pt
└── metrics/
    └── {experiment_id}/
        └── eval_results.json
```

#### Workflow Summary

1. **BigQuery → GCS**: Session-aware splits computed in BQ, exported as sharded Parquet
2. **Local EDA**: Pull small subset for auditing and parameter tuning
3. **Vertex AI Training**: PyTorch container reads from GCS URIs
4. **Evaluation**: Metrics and predictions written back to GCS/BigQuery

---

### Option B: BigQuery Direct + SSH (Hyperbolic Labs / Lambda Labs)

#### Architecture

```
┌─────────────┐                      ┌────────────────────────┐
│  BigQuery   │◀────── queries ──────│  GPU Cloud (SSH)       │
│ (all data)  │                      │  Hyperbolic / Lambda   │
└─────────────┘                      └───────────┬────────────┘
                                                 │
                                     ┌───────────▼────────────┐
                                     │  Local disk / GCS      │
                                     │  (checkpoints, cache)  │
                                     └────────────────────────┘
```

#### Local Cache Layout (on GPU instance)

```
~/neural-project/
├── cache/
│   ├── train.parquet          # Cached after first BQ query
│   ├── val.parquet
│   └── test.parquet
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── src/
│   ├── model.py
│   ├── dataset.py
│   └── augmentation.py
└── train_ssh.py
```

#### Workflow Summary

1. **SSH Setup**: Connect to GPU instance, set up conda environment
2. **First Run**: Query BigQuery, cache data locally as Parquet
3. **Training**: Run training script directly (cached data, fast iteration)
4. **Sync**: Pull checkpoints via scp/rsync, optionally push to GCS

---

## Phase 1: Data Preparation & Exploration (Weeks 1-2)

### Week 1: Data Audit & Preprocessing Pipeline

#### Tasks

| #   | Task                       | Description                                  | Output              |
| --- | -------------------------- | -------------------------------------------- | ------------------- |
| 1.1 | **Data inventory**         | Count sessions, trials, conditions per group | Summary table       |
| 1.2 | **Quality control**        | Identify corrupted/missing trials, outliers  | QC report           |
| 1.3 | **Preprocessing pipeline** | Baseline subtraction, artifact rejection     | Clean dataset       |
| 1.4 | **Spectrogram parameters** | Test different window sizes, overlaps        | Parameter selection |

#### Code: Data Audit Script

```python
import numpy as np
import pandas as pd
from collections import defaultdict

def audit_lfp_dataset(auditory_cortex_df):
    """
    Comprehensive audit of LFP dataset.
    """
    report = {}

    # Basic counts
    report['total_trials'] = len(auditory_cortex_df)
    report['total_sessions'] = auditory_cortex_df['session'].nunique()

    # Per-condition breakdown
    condition_stats = auditory_cortex_df.groupby('condition').agg({
        'session': 'nunique',
        'trial_num': 'count'
    }).rename(columns={'session': 'n_sessions', 'trial_num': 'n_trials'})
    report['condition_breakdown'] = condition_stats

    # Trials per session
    trials_per_session = auditory_cortex_df.groupby('session')['trial_num'].count()
    report['trials_per_session'] = {
        'mean': trials_per_session.mean(),
        'min': trials_per_session.min(),
        'max': trials_per_session.max(),
        'std': trials_per_session.std()
    }

    # Stimulus conditions
    report['unique_frequencies'] = sorted(auditory_cortex_df['frequency'].unique())
    report['unique_amplitudes'] = sorted(auditory_cortex_df['amplitude'].unique())

    # Trace statistics
    trace_lengths = [len(row['trace']) for _, row in auditory_cortex_df.iterrows()]
    report['trace_length'] = {
        'expected': 5001,
        'actual_unique': list(set(trace_lengths)),
        'all_correct': len(set(trace_lengths)) == 1 and trace_lengths[0] == 5001
    }

    # Check for NaN/Inf
    n_nan = 0
    n_inf = 0
    for _, row in auditory_cortex_df.iterrows():
        trace = np.array(row['trace'])
        n_nan += np.isnan(trace).sum()
        n_inf += np.isinf(trace).sum()
    report['data_quality'] = {'n_nan': n_nan, 'n_inf': n_inf}

    return report


def print_audit_report(report):
    """Pretty print the audit report."""
    print("=" * 60)
    print("LFP DATASET AUDIT REPORT")
    print("=" * 60)

    print(f"\n📊 DATASET SIZE")
    print(f"   Total trials: {report['total_trials']}")
    print(f"   Total sessions: {report['total_sessions']}")

    print(f"\n📈 CONDITION BREAKDOWN")
    print(report['condition_breakdown'].to_string())

    print(f"\n📉 TRIALS PER SESSION")
    for k, v in report['trials_per_session'].items():
        print(f"   {k}: {v:.1f}")

    print(f"\n🎵 STIMULUS CONDITIONS")
    print(f"   Frequencies: {report['unique_frequencies']}")
    print(f"   Amplitudes: {report['unique_amplitudes']}")

    print(f"\n✅ DATA QUALITY")
    print(f"   Trace length OK: {report['trace_length']['all_correct']}")
    print(f"   NaN values: {report['data_quality']['n_nan']}")
    print(f"   Inf values: {report['data_quality']['n_inf']}")
    print("=" * 60)
```

#### Code: Spectrogram Parameter Search

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, windows

def compare_spectrogram_params(trace, fs=1000):
    """
    Compare different spectrogram parameters.
    """
    param_sets = [
        {'nperseg': 64, 'noverlap': 56, 'name': 'High time res'},
        {'nperseg': 128, 'noverlap': 120, 'name': 'Balanced'},
        {'nperseg': 256, 'noverlap': 250, 'name': 'High freq res'},
        {'nperseg': 512, 'noverlap': 500, 'name': 'Very high freq res'},
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, params in zip(axes, param_sets):
        window = windows.hann(params['nperseg'])
        f, t, Sxx = spectrogram(
            trace, fs=fs,
            window=window,
            nperseg=params['nperseg'],
            noverlap=params['noverlap']
        )

        # Log scale
        Sxx_log = 10 * np.log10(Sxx + 1e-10)

        # Plot
        im = ax.pcolormesh(t, f, Sxx_log, shading='gouraud', cmap='viridis')
        ax.set_ylim([0, 100])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f"{params['name']}\nnperseg={params['nperseg']}, noverlap={params['noverlap']}")
        ax.axvline(2.0, color='red', linestyle='--', label='Stimulus onset')
        plt.colorbar(im, ax=ax, label='Power (dB)')

    plt.tight_layout()
    plt.savefig('spectrogram_comparison.png', dpi=150)
    plt.show()

    return fig
```

### Week 2: Exploratory Data Analysis

#### Tasks

| #   | Task                  | Description                              | Output               |
| --- | --------------------- | ---------------------------------------- | -------------------- |
| 2.1 | **Group differences** | Compare WT vs FMR1 spectrograms visually | Figure gallery       |
| 2.2 | **Trial variability** | Quantify within-session variance         | Variability metrics  |
| 2.3 | **Baseline model**    | Single-trial 2D ViT baseline             | Baseline accuracy    |
| 2.4 | **Sequence analysis** | Analyze trial-to-trial correlations      | Correlation matrices |

#### Code: Variability Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, windows
from scipy.stats import pearsonr

def analyze_trial_variability(session_df, fs=1000):
    """
    Analyze trial-to-trial variability within a session.
    """
    condition = session_df['condition'].iloc[0]
    session_id = session_df['session'].iloc[0]

    # Compute spectrograms for all trials
    spectrograms = []
    for _, row in session_df.iterrows():
        trace = np.array(row['trace'])
        window = windows.hann(128)
        f, t, Sxx = spectrogram(trace, fs=fs, window=window, nperseg=128, noverlap=120)

        # Focus on evoked response window (2-4s) and low frequencies (1-80 Hz)
        time_mask = (t >= 2) & (t <= 4)
        freq_mask = (f >= 1) & (f <= 80)
        Sxx_crop = np.log10(Sxx[freq_mask][:, time_mask] + 1e-10)
        spectrograms.append(Sxx_crop)

    spectrograms = np.stack(spectrograms)  # (n_trials, freq, time)

    # Compute variability metrics
    mean_spec = spectrograms.mean(axis=0)
    std_spec = spectrograms.std(axis=0)
    cv_spec = std_spec / (np.abs(mean_spec) + 1e-10)  # Coefficient of variation

    # Trial-to-trial correlation
    n_trials = len(spectrograms)
    corr_matrix = np.zeros((n_trials, n_trials))
    for i in range(n_trials):
        for j in range(n_trials):
            corr_matrix[i, j] = pearsonr(
                spectrograms[i].flatten(),
                spectrograms[j].flatten()
            )[0]

    metrics = {
        'session': session_id,
        'condition': condition,
        'n_trials': n_trials,
        'mean_cv': cv_spec.mean(),
        'median_cv': np.median(cv_spec),
        'mean_inter_trial_corr': corr_matrix[np.triu_indices(n_trials, k=1)].mean(),
        'spectrograms': spectrograms,
        'corr_matrix': corr_matrix
    }

    return metrics


def compare_variability_by_condition(auditory_cortex_df):
    """
    Compare trial variability between WT and FMR1.
    """
    wt_metrics = []
    fmr1_metrics = []

    for session_id, session_df in auditory_cortex_df.groupby('session'):
        metrics = analyze_trial_variability(session_df)

        if metrics['condition'] == 'WT':
            wt_metrics.append(metrics)
        else:
            fmr1_metrics.append(metrics)

    # Summary statistics
    print("\n" + "="*50)
    print("TRIAL VARIABILITY COMPARISON")
    print("="*50)

    wt_cv = [m['mean_cv'] for m in wt_metrics]
    fmr1_cv = [m['mean_cv'] for m in fmr1_metrics]

    wt_corr = [m['mean_inter_trial_corr'] for m in wt_metrics]
    fmr1_corr = [m['mean_inter_trial_corr'] for m in fmr1_metrics]

    print(f"\nCoefficient of Variation (higher = more variable):")
    print(f"  WT:   {np.mean(wt_cv):.3f} ± {np.std(wt_cv):.3f}")
    print(f"  FMR1: {np.mean(fmr1_cv):.3f} ± {np.std(fmr1_cv):.3f}")

    print(f"\nInter-trial Correlation (lower = more variable):")
    print(f"  WT:   {np.mean(wt_corr):.3f} ± {np.std(wt_corr):.3f}")
    print(f"  FMR1: {np.mean(fmr1_corr):.3f} ± {np.std(fmr1_corr):.3f}")

    # Statistical test
    from scipy.stats import mannwhitneyu
    stat_cv, p_cv = mannwhitneyu(wt_cv, fmr1_cv)
    stat_corr, p_corr = mannwhitneyu(wt_corr, fmr1_corr)

    print(f"\nMann-Whitney U test:")
    print(f"  CV: U={stat_cv:.1f}, p={p_cv:.4f}")
    print(f"  Corr: U={stat_corr:.1f}, p={p_corr:.4f}")

    return wt_metrics, fmr1_metrics
```

---

## Phase 2: Model Development (Weeks 3-5)

### Week 3: Architecture Implementation

#### Tasks

| #   | Task                    | Description                                  | Output           |
| --- | ----------------------- | -------------------------------------------- | ---------------- |
| 3.1 | **Core architecture**   | Implement Temporal3DViT class                | Model code       |
| 3.2 | **Position embeddings** | Implement factorized 3D positional encodings | Embedding module |
| 3.3 | **Unit tests**          | Test forward pass, gradient flow             | Test suite       |
| 3.4 | **Memory profiling**    | Optimize for GPU memory                      | Memory report    |

#### Architecture Configuration

```python
from dataclasses import dataclass
from typing import Tuple

@dataclass
class Temporal3DViTConfig:
    """Configuration for Temporal 3D ViT."""

    # Input dimensions
    n_trials: int = 8                    # Trials per sequence
    freq_size: int = 64                  # Frequency bins
    time_size: int = 128                 # Time bins

    # Patch dimensions
    patch_trial: int = 2                 # Patch size in trial dim
    patch_freq: int = 8                  # Patch size in freq dim
    patch_time: int = 8                  # Patch size in time dim

    # Model dimensions
    embed_dim: int = 384                 # Hidden dimension
    n_heads: int = 6                     # Attention heads
    n_layers: int = 8                    # Transformer layers
    mlp_ratio: float = 4.0               # FFN expansion ratio

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    drop_path: float = 0.1               # Stochastic depth

    # Output
    n_classes: int = 2                   # WT vs FMR1

    # Training
    layer_scale_init: float = 1e-4       # LayerScale initialization

    @property
    def n_patches(self) -> int:
        return (
            (self.n_trials // self.patch_trial) *
            (self.freq_size // self.patch_freq) *
            (self.time_size // self.patch_time)
        )

    @property
    def patch_dim(self) -> int:
        return self.patch_trial * self.patch_freq * self.patch_time


# Model variants
CONFIGS = {
    'tiny': Temporal3DViTConfig(embed_dim=192, n_heads=3, n_layers=4),
    'small': Temporal3DViTConfig(embed_dim=384, n_heads=6, n_layers=8),
    'base': Temporal3DViTConfig(embed_dim=512, n_heads=8, n_layers=12),
}
```

#### Full Model Implementation with Modern Techniques

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional
import math


class DropPath(nn.Module):
    """Stochastic Depth (drop path) regularization."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class LayerScale(nn.Module):
    """Layer Scale for improved training stability."""

    def __init__(self, dim: int, init_value: float = 1e-4):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class Attention(nn.Module):
    """Multi-head self-attention with optional relative position bias."""

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    """Feed-forward network."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with LayerScale and Stochastic Depth."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: float = 4.,
        dropout: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        layer_scale_init: float = 1e-4
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, n_heads, attn_drop=attn_drop, proj_drop=dropout)
        self.ls1 = LayerScale(dim, layer_scale_init) if layer_scale_init > 0 else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=dropout)
        self.ls2 = LayerScale(dim, layer_scale_init) if layer_scale_init > 0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class Temporal3DViT(nn.Module):
    """
    Full Temporal 3D Vision Transformer implementation.
    """

    def __init__(self, config: Temporal3DViTConfig):
        super().__init__()
        self.config = config

        # Patch embedding via 3D convolution
        self.patch_embed = nn.Conv3d(
            in_channels=1,
            out_channels=config.embed_dim,
            kernel_size=(config.patch_trial, config.patch_freq, config.patch_time),
            stride=(config.patch_trial, config.patch_freq, config.patch_time)
        )

        # Calculate patch grid dimensions
        self.n_patches_k = config.n_trials // config.patch_trial
        self.n_patches_f = config.freq_size // config.patch_freq
        self.n_patches_t = config.time_size // config.patch_time

        # Factorized positional embeddings
        self.pos_embed_k = nn.Parameter(
            torch.zeros(1, self.n_patches_k, config.embed_dim)
        )
        self.pos_embed_f = nn.Parameter(
            torch.zeros(1, self.n_patches_f, config.embed_dim)
        )
        self.pos_embed_t = nn.Parameter(
            torch.zeros(1, self.n_patches_t, config.embed_dim)
        )

        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))

        # Dropout after embedding
        self.pos_drop = nn.Dropout(config.dropout)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, config.drop_path, config.n_layers)]

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=config.embed_dim,
                n_heads=config.n_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                attn_drop=config.attention_dropout,
                drop_path=dpr[i],
                layer_scale_init=config.layer_scale_init
            )
            for i in range(config.n_layers)
        ])

        # Final norm
        self.norm = nn.LayerNorm(config.embed_dim)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim, config.n_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Positional embeddings
        nn.init.trunc_normal_(self.pos_embed_k, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_f, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_t, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Linear layers
        self.apply(self._init_module)

    def _init_module(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _get_pos_embed(self):
        """Compute factorized 3D positional embeddings."""
        # Broadcast sum: (K, F, T) -> (K*F*T,)
        pos = (
            self.pos_embed_k.unsqueeze(2).unsqueeze(3) +  # (1, K, 1, 1, D)
            self.pos_embed_f.unsqueeze(1).unsqueeze(3) +  # (1, 1, F, 1, D)
            self.pos_embed_t.unsqueeze(1).unsqueeze(2)    # (1, 1, 1, T, D)
        )  # (1, K, F, T, D)

        return pos.reshape(1, -1, self.config.embed_dim)  # (1, K*F*T, D)

    def forward(self, x):
        """
        Args:
            x: (B, K, F, T) or (B, 1, K, F, T)
        Returns:
            logits: (B, n_classes)
        """
        if x.dim() == 4:
            x = x.unsqueeze(1)  # Add channel dim

        B = x.shape[0]

        # Patch embed: (B, 1, K, F, T) -> (B, D, K', F', T')
        x = self.patch_embed(x)

        # Flatten spatial dims: (B, D, K', F', T') -> (B, N, D)
        x = x.flatten(2).transpose(1, 2)

        # Add positional embeddings
        x = x + self._get_pos_embed()

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Dropout
        x = self.pos_drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = self.norm(x)

        # Classification from CLS token
        return self.head(x[:, 0])

    def get_attention_maps(self, x):
        """Extract attention maps for interpretability."""
        if x.dim() == 4:
            x = x.unsqueeze(1)

        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self._get_pos_embed()
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_drop(x)

        attention_maps = []
        for block in self.blocks:
            # Store attention before applying block
            with torch.no_grad():
                attn = block.attn
                B, N, C = x.shape
                qkv = attn.qkv(block.norm1(x)).reshape(B, N, 3, attn.n_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                attn_weights = (q @ k.transpose(-2, -1)) * attn.scale
                attn_weights = attn_weights.softmax(dim=-1)
                attention_maps.append(attn_weights)
            x = block(x)

        return attention_maps
```

### Week 4: Data Pipeline

#### Tasks

| #   | Task                        | Description                               | Output                |
| --- | --------------------------- | ----------------------------------------- | --------------------- |
| 4.1 | **BigQuery split & export** | Session-aware splits in BQ, export to GCS | Sharded Parquet files |
| 4.2 | **Dataset class**           | PyTorch Dataset streaming from GCS        | Dataset code          |
| 4.3 | **Data augmentation**       | Time shift, freq mask, noise injection    | Augmentation pipeline |
| 4.4 | **EDA subset extraction**   | Pull small sample for local auditing      | EDA notebook          |

#### Data Strategy: Raw Traces vs Pre-computed Spectrograms

Two approaches for storing data in GCS:

| Approach                      | Pros                                                     | Cons                             |
| ----------------------------- | -------------------------------------------------------- | -------------------------------- |
| **Raw LFP traces**            | Flexible spectrogram params, can tune during experiments | Compute overhead during training |
| **Pre-computed spectrograms** | Faster training iteration, less GPU memory pressure      | Locked into specific params      |

**Recommended**: Store raw traces in Parquet, compute spectrograms on-the-fly during training. This allows parameter tuning (nperseg, noverlap) without re-exporting data. Cache spectrograms locally on Vertex AI worker disk if needed.

#### BigQuery Session-Aware Split Strategy

Session-aware splitting must happen **in BigQuery before export** to ensure all trials from a session stay together:

```sql
-- Step 1: Create session-level split assignments
CREATE OR REPLACE TABLE `project.dataset.session_splits` AS
WITH session_conditions AS (
  SELECT DISTINCT
    session,
    condition,
    -- Deterministic hash for reproducibility
    MOD(ABS(FARM_FINGERPRINT(CONCAT(session, '_seed42'))), 100) AS hash_val
  FROM `project.dataset.lfp_trials`
),
stratified_splits AS (
  SELECT
    session,
    condition,
    hash_val,
    -- Stratified split: ~20% test, ~16% val, ~64% train
    -- Split thresholds applied per-condition for balance
    CASE
      WHEN hash_val < 20 THEN 'test'
      WHEN hash_val < 36 THEN 'val'
      ELSE 'train'
    END AS split
  FROM session_conditions
)
SELECT * FROM stratified_splits;

-- Verify split balance
SELECT
  split,
  condition,
  COUNT(*) as n_sessions
FROM `project.dataset.session_splits`
GROUP BY split, condition
ORDER BY split, condition;
```

```sql
-- Step 2: Export each split to GCS as sharded Parquet
-- Shard by session to keep complete sessions in single files

-- Export TRAIN split
EXPORT DATA OPTIONS (
  uri = 'gs://bucket/neural/v1/train-*.parquet',
  format = 'PARQUET',
  overwrite = true
) AS
SELECT t.*
FROM `project.dataset.lfp_trials` t
JOIN `project.dataset.session_splits` s USING (session)
WHERE s.split = 'train'
ORDER BY session, trial_num;  -- Keep trials ordered within shards

-- Export VAL split
EXPORT DATA OPTIONS (
  uri = 'gs://bucket/neural/v1/val-*.parquet',
  format = 'PARQUET',
  overwrite = true
) AS
SELECT t.*
FROM `project.dataset.lfp_trials` t
JOIN `project.dataset.session_splits` s USING (session)
WHERE s.split = 'val'
ORDER BY session, trial_num;

-- Export TEST split
EXPORT DATA OPTIONS (
  uri = 'gs://bucket/neural/v1/test-*.parquet',
  format = 'PARQUET',
  overwrite = true
) AS
SELECT t.*
FROM `project.dataset.lfp_trials` t
JOIN `project.dataset.session_splits` s USING (session)
WHERE s.split = 'test'
ORDER BY session, trial_num;
```

```sql
-- Step 3: Export small EDA sample for local development
EXPORT DATA OPTIONS (
  uri = 'gs://bucket/neural/v1/eda-sample.parquet',
  format = 'PARQUET',
  overwrite = true
) AS
SELECT t.*
FROM `project.dataset.lfp_trials` t
JOIN `project.dataset.session_splits` s USING (session)
WHERE s.split = 'train'
  AND s.session IN (
    SELECT session FROM `project.dataset.session_splits`
    WHERE split = 'train'
    ORDER BY session
    LIMIT 5  -- 5 sessions for EDA
  );
```

#### Export Metadata to GCS

```python
import json
from google.cloud import bigquery, storage

def export_split_metadata(project_id, dataset_id, bucket_name, version='v1'):
    """Export split metadata for reproducibility tracking."""

    client = bigquery.Client(project=project_id)

    # Query split assignments
    query = f"""
    SELECT session, condition, split
    FROM `{project_id}.{dataset_id}.session_splits`
    ORDER BY session
    """
    df = client.query(query).to_dataframe()

    metadata = {
        'version': version,
        'created_at': datetime.utcnow().isoformat(),
        'split_counts': df.groupby(['split', 'condition']).size().to_dict(),
        'session_assignments': df.set_index('session')['split'].to_dict(),
        'total_sessions': len(df),
        'gcs_paths': {
            'train': f'gs://{bucket_name}/neural/{version}/train-*.parquet',
            'val': f'gs://{bucket_name}/neural/{version}/val-*.parquet',
            'test': f'gs://{bucket_name}/neural/{version}/test-*.parquet',
        }
    }

    # Upload to GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f'neural/{version}/metadata.json')
    blob.upload_from_string(json.dumps(metadata, indent=2))

    print(f"Metadata exported to gs://{bucket_name}/neural/{version}/metadata.json")
    return metadata
```

#### PyTorch Dataset for GCS Streaming

```python
import torch
from torch.utils.data import Dataset, IterableDataset
import pyarrow.parquet as pq
from google.cloud import storage
import io

class GCSTrialSequenceDataset(Dataset):
    """
    PyTorch Dataset that reads trial sequences from GCS Parquet files.
    Computes spectrograms on-the-fly for flexibility.
    """

    def __init__(
        self,
        gcs_prefix: str,  # e.g., 'gs://bucket/neural/v1/train'
        n_trials: int = 8,
        spectrogram_config: dict = None,
        transform = None,
        cache_dir: str = '/tmp/neural_cache'
    ):
        self.gcs_prefix = gcs_prefix
        self.n_trials = n_trials
        self.transform = transform
        self.cache_dir = cache_dir

        self.spec_config = spectrogram_config or {
            'nperseg': 128,
            'noverlap': 120,
            'fs': 1000
        }

        # Parse GCS path
        self.bucket_name = gcs_prefix.replace('gs://', '').split('/')[0]
        self.prefix = '/'.join(gcs_prefix.replace('gs://', '').split('/')[1:])

        # List and load all parquet files
        self._load_data()

        # Build trial sequences
        self._build_sequences()

    def _load_data(self):
        """Load data from GCS Parquet files."""
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.bucket_name)

        blobs = list(bucket.list_blobs(prefix=self.prefix))
        parquet_blobs = [b for b in blobs if b.name.endswith('.parquet')]

        dfs = []
        for blob in parquet_blobs:
            content = blob.download_as_bytes()
            table = pq.read_table(io.BytesIO(content))
            dfs.append(table.to_pandas())

        self.data = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(self.data)} trials from {len(parquet_blobs)} files")

    def _build_sequences(self):
        """Build trial sequences grouped by session."""
        self.sequences = []

        for session_id, session_df in self.data.groupby('session'):
            session_df = session_df.sort_values('trial_num')
            condition = session_df['condition'].iloc[0]
            traces = session_df['trace'].tolist()

            # Create sliding windows of n_trials
            for i in range(0, len(traces) - self.n_trials + 1, self.n_trials // 2):
                self.sequences.append({
                    'session': session_id,
                    'condition': condition,
                    'traces': traces[i:i + self.n_trials],
                    'start_trial': i
                })

        print(f"Built {len(self.sequences)} sequences")

    def _compute_spectrogram(self, trace):
        """Compute spectrogram on-the-fly."""
        from scipy.signal import spectrogram, windows

        window = windows.hann(self.spec_config['nperseg'])
        f, t, Sxx = spectrogram(
            np.array(trace),
            fs=self.spec_config['fs'],
            window=window,
            nperseg=self.spec_config['nperseg'],
            noverlap=self.spec_config['noverlap']
        )

        # Log scale and crop to relevant range
        Sxx_log = np.log10(Sxx + 1e-10)
        freq_mask = f <= 100  # 0-100 Hz

        return Sxx_log[freq_mask, :]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        # Compute spectrograms for each trial
        specs = [self._compute_spectrogram(t) for t in seq['traces']]
        specs = np.stack(specs)  # (n_trials, freq, time)

        # Convert to tensor
        specs = torch.from_numpy(specs).float()

        if self.transform:
            specs = self.transform(specs)

        label = 1 if seq['condition'] == 'FMR1' else 0

        return specs, label
```

#### Option B: BigQuery Direct Dataset (for SSH Workflow)

```python
import os
import pandas as pd
from google.cloud import bigquery
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.signal import spectrogram, windows

class BigQueryTrialSequenceDataset(Dataset):
    """
    PyTorch Dataset that queries BigQuery directly.
    Caches data locally after first load for faster subsequent epochs.
    Use this for SSH-based training on Hyperbolic/Lambda Labs.
    """

    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        split: str,  # 'train', 'val', or 'test'
        n_trials: int = 8,
        spectrogram_config: dict = None,
        transform=None,
        cache_path: str = None  # e.g., '/tmp/neural_cache/train.parquet'
    ):
        self.n_trials = n_trials
        self.transform = transform
        self.spec_config = spectrogram_config or {
            'nperseg': 128,
            'noverlap': 120,
            'fs': 1000
        }

        # Load from cache if exists, otherwise query BigQuery
        if cache_path and os.path.exists(cache_path):
            print(f"Loading from cache: {cache_path}")
            self.data = pd.read_parquet(cache_path)
        else:
            print(f"Querying BigQuery for {split} split...")
            self.data = self._query_bigquery(project_id, dataset_id, split)

            # Cache locally for faster subsequent runs
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                self.data.to_parquet(cache_path)
                print(f"Cached to {cache_path}")

        self._build_sequences()

    def _query_bigquery(self, project_id: str, dataset_id: str, split: str) -> pd.DataFrame:
        """Query BigQuery for split data."""
        client = bigquery.Client(project=project_id)

        query = f"""
        SELECT t.*
        FROM `{project_id}.{dataset_id}.lfp_trials` t
        JOIN `{project_id}.{dataset_id}.session_splits` s USING (session)
        WHERE s.split = '{split}'
        ORDER BY t.session, t.trial_num
        """

        df = client.query(query).to_dataframe()
        print(f"Loaded {len(df)} trials for {split} split")
        return df

    def _build_sequences(self):
        """Build trial sequences grouped by session."""
        self.sequences = []

        for session_id, session_df in self.data.groupby('session'):
            session_df = session_df.sort_values('trial_num')
            condition = session_df['condition'].iloc[0]
            traces = session_df['trace'].tolist()

            # Create sliding windows of n_trials
            for i in range(0, len(traces) - self.n_trials + 1, self.n_trials // 2):
                self.sequences.append({
                    'session': session_id,
                    'condition': condition,
                    'traces': traces[i:i + self.n_trials],
                    'start_trial': i
                })

        print(f"Built {len(self.sequences)} sequences")

    def _compute_spectrogram(self, trace):
        """Compute spectrogram on-the-fly."""
        window = windows.hann(self.spec_config['nperseg'])
        f, t, Sxx = spectrogram(
            np.array(trace),
            fs=self.spec_config['fs'],
            window=window,
            nperseg=self.spec_config['nperseg'],
            noverlap=self.spec_config['noverlap']
        )

        Sxx_log = np.log10(Sxx + 1e-10)
        freq_mask = f <= 100
        return Sxx_log[freq_mask, :]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        specs = [self._compute_spectrogram(t) for t in seq['traces']]
        specs = torch.from_numpy(np.stack(specs)).float()

        if self.transform:
            specs = self.transform(specs)

        label = 1 if seq['condition'] == 'FMR1' else 0
        return specs, label
```

#### Augmentation Pipeline

```python
import torch
import numpy as np

class SpectrogramAugmentation:
    """
    Augmentations for spectrogram sequences.
    """

    def __init__(
        self,
        time_shift_max: int = 10,
        freq_mask_max: int = 8,
        time_mask_max: int = 12,
        noise_std: float = 0.1,
        mixup_alpha: float = 0.2,
        p: float = 0.5
    ):
        self.time_shift_max = time_shift_max
        self.freq_mask_max = freq_mask_max
        self.time_mask_max = time_mask_max
        self.noise_std = noise_std
        self.mixup_alpha = mixup_alpha
        self.p = p

    def time_shift(self, x):
        """Shift along time axis."""
        shift = np.random.randint(-self.time_shift_max, self.time_shift_max + 1)
        return torch.roll(x, shift, dims=-1)

    def freq_mask(self, x):
        """Mask random frequency bands."""
        n_freq = x.shape[-2]
        mask_width = np.random.randint(1, self.freq_mask_max + 1)
        mask_start = np.random.randint(0, n_freq - mask_width)

        x = x.clone()
        x[..., mask_start:mask_start + mask_width, :] = 0
        return x

    def time_mask(self, x):
        """Mask random time segments."""
        n_time = x.shape[-1]
        mask_width = np.random.randint(1, self.time_mask_max + 1)
        mask_start = np.random.randint(0, n_time - mask_width)

        x = x.clone()
        x[..., mask_start:mask_start + mask_width] = 0
        return x

    def add_noise(self, x):
        """Add Gaussian noise."""
        noise = torch.randn_like(x) * self.noise_std
        return x + noise

    def trial_shuffle(self, x):
        """Shuffle trial order (tests if order matters)."""
        perm = torch.randperm(x.shape[0])
        return x[perm]

    def __call__(self, x):
        """Apply random augmentations."""
        if np.random.rand() < self.p:
            x = self.time_shift(x)
        if np.random.rand() < self.p:
            x = self.freq_mask(x)
        if np.random.rand() < self.p:
            x = self.time_mask(x)
        if np.random.rand() < self.p:
            x = self.add_noise(x)
        return x
```

### Week 5: Vertex AI Training Infrastructure

#### Tasks

| #   | Task                    | Description                             | Output                |
| --- | ----------------------- | --------------------------------------- | --------------------- |
| 5.1 | **Docker container**    | PyTorch training image for Vertex AI    | Dockerfile + image    |
| 5.2 | **Training entrypoint** | CLI args for GCS paths, hyperparameters | train.py              |
| 5.3 | **Mixed precision**     | FP16 training with gradient scaling     | AMP integration       |
| 5.4 | **GCS checkpointing**   | Save checkpoints and metrics to GCS     | Checkpoint system     |
| 5.5 | **Job submission**      | Vertex AI custom training job config    | Job submission script |

#### Dockerfile for Vertex AI

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training code
COPY src/ ./src/
COPY train.py .

# Set entrypoint
ENTRYPOINT ["python", "train.py"]
```

```text
# requirements.txt
torch>=2.1.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
pyarrow>=14.0.0
google-cloud-storage>=2.10.0
wandb>=0.15.0
```

#### Training Entrypoint for Vertex AI

```python
# train.py
import argparse
import os
import json
from pathlib import Path
from google.cloud import storage

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Import model and dataset from src/
from src.model import Temporal3DViT, Temporal3DViTConfig
from src.dataset import GCSTrialSequenceDataset
from src.augmentation import SpectrogramAugmentation


def parse_args():
    parser = argparse.ArgumentParser()

    # GCS data paths (passed by Vertex AI)
    parser.add_argument('--train-data', type=str, required=True,
                        help='GCS prefix for training data, e.g., gs://bucket/neural/v1/train')
    parser.add_argument('--val-data', type=str, required=True,
                        help='GCS prefix for validation data')
    parser.add_argument('--test-data', type=str, required=True,
                        help='GCS prefix for test data')

    # Output paths
    parser.add_argument('--output-dir', type=str,
                        default=os.environ.get('AIP_MODEL_DIR', '/tmp/model'),
                        help='GCS path for model artifacts')
    parser.add_argument('--checkpoint-dir', type=str,
                        default=os.environ.get('AIP_CHECKPOINT_DIR', '/tmp/checkpoints'),
                        help='GCS path for checkpoints')

    # Model config
    parser.add_argument('--model-size', type=str, default='small',
                        choices=['tiny', 'small', 'base'])
    parser.add_argument('--n-trials', type=int, default=8)

    # Spectrogram config
    parser.add_argument('--nperseg', type=int, default=128)
    parser.add_argument('--noverlap', type=int, default=120)

    # Training config
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--warmup-epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=15)

    # Hardware
    parser.add_argument('--mixed-precision', action='store_true', default=True)
    parser.add_argument('--num-workers', type=int, default=4)

    # Experiment tracking
    parser.add_argument('--experiment-name', type=str, default='temporal-3d-vit')
    parser.add_argument('--wandb-project', type=str, default='neural-vit')

    return parser.parse_args()


def upload_to_gcs(local_path: str, gcs_path: str):
    """Upload file to GCS."""
    if not gcs_path.startswith('gs://'):
        return  # Local path, skip

    bucket_name = gcs_path.replace('gs://', '').split('/')[0]
    blob_path = '/'.join(gcs_path.replace('gs://', '').split('/')[1:])

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)


def train(args):
    """Main training loop."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize W&B
    import wandb
    wandb.init(
        project=args.wandb_project,
        name=args.experiment_name,
        config=vars(args)
    )

    # Spectrogram config
    spec_config = {
        'nperseg': args.nperseg,
        'noverlap': args.noverlap,
        'fs': 1000
    }

    # Create datasets
    train_dataset = GCSTrialSequenceDataset(
        gcs_prefix=args.train_data,
        n_trials=args.n_trials,
        spectrogram_config=spec_config,
        transform=SpectrogramAugmentation(p=0.5)
    )

    val_dataset = GCSTrialSequenceDataset(
        gcs_prefix=args.val_data,
        n_trials=args.n_trials,
        spectrogram_config=spec_config,
        transform=None
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Get input dimensions from first batch
    sample_spec, _ = train_dataset[0]
    freq_size, time_size = sample_spec.shape[1], sample_spec.shape[2]

    # Create model
    config = Temporal3DViTConfig(
        n_trials=args.n_trials,
        freq_size=freq_size,
        time_size=time_size
    )
    model = Temporal3DViT(config).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    wandb.watch(model)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps
    )

    # Loss and scaler for mixed precision
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler() if args.mixed_precision else None

    # Training loop
    best_val_acc = 0
    patience_counter = 0

    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for specs, labels in train_loader:
            specs = specs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if args.mixed_precision:
                with autocast():
                    logits = model(specs)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(specs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            scheduler.step()

            train_loss += loss.item()
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for specs, labels in val_loader:
                specs = specs.to(device)
                labels = labels.to(device)

                logits = model(specs)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        # Logging
        metrics = {
            'epoch': epoch,
            'train/loss': train_loss / len(train_loader),
            'train/accuracy': train_acc,
            'val/loss': val_loss / len(val_loader),
            'val/accuracy': val_acc,
            'lr': scheduler.get_last_lr()[0]
        }
        wandb.log(metrics)

        print(f"Epoch {epoch}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

        # Checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            # Save best model
            checkpoint_path = '/tmp/best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, checkpoint_path)

            # Upload to GCS
            upload_to_gcs(checkpoint_path, f"{args.checkpoint_dir}/best.pt")
            print(f"New best model saved: val_acc={val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final evaluation on test set
    test_dataset = GCSTrialSequenceDataset(
        gcs_prefix=args.test_data,
        n_trials=args.n_trials,
        spectrogram_config=spec_config,
        transform=None
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Load best model
    checkpoint = torch.load('/tmp/best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for specs, labels in test_loader:
            specs = specs.to(device)
            logits = model(specs)
            preds = logits.argmax(1)

            test_correct += (preds == labels.to(device)).sum().item()
            test_total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    test_acc = test_correct / test_total

    # Save final metrics
    results = {
        'test_accuracy': test_acc,
        'best_val_accuracy': best_val_acc,
        'predictions': all_preds,
        'labels': all_labels
    }

    results_path = '/tmp/eval_results.json'
    with open(results_path, 'w') as f:
        json.dump({k: v if not isinstance(v, list) else v for k, v in results.items()}, f)

    upload_to_gcs(results_path, f"{args.output_dir}/eval_results.json")

    wandb.log({'test/accuracy': test_acc})
    wandb.finish()

    print(f"\nFinal test accuracy: {test_acc:.4f}")


if __name__ == '__main__':
    args = parse_args()
    train(args)
```

#### Vertex AI Job Submission

```python
# submit_training_job.py
from google.cloud import aiplatform

def submit_vertex_training_job(
    project_id: str,
    region: str = 'us-central1',
    experiment_name: str = 'temporal-3d-vit-exp1',
    data_version: str = 'v1',
    bucket_name: str = 'your-bucket',
    machine_type: str = 'n1-standard-8',
    accelerator_type: str = 'NVIDIA_TESLA_T4',
    accelerator_count: int = 1
):
    """Submit a custom training job to Vertex AI."""

    aiplatform.init(project=project_id, location=region)

    # GCS paths
    gcs_base = f'gs://{bucket_name}/neural/{data_version}'

    job = aiplatform.CustomContainerTrainingJob(
        display_name=experiment_name,
        container_uri=f'gcr.io/{project_id}/temporal-3d-vit:latest',
    )

    job.run(
        args=[
            '--train-data', f'{gcs_base}/train',
            '--val-data', f'{gcs_base}/val',
            '--test-data', f'{gcs_base}/test',
            '--output-dir', f'gs://{bucket_name}/neural/models/{experiment_name}',
            '--checkpoint-dir', f'gs://{bucket_name}/neural/checkpoints/{experiment_name}',
            '--experiment-name', experiment_name,
            '--model-size', 'small',
            '--n-trials', '8',
            '--epochs', '100',
            '--batch-size', '16',
            '--lr', '1e-4',
        ],
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        replica_count=1,
    )

    return job


if __name__ == '__main__':
    submit_vertex_training_job(
        project_id='your-project-id',
        bucket_name='your-bucket',
        experiment_name='temporal-3d-vit-v1-small'
    )
```

#### Training Configuration

```python
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Training configuration."""

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    batch_size: int = 16
    epochs: int = 100
    warmup_epochs: int = 10
    min_lr: float = 1e-6

    # Regularization
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 0.0

    # Early stopping
    patience: int = 15

    # Hardware
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = True

    # Logging
    log_every_n_steps: int = 10
    val_every_n_epochs: int = 1
    save_top_k: int = 3

    # GCS paths (set via CLI args in Vertex AI)
    train_gcs_prefix: str = None
    val_gcs_prefix: str = None
    test_gcs_prefix: str = None
    output_gcs_path: str = None
```

---

### Week 5 Alternative: SSH Training (Hyperbolic / Lambda Labs)

Use this approach for interactive development, debugging, or cost-effective training.

#### One-Time Setup on GPU Instance

```bash
# SSH into your GPU instance
ssh user@your-hyperbolic-instance

# Create conda environment
conda create -n neural python=3.10 -y
conda activate neural

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install google-cloud-bigquery pandas-gbq pyarrow scipy wandb

# Set up GCP authentication (upload your service account key first)
# scp service-account-key.json user@instance:~/
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/service-account-key.json"

# Clone your repo
git clone https://github.com/your-org/temporal-3d-neural-vit.git
cd temporal-3d-neural-vit
```

#### SSH Training Script

```python
# train_ssh.py - Run directly on Hyperbolic/Lambda Labs
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from src.model import Temporal3DViT, Temporal3DViTConfig
from src.dataset import BigQueryTrialSequenceDataset
from src.augmentation import SpectrogramAugmentation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-id', type=str, required=True)
    parser.add_argument('--dataset-id', type=str, default='neural')
    parser.add_argument('--n-trials', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--cache-dir', type=str, default='./cache')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--experiment-name', type=str, default='temporal-3d-vit')
    parser.add_argument('--patience', type=int, default=15)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # W&B logging
    import wandb
    wandb.init(project='neural-vit', name=args.experiment_name, config=vars(args))

    # Create datasets - queries BigQuery on first run, caches locally
    train_dataset = BigQueryTrialSequenceDataset(
        project_id=args.project_id,
        dataset_id=args.dataset_id,
        split='train',
        n_trials=args.n_trials,
        transform=SpectrogramAugmentation(p=0.5),
        cache_path=f'{args.cache_dir}/train.parquet'
    )

    val_dataset = BigQueryTrialSequenceDataset(
        project_id=args.project_id,
        dataset_id=args.dataset_id,
        split='val',
        n_trials=args.n_trials,
        cache_path=f'{args.cache_dir}/val.parquet'
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    # Model setup
    sample_spec, _ = train_dataset[0]
    config = Temporal3DViTConfig(
        n_trials=args.n_trials,
        freq_size=sample_spec.shape[1],
        time_size=sample_spec.shape[2]
    )
    model = Temporal3DViT(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_acc = 0
    patience_counter = 0

    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for specs, labels in train_loader:
            specs, labels = specs.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast():
                logits = model(specs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for specs, labels in val_loader:
                specs, labels = specs.to(device), labels.to(device)
                logits = model(specs)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        print(f"Epoch {epoch}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
        wandb.log({
            'epoch': epoch,
            'train/accuracy': train_acc,
            'train/loss': train_loss / len(train_loader),
            'val/accuracy': val_acc,
            'lr': scheduler.get_last_lr()[0]
        })

        # Checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, f'{args.checkpoint_dir}/best.pt')
            print(f"  Saved new best model: {val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final test evaluation
    test_dataset = BigQueryTrialSequenceDataset(
        project_id=args.project_id,
        dataset_id=args.dataset_id,
        split='test',
        n_trials=args.n_trials,
        cache_path=f'{args.cache_dir}/test.parquet'
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Load best model
    checkpoint = torch.load(f'{args.checkpoint_dir}/best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for specs, labels in test_loader:
            specs = specs.to(device)
            logits = model(specs)
            test_correct += (logits.argmax(1) == labels.to(device)).sum().item()
            test_total += labels.size(0)

    test_acc = test_correct / test_total
    wandb.log({'test/accuracy': test_acc})
    wandb.finish()

    print(f"\nFinal Results:")
    print(f"  Best validation accuracy: {best_val_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")


if __name__ == '__main__':
    main()
```

#### Running Training on Hyperbolic Labs

```bash
# SSH into instance
ssh user@hyperbolic-instance
cd ~/temporal-3d-neural-vit
conda activate neural

# Use tmux/screen to persist session if SSH disconnects
tmux new -s training

# Run training
python train_ssh.py \
    --project-id your-gcp-project \
    --dataset-id neural \
    --n-trials 8 \
    --epochs 100 \
    --experiment-name vit-3d-trials8-run1

# Detach from tmux: Ctrl+B, then D
# Reattach later: tmux attach -t training
```

#### Syncing Checkpoints

```bash
# From your local machine - pull checkpoints
scp -r user@hyperbolic-instance:~/temporal-3d-neural-vit/checkpoints ./

# Or push to GCS from the Hyperbolic instance
gsutil cp -r ./checkpoints gs://your-bucket/neural/checkpoints/ssh-experiments/
```

---

## Phase 3: Experiments & Evaluation (Weeks 6-8)

### Week 6: Baseline Experiments

#### Experiment Matrix

Experiments can be run with either workflow:

- **Option A (Vertex AI)**: Submit as managed training jobs
- **Option B (SSH)**: Run directly on Hyperbolic/Lambda Labs

All experiments log metrics to W&B for unified tracking regardless of compute backend.

| Exp ID | Model                 | n_trials | Augmentation | Experiment Name       |
| ------ | --------------------- | -------- | ------------ | --------------------- |
| E1     | 2D ViT (single trial) | 1        | None         | `vit-2d-baseline`     |
| E2     | 2D ViT (single trial) | 1        | Full         | `vit-2d-baseline-aug` |
| E3     | 3D ViT                | 4        | None         | `vit-3d-trials4`      |
| E4     | 3D ViT                | 8        | None         | `vit-3d-trials8`      |
| E5     | 3D ViT                | 16       | None         | `vit-3d-trials16`     |
| E6     | 3D ViT                | 8        | Full         | `vit-3d-trials8-aug`  |

**Running Experiments:**

```bash
# Option A: Vertex AI
python submit_training_job.py --experiment-name vit-3d-trials8

# Option B: SSH (on Hyperbolic/Lambda)
python train_ssh.py --project-id your-project --experiment-name vit-3d-trials8
```

#### GCS Experiment Artifacts

```
gs://bucket/neural/
├── checkpoints/
│   ├── vit-2d-baseline/
│   │   ├── best.pt
│   │   └── last.pt
│   ├── vit-3d-trials8/
│   │   └── ...
│   └── vit-3d-trials8-aug/
│       └── ...
└── metrics/
    ├── vit-2d-baseline/
    │   └── eval_results.json
    └── vit-3d-trials8/
        └── eval_results.json
```

#### Evaluation Metrics

```python
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    precision_recall_fscore_support, confusion_matrix
)

def compute_metrics(y_true, y_pred, y_prob):
    """
    Compute comprehensive classification metrics.
    """
    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

    # ROC-AUC
    if len(np.unique(y_true)) == 2:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1]
    )
    metrics['precision_wt'] = precision[0]
    metrics['recall_wt'] = recall[0]
    metrics['f1_wt'] = f1[0]
    metrics['precision_fmr1'] = precision[1]
    metrics['recall_fmr1'] = recall[1]
    metrics['f1_fmr1'] = f1[1]

    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

    return metrics


def evaluate_model(model, test_loader, device='cuda'):
    """Full evaluation on test set."""
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for specs, labels in test_loader:
            specs = specs.to(device)
            logits = model(specs)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    metrics = compute_metrics(all_labels, all_preds, all_probs)

    return metrics
```

#### Persist Results to BigQuery

```python
from google.cloud import bigquery
from datetime import datetime

def save_experiment_to_bigquery(
    experiment_name: str,
    metrics: dict,
    config: dict,
    project_id: str,
    dataset_id: str = 'experiments'
):
    """
    Save experiment results to BigQuery for tracking and analysis.
    """
    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.{dataset_id}.experiment_results"

    row = {
        'experiment_name': experiment_name,
        'timestamp': datetime.utcnow().isoformat(),
        'accuracy': metrics['accuracy'],
        'balanced_accuracy': metrics['balanced_accuracy'],
        'roc_auc': metrics.get('roc_auc'),
        'f1_wt': metrics['f1_wt'],
        'f1_fmr1': metrics['f1_fmr1'],
        'config': json.dumps(config),
        'confusion_matrix': json.dumps(metrics['confusion_matrix'].tolist())
    }

    errors = client.insert_rows_json(table_id, [row])
    if errors:
        print(f"Error inserting to BigQuery: {errors}")
    else:
        print(f"Results saved to {table_id}")


# BigQuery schema for experiment tracking
EXPERIMENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS `project.experiments.experiment_results` (
    experiment_name STRING,
    timestamp TIMESTAMP,
    accuracy FLOAT64,
    balanced_accuracy FLOAT64,
    roc_auc FLOAT64,
    f1_wt FLOAT64,
    f1_fmr1 FLOAT64,
    config JSON,
    confusion_matrix JSON
);
"""
```

### Week 7: Ablation Studies

#### Ablation Experiments

| Ablation | Description                               | Hypothesis                                 |
| -------- | ----------------------------------------- | ------------------------------------------ |
| A1       | Remove trial position embedding           | Position encoding helps temporal reasoning |
| A2       | Replace factorized with learned pos embed | Factorized is more efficient               |
| A3       | Remove LayerScale                         | LayerScale stabilizes training             |
| A4       | Reduce patch size (2,4,4)                 | Smaller patches capture finer details      |
| A5       | Trial order shuffle                       | Order matters for temporal patterns        |

### Week 8: Analysis & Visualization

#### Tasks

| #   | Task                        | Description                           | Output             |
| --- | --------------------------- | ------------------------------------- | ------------------ |
| 8.1 | **Attention analysis**      | Visualize attention patterns          | Attention maps     |
| 8.2 | **Error analysis**          | Analyze misclassified examples        | Error report       |
| 8.3 | **Embedding visualization** | t-SNE/UMAP of CLS embeddings          | Clustering plots   |
| 8.4 | **Biomarker discovery**     | Which freq-time regions discriminate? | Feature importance |

#### Attention Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_attention_3d(model, sample, config, layer_idx=-1):
    """
    Visualize attention patterns for a single sample.
    """
    # Get attention maps
    attention_maps = model.get_attention_maps(sample.unsqueeze(0))

    # Use specified layer (default: last)
    attn = attention_maps[layer_idx][0]  # (heads, seq_len, seq_len)

    # Average over heads
    attn = attn.mean(dim=0).cpu().numpy()  # (seq_len, seq_len)

    # Extract CLS attention to patches
    cls_attn = attn[0, 1:]  # Skip CLS-to-CLS

    # Reshape to 3D grid
    n_k = config.n_trials // config.patch_trial
    n_f = config.freq_size // config.patch_freq
    n_t = config.time_size // config.patch_time

    cls_attn_3d = cls_attn.reshape(n_k, n_f, n_t)

    # Visualize
    fig, axes = plt.subplots(1, n_k, figsize=(4 * n_k, 4))

    for k in range(n_k):
        ax = axes[k] if n_k > 1 else axes
        im = ax.imshow(cls_attn_3d[k], aspect='auto', cmap='hot', origin='lower')
        ax.set_xlabel('Time patch')
        ax.set_ylabel('Freq patch')
        ax.set_title(f'Trial patches {k*config.patch_trial}-{(k+1)*config.patch_trial-1}')
        plt.colorbar(im, ax=ax)

    plt.suptitle(f'CLS Attention (Layer {layer_idx})', fontsize=14)
    plt.tight_layout()

    return fig, cls_attn_3d
```

---

## Phase 4: Documentation & Deliverables (Weeks 9-10)

### Deliverables Checklist

| #   | Deliverable          | Format / Location                         | Status |
| --- | -------------------- | ----------------------------------------- | ------ |
| D1  | Project report       | PDF/LaTeX                                 | ☐      |
| D2  | Source code          | GitHub repo                               | ☐      |
| D3  | Trained models       | `gs://bucket/neural/checkpoints/*.pt`     | ☐      |
| D4  | Experiment logs      | W&B dashboard                             | ☐      |
| D5  | Experiment results   | BigQuery `experiments.experiment_results` | ☐      |
| D6  | Data splits metadata | `gs://bucket/neural/v1/metadata.json`     | ☐      |
| D7  | Docker image         | `gcr.io/project/temporal-3d-vit:latest`   | ☐      |
| D8  | Presentation         | Slides                                    | ☐      |
| D9  | README               | Markdown                                  | ☐      |

### Report Outline

```
1. Introduction
   1.1 Background on Fragile X Syndrome
   1.2 LFP as a biomarker
   1.3 Why trial-to-trial variability matters

2. Related Work
   2.1 Vision Transformers
   2.2 Video understanding (3D patches)
   2.3 Neural signal classification

3. Methods
   3.1 Dataset description
   3.2 Spectrogram computation
   3.3 Temporal 3D ViT architecture
   3.4 Training procedure

4. Experiments
   4.1 Baseline comparisons
   4.2 Ablation studies
   4.3 Sequence length analysis

5. Results
   5.1 Classification performance
   5.2 Attention analysis
   5.3 Biomarker discovery

6. Discussion
   6.1 Why temporal context helps
   6.2 Limitations
   6.3 Clinical implications

7. Conclusion
```

---

---
