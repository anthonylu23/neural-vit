import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional
import math

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