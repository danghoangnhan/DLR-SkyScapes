"""DenseNet161 encoder for the DenseASPP baseline.

Reference:
    Huang et al., "Densely Connected Convolutional Networks." CVPR 2017.
    https://arxiv.org/abs/1608.06993

The DenseASPP paper (Yang et al., CVPR 2018) uses DenseNet161 as the
backbone with optional dilation in the last block to preserve resolution.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DenseLayer(nn.Module):
    """BN -> ReLU -> 1x1 conv (bottleneck) -> BN -> ReLU -> 3x3 conv -> concat."""

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        bn_size: int = 4,
        dropout_p: float = 0.0,
        dilation: int = 1,
    ):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            bn_size * growth_rate, growth_rate,
            kernel_size=3, padding=dilation, dilation=dilation, bias=False,
        )
        self.dropout_p = dropout_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        if self.dropout_p > 0:
            out = F.dropout(out, p=self.dropout_p, training=self.training)
        return torch.cat([x, out], dim=1)


class _DenseBlock(nn.Module):
    def __init__(
        self,
        n_layers: int,
        in_channels: int,
        growth_rate: int,
        bn_size: int = 4,
        dropout_p: float = 0.0,
        dilation: int = 1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        c = in_channels
        for _ in range(n_layers):
            self.layers.append(_DenseLayer(c, growth_rate, bn_size, dropout_p, dilation))
            c += growth_rate
        self.out_channels = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _Transition(nn.Module):
    """BN -> ReLU -> 1x1 conv (halve channels) -> AvgPool 2x2 (optional)."""

    def __init__(self, in_channels: int, out_channels: int, downsample: bool = True):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool: nn.Module = nn.AvgPool2d(kernel_size=2, stride=2) if downsample else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.conv(self.relu(self.norm(x))))


class DenseNet161Encoder(nn.Module):
    """DenseNet161 encoder producing 5 feature maps.

    Forward returns [c1, c2, c3, c4, c5]:
        c1: post-stem (stride 4, 96 ch)
        c2: post-trans1 (stride 8)
        c3: post-trans2 (stride 16)
        c4: post-trans3 (stride 32 or 16 if output_stride=16)
        c5: post-dense4 (stride 32 or 16)

    Args:
        in_channels: Input image channels.
        growth_rate: Channels added per DenseLayer (48 for DenseNet161).
        init_features: Channels after the stem (96 for DenseNet161).
        block_config: Layer counts per dense block.
        output_stride: 32 (vanilla) or 16 (skip downsample in trans3,
            dilation=2 in dense4 — preserves resolution for DenseASPP).
        dropout_p: Per-layer dropout probability.
    """

    out_channels: list[int]

    def __init__(
        self,
        in_channels: int = 3,
        growth_rate: int = 48,
        init_features: int = 96,
        block_config: tuple[int, int, int, int] = (6, 12, 36, 24),
        output_stride: int = 32,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        if output_stride not in (32, 16):
            raise ValueError(f"output_stride must be 32 or 16; got {output_stride}")
        self.output_stride = output_stride

        # Stem (stride 4)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Dense blocks + transitions
        n_features = init_features
        self.dense1 = _DenseBlock(block_config[0], n_features, growth_rate, dropout_p=dropout_p)
        n_features = self.dense1.out_channels
        trans1_out = n_features // 2
        self.trans1 = _Transition(n_features, trans1_out, downsample=True)

        self.dense2 = _DenseBlock(block_config[1], trans1_out, growth_rate, dropout_p=dropout_p)
        n_features = self.dense2.out_channels
        trans2_out = n_features // 2
        self.trans2 = _Transition(n_features, trans2_out, downsample=True)

        self.dense3 = _DenseBlock(block_config[2], trans2_out, growth_rate, dropout_p=dropout_p)
        n_features = self.dense3.out_channels
        trans3_out = n_features // 2
        # For output_stride=16, skip the trans3 downsample
        self.trans3 = _Transition(
            n_features, trans3_out, downsample=(output_stride == 32),
        )

        # Last dense block — apply dilation if we skipped trans3's downsample
        last_dilation = 1 if output_stride == 32 else 2
        self.dense4 = _DenseBlock(
            block_config[3], trans3_out, growth_rate,
            dropout_p=dropout_p, dilation=last_dilation,
        )
        n_features = self.dense4.out_channels
        self.norm_final = nn.BatchNorm2d(n_features)
        self.relu_final = nn.ReLU(inplace=True)

        self.out_channels = [
            init_features,
            trans1_out,
            trans2_out,
            trans3_out,
            n_features,
        ]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        c1 = self.stem(x)                          # /4
        c2 = self.trans1(self.dense1(c1))          # /8
        c3 = self.trans2(self.dense2(c2))          # /16
        c4 = self.trans3(self.dense3(c3))          # /32 or /16
        c5 = self.relu_final(self.norm_final(self.dense4(c4)))  # /32 or /16
        return [c1, c2, c3, c4, c5]


def densenet161_encoder(
    in_channels: int = 3,
    output_stride: int = 32,
    dropout_p: float = 0.0,
) -> DenseNet161Encoder:
    return DenseNet161Encoder(
        in_channels=in_channels,
        output_stride=output_stride,
        dropout_p=dropout_p,
    )
