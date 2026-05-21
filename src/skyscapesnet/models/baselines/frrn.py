"""FRRN (Pohlen et al., CVPR 2017) — Full-Resolution Residual Networks.

Reference:
    Pohlen, Hermans, Mathias, Leibe.
    "Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes."
    CVPR 2017. https://arxiv.org/abs/1611.08323

Used as a baseline in Tables 2 and 4 of the SkyScapes paper.

Two parallel streams:
- **Residual stream:** stays at full resolution (or near it), constant channel
  count. Carries fine-grained localization information.
- **Pooling stream:** classic encoder-decoder, downsamples / upsamples.

The two streams interact through Full-Resolution Residual Units (FRRUs):
each FRRU pools the residual stream down to the pooling-stream scale, fuses,
runs a few convs, and projects an update back to the residual stream.

This implementation follows FRRN-A. FRRN-B is the same architecture with
more units per scale; switching is a one-arg change in `__init__`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from skyscapesnet.models.outputs import SkyScapesOutput


class _ResidualUnit(nn.Module):
    """Standard pre-activation residual unit applied to the residual stream."""

    def __init__(self, channels: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return x + out


class FRRU(nn.Module):
    """Full-Resolution Residual Unit.

    Args:
        residual_channels: Channels of the (always full-res) residual stream.
        pool_channels: Channels of the pooling stream at the current scale.
        scale: Downsample factor of the pooling stream relative to the residual.
    """

    def __init__(self, residual_channels: int, pool_channels: int, scale: int):
        super().__init__()
        self.scale = scale
        # Fuse pool stream + downsampled residual via 3x3 conv (twice).
        self.conv1 = nn.Conv2d(
            pool_channels + residual_channels, pool_channels,
            kernel_size=3, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(pool_channels)
        self.conv2 = nn.Conv2d(pool_channels, pool_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(pool_channels)
        # Project pool stream back into residual-stream space.
        self.proj = nn.Conv2d(pool_channels, residual_channels, kernel_size=1, bias=False)

    def forward(
        self, y: torch.Tensor, z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Args: y = pool stream (at /scale), z = residual stream (full res)."""
        z_down = F.avg_pool2d(z, kernel_size=self.scale, stride=self.scale)
        y_new = F.relu(self.bn1(self.conv1(torch.cat([y, z_down], dim=1))))
        y_new = F.relu(self.bn2(self.conv2(y_new)))
        z_update = F.interpolate(
            self.proj(y_new), size=z.shape[-2:], mode="bilinear", align_corners=False,
        )
        return y_new, z + z_update


class FRRN(nn.Module):
    """FRRN-A architecture.

    Channel layout (paper FRRN-A):
        residual stream: 48 ch throughout (stem + tail at full res).
        pooling stream:
            /2:  96 ch  (2 FRRUs encoder; 2 FRRUs decoder)
            /4: 192 ch  (2 FRRUs encoder; 2 FRRUs decoder)
            /8: 384 ch  (2 FRRUs encoder; 2 FRRUs decoder)
            /16: 384 ch (2 FRRUs encoder)
    Set `unit_counts` to `(3, 4, 2, 2, 2, 2, 2)` for FRRN-B.
    """

    DEFAULT_SCALES: tuple[int, ...] = (2, 4, 8, 16)
    DEFAULT_POOL_CHANNELS: tuple[int, ...] = (96, 192, 384, 384)

    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 20,
        residual_channels: int = 48,
        scales: tuple[int, ...] | None = None,
        pool_channels: tuple[int, ...] | None = None,
        n_units_per_scale: int = 2,
        n_stem_units: int = 3,
        n_tail_units: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        scales = scales or self.DEFAULT_SCALES
        pool_channels = pool_channels or self.DEFAULT_POOL_CHANNELS
        if len(scales) != len(pool_channels):
            raise ValueError("scales and pool_channels must have the same length")

        # Stem: 5x5 conv -> BN -> ReLU -> N residual units (all at full res).
        self.stem_conv = nn.Conv2d(
            in_channels, residual_channels,
            kernel_size=5, padding=2, bias=False,
        )
        self.stem_bn = nn.BatchNorm2d(residual_channels)
        self.stem_units = nn.Sequential(
            *[_ResidualUnit(residual_channels) for _ in range(n_stem_units)]
        )

        # Build encoder: per-scale stride-2 pool + N FRRUs.
        # Build decoder: per-scale upsample + N FRRUs (skip via residual stream).
        self.encoder_downs = nn.ModuleList()   # spatial downsample for pool stream
        self.encoder_proj = nn.ModuleList()    # 1x1 conv: residual_ch -> pool_ch (entering a new scale)
        self.encoder_frrus = nn.ModuleList()   # nn.ModuleList of nn.ModuleLists
        self.decoder_ups = nn.ModuleList()
        self.decoder_proj = nn.ModuleList()
        self.decoder_frrus = nn.ModuleList()

        prev_pool_ch = residual_channels
        for scale, pc in zip(scales, pool_channels):
            # Encoder side: drop to next scale, then a stack of FRRUs.
            self.encoder_downs.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.encoder_proj.append(
                nn.Conv2d(prev_pool_ch, pc, kernel_size=1, bias=False)
                if prev_pool_ch != pc else nn.Identity()
            )
            self.encoder_frrus.append(
                nn.ModuleList(
                    [FRRU(residual_channels, pc, scale=scale) for _ in range(n_units_per_scale)]
                )
            )
            prev_pool_ch = pc

        # Decoder: walk scales in reverse, skipping the deepest (we exit from it).
        rev_scales = list(reversed(scales[:-1]))
        rev_pool_channels = list(reversed(pool_channels[:-1]))
        prev_pool_ch = pool_channels[-1]
        for scale, pc in zip(rev_scales, rev_pool_channels):
            self.decoder_ups.append(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            )
            self.decoder_proj.append(
                nn.Conv2d(prev_pool_ch, pc, kernel_size=1, bias=False)
                if prev_pool_ch != pc else nn.Identity()
            )
            self.decoder_frrus.append(
                nn.ModuleList(
                    [FRRU(residual_channels, pc, scale=scale) for _ in range(n_units_per_scale)]
                )
            )
            prev_pool_ch = pc

        # Tail: project pool stream back to residual-stream channels (final
        # upsample to full res), then N residual units, then classifier.
        self.tail_proj = nn.Conv2d(prev_pool_ch, residual_channels, kernel_size=1, bias=False)
        self.tail_units = nn.Sequential(
            *[_ResidualUnit(residual_channels) for _ in range(n_tail_units)]
        )
        self.classifier = nn.Conv2d(residual_channels, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> SkyScapesOutput:
        # Stem
        z = F.relu(self.stem_bn(self.stem_conv(x)))
        z = self.stem_units(z)
        y = z  # pool stream starts from the same feature

        # Encoder
        for down, proj, frrus in zip(
            self.encoder_downs, self.encoder_proj, self.encoder_frrus,
        ):
            y = proj(down(y))
            for frru in frrus:
                y, z = frru(y, z)

        # Decoder
        for up, proj, frrus in zip(
            self.decoder_ups, self.decoder_proj, self.decoder_frrus,
        ):
            y = proj(up(y))
            for frru in frrus:
                y, z = frru(y, z)

        # Tail: bring pool stream back up to residual resolution and add.
        y = F.interpolate(self.tail_proj(y), size=z.shape[-2:], mode="bilinear", align_corners=False)
        z = z + y
        z = self.tail_units(z)
        return SkyScapesOutput(seg=self.classifier(z))
