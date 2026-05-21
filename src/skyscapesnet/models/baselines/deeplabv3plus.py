"""DeepLabv3+ (Chen et al., ECCV 2018) — encoder-decoder with ASPP and low-level skip.

Reference:
    Chen, Zhu, Papandreou, Schroff, Adam.
    "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation."
    ECCV 2018. https://arxiv.org/abs/1802.02611

Used as a baseline in Tables 2 and 4 of the SkyScapes paper.

Uses the same `ASPP` module as DeepLabv3, plus a decoder that fuses a
projected low-level feature (c2 = layer1 output) with the ASPP output.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from skyscapesnet.models.outputs import SkyScapesOutput

from .backbones.resnet import resnet101_encoder
from .deeplabv3 import ASPP


class DeepLabV3Plus(nn.Module):
    """DeepLabv3+ with a dilated ResNet101 backbone.

    Args:
        in_channels: Input image channels.
        n_classes: Output classes.
        output_stride: Backbone OS (16 paper-default; 8 for higher resolution).
        aspp_channels: Channels per ASPP branch and decoder hidden.
        low_level_channels: Channels after the low-level (c2) 1x1 projection
            (48 in the paper — kept small so the ASPP signal dominates).
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 20,
        output_stride: int = 16,
        aspp_channels: int = 256,
        low_level_channels: int = 48,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        self.encoder = resnet101_encoder(in_channels=in_channels, output_stride=output_stride)
        c1, c2, c3, c4, c5 = self.encoder.out_channels  # [64, 256, 512, 1024, 2048]

        atrous_rates = (6, 12, 18) if output_stride == 16 else (12, 24, 36)
        self.aspp = ASPP(c5, out_channels=aspp_channels, atrous_rates=atrous_rates)

        self.low_level_proj = nn.Sequential(
            nn.Conv2d(c2, low_level_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_level_channels),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(
                aspp_channels + low_level_channels, aspp_channels,
                kernel_size=3, padding=1, bias=False,
            ),
            nn.BatchNorm2d(aspp_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                aspp_channels, aspp_channels,
                kernel_size=3, padding=1, bias=False,
            ),
            nn.BatchNorm2d(aspp_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(aspp_channels, n_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> SkyScapesOutput:
        input_size = x.shape[-2:]
        feats = self.encoder(x)
        low_level = feats[1]   # c2: /4 spatial, 256 ch
        high_level = feats[-1]  # c5: /16 (OS=16) or /8 (OS=8)

        aspp_out = self.aspp(high_level)
        aspp_out = F.interpolate(
            aspp_out, size=low_level.shape[-2:], mode="bilinear", align_corners=False,
        )
        low_level_out = self.low_level_proj(low_level)
        out = torch.cat([aspp_out, low_level_out], dim=1)
        out = self.decoder(out)
        out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
        return SkyScapesOutput(seg=out)
