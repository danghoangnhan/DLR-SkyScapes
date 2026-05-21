"""DeepLabv3 (Chen et al., 2017) — ASPP-based semantic segmentation.

Reference:
    Chen, Papandreou, Schroff, Adam.
    "Rethinking Atrous Convolution for Semantic Image Segmentation." 2017.
    https://arxiv.org/abs/1706.05587

Used as a baseline in Table 4 (SkyScapes-Lane) of the SkyScapes paper.

The `ASPP` module is exported and reused by DeepLabv3+ (`deeplabv3plus.py`).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from skyscapesnet.models.outputs import SkyScapesOutput

from .backbones.resnet import resnet50_encoder


class ASPPConv(nn.Sequential):
    """One atrous-conv branch: Conv3x3(dilation) -> BN -> ReLU."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=dilation, dilation=dilation, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Module):
    """Image-level pooling branch of ASPP: GAP -> 1x1 conv -> bilinear upsample."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        x = self.project(self.gap(x))
        return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling.

    Five parallel branches: one 1x1 conv, three atrous Conv3x3 at the given
    rates, and one image-level pooling branch. The branches are concatenated
    and projected back to `out_channels` via a 1x1 conv.

    Args:
        in_channels: Channels of the input feature.
        out_channels: Channels per branch and projection output.
        atrous_rates: Dilation rates for the 3 atrous branches.
            Use (6, 12, 18) for output_stride=16; (12, 24, 36) for OS=8.
        dropout_p: Dropout after the projection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        atrous_rates: tuple[int, int, int] = (6, 12, 18),
        dropout_p: float = 0.5,
    ):
        super().__init__()
        branches: list[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        ]
        for rate in atrous_rates:
            branches.append(ASPPConv(in_channels, out_channels, rate))
        branches.append(ASPPPooling(in_channels, out_channels))
        self.branches = nn.ModuleList(branches)

        self.project = nn.Sequential(
            nn.Conv2d(
                len(branches) * out_channels, out_channels,
                kernel_size=1, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [branch(x) for branch in self.branches]
        return self.project(torch.cat(outs, dim=1))


class DeepLabV3(nn.Module):
    """DeepLabv3 with a dilated ResNet50 backbone.

    Args:
        in_channels: Input image channels.
        n_classes: Output classes.
        output_stride: Backbone OS (16 paper-default; 8 for higher resolution).
        aspp_channels: Channels per ASPP branch.
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 20,
        output_stride: int = 16,
        aspp_channels: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        self.encoder = resnet50_encoder(in_channels=in_channels, output_stride=output_stride)
        backbone_out_ch = self.encoder.out_channels[-1]  # 2048

        atrous_rates = (6, 12, 18) if output_stride == 16 else (12, 24, 36)
        self.aspp = ASPP(backbone_out_ch, out_channels=aspp_channels, atrous_rates=atrous_rates)
        self.classifier = nn.Conv2d(aspp_channels, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> SkyScapesOutput:
        input_size = x.shape[-2:]
        feats = self.encoder(x)
        out = self.aspp(feats[-1])
        out = self.classifier(out)
        out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
        return SkyScapesOutput(seg=out)
