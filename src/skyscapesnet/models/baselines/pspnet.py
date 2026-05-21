"""PSPNet (Zhao et al., CVPR 2017) — Pyramid Scene Parsing Network.

Reference:
    Zhao, Shi, Qi, Wang, Jia.
    "Pyramid Scene Parsing Network." CVPR 2017.
    https://arxiv.org/abs/1612.01105

Used as a baseline in Tables 2 and 4 of the SkyScapes paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from skyscapesnet.models.outputs import SkyScapesOutput

from .backbones.resnet import resnet101_encoder


class PyramidPoolingModule(nn.Module):
    """PPM: pool to multiple bin sizes, project, upsample, concat with input.

    Args:
        in_channels: Channels of the input feature.
        pool_sizes: Adaptive-pool output sizes. PSPNet uses (1, 2, 3, 6).
        reduction_channels: Channels per pooled branch after the 1x1 projection.
            Defaults to `in_channels // len(pool_sizes)`.
    """

    def __init__(
        self,
        in_channels: int,
        pool_sizes: tuple[int, ...] = (1, 2, 3, 6),
        reduction_channels: int | None = None,
    ):
        super().__init__()
        if reduction_channels is None:
            reduction_channels = in_channels // len(pool_sizes)
        self.branches = nn.ModuleList()
        for ps in pool_sizes:
            self.branches.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(ps),
                    nn.Conv2d(in_channels, reduction_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(reduction_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.out_channels = in_channels + len(pool_sizes) * reduction_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        outs: list[torch.Tensor] = [x]
        for branch in self.branches:
            outs.append(
                F.interpolate(branch(x), size=(h, w), mode="bilinear", align_corners=False)
            )
        return torch.cat(outs, dim=1)


class PSPNet(nn.Module):
    """PSPNet with a dilated ResNet101 backbone (output_stride=8).

    Args:
        in_channels: Number of input image channels.
        n_classes: Number of segmentation classes.
        output_stride: Backbone OS (8 default — paper-faithful; 16 lighter).
        dropout_p: Dropout in the segmentation head.
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 20,
        output_stride: int = 8,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        self.encoder = resnet101_encoder(in_channels=in_channels, output_stride=output_stride)
        backbone_out_ch = self.encoder.out_channels[-1]  # 2048

        self.ppm = PyramidPoolingModule(in_channels=backbone_out_ch)
        ppm_out_ch = self.ppm.out_channels

        self.head = nn.Sequential(
            nn.Conv2d(ppm_out_ch, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(512, n_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> SkyScapesOutput:
        input_size = x.shape[-2:]
        feats = self.encoder(x)
        out = self.ppm(feats[-1])
        out = self.head(out)
        out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
        return SkyScapesOutput(seg=out)
