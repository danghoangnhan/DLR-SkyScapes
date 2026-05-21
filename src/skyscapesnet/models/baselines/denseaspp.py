"""DenseASPP (Yang et al., CVPR 2018) for semantic segmentation.

Reference:
    Yang, Yu, Zhang, Li, Yang.
    "DenseASPP for Semantic Segmentation in Street Scenes." CVPR 2018.
    https://arxiv.org/abs/1808.06321 (the open-access proceedings version)

Used as a baseline in Tables 2 and 4 of the SkyScapes paper.

The DenseASPP head sits on top of a DenseNet161 backbone (output_stride=16
in this implementation — the paper's OS=8 would require also dilating the
DenseNet161 trans2, which our backbone does not currently expose).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from skyscapesnet.models.outputs import SkyScapesOutput

from .backbones.densenet import densenet161_encoder


class DenseAtrousLayer(nn.Module):
    """One DenseASPP branch.

    BN -> ReLU -> Conv1x1 (bottleneck to `inter_channels`)
    -> BN -> ReLU -> Conv3x3 atrous (-> `out_channels` new features)
    -> Dropout
    The output is concatenated with the input, so subsequent branches receive
    all prior outputs (the "dense" connection pattern).
    """

    def __init__(
        self,
        in_channels: int,
        inter_channels: int,
        out_channels: int,
        dilation: int,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                inter_channels, out_channels,
                kernel_size=3, padding=dilation, dilation=dilation, bias=False,
            ),
            nn.Dropout2d(p=dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new = self.block(x)
        return torch.cat([x, new], dim=1)


class DenseASPPModule(nn.Module):
    """Five atrous-conv branches with dense connections.

    Args:
        in_channels: Channels of the input feature (DenseNet161 c5 = 2208).
        inter_channels: Bottleneck width in each branch (paper: 512).
        out_channels: Channels added by each branch (paper: 128).
        atrous_rates: Dilation rates for the 5 branches (paper: 3, 6, 12, 18, 24).
        dropout_p: Per-branch dropout.
    """

    def __init__(
        self,
        in_channels: int,
        inter_channels: int = 512,
        out_channels: int = 128,
        atrous_rates: tuple[int, ...] = (3, 6, 12, 18, 24),
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.branches = nn.ModuleList()
        c = in_channels
        for rate in atrous_rates:
            self.branches.append(
                DenseAtrousLayer(c, inter_channels, out_channels, rate, dropout_p)
            )
            c += out_channels
        self.out_channels = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for branch in self.branches:
            x = branch(x)
        return x


class DenseASPP(nn.Module):
    """DenseASPP with a DenseNet161 backbone.

    Args:
        in_channels: Input image channels.
        n_classes: Output classes.
        output_stride: DenseNet161 OS (16 default; 32 if you want a lighter run).
        inter_channels: Bottleneck width in each DenseASPP branch.
        out_channels: Channels added per DenseASPP branch.
        atrous_rates: Dilation rates for the 5 DenseASPP branches.
        dropout_p: Dropout inside each DenseASPP branch.
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 20,
        output_stride: int = 16,
        inter_channels: int = 512,
        out_channels: int = 128,
        atrous_rates: tuple[int, ...] = (3, 6, 12, 18, 24),
        dropout_p: float = 0.1,
    ):
        super().__init__()
        if output_stride not in (16, 32):
            raise ValueError(
                "DenseASPP requires DenseNet161 with output_stride in {16, 32}; "
                f"got {output_stride}"
            )
        self.in_channels = in_channels
        self.n_classes = n_classes

        self.encoder = densenet161_encoder(in_channels=in_channels, output_stride=output_stride)
        backbone_out_ch = self.encoder.out_channels[-1]  # 2208

        self.aspp = DenseASPPModule(
            in_channels=backbone_out_ch,
            inter_channels=inter_channels,
            out_channels=out_channels,
            atrous_rates=atrous_rates,
            dropout_p=dropout_p,
        )
        self.classifier = nn.Conv2d(self.aspp.out_channels, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> SkyScapesOutput:
        input_size = x.shape[-2:]
        feats = self.encoder(x)
        out = self.aspp(feats[-1])
        out = self.classifier(out)
        out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
        return SkyScapesOutput(seg=out)
