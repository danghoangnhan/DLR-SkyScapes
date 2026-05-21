"""AdapNet (Valada et al., ICRA 2017) — approximate baseline implementation.

Reference:
    Valada, Vertens, Dhall, Burgard.
    "AdapNet: Adaptive Semantic Segmentation in Adverse Environmental Conditions."
    ICRA 2017.

Used as a baseline in Table 4 (SkyScapes-Lane) of the SkyScapes paper.

This is a **best-effort approximation** — the original AdapNet replaces the
last ResNet stage with custom multi-scale residual units that mix parallel
atrous convolutions at multiple rates. Lacking a canonical public PyTorch
reference, we implement a faithful sketch:

- Dilated ResNet50 backbone (output_stride=8).
- 1x1 reduction conv brings c5 from 2048 ch down to a manageable width
  (default 512) before the multi-scale stack.
- A stack of `MultiscaleResidualBlock`s. Each block has three parallel atrous
  Conv3x3 branches at rates `(1, 2, 3)`, concatenated, then projected back
  with a 1x1 conv and added as a residual.
- 1x1 classifier + bilinear upsample to the input resolution.

Should be treated as "AdapNet-like" rather than a strict reproduction.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from skyscapesnet.models.outputs import SkyScapesOutput

from .backbones.resnet import resnet50_encoder


class MultiscaleResidualBlock(nn.Module):
    """Three parallel atrous 3x3 convs, concat, 1x1 project, residual.

    Args:
        channels: Input and output channels (preserved by the residual).
        atrous_rates: Dilation rates for the parallel branches.
        branch_reduction: Each branch reduces to `channels // branch_reduction`
            channels before being concatenated.
    """

    def __init__(
        self,
        channels: int,
        atrous_rates: tuple[int, ...] = (1, 2, 3),
        branch_reduction: int = 4,
    ):
        super().__init__()
        branch_ch = max(channels // branch_reduction, 1)
        self.branches = nn.ModuleList()
        for rate in atrous_rates:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels, branch_ch,
                        kernel_size=3, padding=rate, dilation=rate, bias=False,
                    ),
                    nn.BatchNorm2d(branch_ch),
                    nn.ReLU(inplace=True),
                )
            )
        self.merge = nn.Sequential(
            nn.Conv2d(len(atrous_rates) * branch_ch, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [branch(x) for branch in self.branches]
        merged = self.merge(torch.cat(outs, dim=1))
        return self.relu(x + merged)


class AdapNet(nn.Module):
    """AdapNet-like network with a dilated ResNet50 backbone.

    Args:
        in_channels: Input image channels.
        n_classes: Output classes.
        output_stride: Backbone OS (8 default, 16 lighter).
        n_multiscale_blocks: Number of stacked MultiscaleResidualBlocks on top of c5.
        atrous_rates: Dilation rates inside each MultiscaleResidualBlock.
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 20,
        output_stride: int = 8,
        bottleneck_channels: int = 512,
        n_multiscale_blocks: int = 4,
        atrous_rates: tuple[int, ...] = (1, 2, 3),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        self.encoder = resnet50_encoder(in_channels=in_channels, output_stride=output_stride)
        c5_ch = self.encoder.out_channels[-1]  # 2048

        # Reduce c5 width before the parallel-atrous stack — keeps params sane.
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c5_ch, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
        )
        self.multiscale = nn.Sequential(
            *[
                MultiscaleResidualBlock(bottleneck_channels, atrous_rates=atrous_rates)
                for _ in range(n_multiscale_blocks)
            ]
        )
        self.classifier = nn.Conv2d(bottleneck_channels, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> SkyScapesOutput:
        input_size = x.shape[-2:]
        feats = self.encoder(x)
        out = self.bottleneck(feats[-1])
        out = self.multiscale(out)
        out = self.classifier(out)
        out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
        return SkyScapesOutput(seg=out)
