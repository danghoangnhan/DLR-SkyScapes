"""GCN (Peng et al., CVPR 2017) — "Large Kernel Matters".

Reference:
    Peng, Zhang, Yu, Luo, Sun.
    "Large Kernel Matters — Improve Semantic Segmentation by Global Convolutional Network."
    CVPR 2017. https://arxiv.org/abs/1703.02719

Used as a baseline in Tables 2 and 4 of the SkyScapes paper.

Each ResNet stage feature is passed through a Global Convolutional Network
(GCN) block that approximates a large kxk conv as two parallel separable
paths (1xk -> kx1 and kx1 -> 1xk). A Boundary Refinement (BR) residual
block then sharpens predictions; BR is reused at every fusion step.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from skyscapesnet.models.outputs import SkyScapesOutput

from .backbones.resnet import resnet152_encoder


class GCNBlock(nn.Module):
    """Two parallel separable-conv paths approximating a kxk convolution.

    Args:
        in_channels: Input channels (a ResNet stage output).
        out_channels: Output channels (typically n_classes).
        kernel_size: Large kernel size (15 in the paper).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 15):
        super().__init__()
        pad = kernel_size // 2
        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (kernel_size, 1), padding=(pad, 0)),
            nn.Conv2d(out_channels, out_channels, (1, kernel_size), padding=(0, pad)),
        )
        self.right = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0, pad)),
            nn.Conv2d(out_channels, out_channels, (kernel_size, 1), padding=(pad, 0)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.left(x) + self.right(x)


class BoundaryRefinement(nn.Module):
    """Residual refinement: x + Conv3x3 -> ReLU -> Conv3x3."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class GCN(nn.Module):
    """"Large Kernel Matters" segmentation network on a ResNet152 backbone.

    Args:
        in_channels: Input image channels.
        n_classes: Output classes.
        kernel_size: GCN large-kernel size (15 in the paper).
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 20,
        kernel_size: int = 15,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        self.encoder = resnet152_encoder(in_channels=in_channels, output_stride=32)
        c1, c2, c3, c4, c5 = self.encoder.out_channels  # [64, 256, 512, 1024, 2048]

        # GCN + BR at each scale
        self.gcn2 = GCNBlock(c2, n_classes, kernel_size)
        self.gcn3 = GCNBlock(c3, n_classes, kernel_size)
        self.gcn4 = GCNBlock(c4, n_classes, kernel_size)
        self.gcn5 = GCNBlock(c5, n_classes, kernel_size)

        self.br2 = BoundaryRefinement(n_classes)
        self.br3 = BoundaryRefinement(n_classes)
        self.br4 = BoundaryRefinement(n_classes)
        self.br5 = BoundaryRefinement(n_classes)

        # BR after each fusion (3 fusion steps) and one final BR at input resolution
        self.br_fuse4 = BoundaryRefinement(n_classes)
        self.br_fuse3 = BoundaryRefinement(n_classes)
        self.br_fuse2 = BoundaryRefinement(n_classes)
        self.br_final = BoundaryRefinement(n_classes)

    def forward(self, x: torch.Tensor) -> SkyScapesOutput:
        input_size = x.shape[-2:]
        _, c2, c3, c4, c5 = self.encoder(x)

        s5 = self.br5(self.gcn5(c5))  # /32
        s4 = self.br4(self.gcn4(c4))  # /16
        s3 = self.br3(self.gcn3(c3))  # /8
        s2 = self.br2(self.gcn2(c2))  # /4

        # Bottom-up fusion: upsample lower-res score map, add to next-finer, BR.
        out = F.interpolate(s5, size=s4.shape[-2:], mode="bilinear", align_corners=False) + s4
        out = self.br_fuse4(out)
        out = F.interpolate(out, size=s3.shape[-2:], mode="bilinear", align_corners=False) + s3
        out = self.br_fuse3(out)
        out = F.interpolate(out, size=s2.shape[-2:], mode="bilinear", align_corners=False) + s2
        out = self.br_fuse2(out)

        out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
        out = self.br_final(out)
        return SkyScapesOutput(seg=out)
