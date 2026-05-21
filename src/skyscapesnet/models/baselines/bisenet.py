"""BiSeNet (Yu et al., ECCV 2018) — Bilateral Segmentation Network.

Reference:
    Yu, Wang, Peng, Gao, Yu, Sang.
    "BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation."
    ECCV 2018. https://arxiv.org/abs/1808.00897

Used as a baseline in Tables 2 (SkyScapes-Dense) and 4 (SkyScapes-Lane) of the
SkyScapes paper.

Two parallel paths:
- **Spatial Path (SP):** 3 stride-2 Conv-BN-ReLU stages preserving spatial detail.
- **Context Path (CP):** dilated ResNet50 backbone with Attention Refinement
  Modules (ARM) at /16 and /32, plus a global-pool "tail" branch.
A Feature Fusion Module (FFM) with channel attention combines the two paths
before the final classifier.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from skyscapesnet.models.outputs import SkyScapesOutput

from .backbones.resnet import resnet50_encoder


class _ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
    ):
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class SpatialPath(nn.Module):
    """3 stride-2 Conv-BN-ReLU stages producing a /8 spatial feature map."""

    def __init__(self, in_channels: int = 3, out_channels: int = 256):
        super().__init__()
        c = 64
        self.stage1 = _ConvBNReLU(in_channels, c, kernel_size=7, stride=2, padding=3)
        self.stage2 = _ConvBNReLU(c, c * 2, kernel_size=3, stride=2)
        self.stage3 = _ConvBNReLU(c * 2, c * 4, kernel_size=3, stride=2)
        self.project = _ConvBNReLU(c * 4, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.stage3(self.stage2(self.stage1(x))))


class AttentionRefinementModule(nn.Module):
    """GAP -> Conv 1x1 -> BN -> Sigmoid -> multiply with input."""

    def __init__(self, channels: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.sigmoid(self.bn(self.conv(self.gap(x))))
        return x * attention


class FeatureFusionModule(nn.Module):
    """Concat SP + CP features, fuse via 1x1 conv, apply channel attention."""

    def __init__(self, sp_channels: int, cp_channels: int, out_channels: int):
        super().__init__()
        self.fuse = _ConvBNReLU(sp_channels + cp_channels, out_channels, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(out_channels, out_channels // 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sp: torch.Tensor, cp: torch.Tensor) -> torch.Tensor:
        fused = self.fuse(torch.cat([sp, cp], dim=1))
        attention = self.sigmoid(self.conv2(self.relu(self.conv1(self.gap(fused)))))
        return fused + fused * attention


class BiSeNet(nn.Module):
    """BiSeNet with a ResNet50 context-path backbone.

    Args:
        in_channels: Input image channels.
        n_classes: Output classes.
        sp_channels: Spatial Path output channels (default 256).
        cp_channels: Channels after CP-side ARMs / tail projection (default 128).
        ffm_channels: Output channels of the Feature Fusion Module.
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 20,
        sp_channels: int = 256,
        cp_channels: int = 128,
        ffm_channels: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        # Spatial Path: stays at /8 throughout
        self.spatial_path = SpatialPath(in_channels=in_channels, out_channels=sp_channels)

        # Context Path: standard ResNet50, then ARMs and tail
        self.encoder = resnet50_encoder(in_channels=in_channels, output_stride=32)
        _, _, _, c4_ch, c5_ch = self.encoder.out_channels  # 1024, 2048

        # Project c4 / c5 to cp_channels before ARM
        self.proj_c4 = _ConvBNReLU(c4_ch, cp_channels, kernel_size=1)
        self.proj_c5 = _ConvBNReLU(c5_ch, cp_channels, kernel_size=1)
        self.arm_c4 = AttentionRefinementModule(cp_channels)
        self.arm_c5 = AttentionRefinementModule(cp_channels)

        # Tail: GAP on c5 -> 1x1 conv to cp_channels (broadcast back to c5 size)
        self.tail = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            _ConvBNReLU(c5_ch, cp_channels, kernel_size=1),
        )

        # Fusion + classifier
        self.ffm = FeatureFusionModule(
            sp_channels=sp_channels, cp_channels=cp_channels, out_channels=ffm_channels,
        )
        self.classifier = nn.Conv2d(ffm_channels, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> SkyScapesOutput:
        input_size = x.shape[-2:]

        sp = self.spatial_path(x)  # /8, sp_channels

        feats = self.encoder(x)
        c4, c5 = feats[3], feats[4]  # /16 and /32 (output_stride=32)

        # Tail: broadcast global context back to c5 size, add to projected c5
        c5_proj = self.proj_c5(c5)
        tail = F.interpolate(self.tail(c5), size=c5_proj.shape[-2:], mode="bilinear", align_corners=False)
        c5_arm = self.arm_c5(c5_proj + tail)

        c4_arm = self.arm_c4(self.proj_c4(c4))

        # Combine /32 → /16, then upsample /16 → /8 (match SP)
        cp = F.interpolate(c5_arm, size=c4_arm.shape[-2:], mode="bilinear", align_corners=False)
        cp = cp + c4_arm
        cp = F.interpolate(cp, size=sp.shape[-2:], mode="bilinear", align_corners=False)

        fused = self.ffm(sp, cp)
        out = self.classifier(fused)
        out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
        return SkyScapesOutput(seg=out)
