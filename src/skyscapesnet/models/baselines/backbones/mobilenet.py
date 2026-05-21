"""MobileNetV2 encoder for the Mobile-U-Net baseline.

Reference:
    Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks."
    CVPR 2018. https://arxiv.org/abs/1801.04381

Stage boundaries are chosen so that forward returns 5 feature maps at strides
/2, /4, /8, /16, /32 — matching U-Net's encoder skip layout exactly.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class InvertedResidual(nn.Module):
    """MobileNetV2 inverted-residual block: 1x1 expand -> 3x3 dw -> 1x1 project."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int,
    ):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"stride must be 1 or 2; got {stride}")
        hidden = int(round(in_channels * expand_ratio))
        self.use_residual = stride == 1 and in_channels == out_channels

        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
            ]
        layers += [
            nn.Conv2d(
                hidden, hidden,
                kernel_size=3, stride=stride, padding=1,
                groups=hidden, bias=False,
            ),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


class MobileNetV2Encoder(nn.Module):
    """MobileNetV2 encoder exposing 5 feature maps at strides /2, /4, /8, /16, /32.

    Stage layout:
        stage1 (/2):  Conv3x3 s2 (3->32) + IR(32->16, s1, t1)
        stage2 (/4):  IR(16->24, s2, t6) + IR(24->24, s1, t6)
        stage3 (/8):  IR(24->32, s2, t6) + 2x IR(32->32, s1, t6)
        stage4 (/16): IR(32->64, s2, t6) + 3x IR(64->64, s1, t6)
                      + IR(64->96, s1, t6) + 2x IR(96->96, s1, t6)
        stage5 (/32): IR(96->160, s2, t6) + 2x IR(160->160, s1, t6)
                      + IR(160->320, s1, t6)
    """

    out_channels: list[int] = [16, 24, 32, 96, 320]

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            InvertedResidual(32, 16, stride=1, expand_ratio=1),
        )
        self.stage2 = self._stage(16, 24, n=2, stride=2, t=6)
        self.stage3 = self._stage(24, 32, n=3, stride=2, t=6)
        self.stage4 = nn.Sequential(
            *self._blocks(32, 64, n=4, stride=2, t=6),
            *self._blocks(64, 96, n=3, stride=1, t=6),
        )
        self.stage5 = nn.Sequential(
            *self._blocks(96, 160, n=3, stride=2, t=6),
            *self._blocks(160, 320, n=1, stride=1, t=6),
        )

    @staticmethod
    def _stage(in_c: int, out_c: int, n: int, stride: int, t: int) -> nn.Sequential:
        return nn.Sequential(*MobileNetV2Encoder._blocks(in_c, out_c, n, stride, t))

    @staticmethod
    def _blocks(
        in_c: int, out_c: int, n: int, stride: int, t: int,
    ) -> list[nn.Module]:
        layers: list[nn.Module] = [
            InvertedResidual(in_c, out_c, stride=stride, expand_ratio=t)
        ]
        for _ in range(1, n):
            layers.append(InvertedResidual(out_c, out_c, stride=1, expand_ratio=t))
        return layers

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return [c1, c2, c3, c4, c5]


def mobilenetv2_encoder(in_channels: int = 3) -> MobileNetV2Encoder:
    return MobileNetV2Encoder(in_channels=in_channels)
