"""Dilated ResNet encoder (ResNet50 / 101 / 152) for segmentation baselines.

Reference:
    He et al., "Deep Residual Learning for Image Recognition." CVPR 2016.
    https://arxiv.org/abs/1512.03385

The dilated variants follow the standard practice introduced by DeepLab:
    output_stride=32: vanilla ResNet (stage4 stride 2, no dilation).
    output_stride=16: stage4 stride 1, dilation 2 — preserves /16 resolution.
    output_stride=8:  stage3 stride 1 dilation 2, stage4 stride 1 dilation 4.

Used by PSPNet, DeepLabv3, DeepLabv3+, GCN, RefineNet, BiSeNet.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    """ResNet bottleneck block: 1x1 -> 3x3 (optional dilation) -> 1x1, with identity skip."""

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module | None = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes,
            kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNetEncoder(nn.Module):
    """Dilated ResNet encoder producing 5 feature maps.

    Forward returns [c1, c2, c3, c4, c5] where:
        c1: post-stem (stride 2, 64 ch)
        c2: post-layer1 (stride 4, 256 ch)
        c3: post-layer2 (stride 8, 512 ch)
        c4: post-layer3 (stride 16 or 8 if dilated, 1024 ch)
        c5: post-layer4 (stride 32, 16, or 8, 2048 ch)

    Args:
        n_blocks: Per-layer block counts. ResNet50=[3,4,6,3], 101=[3,4,23,3], 152=[3,8,36,3].
        in_channels: Input image channels.
        output_stride: 32 | 16 | 8.
    """

    out_channels: list[int] = [64, 256, 512, 1024, 2048]

    def __init__(
        self,
        n_blocks: list[int],
        in_channels: int = 3,
        output_stride: int = 32,
    ):
        super().__init__()
        if output_stride not in (32, 16, 8):
            raise ValueError(f"output_stride must be 32, 16, or 8; got {output_stride}")
        self.output_stride = output_stride

        # Stem (stride 4)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if output_stride == 32:
            strides = [1, 2, 2, 2]
            dilations = [1, 1, 1, 1]
        elif output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        else:  # 8
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]

        self._in_channels = 64
        self.layer1 = self._make_layer(64, n_blocks[0], stride=strides[0], dilation=dilations[0])
        self.layer2 = self._make_layer(128, n_blocks[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(256, n_blocks[2], stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_layer(512, n_blocks[3], stride=strides[3], dilation=dilations[3])

    def _make_layer(
        self, planes: int, n_blocks: int, stride: int, dilation: int,
    ) -> nn.Sequential:
        out_channels = planes * Bottleneck.expansion
        downsample: nn.Module | None = None
        if stride != 1 or self._in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self._in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        layers = [Bottleneck(self._in_channels, planes, stride, dilation, downsample)]
        self._in_channels = out_channels
        for _ in range(1, n_blocks):
            layers.append(Bottleneck(self._in_channels, planes, stride=1, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        c1 = self.relu(self.bn1(self.conv1(x)))   # /2
        x = self.maxpool(c1)                       # /4
        c2 = self.layer1(x)                        # /4
        c3 = self.layer2(c2)                       # /8
        c4 = self.layer3(c3)                       # /16 or /8 (dilated)
        c5 = self.layer4(c4)                       # /32, /16, or /8
        return [c1, c2, c3, c4, c5]


def resnet50_encoder(in_channels: int = 3, output_stride: int = 32) -> ResNetEncoder:
    return ResNetEncoder([3, 4, 6, 3], in_channels, output_stride)


def resnet101_encoder(in_channels: int = 3, output_stride: int = 32) -> ResNetEncoder:
    return ResNetEncoder([3, 4, 23, 3], in_channels, output_stride)


def resnet152_encoder(in_channels: int = 3, output_stride: int = 32) -> ResNetEncoder:
    return ResNetEncoder([3, 8, 36, 3], in_channels, output_stride)
