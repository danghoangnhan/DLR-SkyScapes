"""Full-Resolution Separable Residual (FRSR) module for SkyScapesNet.

Inspired by FRRU from FRRN (Pohlen et al., CVPR 2017), but uses separable
convolutions. FRSR has two processing streams:
- Residual stream: maintains full resolution for better localization
- Pooling stream: downsamples for better recognition, then upsamples back

Reference: "SkyScapes — Fine-Grained Semantic Understanding of Aerial Scenes"
(Azimi et al., ICCV 2019), Section 4 and Figure 4.
"""

import torch.nn as nn
import torch.nn.functional as F

from .layers import SeparableConv2d


class FRSR(nn.Module):
    """Full-Resolution Separable Residual unit.

    Two parallel streams:
    1. Residual stream: keeps features at full resolution for localization
    2. Pooling stream: downsamples → separable convs → upsamples → adds to residual

    Args:
        channels: Number of input/output channels.
        pool_size: Downsampling factor for the pooling stream.
    """

    def __init__(self, channels, pool_size=2):
        super().__init__()
        self.pool_size = pool_size

        # Pooling stream
        self.pool_stream = nn.Sequential(
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_size),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            SeparableConv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            SeparableConv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )

        # Residual stream: lightweight processing to maintain full resolution
        self.residual_stream = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            SeparableConv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        h, w = x.shape[2:]

        # Residual stream at full resolution
        residual = self.residual_stream(x)

        # Pooling stream: downsample → process → upsample
        pooled = self.pool_stream(x)
        pooled = F.interpolate(pooled, size=(h, w), mode="nearest")

        # Combine: add pooling features to residual
        return residual + pooled
