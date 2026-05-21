"""Large-Kernel Boundary Refinement (LKBR) for SkyScapesNet.

Applied to skip connections to improve boundary accuracy for tiny objects.
Uses two parallel streams with large kernels for boundary refinement.

Reference: "SkyScapes — Fine-Grained Semantic Understanding of Aerial Scenes"
(Azimi et al., ICCV 2019), Section 4 and Figure 4.
Inspired by GCN (Peng et al., CVPR 2017).
"""

import torch
import torch.nn as nn

from .layers import SeparableConv2d


class LKBR(nn.Module):
    """Large-Kernel Boundary Refinement module.

    Two parallel streams applied to skip connection features:
    - Stream 1 (large kernel): Conv3x1 → Conv1x3 → element-wise Add
    - Stream 2 (local): Conv3x3 → SeparableConv3x3 → ReLU

    Output: Add(stream1, stream2) → Cat with original skip features

    This doubles the channel count of the skip connection (cat with original).

    Args:
        channels: Number of input channels in the skip connection.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # Stream 1: large-kernel decomposition (3x1 + 1x3)
        self.stream1_h = nn.Conv2d(
            channels, channels, kernel_size=(3, 1), padding=(1, 0), bias=False,
        )
        self.stream1_v = nn.Conv2d(
            channels, channels, kernel_size=(1, 3), padding=(0, 1), bias=False,
        )

        # Stream 2: local refinement
        self.stream2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            SeparableConv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, skip):
        # Stream 1: horizontal + vertical large kernel → add
        s1 = self.stream1_h(skip) + self.stream1_v(skip)

        # Stream 2: local conv refinement
        s2 = self.stream2(skip)

        # Combine streams and concatenate with original
        refined = s1 + s2
        return torch.cat([skip, refined], dim=1)
