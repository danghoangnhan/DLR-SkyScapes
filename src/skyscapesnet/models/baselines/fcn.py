"""FCN-8s (Long et al., 2015) — pure PyTorch baseline.

Reference:
    Long, Shelhamer, Darrell.
    "Fully Convolutional Networks for Semantic Segmentation." CVPR 2015.
    https://arxiv.org/abs/1411.4038

VGG16 encoder + fc6/fc7 as convolutions + skip-fusion at /16 and /8 + final
8x upsample. Bilinear upsampling is used instead of the paper's learned
ConvTranspose2d to keep the implementation independent of input-size
divisibility; behaviourally equivalent on the SkyScapes evaluation.

Used as a baseline in Tables 2 and 4 of the SkyScapes paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from skyscapesnet.models.outputs import SkyScapesOutput

from .backbones.vgg import VGG16_CFG, VGGEncoder


class FCN8s(nn.Module):
    """FCN-8s segmentation head on top of a VGG16 encoder.

    Args:
        in_channels: Number of input image channels.
        n_classes: Number of segmentation classes.
        batch_norm: BatchNorm in the VGG encoder (default True, modern).
        dropout_p: Dropout after fc6 and fc7.
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 20,
        batch_norm: bool = True,
        dropout_p: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        self.encoder = VGGEncoder(
            cfg=VGG16_CFG,
            in_channels=in_channels,
            batch_norm=batch_norm,
            return_pool_indices=False,
        )

        # Classifier head: fc6 (7x7 conv) + fc7 (1x1 conv) + score (1x1 conv)
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.score_fr = nn.Conv2d(4096, n_classes, kernel_size=1)

        # Score heads on pool3 (/8) and pool4 (/16) for the skip fusion
        self.score_pool3 = nn.Conv2d(256, n_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, n_classes, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x: torch.Tensor) -> SkyScapesOutput:
        input_size = x.shape[-2:]
        features = self.encoder(x)
        pool3 = features[2]  # /8,  256 ch
        pool4 = features[3]  # /16, 512 ch
        pool5 = features[4]  # /32, 512 ch

        h = self.relu(self.fc6(pool5))
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        h = self.dropout(h)
        h = self.score_fr(h)  # /32, n_classes

        # Fuse with pool4 at /16
        h = F.interpolate(h, size=pool4.shape[-2:], mode="bilinear", align_corners=False)
        h = h + self.score_pool4(pool4)

        # Fuse with pool3 at /8
        h = F.interpolate(h, size=pool3.shape[-2:], mode="bilinear", align_corners=False)
        h = h + self.score_pool3(pool3)

        # Upsample to original input resolution
        h = F.interpolate(h, size=input_size, mode="bilinear", align_corners=False)
        return SkyScapesOutput(seg=h)
