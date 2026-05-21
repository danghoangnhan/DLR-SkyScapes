"""U-Net (Ronneberger et al., 2015) — pure PyTorch baseline for SkyScapes.

Reference:
    Ronneberger, Fischer, Brox.
    "U-Net: Convolutional Networks for Biomedical Image Segmentation."
    MICCAI 2015. https://arxiv.org/abs/1505.04597

Used as a baseline in Tables 2 and 4 of the SkyScapes paper (Azimi et al., ICCV 2019).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from skyscapesnet.models.outputs import SkyScapesOutput


class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x 2."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """MaxPool 2x2 then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """Upsample, concatenate skip, then DoubleConv.

    Generic decoder up block supporting asymmetric encoder/skip channels.
    With `bilinear=True` upsampling is parameter-free and DoubleConv halves
    channels via its intermediate width. With `bilinear=False` the up step
    is a ConvTranspose2d that also halves channels (paper-faithful).

    Args:
        in_channels: Channels of the incoming feature (the one being upsampled).
        skip_channels: Channels of the skip feature being concatenated.
        out_channels: Output channels after DoubleConv.
        bilinear: Use bilinear upsampling (True) or ConvTranspose2d (False).
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        bilinear: bool = True,
    ):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            concat_channels = in_channels + skip_channels
            self.conv = DoubleConv(
                concat_channels, out_channels, mid_channels=concat_channels // 2,
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2,
            )
            concat_channels = in_channels // 2 + skip_channels
            self.conv = DoubleConv(concat_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dy = skip.size(-2) - x.size(-2)
        dx = skip.size(-1) - x.size(-1)
        if dx or dy:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """U-Net for semantic segmentation.

    Encoder: 5 stages, channels doubling from `base_channels`.
    Decoder: 4 upsampling stages with skip connections to the matching encoder stage.

    Args:
        in_channels: Number of input image channels.
        n_classes: Number of segmentation classes.
        base_channels: Channel width of the first encoder stage. Doubles each stage.
        bilinear: If True, use bilinear upsampling in the decoder (fewer params).
            If False, use ConvTranspose2d (paper-faithful).
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 20,
        base_channels: int = 64,
        bilinear: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_channels = base_channels
        self.bilinear = bilinear

        c = base_channels
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16 // factor)

        # Decoder: Up takes (in_channels, skip_channels, out_channels)
        self.up1 = Up(c * 16 // factor, c * 8, c * 8 // factor, bilinear)
        self.up2 = Up(c * 8 // factor, c * 4, c * 4 // factor, bilinear)
        self.up3 = Up(c * 4 // factor, c * 2, c * 2 // factor, bilinear)
        self.up4 = Up(c * 2 // factor, c, c, bilinear)
        self.outc = nn.Conv2d(c, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> SkyScapesOutput:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return SkyScapesOutput(seg=self.outc(x))
