"""Mobile-U-Net — U-Net decoder on a MobileNetV2 encoder.

The SkyScapes paper (Tables 2 and 4) reports a custom "Mobile-U-Net*" baseline.
The paper doesn't specify the decoder, so we use the standard U-Net decoder
(matching `unet.py`) but with separate `(in, skip, out)` channel
parameterization to handle MobileNetV2's asymmetric encoder widths.

Encoder strides /2, /4, /8, /16, /32 map directly to U-Net's 5 skip levels;
a final 2x upsample restores /1.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from skyscapesnet.models.outputs import SkyScapesOutput

from .backbones.mobilenet import MobileNetV2Encoder
from .unet import DoubleConv, Up


class MobileUNet(nn.Module):
    """U-Net with a MobileNetV2 encoder.

    Args:
        in_channels: Number of input image channels.
        n_classes: Number of segmentation classes.
        decoder_channels: Output channels of the 4 decoder Up blocks
            (one per skip level, in /16 → /2 order).
        bilinear: Use bilinear upsampling (True) or ConvTranspose2d (False)
            inside each Up block and for the final /2 → /1 step.
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 20,
        decoder_channels: tuple[int, int, int, int] = (96, 64, 32, 16),
        bilinear: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.encoder = MobileNetV2Encoder(in_channels=in_channels)
        c1, c2, c3, c4, c5 = self.encoder.out_channels  # [16, 24, 32, 96, 320]
        d1, d2, d3, d4 = decoder_channels

        # Decoder: 4 Up blocks bridging encoder stages /32→/16→/8→/4→/2
        self.up1 = Up(in_channels=c5, skip_channels=c4, out_channels=d1, bilinear=bilinear)
        self.up2 = Up(in_channels=d1, skip_channels=c3, out_channels=d2, bilinear=bilinear)
        self.up3 = Up(in_channels=d2, skip_channels=c2, out_channels=d3, bilinear=bilinear)
        self.up4 = Up(in_channels=d3, skip_channels=c1, out_channels=d4, bilinear=bilinear)

        # Final /2 → /1 upsample + conv (no skip from the original input).
        if bilinear:
            self.final_up: nn.Module = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True,
            )
        else:
            self.final_up = nn.ConvTranspose2d(d4, d4, kernel_size=2, stride=2)
        self.final_conv = DoubleConv(d4, d4)
        self.outc = nn.Conv2d(d4, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> SkyScapesOutput:
        input_size = x.shape[-2:]
        c1, c2, c3, c4, c5 = self.encoder(x)
        out = self.up1(c5, c4)
        out = self.up2(out, c3)
        out = self.up3(out, c2)
        out = self.up4(out, c1)
        out = self.final_up(out)
        # Guard against off-by-one in spatial size for odd-sized inputs.
        if out.shape[-2:] != input_size:
            out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
        out = self.final_conv(out)
        return SkyScapesOutput(seg=self.outc(out))
