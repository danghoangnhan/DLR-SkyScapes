"""RefineNet (Lin et al., CVPR 2017) — multi-path refinement segmentation.

Reference:
    Lin, Milan, Shen, Reid.
    "RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation."
    CVPR 2017. https://arxiv.org/abs/1611.06612

Used as a baseline in Table 2 (SkyScapes-Dense) of the SkyScapes paper.

Architecture (RefineNet-4-cascaded with ResNet101):
- ResNet101 stages c2 (/4), c3 (/8), c4 (/16), c5 (/32).
- Project each stage to a uniform `channels` (256) via 1x1 convs.
- Four RefineNet blocks cascaded from /32 → /4. Each block applies:
    RCU (Residual Conv Unit) ×2 on each input
    → MRF (Multi-Resolution Fusion) if multi-input
    → CRP (Chained Residual Pooling)
    → RCU (output).
- Classifier 1x1 → bilinear upsample to input.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from skyscapesnet.models.outputs import SkyScapesOutput

from .backbones.resnet import resnet101_encoder


class RCU(nn.Module):
    """Residual Conv Unit: ReLU -> Conv 3x3 -> ReLU -> Conv 3x3, residual."""

    def __init__(self, channels: int):
        super().__init__()
        # Note: ReLU is applied BEFORE the first conv (pre-activation) so the residual
        # skip connection can carry full-range activations; matches the paper.
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        return out + residual


class MultiResolutionFusion(nn.Module):
    """Project each input to `out_channels`, upsample to the highest input size, sum."""

    def __init__(self, out_channels: int, *in_channels: int):
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Conv2d(c, out_channels, kernel_size=3, padding=1, bias=False) for c in in_channels]
        )

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        projected = [proj(x) for proj, x in zip(self.projections, inputs)]
        # Target spatial size: the largest input (finest resolution).
        target_size = max((p.shape[-2:] for p in projected), key=lambda hw: hw[0] * hw[1])
        upsampled = [
            p if p.shape[-2:] == target_size
            else F.interpolate(p, size=target_size, mode="bilinear", align_corners=False)
            for p in projected
        ]
        return sum(upsampled)


class ChainedResidualPooling(nn.Module):
    """ReLU then a chain of (MaxPool 5x5 stride 1, Conv 3x3) blocks summed to the input."""

    def __init__(self, channels: int, n_pools: int = 4):
        super().__init__()
        self.pools = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=5, stride=1, padding=2) for _ in range(n_pools)]
        )
        self.convs = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False) for _ in range(n_pools)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x)
        out = x
        for pool, conv in zip(self.pools, self.convs):
            x = conv(pool(x))
            out = out + x
        return out


class RefineBlock(nn.Module):
    """RefineNet block: RCU(s) per input -> MRF -> CRP -> RCU.

    Args:
        out_channels: Operating channel count of the block.
        in_channels_list: Channels of each incoming feature.
    """

    def __init__(self, out_channels: int, *in_channels_list: int):
        super().__init__()
        # Two RCUs on each input (per-input refinement)
        self.input_rcus = nn.ModuleList(
            [nn.Sequential(RCU(c), RCU(c)) for c in in_channels_list]
        )
        # MRF if more than one input; otherwise just project to out_channels.
        if len(in_channels_list) > 1:
            self.fusion = MultiResolutionFusion(out_channels, *in_channels_list)
        else:
            in_ch = in_channels_list[0]
            self.fusion = (
                nn.Identity() if in_ch == out_channels
                else nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1, bias=False)
            )
        self.crp = ChainedResidualPooling(out_channels)
        self.output_rcu = RCU(out_channels)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        refined = [rcu(x) for rcu, x in zip(self.input_rcus, inputs)]
        if isinstance(self.fusion, MultiResolutionFusion):
            fused = self.fusion(*refined)
        else:
            fused = self.fusion(refined[0])
        return self.output_rcu(self.crp(fused))


class RefineNet(nn.Module):
    """RefineNet-4-cascaded with a ResNet101 backbone.

    Args:
        in_channels: Input image channels.
        n_classes: Output classes.
        channels: Operating channel count across all RefineNet blocks (256 default).
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 20,
        channels: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        self.encoder = resnet101_encoder(in_channels=in_channels, output_stride=32)
        _, c2_ch, c3_ch, c4_ch, c5_ch = self.encoder.out_channels  # 256, 512, 1024, 2048

        # Adapt convs project each ResNet stage to `channels`.
        self.adapt_c2 = nn.Conv2d(c2_ch, channels, kernel_size=1, bias=False)
        self.adapt_c3 = nn.Conv2d(c3_ch, channels, kernel_size=1, bias=False)
        self.adapt_c4 = nn.Conv2d(c4_ch, channels, kernel_size=1, bias=False)
        self.adapt_c5 = nn.Conv2d(c5_ch, channels, kernel_size=1, bias=False)

        # Four cascaded RefineNet blocks
        self.refine4 = RefineBlock(channels, channels)                 # input: c5
        self.refine3 = RefineBlock(channels, channels, channels)       # inputs: refine4 + c4
        self.refine2 = RefineBlock(channels, channels, channels)       # inputs: refine3 + c3
        self.refine1 = RefineBlock(channels, channels, channels)       # inputs: refine2 + c2

        self.classifier = nn.Conv2d(channels, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> SkyScapesOutput:
        input_size = x.shape[-2:]
        _, c2, c3, c4, c5 = self.encoder(x)

        p2 = self.adapt_c2(c2)
        p3 = self.adapt_c3(c3)
        p4 = self.adapt_c4(c4)
        p5 = self.adapt_c5(c5)

        r4 = self.refine4(p5)
        r3 = self.refine3(r4, p4)
        r2 = self.refine2(r3, p3)
        r1 = self.refine1(r2, p2)

        out = self.classifier(r1)
        out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
        return SkyScapesOutput(seg=out)
