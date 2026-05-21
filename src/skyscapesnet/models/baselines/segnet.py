"""SegNet (Badrinarayanan et al., 2017) — pure PyTorch baseline.

Reference:
    Badrinarayanan, Kendall, Cipolla.
    "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation."
    IEEE TPAMI 2017. https://arxiv.org/abs/1511.00561

Encoder: VGG16's 13 conv layers, pooling with stored max-pool indices.
Decoder: symmetric mirror — MaxUnpool2d using the stored indices, then matching
convs. The final conv outputs class logits with no BN/ReLU.

Used as a baseline in Table 2 (SkyScapes-Dense) of the SkyScapes paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from skyscapesnet.models.outputs import SkyScapesOutput

from .backbones.vgg import VGG16_CFG, VGGEncoder


def _build_decoder_stage_cfg(
    encoder_cfg: list[list[int]], n_classes: int,
) -> list[list[int]]:
    """Mirror the encoder cfg to determine the decoder's per-stage conv channels.

    The first K-1 convs of each decoder stage preserve channel count; the K-th
    drops to the matching previous encoder stage's output (or n_classes for the
    final decoder stage).
    """
    encoder_stage_outs = [stage[-1] for stage in encoder_cfg]
    decoder_stage_cfgs: list[list[int]] = []
    n_stages = len(encoder_cfg)
    for i, stage in enumerate(reversed(encoder_cfg)):
        n_convs = len(stage)
        in_c = encoder_stage_outs[-(i + 1)]
        if i == n_stages - 1:
            target_out_c = n_classes
        else:
            target_out_c = encoder_stage_outs[-(i + 2)]
        decoder_stage_cfgs.append([in_c] * (n_convs - 1) + [target_out_c])
    return decoder_stage_cfgs


def _build_decoder_block(
    in_channels: int,
    convs_channels: list[int],
    batch_norm: bool,
    is_final_stage: bool,
) -> nn.Sequential:
    """Build one decoder stage as Sequential(Conv-BN-ReLU x K), where the very
    last conv of the very last stage is bare (no BN/ReLU)."""
    layers: list[nn.Module] = []
    c = in_channels
    for j, out_c in enumerate(convs_channels):
        is_final_conv = is_final_stage and (j == len(convs_channels) - 1)
        layers.append(
            nn.Conv2d(
                c, out_c,
                kernel_size=3, padding=1,
                bias=is_final_conv or not batch_norm,
            )
        )
        if not is_final_conv:
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
        c = out_c
    return nn.Sequential(*layers)


class SegNet(nn.Module):
    """SegNet for semantic segmentation.

    Args:
        in_channels: Number of input image channels.
        n_classes: Number of segmentation classes.
        encoder_cfg: Per-stage conv-channel counts (VGG16 by default).
        batch_norm: Insert BatchNorm after each conv.
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 20,
        encoder_cfg: list[list[int]] | None = None,
        batch_norm: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        cfg = encoder_cfg if encoder_cfg is not None else VGG16_CFG
        self.encoder = VGGEncoder(
            cfg=cfg,
            in_channels=in_channels,
            batch_norm=batch_norm,
            return_pool_indices=True,
        )

        decoder_stage_cfgs = _build_decoder_stage_cfg(cfg, n_classes)
        encoder_stage_outs = [stage[-1] for stage in cfg]
        n_stages = len(cfg)

        self.decoder_unpools = nn.ModuleList(
            [nn.MaxUnpool2d(kernel_size=2, stride=2) for _ in range(n_stages)]
        )
        self.decoder_blocks = nn.ModuleList()
        for i, stage_cfg in enumerate(decoder_stage_cfgs):
            in_c = encoder_stage_outs[-(i + 1)]
            self.decoder_blocks.append(
                _build_decoder_block(
                    in_c, stage_cfg, batch_norm, is_final_stage=(i == n_stages - 1),
                )
            )

    def forward(self, x: torch.Tensor) -> SkyScapesOutput:
        features, indices, sizes = self.encoder.forward_with_sizes(x)
        # features[-1] is the deepest post-pool feature (/32 for VGG16)
        out = features[-1]
        for i, (unpool, block) in enumerate(zip(self.decoder_unpools, self.decoder_blocks)):
            idx = indices[-(i + 1)]
            target_size = sizes[-(i + 1)]
            out = unpool(out, idx, output_size=target_size)
            out = block(out)
        return SkyScapesOutput(seg=out)
