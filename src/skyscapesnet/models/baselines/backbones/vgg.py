"""VGG encoder for FCN-8s and SegNet.

Reference:
    Simonyan & Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition."
    ICLR 2015. https://arxiv.org/abs/1409.1556
"""
from __future__ import annotations

import torch
import torch.nn as nn

VGG16_CFG: list[list[int]] = [
    [64, 64],
    [128, 128],
    [256, 256, 256],
    [512, 512, 512],
    [512, 512, 512],
]


class VGGEncoder(nn.Module):
    """5-stage VGG encoder, configurable per-stage conv channel widths.

    Each stage applies K (Conv3x3 -> BN -> ReLU) layers followed by
    MaxPool2d(2). Forward returns the 5 post-pool feature maps at strides
    /2, /4, /8, /16, /32, optionally paired with the pool indices needed by
    SegNet's MaxUnpool2d decoder.

    Args:
        cfg: Per-stage conv-channel counts. See `VGG16_CFG`.
        in_channels: Input image channels.
        batch_norm: Insert BatchNorm after each conv (default True; original
            VGG had no BN — kept for training stability without pretraining).
        return_pool_indices: Pool with return_indices=True (SegNet decoder).
    """

    def __init__(
        self,
        cfg: list[list[int]],
        in_channels: int = 3,
        batch_norm: bool = True,
        return_pool_indices: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.return_pool_indices = return_pool_indices

        self.stages = nn.ModuleList()
        self.pools = nn.ModuleList()
        c = in_channels
        for stage_channels in cfg:
            layers: list[nn.Module] = []
            for out_c in stage_channels:
                layers.append(
                    nn.Conv2d(c, out_c, kernel_size=3, padding=1, bias=not batch_norm)
                )
                if batch_norm:
                    layers.append(nn.BatchNorm2d(out_c))
                layers.append(nn.ReLU(inplace=True))
                c = out_c
            self.stages.append(nn.Sequential(*layers))
            self.pools.append(
                nn.MaxPool2d(kernel_size=2, stride=2, return_indices=return_pool_indices)
            )

        self.out_channels: list[int] = [stage[-1] for stage in cfg]

    def forward(
        self, x: torch.Tensor
    ) -> list[torch.Tensor] | tuple[list[torch.Tensor], list[torch.Tensor]]:
        features: list[torch.Tensor] = []
        indices: list[torch.Tensor] = []
        for stage, pool in zip(self.stages, self.pools):
            x = stage(x)
            if self.return_pool_indices:
                x, idx = pool(x)
                indices.append(idx)
            else:
                x = pool(x)
            features.append(x)
        if self.return_pool_indices:
            return features, indices
        return features

    def forward_with_sizes(
        self, x: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Size]]:
        """Forward also returning pre-pool spatial sizes (for SegNet's MaxUnpool2d).

        Only valid when `return_pool_indices=True`. Returns
        (post_pool_features, pool_indices, pre_pool_sizes), where
        `pre_pool_sizes[i]` is the shape MaxUnpool2d should restore at stage i.
        """
        if not self.return_pool_indices:
            raise RuntimeError(
                "forward_with_sizes requires return_pool_indices=True"
            )
        features: list[torch.Tensor] = []
        indices: list[torch.Tensor] = []
        pre_pool_sizes: list[torch.Size] = []
        for stage, pool in zip(self.stages, self.pools):
            x = stage(x)
            pre_pool_sizes.append(x.shape)
            x, idx = pool(x)
            indices.append(idx)
            features.append(x)
        return features, indices, pre_pool_sizes


def vgg16_encoder(
    in_channels: int = 3,
    batch_norm: bool = True,
    return_pool_indices: bool = False,
) -> VGGEncoder:
    """VGG16-style encoder (13 conv layers across 5 stages).

    Used by FCN-8s (default kwargs) and SegNet (`return_pool_indices=True`).
    """
    return VGGEncoder(
        cfg=VGG16_CFG,
        in_channels=in_channels,
        batch_norm=batch_norm,
        return_pool_indices=return_pool_indices,
    )
