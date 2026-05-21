"""Output container for SkyScapesNet forward pass.

Dataclass + pytree registration so torch.compile / torch.func / Lightning
all see it as a normal tensor container.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor
from torch.utils import _pytree as pytree


@dataclass
class SkyScapesOutput:
    """Multi-task forward output.

    Attributes:
        seg: (N, C_seg, H, W) — always present.
        multi_edge: (N, C_seg, H, W) — Dense-task only.
        binary_edge: (N, 1, H, W) — Dense-task only.
    """
    seg: Tensor
    multi_edge: Tensor | None = None
    binary_edge: Tensor | None = None


def _flatten(out: SkyScapesOutput):
    values = [getattr(out, f.name) for f in fields(out)]
    keys = [f.name for f in fields(out)]
    # Filter out None so the flat list only contains real tensors.
    flat = [v for v in values if v is not None]
    none_mask = [v is None for v in values]
    return flat, (keys, none_mask)


def _unflatten(flat, context):
    keys, none_mask = context
    kwargs = {}
    it = iter(flat)
    for k, is_none in zip(keys, none_mask):
        kwargs[k] = None if is_none else next(it)
    return SkyScapesOutput(**kwargs)


pytree.register_pytree_node(SkyScapesOutput, _flatten, _unflatten)
