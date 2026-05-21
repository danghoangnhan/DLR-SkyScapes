"""Sliding-window stitching callback (issue #8).

Aggregates softmaxed logits from overlapping predict patches into a per-tile
accumulator, averages overlap regions, argmaxes, writes mask + viz to disk.
"""
from __future__ import annotations

from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _accumulate(accumulator: torch.Tensor, counts: torch.Tensor,
                logits: torch.Tensor, top: int, left: int) -> None:
    """Add softmax(logits) into accumulator at [top:top+ph, left:left+pw]."""
    probs = F.softmax(logits, dim=1)
    _, c, ph, pw = probs.shape
    accumulator[:, :, top:top + ph, left:left + pw] += probs.squeeze(0)
    counts[top:top + ph, left:left + pw] += 1.0


def _finalize(accumulator: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Average accumulator by counts and argmax to a single (H, W) mask."""
    # Avoid division by zero in regions with no patch coverage
    safe_counts = counts.clamp(min=1).unsqueeze(0).unsqueeze(0)
    averaged = accumulator / safe_counts
    return averaged.argmax(dim=1).squeeze(0)


class StitchingCallback(L.Callback):
    """Lightning predict-mode callback. Writes one PNG per source tile."""

    def __init__(self, output_dir: str = "predictions"):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._accumulators: dict[int, torch.Tensor] = {}
        self._counts: dict[int, torch.Tensor] = {}

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        logits = outputs["logits"]            # (N, C, h, w)
        coords = outputs["coords"]            # (N, 4) - top, left, h, w
        tile_ids = outputs["tile_id"]         # (N,)
        tile_shapes = outputs.get("tile_shape")  # per-tile (H, W) lookup

        for i in range(logits.shape[0]):
            tile_id = int(tile_ids[i].item())
            top, left, ph, pw = coords[i].tolist()
            if tile_id not in self._accumulators:
                if tile_shapes is None:
                    raise RuntimeError("tile_shape missing from predict batch")
                H, W = tile_shapes[tile_id].tolist()
                C = logits.shape[1]
                self._accumulators[tile_id] = torch.zeros(1, C, H, W, device=logits.device)
                self._counts[tile_id] = torch.zeros(H, W, device=logits.device)

            _accumulate(self._accumulators[tile_id], self._counts[tile_id],
                        logits[i:i + 1], top, left)

    def on_predict_end(self, trainer, pl_module):
        for tile_id, acc in self._accumulators.items():
            pred = _finalize(acc.cpu(), self._counts[tile_id].cpu()).numpy().astype(np.uint8)
            Image.fromarray(pred).save(self.output_dir / f"tile_{tile_id:04d}_mask.png")
        self._accumulators.clear()
        self._counts.clear()
