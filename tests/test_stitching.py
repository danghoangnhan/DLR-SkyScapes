"""Verify StitchingCallback reconstructs a known-input tile."""
import torch

from skyscapesnet.callbacks.stitching import _accumulate, _finalize


def test_overlapping_patches_average_correctly():
    """Two overlapping patches with constant class logits should produce
    a clean argmax over the union region."""
    accumulator = torch.zeros(1, 3, 32, 64)  # (n_classes=3, H, W)
    counts = torch.zeros(32, 64)

    # Patch 1: cells [0:32, 0:32] = class 0
    logits1 = torch.zeros(1, 3, 32, 32)
    logits1[0, 0] = 5.0
    _accumulate(accumulator, counts, logits1, top=0, left=0)

    # Patch 2: cells [0:32, 16:48] = class 0 too
    logits2 = torch.zeros(1, 3, 32, 32)
    logits2[0, 0] = 5.0
    _accumulate(accumulator, counts, logits2, top=0, left=16)

    pred = _finalize(accumulator, counts)
    assert pred.shape == (32, 64)
    assert (pred == 0).all()
