"""Tests for segmentation and edge metrics."""
import torch

from skyscapesnet.metrics.edge import EdgeF1


def test_edge_f1_perfect_prediction_is_1():
    f1 = EdgeF1()
    # Perfect prediction: binary edge map, ground truth identical
    target = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=torch.long)
    pred = target.float().unsqueeze(0).unsqueeze(0)  # (1, 1, 2, 4) — logits = sigmoid input
    pred = (pred * 10) - 5  # very confident logits
    f1.update(pred, target.unsqueeze(0))
    assert f1.compute().item() > 0.99
