"""Parity test: new torchmetrics-backed SemSegMetrics must agree with old ConfusionMatrix."""
import torch
import numpy as np

from skyscapesnet.utils.metrics import ConfusionMatrix
from skyscapesnet.metrics.seg import SemSegMetrics
from skyscapesnet.metrics.edge import EdgeF1


def test_miou_parity_with_legacy_confusion_matrix():
    n_classes = 5
    torch.manual_seed(0)
    pred = torch.randint(0, n_classes, (4, 32, 32))
    target = torch.randint(0, n_classes, (4, 32, 32))

    # Legacy
    cm = ConfusionMatrix(n_classes)
    cm.update(pred, target)
    legacy_miou = cm.mean_iou()
    legacy_acc = cm.pixel_accuracy()

    # New
    metrics = SemSegMetrics(n_classes=n_classes)
    metrics.update(pred, target)
    new_miou = metrics["miou"].compute().item()
    new_acc = metrics["pixel_acc"].compute().item()

    assert abs(legacy_miou - new_miou) < 1e-6, f"{legacy_miou} vs {new_miou}"
    assert abs(legacy_acc - new_acc) < 1e-6, f"{legacy_acc} vs {new_acc}"


def test_edge_f1_perfect_prediction_is_1():
    f1 = EdgeF1()
    # Perfect prediction: binary edge map, ground truth identical
    target = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=torch.long)
    pred = target.float().unsqueeze(0).unsqueeze(0)  # (1, 1, 2, 4) — logits = sigmoid input
    pred = (pred * 10) - 5  # very confident logits
    f1.update(pred, target.unsqueeze(0))
    assert f1.compute().item() > 0.99
