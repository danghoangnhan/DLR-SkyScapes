"""Test single-head SegLoss for Lane-13 / Category-11."""
import torch

from skyscapesnet.losses.loss import SegLoss
from skyscapesnet.models.outputs import SkyScapesOutput


def test_seg_loss_runs_and_backprops():
    criterion = SegLoss(n_classes=13)
    seg = torch.randn(2, 13, 32, 32, requires_grad=True)
    target = torch.randint(0, 13, (2, 32, 32))
    out = SkyScapesOutput(seg=seg)
    loss, components = criterion(out, target)
    assert loss.requires_grad
    loss.backward()
    assert "seg_ce" in components and "seg_iou" in components
