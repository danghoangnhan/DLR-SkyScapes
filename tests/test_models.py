"""Verification tests for all models."""

import tempfile

import torch
import pytest

from skyscapesnet.models.fc_densenet import FCDenseNet
from skyscapesnet.models.craspp import CRASPP
from skyscapesnet.models.skyscapesnet import SkyScapesNet
from skyscapesnet.losses.loss import MultiTaskLoss
from skyscapesnet.utils.metrics import ConfusionMatrix


def test_fc_densenet103():
    model = FCDenseNet(in_channels=3, n_classes=20)
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    assert out.shape == (1, 20, 256, 256), f"Expected (1, 20, 256, 256), got {out.shape}"
    print(f"[PASS] FC-DenseNet103 | Output: {out.shape} | Params: {n_params:,}")


def test_fc_densenet_backbone():
    model = FCDenseNet(in_channels=3, n_classes=None)
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    assert out.shape[0] == 1 and out.shape[2] == 256 and out.shape[3] == 256
    print(f"[PASS] FC-DenseNet103 backbone | Output: {out.shape} (raw features)")


def test_craspp():
    model = CRASPP(in_channels=656, out_channels=240)
    model.eval()  # avoid BN issue with 1x1 spatial
    x = torch.randn(2, 656, 8, 8)
    out, new = model(x)
    assert out.shape == (2, 240, 8, 8), f"Expected (2, 240, 8, 8), got {out.shape}"
    print(f"[PASS] CRASPP | Output: {out.shape}")


def test_skyscapesnet():
    model = SkyScapesNet(in_channels=3, n_classes=20)
    model.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    seg, multi_edge, binary_edge = out.seg, out.multi_edge, out.binary_edge
    n_params = sum(p.numel() for p in model.parameters())
    assert seg.shape == (1, 20, 256, 256), f"seg: {seg.shape}"
    assert multi_edge.shape == (1, 20, 256, 256), f"multi_edge: {multi_edge.shape}"
    assert binary_edge.shape == (1, 1, 256, 256), f"binary_edge: {binary_edge.shape}"
    print(f"[PASS] SkyScapesNet | Seg: {seg.shape} | Multi-Edge: {multi_edge.shape} | "
          f"Binary-Edge: {binary_edge.shape} | Params: {n_params:,}")


def test_losses():
    from skyscapesnet.models.outputs import SkyScapesOutput
    criterion = MultiTaskLoss(n_classes=20)
    seg_pred = torch.randn(2, 20, 64, 64, requires_grad=True)
    multi_edge_pred = torch.randn(2, 20, 64, 64, requires_grad=True)
    binary_edge_pred = torch.randn(2, 1, 64, 64, requires_grad=True)
    out = SkyScapesOutput(seg=seg_pred, multi_edge=multi_edge_pred, binary_edge=binary_edge_pred)
    seg_target = torch.randint(0, 20, (2, 64, 64))

    loss, loss_dict = criterion(out, seg_target)
    assert loss.requires_grad, "Loss should require grad for backprop"
    print(f"[PASS] MultiTaskLoss | Total: {loss.item():.4f} | Components: {loss_dict}")


def test_metrics():
    cm = ConfusionMatrix(5)
    pred = torch.tensor([0, 1, 2, 3, 4, 0, 1])
    target = torch.tensor([0, 1, 2, 3, 4, 1, 0])
    cm.update(pred, target)
    miou = cm.mean_iou()
    acc = cm.pixel_accuracy()
    print(f"[PASS] ConfusionMatrix | mIoU: {miou:.4f} | Acc: {acc:.4f}")


def test_skyscapesnet_without_edge_heads():
    model = SkyScapesNet(in_channels=3, n_classes=13, growth_rate=16, use_edge_heads=False)
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.seg.shape == (1, 13, 64, 64)
    assert out.multi_edge is None
    assert out.binary_edge is None


def test_hub_roundtrip():
    # Create model and get a reference output
    model = SkyScapesNet(in_channels=3, n_classes=20, growth_rate=16)  # small for speed
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        orig_seg = model(x).seg

    # Save and reload
    with tempfile.TemporaryDirectory() as tmp_dir:
        model.save_pretrained(tmp_dir)
        loaded = SkyScapesNet.from_pretrained(tmp_dir)
        loaded.eval()
        with torch.no_grad():
            loaded_seg = loaded(x).seg

    assert torch.allclose(orig_seg, loaded_seg, atol=1e-6), "Weights not preserved!"
    print(f"[PASS] PyTorchModelHubMixin save/load round-trip | Outputs match")
