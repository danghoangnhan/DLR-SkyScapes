"""Smoke tests for the pure-PyTorch baseline implementations."""
from __future__ import annotations

import pytest
import torch

from skyscapesnet.losses.loss import SegLoss
from skyscapesnet.models.baselines import (
    AdapNet,
    BiSeNet,
    DeepLabV3,
    DeepLabV3Plus,
    DenseASPP,
    FCN8s,
    FRRN,
    GCN,
    MobileUNet,
    PSPNet,
    RefineNet,
    SegNet,
    UNet,
)
from skyscapesnet.models.outputs import SkyScapesOutput


@pytest.mark.parametrize("bilinear", [True, False])
def test_unet_forward_shape(bilinear: bool) -> None:
    model = UNet(in_channels=3, n_classes=20, base_channels=32, bilinear=bilinear)
    model.eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, SkyScapesOutput)
    assert out.seg.shape == (1, 20, 128, 128)
    assert out.multi_edge is None
    assert out.binary_edge is None
    assert model.n_classes == 20


def test_unet_handles_non_power_of_two_input() -> None:
    """Spatial dims that aren't divisible by 16 should still round-trip via pad."""
    model = UNet(in_channels=3, n_classes=13, base_channels=16)
    model.eval()
    x = torch.randn(1, 3, 130, 134)
    with torch.no_grad():
        out = model(x)
    assert out.seg.shape == (1, 13, 130, 134)


def test_unet_loss_backward() -> None:
    """End-to-end loss + backward — same call site that SkyScapesLitModule uses."""
    model = UNet(in_channels=3, n_classes=20, base_channels=16)
    loss_fn = SegLoss(n_classes=20)
    x = torch.randn(2, 3, 64, 64)
    target = torch.randint(0, 20, (2, 64, 64))
    out = model(x)
    loss, components = loss_fn(out, target)
    loss.backward()
    assert loss.requires_grad
    assert "seg_ce" in components and "seg_iou" in components
    # at least one parameter should have received a gradient
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_unet_param_count_sane() -> None:
    """Default config should land near the literature's ~31M parameters."""
    model = UNet(in_channels=3, n_classes=20, base_channels=64, bilinear=False)
    n_params = sum(p.numel() for p in model.parameters())
    # canonical U-Net (5-stage, base=64, ConvTranspose) is ~31M params; allow ±5M
    assert 26_000_000 < n_params < 36_000_000, f"got {n_params:,}"


# --- SegNet ---------------------------------------------------------

def test_segnet_forward_shape() -> None:
    model = SegNet(in_channels=3, n_classes=20)
    model.eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, SkyScapesOutput)
    assert out.seg.shape == (1, 20, 128, 128)
    assert out.multi_edge is None and out.binary_edge is None
    assert model.n_classes == 20


def test_segnet_loss_backward() -> None:
    model = SegNet(in_channels=3, n_classes=13)
    loss_fn = SegLoss(n_classes=13)
    x = torch.randn(2, 3, 64, 64)
    target = torch.randint(0, 13, (2, 64, 64))
    out = model(x)
    loss, _ = loss_fn(out, target)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_segnet_param_count_sane() -> None:
    """SegNet (VGG16-BN encoder + symmetric decoder) is ~29M params."""
    model = SegNet(in_channels=3, n_classes=20)
    n_params = sum(p.numel() for p in model.parameters())
    assert 25_000_000 < n_params < 33_000_000, f"got {n_params:,}"


# --- FCN-8s ---------------------------------------------------------

def test_fcn8s_forward_shape() -> None:
    model = FCN8s(in_channels=3, n_classes=20)
    model.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, SkyScapesOutput)
    assert out.seg.shape == (1, 20, 256, 256)
    assert out.multi_edge is None and out.binary_edge is None


def test_fcn8s_handles_non_multiple_of_32() -> None:
    """Bilinear upsampling lets FCN-8s accept arbitrary spatial sizes."""
    model = FCN8s(in_channels=3, n_classes=13)
    model.eval()
    x = torch.randn(1, 3, 130, 134)
    with torch.no_grad():
        out = model(x)
    assert out.seg.shape == (1, 13, 130, 134)


def test_fcn8s_loss_backward() -> None:
    model = FCN8s(in_channels=3, n_classes=20)
    loss_fn = SegLoss(n_classes=20)
    x = torch.randn(2, 3, 96, 96)
    target = torch.randint(0, 20, (2, 96, 96))
    out = model(x)
    loss, _ = loss_fn(out, target)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_fcn8s_param_count_sane() -> None:
    """FCN-8s (VGG16-BN + fc6 7x7 + fc7 1x1 + skip-fusion) is ~134M params."""
    model = FCN8s(in_channels=3, n_classes=20)
    n_params = sum(p.numel() for p in model.parameters())
    assert 130_000_000 < n_params < 140_000_000, f"got {n_params:,}"


# --- Mobile-U-Net ---------------------------------------------------

@pytest.mark.parametrize("bilinear", [True, False])
def test_mobile_unet_forward_shape(bilinear: bool) -> None:
    model = MobileUNet(in_channels=3, n_classes=20, bilinear=bilinear)
    model.eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, SkyScapesOutput)
    assert out.seg.shape == (1, 20, 128, 128)
    assert out.multi_edge is None and out.binary_edge is None


def test_mobile_unet_loss_backward() -> None:
    model = MobileUNet(in_channels=3, n_classes=13)
    loss_fn = SegLoss(n_classes=13)
    x = torch.randn(2, 3, 64, 64)
    target = torch.randint(0, 13, (2, 64, 64))
    out = model(x)
    loss, _ = loss_fn(out, target)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_mobile_unet_param_count_sane() -> None:
    """MobileNetV2 (~2M) + lightweight U-Net decoder lands well under 10M."""
    model = MobileUNet(in_channels=3, n_classes=20)
    n_params = sum(p.numel() for p in model.parameters())
    assert 2_500_000 < n_params < 10_000_000, f"got {n_params:,}"


# --- PSPNet ---------------------------------------------------------

def test_pspnet_forward_shape() -> None:
    model = PSPNet(in_channels=3, n_classes=20, output_stride=8)
    model.eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, SkyScapesOutput)
    assert out.seg.shape == (1, 20, 128, 128)
    assert out.multi_edge is None and out.binary_edge is None


def test_pspnet_loss_backward() -> None:
    model = PSPNet(in_channels=3, n_classes=13, output_stride=16)
    loss_fn = SegLoss(n_classes=13)
    x = torch.randn(2, 3, 64, 64)
    target = torch.randint(0, 13, (2, 64, 64))
    out = model(x)
    loss, _ = loss_fn(out, target)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_pspnet_param_count_sane() -> None:
    """ResNet101 (~42.5M) + PPM (~4M) + head 4096→512 3x3 (~19M) ≈ 65M."""
    model = PSPNet(in_channels=3, n_classes=20, output_stride=8)
    n_params = sum(p.numel() for p in model.parameters())
    assert 55_000_000 < n_params < 75_000_000, f"got {n_params:,}"


# --- DeepLabv3 ------------------------------------------------------

def test_deeplabv3_forward_shape() -> None:
    model = DeepLabV3(in_channels=3, n_classes=20, output_stride=16)
    model.eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, SkyScapesOutput)
    assert out.seg.shape == (1, 20, 128, 128)


def test_deeplabv3_loss_backward() -> None:
    model = DeepLabV3(in_channels=3, n_classes=13)
    loss_fn = SegLoss(n_classes=13)
    x = torch.randn(2, 3, 64, 64)
    target = torch.randint(0, 13, (2, 64, 64))
    out = model(x)
    loss, _ = loss_fn(out, target)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_deeplabv3_param_count_sane() -> None:
    """ResNet50 (~23.5M) + ASPP (~16M) ≈ 40M."""
    model = DeepLabV3(in_channels=3, n_classes=20)
    n_params = sum(p.numel() for p in model.parameters())
    assert 35_000_000 < n_params < 50_000_000, f"got {n_params:,}"


# --- DeepLabv3+ -----------------------------------------------------

def test_deeplabv3plus_forward_shape() -> None:
    model = DeepLabV3Plus(in_channels=3, n_classes=20, output_stride=16)
    model.eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, SkyScapesOutput)
    assert out.seg.shape == (1, 20, 128, 128)


def test_deeplabv3plus_loss_backward() -> None:
    model = DeepLabV3Plus(in_channels=3, n_classes=13)
    loss_fn = SegLoss(n_classes=13)
    x = torch.randn(2, 3, 64, 64)
    target = torch.randint(0, 13, (2, 64, 64))
    out = model(x)
    loss, _ = loss_fn(out, target)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_deeplabv3plus_param_count_sane() -> None:
    """ResNet101 (~42.5M) + ASPP (~16M) + decoder ≈ 60M."""
    model = DeepLabV3Plus(in_channels=3, n_classes=20)
    n_params = sum(p.numel() for p in model.parameters())
    assert 55_000_000 < n_params < 70_000_000, f"got {n_params:,}"


# --- DenseASPP ------------------------------------------------------

def test_denseaspp_forward_shape() -> None:
    model = DenseASPP(in_channels=3, n_classes=20, output_stride=16)
    model.eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, SkyScapesOutput)
    assert out.seg.shape == (1, 20, 128, 128)


def test_denseaspp_loss_backward() -> None:
    model = DenseASPP(in_channels=3, n_classes=13, output_stride=16)
    loss_fn = SegLoss(n_classes=13)
    x = torch.randn(2, 3, 64, 64)
    target = torch.randint(0, 13, (2, 64, 64))
    out = model(x)
    loss, _ = loss_fn(out, target)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_denseaspp_param_count_sane() -> None:
    """DenseNet161 (~26.5M) + DenseASPP head (~9M) ≈ 36M."""
    model = DenseASPP(in_channels=3, n_classes=20, output_stride=16)
    n_params = sum(p.numel() for p in model.parameters())
    assert 30_000_000 < n_params < 45_000_000, f"got {n_params:,}"


# --- GCN ------------------------------------------------------------

def test_gcn_forward_shape() -> None:
    model = GCN(in_channels=3, n_classes=20, kernel_size=7)  # smaller kernel for the test
    model.eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, SkyScapesOutput)
    assert out.seg.shape == (1, 20, 128, 128)


def test_gcn_loss_backward() -> None:
    model = GCN(in_channels=3, n_classes=13, kernel_size=7)
    loss_fn = SegLoss(n_classes=13)
    x = torch.randn(2, 3, 64, 64)
    target = torch.randint(0, 13, (2, 64, 64))
    out = model(x)
    loss, _ = loss_fn(out, target)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_gcn_param_count_sane() -> None:
    """ResNet152 (~58.1M) + 4 GCNs + 8 BRs ≈ 60-62M with k=15."""
    model = GCN(in_channels=3, n_classes=20, kernel_size=15)
    n_params = sum(p.numel() for p in model.parameters())
    assert 55_000_000 < n_params < 70_000_000, f"got {n_params:,}"


# --- BiSeNet --------------------------------------------------------

def test_bisenet_forward_shape() -> None:
    model = BiSeNet(in_channels=3, n_classes=20)
    model.eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, SkyScapesOutput)
    assert out.seg.shape == (1, 20, 128, 128)


def test_bisenet_loss_backward() -> None:
    model = BiSeNet(in_channels=3, n_classes=13)
    loss_fn = SegLoss(n_classes=13)
    x = torch.randn(2, 3, 64, 64)
    target = torch.randint(0, 13, (2, 64, 64))
    out = model(x)
    loss, _ = loss_fn(out, target)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_bisenet_param_count_sane() -> None:
    """ResNet50 context path (~23.5M) + spatial path + ARM/FFM heads."""
    model = BiSeNet(in_channels=3, n_classes=20)
    n_params = sum(p.numel() for p in model.parameters())
    assert 22_000_000 < n_params < 35_000_000, f"got {n_params:,}"


# --- RefineNet ------------------------------------------------------

def test_refinenet_forward_shape() -> None:
    model = RefineNet(in_channels=3, n_classes=20)
    model.eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, SkyScapesOutput)
    assert out.seg.shape == (1, 20, 128, 128)


def test_refinenet_loss_backward() -> None:
    model = RefineNet(in_channels=3, n_classes=13, channels=128)
    loss_fn = SegLoss(n_classes=13)
    x = torch.randn(2, 3, 64, 64)
    target = torch.randint(0, 13, (2, 64, 64))
    out = model(x)
    loss, _ = loss_fn(out, target)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_refinenet_param_count_sane() -> None:
    """ResNet101 (~42.5M) + 4 RefineNet blocks at 256 ch.

    Each block contributes ~9M (2 input-RCUs × 1.2M, MRF ~1.2M, CRP ~2.4M,
    output RCU ~1.2M), so the head totals ~35M on top of ResNet101.
    """
    model = RefineNet(in_channels=3, n_classes=20, channels=256)
    n_params = sum(p.numel() for p in model.parameters())
    assert 70_000_000 < n_params < 85_000_000, f"got {n_params:,}"


# --- FRRN -----------------------------------------------------------

def test_frrn_forward_shape() -> None:
    model = FRRN(in_channels=3, n_classes=20)
    model.eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, SkyScapesOutput)
    assert out.seg.shape == (1, 20, 128, 128)


def test_frrn_loss_backward() -> None:
    model = FRRN(in_channels=3, n_classes=13)
    loss_fn = SegLoss(n_classes=13)
    x = torch.randn(2, 3, 64, 64)
    target = torch.randint(0, 13, (2, 64, 64))
    out = model(x)
    loss, _ = loss_fn(out, target)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_frrn_param_count_sane() -> None:
    """FRRN-A: residual stream (48 ch) + pooling stream (96/192/384/384 ch) ≈ 17M."""
    model = FRRN(in_channels=3, n_classes=20)
    n_params = sum(p.numel() for p in model.parameters())
    assert 14_000_000 < n_params < 22_000_000, f"got {n_params:,}"


# --- AdapNet --------------------------------------------------------

def test_adapnet_forward_shape() -> None:
    model = AdapNet(in_channels=3, n_classes=20, output_stride=8)
    model.eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, SkyScapesOutput)
    assert out.seg.shape == (1, 20, 128, 128)


def test_adapnet_loss_backward() -> None:
    model = AdapNet(in_channels=3, n_classes=13, output_stride=16, n_multiscale_blocks=2)
    loss_fn = SegLoss(n_classes=13)
    x = torch.randn(2, 3, 64, 64)
    target = torch.randint(0, 13, (2, 64, 64))
    out = model(x)
    loss, _ = loss_fn(out, target)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_adapnet_param_count_sane() -> None:
    """ResNet50 (~23.5M) + 2048->512 bottleneck + 4 multiscale-residual blocks ≈ 32M."""
    model = AdapNet(in_channels=3, n_classes=20, n_multiscale_blocks=4)
    n_params = sum(p.numel() for p in model.parameters())
    assert 25_000_000 < n_params < 40_000_000, f"got {n_params:,}"
