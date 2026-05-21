"""Smoke tests for the pure-PyTorch baseline encoder backbones."""
from __future__ import annotations

import pytest
import torch

from skyscapesnet.models.baselines.backbones import (
    densenet161_encoder,
    mobilenetv2_encoder,
    resnet50_encoder,
    resnet101_encoder,
    resnet152_encoder,
    vgg16_encoder,
)


# --- VGG -------------------------------------------------------------

def test_vgg16_encoder_shapes() -> None:
    encoder = vgg16_encoder()
    encoder.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        feats = encoder(x)
    assert isinstance(feats, list) and len(feats) == 5
    expected_strides = [2, 4, 8, 16, 32]
    expected_channels = [64, 128, 256, 512, 512]
    for f, s, c in zip(feats, expected_strides, expected_channels):
        assert f.shape == (1, c, 256 // s, 256 // s)


def test_vgg16_encoder_pool_indices() -> None:
    encoder = vgg16_encoder(return_pool_indices=True)
    encoder.eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        feats, indices = encoder(x)
    assert len(feats) == 5 and len(indices) == 5
    # MaxUnpool2d should accept these directly
    for f, idx in zip(feats, indices):
        assert idx.shape == f.shape


# --- ResNet ----------------------------------------------------------

@pytest.mark.parametrize("factory", [resnet50_encoder, resnet101_encoder, resnet152_encoder])
def test_resnet_encoder_default_stride(factory) -> None:
    encoder = factory()
    encoder.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        feats = encoder(x)
    assert len(feats) == 5
    # c1=/2, c2=/4, c3=/8, c4=/16, c5=/32
    assert feats[0].shape == (1, 64, 128, 128)
    assert feats[1].shape == (1, 256, 64, 64)
    assert feats[2].shape == (1, 512, 32, 32)
    assert feats[3].shape == (1, 1024, 16, 16)
    assert feats[4].shape == (1, 2048, 8, 8)


@pytest.mark.parametrize("output_stride,c4_size,c5_size", [(16, 16, 16), (8, 32, 32)])
def test_resnet_encoder_dilated(output_stride: int, c4_size: int, c5_size: int) -> None:
    encoder = resnet50_encoder(output_stride=output_stride)
    encoder.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        feats = encoder(x)
    assert feats[3].shape[-1] == c4_size
    assert feats[4].shape[-1] == c5_size


# --- MobileNetV2 -----------------------------------------------------

def test_mobilenetv2_encoder_shapes() -> None:
    encoder = mobilenetv2_encoder()
    encoder.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        feats = encoder(x)
    assert len(feats) == 5
    expected_strides = [2, 4, 8, 16, 32]
    expected_channels = [16, 24, 32, 96, 320]
    for f, s, c in zip(feats, expected_strides, expected_channels):
        assert f.shape == (1, c, 256 // s, 256 // s)


# --- DenseNet161 -----------------------------------------------------

@pytest.mark.parametrize("output_stride", [32, 16])
def test_densenet161_encoder(output_stride: int) -> None:
    encoder = densenet161_encoder(output_stride=output_stride)
    encoder.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        feats = encoder(x)
    assert len(feats) == 5
    assert feats[0].shape == (1, 96, 64, 64)        # /4
    assert feats[1].shape == (1, 192, 32, 32)       # /8
    assert feats[2].shape == (1, 384, 16, 16)       # /16
    if output_stride == 32:
        assert feats[3].shape[-1] == 8              # /32 after trans3
        assert feats[4].shape[-1] == 8
    else:
        assert feats[3].shape[-1] == 16             # trans3 didn't downsample
        assert feats[4].shape[-1] == 16


def test_backbone_param_counts_sane() -> None:
    """Spot-check param counts against the literature values."""
    counts = {
        "vgg16_bn": sum(p.numel() for p in vgg16_encoder().parameters()),
        "resnet50": sum(p.numel() for p in resnet50_encoder().parameters()),
        "resnet101": sum(p.numel() for p in resnet101_encoder().parameters()),
        "mobilenetv2": sum(p.numel() for p in mobilenetv2_encoder().parameters()),
        "densenet161": sum(p.numel() for p in densenet161_encoder().parameters()),
    }
    # Bounds chosen with ±10% margin around literature values (encoder only,
    # no classifier head).
    assert 14_000_000 < counts["vgg16_bn"] < 16_000_000, counts["vgg16_bn"]
    assert 22_000_000 < counts["resnet50"] < 25_000_000, counts["resnet50"]
    assert 41_000_000 < counts["resnet101"] < 45_000_000, counts["resnet101"]
    assert 1_500_000 < counts["mobilenetv2"] < 3_500_000, counts["mobilenetv2"]
    assert 25_000_000 < counts["densenet161"] < 30_000_000, counts["densenet161"]
