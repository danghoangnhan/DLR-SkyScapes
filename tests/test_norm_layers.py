"""Test norm_layer parameter routing in SkyScapesNet."""
import pytest
import torch

from skyscapesnet.models.skyscapesnet import SkyScapesNet


@pytest.mark.parametrize("norm_layer", ["batch", "group", "instance"])
def test_skyscapesnet_forward_with_norm(norm_layer):
    model = SkyScapesNet(
        in_channels=3, n_classes=20, growth_rate=16, norm_layer=norm_layer,
    )
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.seg.shape == (1, 20, 64, 64)


def test_default_norm_is_batch():
    """Default behavior must not change."""
    model = SkyScapesNet(in_channels=3, n_classes=20, growth_rate=16)
    # Find one of the FDB layers and check its norm
    fdb = model.encoder_fdbs[0]
    sep_layer = fdb.layers[0]
    norm = sep_layer.layer[0]
    assert isinstance(norm, torch.nn.BatchNorm2d)


def test_group_norm_replaces_batchnorm():
    model = SkyScapesNet(in_channels=3, n_classes=20, growth_rate=16, norm_layer="group")
    fdb = model.encoder_fdbs[0]
    sep_layer = fdb.layers[0]
    norm = sep_layer.layer[0]
    assert isinstance(norm, torch.nn.GroupNorm)
