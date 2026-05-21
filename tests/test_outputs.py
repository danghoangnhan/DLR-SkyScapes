"""Test the SkyScapesOutput dataclass."""
import torch
import torch.utils._pytree as pytree

from skyscapesnet.models.outputs import SkyScapesOutput


def test_dense_output_has_three_heads():
    seg = torch.zeros(1, 20, 8, 8)
    multi_edge = torch.zeros(1, 20, 8, 8)
    binary_edge = torch.zeros(1, 1, 8, 8)
    out = SkyScapesOutput(seg=seg, multi_edge=multi_edge, binary_edge=binary_edge)
    assert out.seg.shape == (1, 20, 8, 8)
    assert out.multi_edge is not None
    assert out.binary_edge is not None


def test_seg_only_output_has_none_edge_heads():
    seg = torch.zeros(1, 13, 8, 8)
    out = SkyScapesOutput(seg=seg)
    assert out.multi_edge is None
    assert out.binary_edge is None


def test_output_is_a_registered_pytree_node():
    """Required so torch.compile / torch.func can traverse the dataclass."""
    out = SkyScapesOutput(
        seg=torch.zeros(1, 20, 4, 4),
        multi_edge=torch.zeros(1, 20, 4, 4),
        binary_edge=torch.zeros(1, 1, 4, 4),
    )
    flat, spec = pytree.tree_flatten(out)
    # Three tensors should flatten out
    assert len(flat) == 3
    reconstructed = pytree.tree_unflatten(flat, spec)
    assert isinstance(reconstructed, SkyScapesOutput)
    assert torch.equal(reconstructed.seg, out.seg)
