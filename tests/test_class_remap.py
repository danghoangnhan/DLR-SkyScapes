"""Validate Dense-20 / Lane-13 / Category-11 class-remap LUTs."""
import pytest

from skyscapesnet.data.class_maps import (
    DENSE_20_MAP, LANE_13_MAP, CATEGORY_11_MAP,
)


@pytest.mark.parametrize("lut,n_classes", [
    (DENSE_20_MAP, 20),
    (LANE_13_MAP, 13),
    (CATEGORY_11_MAP, 11),
])
def test_lut_covers_31_source_classes(lut, n_classes):
    """Each LUT must remap every 0..30 source id to a valid target id."""
    assert set(lut.keys()) == set(range(31)), f"Missing source ids: {set(range(31)) - set(lut.keys())}"
    for src, dst in lut.items():
        assert 0 <= dst < n_classes, f"src {src} → dst {dst} out of range [0, {n_classes})"
