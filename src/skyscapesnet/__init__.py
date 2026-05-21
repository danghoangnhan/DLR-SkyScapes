"""SkyScapesNet: aerial semantic segmentation."""

from skyscapesnet.models.skyscapesnet import SkyScapesNet
from skyscapesnet.models.fc_densenet import FCDenseNet

__all__ = ["SkyScapesNet", "FCDenseNet"]
