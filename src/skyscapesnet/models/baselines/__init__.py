"""Pure-PyTorch baseline implementations for the SkyScapes benchmark tables.

Each baseline is an nn.Module that:
- accepts (N, in_channels, H, W) tensors,
- exposes an `n_classes` attribute,
- returns a SkyScapesOutput with only `.seg` populated.

Pair with SegLoss (not MultiTaskLoss) via the YAML config — none of the
baselines have edge-detection heads.
"""
from .adapnet import AdapNet
from .bisenet import BiSeNet
from .deeplabv3 import DeepLabV3
from .deeplabv3plus import DeepLabV3Plus
from .denseaspp import DenseASPP
from .fcn import FCN8s
from .frrn import FRRN
from .gcn import GCN
from .mobile_unet import MobileUNet
from .pspnet import PSPNet
from .refinenet import RefineNet
from .segnet import SegNet
from .unet import UNet

__all__ = [
    "AdapNet", "BiSeNet", "DeepLabV3", "DeepLabV3Plus", "DenseASPP",
    "FCN8s", "FRRN", "GCN", "MobileUNet", "PSPNet", "RefineNet",
    "SegNet", "UNet",
]
