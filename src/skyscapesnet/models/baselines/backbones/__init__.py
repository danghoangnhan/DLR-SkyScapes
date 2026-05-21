"""Pure-PyTorch encoder backbones shared across baseline architectures."""
from .densenet import DenseNet161Encoder, densenet161_encoder
from .mobilenet import MobileNetV2Encoder, mobilenetv2_encoder
from .resnet import (
    Bottleneck,
    ResNetEncoder,
    resnet50_encoder,
    resnet101_encoder,
    resnet152_encoder,
)
from .vgg import VGG16_CFG, VGGEncoder, vgg16_encoder

__all__ = [
    # VGG
    "VGGEncoder", "VGG16_CFG", "vgg16_encoder",
    # ResNet
    "Bottleneck", "ResNetEncoder",
    "resnet50_encoder", "resnet101_encoder", "resnet152_encoder",
    # MobileNet
    "MobileNetV2Encoder", "mobilenetv2_encoder",
    # DenseNet
    "DenseNet161Encoder", "densenet161_encoder",
]
