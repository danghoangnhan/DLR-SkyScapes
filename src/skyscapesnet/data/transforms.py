"""Joint transforms for image + mask augmentation."""

import random
import numpy as np
from PIL import Image, ImageFilter


class JointCompose:
    """Compose multiple joint transforms."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class JointRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return image, mask


class JointRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return image, mask


class JointRandomRotation:
    """Random 90-degree rotation (0, 90, 180, 270)."""

    def __call__(self, image, mask):
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            image = image.rotate(angle, expand=False)
            mask = mask.rotate(angle, expand=False, resample=Image.NEAREST)
        return image, mask


class JointColorJitter:
    """Color jitter applied only to the image (not the mask)."""

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, image, mask):
        from torchvision.transforms import ColorJitter
        jitter = ColorJitter(
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
        )
        image = jitter(image)
        return image, mask
