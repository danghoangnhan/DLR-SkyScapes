"""GPU-side joint image+mask augmentation via kornia.

Applied in SkyScapesDataModule.on_after_batch_transfer.
"""
from typing import Literal

import kornia.augmentation as K
import torch
import torch.nn as nn


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TrainAugmentations(nn.Module):
    """Geometric augs apply to image + mask jointly; photometric only to image."""

    def __init__(self):
        super().__init__()
        self.geom = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomRotation90(times=(0, 3), p=0.5),
            data_keys=["image", "mask"],
            same_on_batch=False,
        )
        self.photo = K.AugmentationSequential(
            K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            K.RandomGaussianNoise(mean=0.0, std=0.02, p=0.3),
            K.Normalize(mean=torch.tensor(IMAGENET_MEAN), std=torch.tensor(IMAGENET_STD)),
            data_keys=["image"],
        )

    def forward(self, image: torch.Tensor, mask: torch.Tensor):
        # image: (N, 3, H, W) float in [0, 1]; mask: (N, H, W) long
        # kornia wants mask as float (N, 1, H, W) for joint augmentation
        mask_f = mask.unsqueeze(1).float()
        image, mask_f = self.geom(image, mask_f)
        image = self.photo(image)
        mask = mask_f.squeeze(1).long()
        return image, mask


class EvalAugmentations(nn.Module):
    """Eval: normalize only, no random augs."""

    def __init__(self):
        super().__init__()
        self.photo = K.Normalize(
            mean=torch.tensor(IMAGENET_MEAN), std=torch.tensor(IMAGENET_STD),
        )

    def forward(self, image: torch.Tensor, mask: torch.Tensor):
        return self.photo(image), mask
