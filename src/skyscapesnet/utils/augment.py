"""Data augmentation pipeline for aerial imagery using Albumentations.

Designed for the DLR-SkyScapes dataset with multi-target support
(image, segmentation mask, and edge maps).
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(crop_size=256):
    """Training augmentation pipeline.

    Includes geometric and color augmentations suitable for aerial imagery
    (rotation-invariant, since aerial views have no canonical orientation).
    """
    return A.Compose([
        A.RandomCrop(height=crop_size, width=crop_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5,
        ),
        A.GaussNoise(p=0.3),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ], additional_targets={
        "multi_edge_mask": "mask",
        "binary_edge_mask": "mask",
    })


def get_val_transforms(crop_size=256):
    """Validation transform pipeline (no augmentation, just normalize + crop)."""
    return A.Compose([
        A.CenterCrop(height=crop_size, width=crop_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ], additional_targets={
        "multi_edge_mask": "mask",
        "binary_edge_mask": "mask",
    })
