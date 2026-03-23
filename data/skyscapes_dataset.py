"""DLR-SkyScapes dataset loader.

Loads aerial images and semantic segmentation masks for the SkyScapes dataset.
Supports SkyScapes-Dense (20 classes) task variant.

The 31 original fine-grained classes are mapped to 20 classes for Dense task
by merging all 12 lane-marking types into a single "lane-marking" class.

Reference: "SkyScapes — Fine-Grained Semantic Understanding of Aerial Scenes"
(Azimi et al., ICCV 2019), Section 2.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


# =============================================================================
# 31 original SkyScapes classes (Section 2.1 of the paper)
# =============================================================================

SKYSCAPES_31_CLASSES = [
    # Non-lane classes (19 classes → become ids 0-18 in Dense-20)
    {"name": "low_vegetation",       "id": 0},
    {"name": "paved_road",           "id": 1},
    {"name": "non_paved_road",       "id": 2},
    {"name": "paved_parking_place",  "id": 3},
    {"name": "non_paved_parking_place", "id": 4},
    {"name": "bike_way",             "id": 5},
    {"name": "sidewalk",             "id": 6},
    {"name": "entrance_exit",        "id": 7},
    {"name": "danger_area",          "id": 8},
    {"name": "building",             "id": 9},
    {"name": "car",                  "id": 10},
    {"name": "trailer",              "id": 11},
    {"name": "van",                  "id": 12},
    {"name": "truck",                "id": 13},
    {"name": "large_truck",          "id": 14},
    {"name": "bus",                  "id": 15},
    {"name": "clutter",              "id": 16},
    {"name": "impervious_surface",   "id": 17},
    {"name": "tree",                 "id": 18},
    # 12 lane-marking types (all merged into id 19 for Dense-20)
    {"name": "dash_line",            "id": 19},
    {"name": "long_line",            "id": 20},
    {"name": "small_dash_line",      "id": 21},
    {"name": "turn_sign",            "id": 22},
    {"name": "plus_sign",            "id": 23},
    {"name": "other_signs",          "id": 24},
    {"name": "crosswalk",            "id": 25},
    {"name": "stop_line",            "id": 26},
    {"name": "zebra_zone",           "id": 27},
    {"name": "no_parking_zone",      "id": 28},
    {"name": "parking_zone",         "id": 29},
    {"name": "other_lane_markings",  "id": 30},
]

# =============================================================================
# SkyScapes-Dense: 20 classes (lane markings merged into single class)
# =============================================================================

SKYSCAPES_DENSE_CLASSES = [
    "low_vegetation", "paved_road", "non_paved_road", "paved_parking_place",
    "non_paved_parking_place", "bike_way", "sidewalk", "entrance_exit",
    "danger_area", "building", "car", "trailer", "van", "truck",
    "large_truck", "bus", "clutter", "impervious_surface", "tree",
    "lane_marking",  # merged from 12 sub-types
]

NUM_CLASSES_DENSE = 20

# Map from 31-class ids to 20-class Dense ids
DENSE_ID_MAP = {}
for i in range(19):
    DENSE_ID_MAP[i] = i
for i in range(19, 31):
    DENSE_ID_MAP[i] = 19  # all lane markings → class 19


def rgb_mask_to_class_ids(mask_rgb, color_to_id):
    """Convert an RGB label mask to a class-index mask.

    Args:
        mask_rgb: numpy array (H, W, 3) with RGB values.
        color_to_id: dict mapping (R, G, B) -> class_id.

    Returns:
        numpy array (H, W) with integer class indices.
    """
    h, w = mask_rgb.shape[:2]
    class_mask = np.zeros((h, w), dtype=np.int64)

    for color, class_id in color_to_id.items():
        match = np.all(mask_rgb == np.array(color, dtype=np.uint8), axis=-1)
        class_mask[match] = class_id

    return class_mask


class SkyScapesDataset(Dataset):
    """DLR-SkyScapes semantic segmentation dataset.

    Expected directory structure:
        root/
        ├── images/
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/

    Images are 5616×3744 pixels. They are cropped to patch_size (default 512)
    with configurable overlap for training.

    Args:
        root_dir: Path to dataset root.
        split: 'train' or 'val'.
        transform: Optional callable for joint (image, mask) augmentation.
        patch_size: Crop size (default 512, as per the paper).
        task: 'dense' (20 classes) or 'raw' (31 classes).
        color_to_id: Optional dict mapping RGB→class_id for the raw annotations.
    """

    def __init__(self, root_dir, split="train", transform=None, patch_size=512,
                 task="dense", color_to_id=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.patch_size = patch_size
        self.task = task
        self.color_to_id = color_to_id  # None = assume masks are already class-indexed

        if task == "dense":
            self.n_classes = NUM_CLASSES_DENSE
            self.id_remap = DENSE_ID_MAP
        else:
            self.n_classes = 31
            self.id_remap = None

        self.image_dir = self.root_dir / "images" / split
        self.label_dir = self.root_dir / "labels" / split

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")

        # Find image files
        self.image_files = sorted(self.image_dir.glob("*.png"))
        if not self.image_files:
            self.image_files = sorted(self.image_dir.glob("*.tif"))
        if not self.image_files:
            self.image_files = sorted(self.image_dir.glob("*.jpg"))

        # Match label files
        self.label_files = []
        for img_path in self.image_files:
            stem = img_path.stem
            label_path = self.label_dir / f"{stem}.png"
            if not label_path.exists():
                label_path = self.label_dir / f"{stem}_label.png"
            self.label_files.append(label_path)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert("RGB")
        mask_img = Image.open(self.label_files[idx])

        if self.transform is not None:
            image, mask_img = self.transform(image, mask_img)

        # Random crop to patch_size
        if self.patch_size is not None:
            image, mask_img = self._random_crop(image, mask_img, self.patch_size)

        # Convert image to tensor
        image_np = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # (3, H, W)

        # Convert mask to class indices
        if self.color_to_id is not None:
            mask_np = np.array(mask_img.convert("RGB"), dtype=np.uint8)
            class_mask = rgb_mask_to_class_ids(mask_np, self.color_to_id)
        else:
            # Assume mask is already single-channel class-indexed
            class_mask = np.array(mask_img, dtype=np.int64)

        # Remap to task-specific classes (e.g., Dense-20)
        if self.id_remap is not None:
            remapped = np.zeros_like(class_mask)
            for src_id, dst_id in self.id_remap.items():
                remapped[class_mask == src_id] = dst_id
            class_mask = remapped

        mask_tensor = torch.from_numpy(class_mask)  # (H, W)

        return image_tensor, mask_tensor

    @staticmethod
    def _random_crop(image, mask, size):
        w, h = image.size
        if w < size or h < size:
            raise ValueError(f"Image size ({w}x{h}) smaller than patch_size ({size})")
        x = np.random.randint(0, w - size + 1)
        y = np.random.randint(0, h - size + 1)
        image = image.crop((x, y, x + size, y + size))
        mask = mask.crop((x, y, x + size, y + size))
        return image, mask
