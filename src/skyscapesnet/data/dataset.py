"""DLR-SkyScapes dataset (TorchGeo NonGeoDataset)."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchgeo.datasets import NonGeoDataset

from skyscapesnet.data.class_maps import (
    DENSE_20_MAP, LANE_13_MAP, CATEGORY_11_MAP,
)


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


class SkyScapesDataset(NonGeoDataset):
    """Single dataset that serves Dense-20 / Lane-13 / Category-11 via task arg.

    Expected directory structure:
        root/images/{train,val}/*.{png,tif,jpg}
        root/labels/{train,val}/*.{png}    (single-channel class-indexed)

    Args:
        root: Path to dataset root.
        split: "train" or "val".
        task: "dense" | "lane" | "category".
        color_to_id: Optional RGB->id LUT if labels are RGB-encoded.
    """

    TASK_REMAPS = {
        "dense": DENSE_20_MAP,
        "lane": LANE_13_MAP,
        "category": CATEGORY_11_MAP,
    }
    TASK_N_CLASSES = {"dense": 20, "lane": 13, "category": 11}

    def __init__(self, root: str, split: str, task: str = "dense",
                 color_to_id: dict | None = None):
        super().__init__()
        if task not in self.TASK_REMAPS:
            raise ValueError(f"Unknown task: {task!r}. Must be one of {list(self.TASK_REMAPS)}")
        self.root = Path(root)
        self.split = split
        self.task = task
        self.id_remap = self.TASK_REMAPS[task]
        self.color_to_id = color_to_id

        self.image_dir = self.root / "images" / split
        self.label_dir = self.root / "labels" / split
        if not self.image_dir.exists():
            raise FileNotFoundError(self.image_dir)
        if not self.label_dir.exists():
            raise FileNotFoundError(self.label_dir)

        self.image_files = sorted(
            p for ext in ("png", "tif", "jpg")
            for p in self.image_dir.glob(f"*.{ext}")
        )
        self.label_files = [
            self.label_dir / f"{p.stem}.png" for p in self.image_files
        ]

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image = np.array(Image.open(self.image_files[idx]).convert("RGB"))

        if self.color_to_id is not None:
            mask = rgb_mask_to_class_ids(
                np.array(Image.open(self.label_files[idx]).convert("RGB")),
                self.color_to_id,
            )
        else:
            mask = np.array(Image.open(self.label_files[idx]))

        # Remap raw 31-class ids to task-specific target ids
        remapped = np.zeros_like(mask, dtype=np.int64)
        for src, dst in self.id_remap.items():
            remapped[mask == src] = dst

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()  # (3, H, W) uint8
        mask_tensor = torch.from_numpy(remapped)                              # (H, W) int64

        return {"image": image_tensor, "mask": mask_tensor}
