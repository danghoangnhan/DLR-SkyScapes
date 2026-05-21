"""Sliding-window patch extraction for full-resolution prediction (issue #8)."""
import torch


class SlidingWindowCollate:
    """Collates a batch of full tiles into a batch of overlapping patches.

    Output batch keys: image, mask, coords (N, 4: top, left, h, w), tile_id (N).
    """

    def __init__(self, patch_size: int, stride: int):
        self.patch_size = patch_size
        self.stride = stride

    def __call__(self, samples):
        all_imgs, all_masks, all_coords, all_tile_ids = [], [], [], []
        for tile_id, sample in enumerate(samples):
            img, mask = sample["image"], sample["mask"]
            _, h, w = img.shape
            for top in range(0, h, self.stride):
                for left in range(0, w, self.stride):
                    bottom = min(top + self.patch_size, h)
                    right = min(left + self.patch_size, w)
                    # Skip degenerate patches
                    if bottom - top < self.patch_size or right - left < self.patch_size:
                        continue
                    all_imgs.append(img[:, top:bottom, left:right])
                    all_masks.append(mask[top:bottom, left:right])
                    all_coords.append([top, left, self.patch_size, self.patch_size])
                    all_tile_ids.append(tile_id)
        return {
            "image": torch.stack(all_imgs).float() / 255.0,
            "mask": torch.stack(all_masks),
            "coords": torch.tensor(all_coords),
            "tile_id": torch.tensor(all_tile_ids),
            "tile_shape": torch.tensor([[s["image"].shape[1], s["image"].shape[2]] for s in samples]),
        }
