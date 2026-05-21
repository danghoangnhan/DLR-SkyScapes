"""LightningDataModule for SkyScapes."""
from __future__ import annotations

import lightning as L
import torch
from torch.utils.data import DataLoader

from skyscapesnet.data.dataset import SkyScapesDataset
from skyscapesnet.data.gpu_transforms import TrainAugmentations, EvalAugmentations


class _PatchSampler(torch.utils.data.Sampler):
    """Yields (tile_idx, top, left) tuples for random crops within tiles.

    Lightweight RandomGeoSampler analogue that works with NonGeoDataset.
    """

    def __init__(self, dataset: SkyScapesDataset, patch_size: int, n_samples: int):
        self.dataset = dataset
        self.patch_size = patch_size
        self.n_samples = n_samples

    def __iter__(self):
        for _ in range(self.n_samples):
            tile_idx = int(torch.randint(0, len(self.dataset), (1,)).item())
            yield tile_idx

    def __len__(self):
        return self.n_samples


class _PatchCollate:
    """Crops a random patch out of each loaded tile."""

    def __init__(self, patch_size: int):
        self.patch_size = patch_size

    def __call__(self, batch):
        out_imgs, out_masks = [], []
        for sample in batch:
            img, mask = sample["image"], sample["mask"]
            _, h, w = img.shape
            top = int(torch.randint(0, max(1, h - self.patch_size + 1), (1,)).item())
            left = int(torch.randint(0, max(1, w - self.patch_size + 1), (1,)).item())
            out_imgs.append(img[:, top:top + self.patch_size, left:left + self.patch_size])
            out_masks.append(mask[top:top + self.patch_size, left:left + self.patch_size])
        return {
            "image": torch.stack(out_imgs).float() / 255.0,
            "mask": torch.stack(out_masks),
        }


class SkyScapesDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str,
        task: str = "dense",
        patch_size: int = 512,
        batch_size: int = 1,
        num_workers: int = 4,
        train_samples_per_epoch: int = 1000,
        sliding_stride: int = 256,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_aug = TrainAugmentations()
        self.eval_aug = EvalAugmentations()

    def setup(self, stage: str):
        if stage in ("fit", None):
            self.train_ds = SkyScapesDataset(self.hparams.root, split="train", task=self.hparams.task)
            self.val_ds = SkyScapesDataset(self.hparams.root, split="val", task=self.hparams.task)
        if stage in ("test", "predict"):
            self.val_ds = SkyScapesDataset(self.hparams.root, split="val", task=self.hparams.task)

    def train_dataloader(self):
        sampler = _PatchSampler(self.train_ds, self.hparams.patch_size, self.hparams.train_samples_per_epoch)
        return DataLoader(
            self.train_ds, sampler=sampler,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=_PatchCollate(self.hparams.patch_size),
            pin_memory=True,
        )

    def val_dataloader(self):
        # Center crop for now; sliding window for predict only.
        sampler = _PatchSampler(self.val_ds, self.hparams.patch_size, n_samples=len(self.val_ds))
        return DataLoader(
            self.val_ds, sampler=sampler,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=_PatchCollate(self.hparams.patch_size),
            pin_memory=True,
        )

    def predict_dataloader(self):
        """Sliding-window with overlap. Yields {image, mask, coords, tile_id}."""
        from skyscapesnet.data.sliding import SlidingWindowCollate
        sampler = torch.utils.data.SequentialSampler(self.val_ds)
        return DataLoader(
            self.val_ds, sampler=sampler,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=SlidingWindowCollate(self.hparams.patch_size, self.hparams.sliding_stride),
            pin_memory=True,
        )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.trainer is None:
            return batch
        image, mask = batch["image"], batch["mask"]
        if self.trainer.training:
            image, mask = self.train_aug(image, mask)
        else:
            image, mask = self.eval_aug(image, mask)
        batch["image"], batch["mask"] = image, mask
        return batch
