"""Smoke test for SkyScapesDataModule with a tiny synthetic dataset on disk."""
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from skyscapesnet.data.datamodule import SkyScapesDataModule


@pytest.fixture
def tiny_dataset(tmp_path):
    """Builds a minimal {images,labels}/{train,val}/x.png tree on disk."""
    for split in ("train", "val"):
        img_dir = tmp_path / "images" / split
        lbl_dir = tmp_path / "labels" / split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for i in range(2):
            img = (np.random.rand(1024, 1024, 3) * 255).astype(np.uint8)
            lbl = np.random.randint(0, 31, (1024, 1024), dtype=np.uint8)
            Image.fromarray(img).save(img_dir / f"tile_{i}.png")
            Image.fromarray(lbl).save(lbl_dir / f"tile_{i}.png")
    return tmp_path


@pytest.mark.parametrize("task,n_classes", [("dense", 20), ("lane", 13), ("category", 11)])
def test_train_dataloader_yields_correct_shapes(tiny_dataset, task, n_classes):
    dm = SkyScapesDataModule(
        root=str(tiny_dataset), task=task, patch_size=256, batch_size=2,
        num_workers=0, train_samples_per_epoch=4,
    )
    dm.setup("fit")
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert batch["image"].shape == (2, 3, 256, 256)
    assert batch["mask"].shape == (2, 256, 256)
    assert batch["mask"].max() < n_classes
