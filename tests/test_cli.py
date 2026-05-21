"""End-to-end CLI smoke test."""
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def synthetic_root(tmp_path):
    for split in ("train", "val"):
        img_dir = tmp_path / "images" / split
        lbl_dir = tmp_path / "labels" / split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for i in range(2):
            img = (np.random.rand(640, 640, 3) * 255).astype(np.uint8)
            lbl = np.random.randint(0, 31, (640, 640), dtype=np.uint8)
            Image.fromarray(img).save(img_dir / f"tile_{i}.png")
            Image.fromarray(lbl).save(lbl_dir / f"tile_{i}.png")
    return tmp_path


def test_fit_fast_dev_run(synthetic_root, tmp_path):
    cmd = [
        sys.executable, "-m", "skyscapesnet.cli", "fit",
        "--config", "configs/base.yaml",
        "--config", "configs/dense.yaml",
        f"--data.root={synthetic_root}",
        "--data.patch_size=256",
        "--data.train_samples_per_epoch=2",
        "--data.num_workers=0",
        "--trainer.fast_dev_run=1",
        f"--trainer.default_root_dir={tmp_path}",
        f"--trainer.logger.init_args.save_dir={tmp_path}",
        "--model.model.init_args.growth_rate=16",
        "--model.class_weights_path=null",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, f"STDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
