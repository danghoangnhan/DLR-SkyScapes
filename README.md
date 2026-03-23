# SkyScapesNet-Dense

[![Paper](https://img.shields.io/badge/ICCV%202019-Paper-blue)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Azimi_SkyScapes__Fine-Grained_Semantic_Understanding_of_Aerial_Scenes_ICCV_2019_paper.pdf)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace%20Hub-yellow)](https://huggingface.co)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org)

Unofficial PyTorch implementation of **SkyScapesNet** from *"SkyScapes -- Fine-Grained Semantic Understanding of Aerial Scenes"* (Azimi et al., ICCV 2019).

Multi-task model for aerial semantic segmentation targeting **SkyScapes-Dense** (20 classes, 13cm/pixel resolution).

## Architecture

SkyScapesNet is a modified FC-DenseNet-103 with 5 novel components (148M parameters):

| Component | Description |
|-----------|-------------|
| **FDB** (Fully Dense Block) | Dense blocks with separable convolutions + residual projections |
| **FRSR** | Full-Resolution Separable Residual units for localization |
| **CRASPP** | Cascading forward (18-12-6-1) + reverse (1-6-12-18) ASPP at bottleneck |
| **LKBR** | Large-Kernel Boundary Refinement on skip connections |
| **Multi-task branching** | 3 decoder branches split after 2nd upsampling |

**Outputs:** semantic segmentation (20 cls) + multi-class edges (20 cls) + binary edges (1 ch)

Paper result: **40.13 mIoU** on SkyScapes-Dense (vs 37.78 FC-DenseNet-103 baseline).

## Quick Start

```bash
# Install
uv sync

# Smoke test (synthetic data, 3 epochs)
python train.py --smoke_test --model skyscapesnet

# Train on real data
python train.py --data_root /path/to/skyscapes --model skyscapesnet

# Evaluate
python evaluate.py --data_root /path/to/skyscapes --checkpoint checkpoints/best.pth

# Run model tests
python test_models.py
```

## HuggingFace Hub

```python
from models import SkyScapesNet

# Save locally
model = SkyScapesNet(n_classes=20)
model.save_pretrained("./my-model")

# Load from local dir or HF Hub
model = SkyScapesNet.from_pretrained("./my-model")
model = SkyScapesNet.from_pretrained("username/skyscapesnet-dense")

# Push to Hub
model.push_to_hub("username/skyscapesnet-dense")
```

## Project Structure

```
SkyScapesNet-Dense/
├── models/
│   ├── layers.py           # SeparableConv2d, FDB, DoS, UpS + Tiramisu blocks
│   ├── fc_densenet.py      # FC-DenseNet-103 baseline
│   ├── skyscapesnet.py     # Full SkyScapesNet (multi-task, 148M params)
│   ├── craspp.py           # Concatenated Reverse ASPP
│   ├── frsr.py             # Full-Resolution Separable Residual
│   ├── lkbr.py             # Large-Kernel Boundary Refinement
│   └── hub.py              # HuggingFace Hub mixin
├── data/
│   ├── skyscapes_dataset.py # Dataset loader (31→20 class mapping)
│   └── transforms.py       # Joint image+mask augmentations
├── losses/
│   └── loss.py             # CE + Soft-IoU + Soft-Dice, multi-task loss
├── utils/
│   ├── metrics.py          # ConfusionMatrix, mIoU, pixel accuracy
│   └── augment.py          # Albumentations pipeline
├── train.py                # Training (LR=1e-4, batch=1, 60 epochs, 512x512)
├── evaluate.py             # Per-class IoU evaluation
├── test_models.py          # Model verification (7 tests)
└── pyproject.toml          # Dependencies (uv)
```

## Training Settings (from paper)

| Setting | Value |
|---------|-------|
| Crop size | 512 x 512 |
| Learning rate | 0.0001 |
| Batch size | 1 |
| Epochs | 60 |
| Loss | Cross-entropy + Soft-IoU |
| Augmentation | Horizontal + vertical flip |

## Dataset

[DLR-SkyScapes](https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/dlr-skyscapes) -- 16 aerial images (5616x3744 px, 13cm/pixel), 31 classes, 50%/12.5%/37.5% train/val/test split.

SkyScapes-Dense merges 12 lane-marking types into one class (20 total).

## Citation

```bibtex
@InProceedings{Azimi_2019_ICCV,
    author    = {Azimi, Seyed Majid and Henry, Corentin and Sommer, Lars and Schumann, Arne and Vig, Eleonora},
    title     = {SkyScapes -- Fine-Grained Semantic Understanding of Aerial Scenes},
    booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2019}
}
```
