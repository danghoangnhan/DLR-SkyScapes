"""Compute and cache per-class weights for a SkyScapes task.

Usage:
    python scripts/compute_class_weights.py \
        --data_root /path/to/skyscapes \
        --task dense \
        --output cache/class_weights_dense.json
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from skyscapesnet.data.dataset import SkyScapesDataset
from skyscapesnet.losses.loss import compute_class_weights


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--task", type=str, choices=["dense", "lane", "category"],
                        required=True)
    parser.add_argument("--method", type=str, default="inverse_freq",
                        choices=["inverse_freq", "median_freq"])
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


N_CLASSES = {"dense": 20, "lane": 13, "category": 11}


class _TupleAdapter:
    """Adapt the new dict-yielding SkyScapesDataset to (image, mask) tuples."""

    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        sample = self._dataset[idx]
        return sample["image"], sample["mask"]


def main():
    args = parse_args()
    dataset = _TupleAdapter(
        SkyScapesDataset(args.data_root, split="train", task=args.task)
    )
    n_classes = N_CLASSES[args.task]
    weights = compute_class_weights(dataset, n_classes, method=args.method)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "task": args.task,
        "method": args.method,
        "n_classes": n_classes,
        "weights": weights.tolist(),
    }, indent=2))
    print(f"Wrote {n_classes}-class weights to {out_path}")


if __name__ == "__main__":
    main()
