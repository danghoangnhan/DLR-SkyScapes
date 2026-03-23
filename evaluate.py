"""Evaluation script for SkyScapesNet / FC-DenseNet103.

Usage:
    python evaluate.py --data_root /path/to/skyscapes --checkpoint checkpoints/best.pth
"""

import argparse

import torch
from torch.utils.data import DataLoader

from models.fc_densenet import FCDenseNet
from models.skyscapesnet import SkyScapesNet
from data.skyscapes_dataset import SkyScapesDataset, NUM_CLASSES_DENSE, SKYSCAPES_DENSE_CLASSES
from utils.metrics import ConfusionMatrix


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SkyScapesNet")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="skyscapesnet",
                        choices=["fc_densenet103", "skyscapesnet"])
    parser.add_argument("--n_classes", type=int, default=NUM_CLASSES_DENSE)
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--split", type=str, default="val")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    if args.model == "skyscapesnet":
        model = SkyScapesNet(n_classes=args.n_classes)
    else:
        model = FCDenseNet.densenet103(n_classes=args.n_classes)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    is_multitask = args.model == "skyscapesnet"

    # Data
    dataset = SkyScapesDataset(args.data_root, split=args.split, patch_size=args.patch_size)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Evaluate
    cm = ConfusionMatrix(args.n_classes)

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)

            if is_multitask:
                seg, _, _ = model(images)
                preds = seg.argmax(dim=1)
            else:
                logits = model(images)
                preds = logits.argmax(dim=1)

            cm.update(preds, masks)

    # Report
    per_class_iou = cm.per_class_iou()
    miou = cm.mean_iou()
    pixel_acc = cm.pixel_accuracy()

    print(f"\nOverall Pixel Accuracy: {pixel_acc:.4f}")
    print(f"Mean IoU: {miou:.4f}\n")
    print(f"{'Class':<30} {'IoU':>8}")
    print("-" * 40)
    for i, name in enumerate(SKYSCAPES_DENSE_CLASSES):
        if i < len(per_class_iou):
            print(f"{name:<30} {per_class_iou[i]:>8.4f}")


if __name__ == "__main__":
    main()
