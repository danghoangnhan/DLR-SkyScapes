"""Training script for SkyScapesNet / FC-DenseNet103.

Paper settings: LR=0.0001, batch_size=1, 60 epochs, 512×512 crops.

Usage:
    python train.py --data_root /path/to/skyscapes --model skyscapesnet
    python train.py --smoke_test --model skyscapesnet
    python train.py --smoke_test --model fc_densenet103
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from models.fc_densenet import FCDenseNet
from models.skyscapesnet import SkyScapesNet
from data.skyscapes_dataset import SkyScapesDataset, NUM_CLASSES_DENSE
from data.transforms import (
    JointCompose, JointRandomHorizontalFlip,
    JointRandomVerticalFlip, JointColorJitter,
)
from losses.loss import MultiTaskLoss, WeightedCrossEntropyLoss, SoftIoULoss
from utils.metrics import ConfusionMatrix


class SyntheticDataset(Dataset):
    """Random data for smoke-testing the training pipeline."""

    def __init__(self, n_samples=32, img_size=512, n_classes=20):
        self.n_samples = n_samples
        self.img_size = img_size
        self.n_classes = n_classes

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        image = torch.randn(3, self.img_size, self.img_size)
        mask = torch.randint(0, self.n_classes, (self.img_size, self.img_size))
        return image, mask


def parse_args():
    parser = argparse.ArgumentParser(description="Train SkyScapesNet")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Path to DLR-SkyScapes dataset root")
    parser.add_argument("--model", type=str, default="skyscapesnet",
                        choices=["fc_densenet103", "skyscapesnet"])
    parser.add_argument("--n_classes", type=int, default=NUM_CLASSES_DENSE)
    parser.add_argument("--patch_size", type=int, default=512,
                        help="Crop size (paper uses 512)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (paper uses 1)")
    parser.add_argument("--epochs", type=int, default=60,
                        help="Number of epochs (paper uses 60)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (paper uses 0.0001)")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--amp", action="store_true",
                        help="Use automatic mixed precision")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Quick test with synthetic data (3 epochs)")
    return parser.parse_args()


def build_model(args):
    if args.model == "skyscapesnet":
        return SkyScapesNet(
            in_channels=3, n_classes=args.n_classes, dropout_p=args.dropout,
        )
    elif args.model == "fc_densenet103":
        return FCDenseNet.densenet103(n_classes=args.n_classes, dropout_p=args.dropout)
    else:
        raise ValueError(f"Unknown model: {args.model}")


def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    use_amp, is_multitask):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for images, masks in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        with autocast("cuda", enabled=use_amp):
            if is_multitask:
                seg, multi_edge, binary_edge = model(images)
                loss, _ = criterion(seg, multi_edge, binary_edge, masks)
            else:
                logits = model(images)
                loss = criterion(logits, masks)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, n_classes, is_multitask):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    cm = ConfusionMatrix(n_classes)

    for images, masks in tqdm(loader, desc="Validating", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        if is_multitask:
            seg, multi_edge, binary_edge = model(images)
            loss, _ = criterion(seg, multi_edge, binary_edge, masks)
            preds = seg.argmax(dim=1)
        else:
            logits = model(images)
            loss = criterion(logits, masks)
            preds = logits.argmax(dim=1)

        total_loss += loss.item()
        n_batches += 1
        cm.update(preds, masks)

    avg_loss = total_loss / max(n_batches, 1)
    miou = cm.mean_iou()
    pixel_acc = cm.pixel_accuracy()

    return avg_loss, miou, pixel_acc


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    is_multitask = args.model == "skyscapesnet"

    # Data
    if args.smoke_test or args.data_root is None:
        print("Using synthetic data for smoke test")
        if args.smoke_test:
            args.epochs = min(args.epochs, 3)
        train_dataset = SyntheticDataset(
            n_samples=8, img_size=args.patch_size, n_classes=args.n_classes,
        )
        val_dataset = SyntheticDataset(
            n_samples=4, img_size=args.patch_size, n_classes=args.n_classes,
        )
    else:
        train_transform = JointCompose([
            JointRandomHorizontalFlip(),
            JointRandomVerticalFlip(),
            JointColorJitter(),
        ])
        train_dataset = SkyScapesDataset(
            args.data_root, split="train",
            transform=train_transform, patch_size=args.patch_size,
        )
        val_dataset = SkyScapesDataset(
            args.data_root, split="val", patch_size=args.patch_size,
        )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Model
    model = build_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model} | Parameters: {n_params:,}")

    # Optimizer (paper uses LR=0.0001, doesn't specify optimizer — use Adam)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    # Loss
    if is_multitask:
        criterion = MultiTaskLoss(n_classes=args.n_classes, ignore_index=255)
    else:
        ce_loss = WeightedCrossEntropyLoss(ignore_index=255)
        iou_loss = SoftIoULoss(args.n_classes, ignore_index=255)
        criterion = lambda logits, targets: ce_loss(logits, targets) + iou_loss(logits, targets)

    scaler = GradScaler("cuda", enabled=args.amp)

    start_epoch = 0
    best_miou = 0.0

    # Resume
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_miou = ckpt.get("best_miou", 0.0)
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            args.amp, is_multitask,
        )
        val_loss, miou, pixel_acc = validate(
            model, val_loader, criterion, device, args.n_classes, is_multitask,
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"mIoU: {miou:.4f} | Acc: {pixel_acc:.4f} | "
            f"LR: {lr:.6f} | Time: {elapsed:.1f}s"
        )

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_miou": max(best_miou, miou),
        }
        torch.save(ckpt, os.path.join(args.save_dir, "last.pth"))

        if miou > best_miou:
            best_miou = miou
            torch.save(ckpt, os.path.join(args.save_dir, "best.pth"))
            print(f"  -> New best mIoU: {best_miou:.4f}")

    print(f"Training complete. Best mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()
