"""Loss functions for SkyScapesNet training.

The paper uses cross-entropy + Soft-IoU loss (or Soft-Dice loss),
with scheduled class weighting that gradually evolves during training.

Reference: "SkyScapes — Fine-Grained Semantic Understanding of Aerial Scenes"
(Azimi et al., ICCV 2019), Section 4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftIoULoss(nn.Module):
    """Soft IoU Loss (differentiable approximation of 1 - IoU).

    Computes per-class soft IoU using softmax probabilities and averages.

    Args:
        n_classes: Number of classes.
        ignore_index: Class index to ignore.
    """

    def __init__(self, n_classes, ignore_index=255):
        super().__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)  # (N, C, H, W)

        # Create one-hot target
        valid = targets != self.ignore_index
        targets_clean = targets.clone()
        targets_clean[~valid] = 0
        one_hot = F.one_hot(targets_clean, self.n_classes)  # (N, H, W, C)
        one_hot = one_hot.permute(0, 3, 1, 2).float()  # (N, C, H, W)

        # Mask out ignored pixels
        valid_mask = valid.unsqueeze(1).float()  # (N, 1, H, W)
        one_hot = one_hot * valid_mask
        probs = probs * valid_mask

        # Per-class soft IoU
        intersection = (probs * one_hot).sum(dim=(0, 2, 3))
        union = (probs + one_hot - probs * one_hot).sum(dim=(0, 2, 3))

        iou = (intersection + 1e-6) / (union + 1e-6)

        # Only average over classes that have at least one pixel
        present = one_hot.sum(dim=(0, 2, 3)) > 0
        if present.any():
            return 1.0 - iou[present].mean()
        return torch.tensor(0.0, device=logits.device)


class SoftDiceLoss(nn.Module):
    """Soft Dice Loss (differentiable approximation of 1 - Dice coefficient).

    Args:
        n_classes: Number of classes.
        ignore_index: Class index to ignore.
    """

    def __init__(self, n_classes, ignore_index=255):
        super().__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)

        valid = targets != self.ignore_index
        targets_clean = targets.clone()
        targets_clean[~valid] = 0
        one_hot = F.one_hot(targets_clean, self.n_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).float()

        valid_mask = valid.unsqueeze(1).float()
        one_hot = one_hot * valid_mask
        probs = probs * valid_mask

        intersection = (probs * one_hot).sum(dim=(0, 2, 3))
        cardinality = (probs + one_hot).sum(dim=(0, 2, 3))

        dice = (2.0 * intersection + 1e-6) / (cardinality + 1e-6)

        present = one_hot.sum(dim=(0, 2, 3)) > 0
        if present.any():
            return 1.0 - dice[present].mean()
        return torch.tensor(0.0, device=logits.device)


class WeightedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with per-class weights.

    Args:
        class_weights: Optional tensor of per-class weights.
        ignore_index: Class index to ignore (default: 255).
    """

    def __init__(self, class_weights=None, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits, targets):
        return F.cross_entropy(
            logits, targets,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
        )


class ScheduledClassWeights:
    """Scheduled class weighting that evolves during training.

    Weights start uniform and gradually move to the target weights
    over a specified number of epochs.

    Args:
        target_weights: Final per-class weights (tensor).
        warmup_epochs: Number of epochs over which to ramp from uniform to target.
    """

    def __init__(self, target_weights, warmup_epochs=20):
        self.target_weights = target_weights
        self.warmup_epochs = warmup_epochs
        n_classes = len(target_weights)
        self.uniform_weights = torch.ones(n_classes) / n_classes * target_weights.sum()

    def get_weights(self, epoch):
        if epoch >= self.warmup_epochs:
            return self.target_weights
        alpha = epoch / self.warmup_epochs
        return (1 - alpha) * self.uniform_weights + alpha * self.target_weights


class MultiTaskLoss(nn.Module):
    """Combined multi-task loss for SkyScapesNet.

    Total loss = seg_loss + lambda_multi * multi_edge_loss + lambda_binary * binary_edge_loss

    Segmentation uses cross-entropy + Soft-IoU.
    Edge detection uses binary cross-entropy.

    Args:
        n_classes: Number of segmentation classes.
        class_weights: Optional per-class weights for segmentation CE.
        lambda_multi: Weight for multi-class edge loss.
        lambda_binary: Weight for binary edge loss.
        ignore_index: Class index to ignore.
    """

    def __init__(self, n_classes, class_weights=None, lambda_multi=1.0,
                 lambda_binary=1.0, ignore_index=255):
        super().__init__()
        self.ce_loss = WeightedCrossEntropyLoss(class_weights, ignore_index)
        self.iou_loss = SoftIoULoss(n_classes, ignore_index)
        self.lambda_multi = lambda_multi
        self.lambda_binary = lambda_binary
        self.ignore_index = ignore_index

    def forward(self, seg_logits, multi_edge_logits, binary_edge_logits,
                seg_targets, edge_targets=None):
        """
        Args:
            seg_logits: (N, C, H, W) segmentation logits.
            multi_edge_logits: (N, C, H, W) multi-class edge logits.
            binary_edge_logits: (N, 1, H, W) binary edge logits.
            seg_targets: (N, H, W) integer class labels.
            edge_targets: (N, H, W) integer edge labels (optional).
                If None, edge labels are derived from seg_targets boundaries.
        """
        # Segmentation loss: CE + Soft-IoU
        seg_ce = self.ce_loss(seg_logits, seg_targets)
        seg_iou = self.iou_loss(seg_logits, seg_targets)
        seg_loss = seg_ce + seg_iou

        # Edge targets: derive from segmentation labels if not provided
        if edge_targets is None:
            edge_targets = self._compute_edge_targets(seg_targets)

        # Multi-class edge loss
        multi_edge_loss = F.cross_entropy(
            multi_edge_logits, edge_targets, ignore_index=self.ignore_index,
        )

        # Binary edge loss
        binary_targets = (edge_targets > 0).float().unsqueeze(1)  # (N, 1, H, W)
        valid = (edge_targets != self.ignore_index).float().unsqueeze(1)
        binary_edge_loss = F.binary_cross_entropy_with_logits(
            binary_edge_logits, binary_targets, weight=valid,
        )

        total = seg_loss + self.lambda_multi * multi_edge_loss + self.lambda_binary * binary_edge_loss
        return total, {
            "seg_ce": seg_ce.item(),
            "seg_iou": seg_iou.item(),
            "multi_edge": multi_edge_loss.item(),
            "binary_edge": binary_edge_loss.item(),
        }

    @staticmethod
    def _compute_edge_targets(seg_targets):
        """Derive edge labels from segmentation labels using Laplacian-like filter."""
        # Pad and compute boundaries via neighbor comparison
        t = seg_targets.float().unsqueeze(1)  # (N, 1, H, W)
        # Simple Sobel-like boundary detection
        kernel = torch.tensor(
            [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
            dtype=torch.float32, device=t.device,
        ).view(1, 1, 3, 3)
        edges = F.conv2d(t, kernel, padding=1)
        edge_mask = (edges.abs() > 0).squeeze(1)  # (N, H, W)

        # Edge pixels keep their class label, non-edge pixels become 0
        edge_labels = seg_targets.clone()
        edge_labels[~edge_mask] = 0
        return edge_labels


def compute_class_weights(dataset, n_classes, method="inverse_freq"):
    """Compute per-class weights from a dataset.

    Args:
        dataset: Dataset yielding (image, mask) pairs.
        n_classes: Number of classes.
        method: 'inverse_freq' or 'median_freq'.

    Returns:
        Tensor of shape (n_classes,) with per-class weights.
    """
    counts = torch.zeros(n_classes, dtype=torch.float64)

    for _, mask in dataset:
        for c in range(n_classes):
            counts[c] += (mask == c).sum().item()

    total = counts.sum()
    if method == "inverse_freq":
        weights = total / (n_classes * counts.clamp(min=1))
    elif method == "median_freq":
        freq = counts / total
        median_freq = freq[freq > 0].median()
        weights = median_freq / freq.clamp(min=1e-10)
    else:
        raise ValueError(f"Unknown method: {method}")

    return weights.float()
