"""Evaluation metrics for semantic segmentation."""

import torch
import numpy as np


class ConfusionMatrix:
    """Streaming confusion matrix for computing IoU and pixel accuracy."""

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

    def update(self, pred, target):
        """Update confusion matrix with a batch of predictions.

        Args:
            pred: Predicted class indices (N, H, W) or (H, W).
            target: Ground truth class indices, same shape as pred.
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()

        pred = pred.flatten()
        target = target.flatten()

        # Filter out ignore indices (e.g., 255)
        valid = target < self.n_classes
        pred = pred[valid]
        target = target[valid]

        indices = target * self.n_classes + pred
        self.matrix += np.bincount(
            indices, minlength=self.n_classes ** 2
        ).reshape(self.n_classes, self.n_classes)

    def reset(self):
        self.matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)

    def per_class_iou(self):
        """Compute per-class Intersection over Union."""
        tp = np.diag(self.matrix)
        fp = self.matrix.sum(axis=0) - tp
        fn = self.matrix.sum(axis=1) - tp
        denom = tp + fp + fn
        iou = np.where(denom > 0, tp / denom, 0.0)
        return iou

    def mean_iou(self):
        """Compute mean IoU across classes with at least one sample."""
        iou = self.per_class_iou()
        valid = (self.matrix.sum(axis=1) + self.matrix.sum(axis=0) - np.diag(self.matrix)) > 0
        return iou[valid].mean() if valid.any() else 0.0

    def pixel_accuracy(self):
        """Compute overall pixel accuracy."""
        correct = np.diag(self.matrix).sum()
        total = self.matrix.sum()
        return correct / total if total > 0 else 0.0
