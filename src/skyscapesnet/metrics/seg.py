"""Semantic segmentation metrics, torchmetrics-backed."""
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassJaccardIndex,
    MulticlassAccuracy,
)


class SemSegMetrics(MetricCollection):
    """mIoU + per-class IoU + pixel accuracy.

    Drop-in replacement for utils/metrics.py:ConfusionMatrix.
    """

    def __init__(self, n_classes: int, ignore_index: int = 255):
        super().__init__({
            "miou": MulticlassJaccardIndex(
                num_classes=n_classes, average="macro", ignore_index=ignore_index,
            ),
            "per_class_iou": MulticlassJaccardIndex(
                num_classes=n_classes, average="none", ignore_index=ignore_index,
            ),
            "pixel_acc": MulticlassAccuracy(
                num_classes=n_classes, average="micro", ignore_index=ignore_index,
            ),
        })
