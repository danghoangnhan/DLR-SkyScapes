"""Edge-detection metrics."""
from torchmetrics import Metric
from torchmetrics.classification import BinaryF1Score
import torch


class EdgeF1(Metric):
    """Pixel-wise F1 for binary edge detection.

    Thin wrapper around torchmetrics.BinaryF1Score that accepts (N, 1, H, W) logits.
    """

    is_differentiable: bool = False
    higher_is_better: bool = True

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.f1 = BinaryF1Score(threshold=threshold)

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        # logits: (N, 1, H, W) -> squeeze + sigmoid; target: (N, H, W) binary
        probs = torch.sigmoid(logits.squeeze(1))
        self.f1.update(probs, target.long())

    def compute(self) -> torch.Tensor:
        return self.f1.compute()
