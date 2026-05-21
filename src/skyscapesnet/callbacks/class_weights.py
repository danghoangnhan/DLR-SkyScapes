"""Schedule per-class weights from uniform -> target over warmup_epochs (issue #4)."""
import lightning as L
import torch


class ScheduledClassWeightsCallback(L.Callback):
    """Linearly ramps the loss's class_weights from uniform to target over warmup_epochs.

    Reads ``pl_module.target_weights`` (Buffer registered by SkyScapesLitModule) and
    writes the scheduled weights into ``pl_module.loss.class_weights`` in-place at the
    start of every training epoch.
    """

    def __init__(self, warmup_epochs: int = 20):
        super().__init__()
        self.warmup_epochs = warmup_epochs

    def on_train_epoch_start(self, trainer, pl_module):
        target = pl_module.target_weights
        n = target.numel()
        alpha = min(trainer.current_epoch / max(self.warmup_epochs, 1), 1.0)
        uniform = torch.ones_like(target) / n * target.sum()
        weights = (1 - alpha) * uniform + alpha * target

        # Mutate the buffer in-place to preserve device + identity.
        pl_module.loss.class_weights.copy_(weights)

        # The CE submodule may hold its own reference - sync that too if applicable.
        if hasattr(pl_module.loss, "ce_loss") and hasattr(pl_module.loss.ce_loss, "class_weights"):
            ce_w = pl_module.loss.ce_loss.class_weights
            if ce_w is not None and ce_w.shape == weights.shape:
                ce_w.copy_(weights)
            else:
                pl_module.loss.ce_loss.class_weights = weights.clone().to(target.device)
