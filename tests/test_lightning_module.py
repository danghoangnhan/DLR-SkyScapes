"""Smoke test for SkyScapesLitModule end-to-end."""
import torch

from skyscapesnet.models.skyscapesnet import SkyScapesNet
from skyscapesnet.models.lightning_module import SkyScapesLitModule
from skyscapesnet.losses.loss import MultiTaskLoss


def test_training_step_returns_scalar_loss():
    model = SkyScapesNet(in_channels=3, n_classes=20, growth_rate=16)
    loss = MultiTaskLoss(n_classes=20)
    lit = SkyScapesLitModule(model=model, loss=loss, task="dense")
    batch = {"image": torch.randn(1, 3, 64, 64), "mask": torch.randint(0, 20, (1, 64, 64))}
    out = lit.training_step(batch, 0)
    assert out.requires_grad
    assert out.dim() == 0


def test_configure_optimizers_returns_dict():
    model = SkyScapesNet(in_channels=3, n_classes=20, growth_rate=16)
    loss = MultiTaskLoss(n_classes=20)
    lit = SkyScapesLitModule(
        model=model, loss=loss, task="dense",
        optimizer_cfg={"class_path": "torch.optim.Adam", "init_args": {"lr": 1e-4}},
        scheduler_cfg={"class_path": "torch.optim.lr_scheduler.CosineAnnealingLR",
                       "init_args": {"T_max": 60}},
    )
    opt_cfg = lit.configure_optimizers()
    assert "optimizer" in opt_cfg
