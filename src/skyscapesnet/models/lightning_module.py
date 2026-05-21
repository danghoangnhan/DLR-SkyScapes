"""LightningModule wrapper for SkyScapesNet / FCDenseNet."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Literal

import lightning as L
import torch
import torch.nn as nn

from skyscapesnet.metrics.seg import SemSegMetrics


def _instantiate(cfg: dict | None, default_kwargs: dict | None = None, **extra) -> Any:
    """Instantiate from a LightningCLI-style {class_path, init_args} dict."""
    if cfg is None:
        return None
    cls_path = cfg["class_path"]
    module_name, cls_name = cls_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), cls_name)
    init_args = {**(default_kwargs or {}), **cfg.get("init_args", {}), **extra}
    return cls(**init_args)


class SkyScapesLitModule(L.LightningModule):
    """LightningModule wrapper around an nn.Module + loss.

    Args:
        model: nn.Module returning a SkyScapesOutput.
        loss: nn.Module taking (output, mask) → (scalar, components dict).
        task: One of "dense" | "lane" | "category".
        optimizer_cfg: {class_path, init_args} dict for the optimizer.
        scheduler_cfg: Optional {class_path, init_args} dict for the LR scheduler.
        class_weights_path: Optional path to a JSON produced by
            scripts/compute_class_weights.py.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        task: Literal["dense", "lane", "category"] = "dense",
        optimizer_cfg: dict | None = None,
        scheduler_cfg: dict | None = None,
        class_weights_path: str | None = None,
    ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.task = task
        self.optimizer_cfg = optimizer_cfg or {
            "class_path": "torch.optim.Adam",
            "init_args": {"lr": 1e-4, "weight_decay": 1e-4},
        }
        self.scheduler_cfg = scheduler_cfg
        self.class_weights_path = class_weights_path

        n_classes = getattr(model, "n_classes", None)
        if n_classes is None:
            raise ValueError("Model must expose n_classes attribute")
        self.n_classes = n_classes
        self.val_metrics = SemSegMetrics(n_classes=n_classes)
        self.test_metrics = SemSegMetrics(n_classes=n_classes)

        # Placeholder for the cached target weights; populated in setup().
        self.register_buffer("target_weights", torch.ones(n_classes), persistent=False)

    def setup(self, stage: str) -> None:
        if self.class_weights_path:
            data = json.loads(Path(self.class_weights_path).read_text())
            w = torch.tensor(data["weights"], dtype=torch.float32)
            self.target_weights = w.to(self.target_weights.device)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        out = self.model(batch["image"])
        loss, components = self.loss(out, batch["mask"])
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        for k, v in components.items():
            self.log(f"train/{k}", v, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.model(batch["image"])
        loss, _ = self.loss(out, batch["mask"])
        self.log("val/loss", loss, on_step=False, on_epoch=True)
        preds = out.seg.argmax(dim=1)
        self.val_metrics.update(preds, batch["mask"])

    def on_validation_epoch_end(self):
        result = self.val_metrics.compute()
        self.log("val/miou", result["miou"], prog_bar=True)
        self.log("val/pixel_acc", result["pixel_acc"])
        for i, iou in enumerate(result["per_class_iou"]):
            self.log(f"val/iou_class_{i}", iou)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        out = self.model(batch["image"])
        preds = out.seg.argmax(dim=1)
        self.test_metrics.update(preds, batch["mask"])

    def on_test_epoch_end(self):
        result = self.test_metrics.compute()
        self.log("test/miou", result["miou"])
        self.log("test/pixel_acc", result["pixel_acc"])
        self.test_metrics.reset()

    def predict_step(self, batch, batch_idx):
        out = self.model(batch["image"])
        return {
            "logits": out.seg,
            "coords": batch["coords"],
            "tile_id": batch["tile_id"],
            "tile_shape": batch.get("tile_shape"),
        }

    def configure_optimizers(self):
        optimizer = _instantiate(self.optimizer_cfg, params=self.parameters())
        config: dict[str, Any] = {"optimizer": optimizer}
        if self.scheduler_cfg is not None:
            scheduler = _instantiate(self.scheduler_cfg, optimizer=optimizer)
            config["lr_scheduler"] = scheduler
        return config
