# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Unofficial PyTorch implementation of **SkyScapesNet** (Azimi et al., ICCV 2019) for fine-grained aerial semantic segmentation on the DLR-SkyScapes dataset. Two models are supported: `SkyScapesNet` (the full multi-task ~148M-param model) and `FCDenseNet` (the FC-DenseNet-103 / Tiramisu baseline).

Package manager: **uv** (`uv.lock`, `pyproject.toml`). Python 3.10+.

## Commands

```bash
# Install dependencies into .venv
uv sync

# Verify all models / losses / metrics (custom script, NOT pytest)
python test_models.py

# Smoke-test training pipeline with synthetic data (3 epochs)
python train.py --smoke_test --model skyscapesnet
python train.py --smoke_test --model fc_densenet103

# Train on real data (paper settings: LR=1e-4, batch=1, 60 epochs, 512x512)
python train.py --data_root /path/to/skyscapes --model skyscapesnet [--amp]

# Resume training
python train.py --data_root ... --resume checkpoints/last.pth

# Evaluate (per-class IoU + mIoU + pixel acc)
python evaluate.py --data_root /path/to/skyscapes --checkpoint checkpoints/best.pth
```

`.claude/settings.local.json` allowlists `.venv/bin/python *` — prefer invoking that explicitly over relying on shell activation. `test_models.py` is a hand-rolled runner that prints `[PASS]/[FAIL]` per test, not a pytest suite — there is no `pytest -k` equivalent; comment out entries in its `tests` list to run a subset.

## Architecture

### The dense-block return convention (critical)

Both `DenseBlock` (Tiramisu) and `FullyDenseBlock` (SkyScapesNet, `models/layers.py`) return a **tuple `(all_features, new_features_only)`**. Callers must pick the right one — this distinction propagates through every model:

- **Encoder**: uses `all_features` for the skip connection (full dense concat).
- **Bottleneck / Decoder**: uses `new_features_only` so channel counts don't explode.
- `CRASPP.forward` returns `(out, out)` to match this interface so it can be a drop-in bottleneck replacement.

When adding any block that participates in this pipeline, return the same `(all, new)` tuple.

### SkyScapesNet topology (models/skyscapesnet.py)

Modified FC-DenseNet-103 with five novel components. Layer config is `[4,5,7,10,12]` encoder, `15` bottleneck, `[12,10,7,5,4]` decoder, `growth_rate=32`, `n_init_features=48`.

| Component | File | Role |
|---|---|---|
| FDB (FullyDenseBlock) | `models/layers.py` | Replaces DenseBlock; SeparableLayer + residual 1x1 projections from prior new-features |
| DoS / UpS | `models/layers.py` | Replace TransitionDown/Up; UpS adds transposed-conv path + nearest-neighbour-projection path before concat |
| FRSR | `models/frsr.py` | Per-encoder-stage full-resolution residual (dual stream) applied AFTER FDB, BEFORE downsample, and the FRSR output becomes the skip |
| CRASPP | `models/craspp.py` | Bottleneck; cascading forward (pool→a18→a12→a6→1x1) + reverse (1x1→a6→a12→a18), concatenated then projected |
| LKBR | `models/lkbr.py` | Applied to every decoder skip; concatenates `[skip, refined]`, **doubling skip channel count** |

**Skip channel math gotcha**: because LKBR concatenates original + refined, every decoder concat is `concat_ch = upsample_channels + skip_ch * 2` (not `+ skip_ch`). Get this wrong and shapes silently work but parameter counts blow up.

**Skip indexing**: encoder appends in shallow-to-deep order, so `skips[0]` is shallowest (skip1, highest resolution) and `skips[-1]` is deepest (skip5). The shared decoder consumes `skip5, skip4`; the three task branches each consume `skip3, skip2, skip1`.

**Multi-task branching**: after 2 shared upsampling steps the feature map splits into three `DecoderBranch` instances (each = 3 more upsampling steps with their own FDBs/LKBRs):
1. `seg_branch` → `(N, n_classes, H, W)`
2. `multi_edge_branch` → `(N, n_classes, H, W)`
3. `binary_edge_branch` → `(N, 1, H, W)`

`forward()` always returns the 3-tuple; consumers (`evaluate.py`, `validate()`) use only `seg`.

### Training fork on `is_multitask`

`train.py` keys everything off `is_multitask = args.model == "skyscapesnet"`:
- Multitask: `MultiTaskLoss(seg, multi_edge, binary_edge, mask)` returns `(loss, dict)`. Edge targets are **auto-derived from seg targets via a Sobel-like 3x3 kernel** when not passed (`losses/loss.py` `_compute_edge_targets`).
- Baseline: a `lambda` combining `WeightedCrossEntropyLoss + SoftIoULoss`.

`ignore_index=255` is the project-wide convention for masked pixels — `SoftIoULoss`, `SoftDiceLoss`, `WeightedCrossEntropyLoss`, and `ConfusionMatrix` all honour it (the metric drops any `target >= n_classes`).

### Dataset (data/skyscapes_dataset.py)

DLR-SkyScapes has 31 fine-grained classes; **SkyScapes-Dense merges the 12 lane-marking types into one** → 20 classes via `DENSE_ID_MAP`. Use `task="dense"` (default) for this; `task="raw"` keeps all 31.

Expected layout: `root/{images,labels}/{train,val}/`. Labels are read as single-channel class-indexed by default; pass `color_to_id={(R,G,B): id}` if your masks are RGB-encoded (then `rgb_mask_to_class_ids` does the conversion). Images get random-cropped to `patch_size` (default 512); source aerial tiles are 5616x3744, so the crop must fit.

### Two augmentation pipelines (only one is wired up)

- `data/transforms.py` — PIL-based `JointCompose` / `JointRandomHorizontalFlip` / etc. **This is what `train.py` actually uses.**
- `utils/augment.py` — Albumentations multi-target pipeline (with `multi_edge_mask`, `binary_edge_mask` additional targets). **Not currently called by `train.py`** — present for future multi-target training where edge masks come from disk rather than being derived from seg labels.

### HuggingFace Hub integration

`SkyScapesNet` and `FCDenseNet` both inherit `PyTorchModelHubMixin`. Constructor kwargs are auto-serialised into `config.json` by the mixin, so any new `__init__` parameter must be JSON-serialisable and have a sensible default. `test_hub_roundtrip` in `test_models.py` verifies save/load preserves weights exactly.

## Non-obvious gotchas

- **BN with 1x1 spatial**: `CRASPP.image_pool` deliberately omits BatchNorm after `AdaptiveAvgPool2d(1)` — BN over a 1x1 feature map breaks with `batch_size=1` (the paper's default). Several tests call `model.eval()` for the same reason.
- **CRASPP channel arithmetic**: each cascading stage cats the prior accumulated tensor with its new output, so input channels to each atrous conv grow by `mid_channels` per stage. Final projection consumes `9 * mid_channels` (5 forward + 4 reverse). Audit these counts when changing `mid_channels` or the dilation chain.
- **Spatial mismatch handling in `UpsamplingBlock` / `TransitionUp`**: transposed-conv output is cropped to skip dimensions before concat. If you change padding/stride, recheck — silent wrong-size cropping will not error.
- **Edge target derivation is approximate**: `MultiTaskLoss._compute_edge_targets` uses a simple Laplacian-like kernel on class indices. For real evaluation against the paper, supply explicit edge GT via the `edge_targets` argument.
- **Optimizer is not specified by the paper**: code uses Adam + CosineAnnealingLR — keep this in mind when comparing to paper numbers (40.13 mIoU target).
