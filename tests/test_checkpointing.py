"""Verify gradient checkpointing produces gradients aligned with the non-checkpointed path."""
import torch
import torch.nn.functional as F

from skyscapesnet.models.skyscapesnet import SkyScapesNet


def test_checkpointing_preserves_gradient_direction():
    torch.manual_seed(0)
    model_no_ckpt = SkyScapesNet(in_channels=3, n_classes=20, growth_rate=16, use_checkpointing=False)
    torch.manual_seed(0)
    model_ckpt = SkyScapesNet(in_channels=3, n_classes=20, growth_rate=16, use_checkpointing=True)

    # Identical weights at initialization
    model_ckpt.load_state_dict(model_no_ckpt.state_dict())

    model_no_ckpt.train()
    model_ckpt.train()

    x = torch.randn(1, 3, 64, 64, requires_grad=False)
    target = torch.randint(0, 20, (1, 64, 64))

    # Seed before each forward so dropout / any stochastic ops use the
    # same RNG draws for both models. Without this, dropout at p=0.2
    # decorrelates the two gradient vectors entirely.
    for m in (model_no_ckpt, model_ckpt):
        torch.manual_seed(123)
        m.zero_grad()
        out = m(x)
        loss = F.cross_entropy(out.seg, target)
        loss.backward()

    # Compare gradients of the first conv as a sentinel
    g1 = model_no_ckpt.initial_conv.weight.grad.flatten()
    g2 = model_ckpt.initial_conv.weight.grad.flatten()
    cos = F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).item()
    assert cos >= 0.99, f"cosine similarity {cos:.4f} indicates checkpointing changed gradients"
