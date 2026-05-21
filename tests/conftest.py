"""Shared pytest fixtures."""
import torch
import pytest


@pytest.fixture
def fixed_seed():
    """Seed torch RNG for reproducibility."""
    torch.manual_seed(42)
    yield
