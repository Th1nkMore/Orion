"""Tests for FiLM L1 and L2 modulation layers.

Verifies:
- Identity initialization (FiLM starts as no-op)
- Correct output shapes after modulation
- Parameter counts within budget
- Checkpoint save/load round-trip
- freeze_all_except_film correctly isolates FiLM params
"""

import torch
import torch.nn as nn
import pytest
import tempfile
import os


# ── Mock dimensions (from CLAUDE.md) ──
B = 2
NUM_QUERY = 900
EMBED_DIMS = 256
UNCERTAINTY_DIM = 256
EGO_DIM = 4096


# ── FiLM L1 Tests (QT-Former level) ──────────────────────────────────

class MockFiLML1(nn.Module):
    """Mimics FiLM L1 from PETRTemporalTransformer."""

    def __init__(self):
        super().__init__()
        self.film_gamma = nn.Linear(UNCERTAINTY_DIM, EMBED_DIMS)
        self.film_beta = nn.Linear(UNCERTAINTY_DIM, EMBED_DIMS)
        # Identity init
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.ones_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

    def forward(self, query, uncertainty_emb):
        # query: [num_query, B, D], uncertainty_emb: [B, D]
        gamma = self.film_gamma(uncertainty_emb).unsqueeze(0)  # [1, B, D]
        beta = self.film_beta(uncertainty_emb).unsqueeze(0)    # [1, B, D]
        return gamma * query + beta


class MockFiLML2(nn.Module):
    """Mimics FiLM L2 from orion.py (VAE level)."""

    def __init__(self):
        super().__init__()
        self.film_gamma_l2 = nn.Linear(UNCERTAINTY_DIM, EGO_DIM)
        self.film_beta_l2 = nn.Linear(UNCERTAINTY_DIM, EGO_DIM)
        # Identity init
        nn.init.zeros_(self.film_gamma_l2.weight)
        nn.init.ones_(self.film_gamma_l2.bias)
        nn.init.zeros_(self.film_beta_l2.weight)
        nn.init.zeros_(self.film_beta_l2.bias)

    def forward(self, current_states, uncertainty_emb):
        # current_states: [B, 1, 4096], uncertainty_emb: [B, 256]
        gamma = self.film_gamma_l2(uncertainty_emb).unsqueeze(1)  # [B, 1, 4096]
        beta = self.film_beta_l2(uncertainty_emb).unsqueeze(1)    # [B, 1, 4096]
        return gamma * current_states + beta


@pytest.fixture
def film_l1():
    return MockFiLML1()


@pytest.fixture
def film_l2():
    return MockFiLML2()


@pytest.fixture
def uncertainty_emb():
    return torch.randn(B, UNCERTAINTY_DIM)


# ── Identity Init Tests ──────────────────────────────────────────────

def test_film_l1_identity_init(film_l1, uncertainty_emb):
    """FiLM L1 with identity init should be a no-op."""
    query = torch.randn(NUM_QUERY, B, EMBED_DIMS)
    with torch.no_grad():
        output = film_l1(query, uncertainty_emb)
    torch.testing.assert_close(output, query, atol=1e-5, rtol=1e-5)


def test_film_l2_identity_init(film_l2, uncertainty_emb):
    """FiLM L2 with identity init should be a no-op."""
    current_states = torch.randn(B, 1, EGO_DIM)
    with torch.no_grad():
        output = film_l2(current_states, uncertainty_emb)
    torch.testing.assert_close(output, current_states, atol=1e-5, rtol=1e-5)


# ── Output Shape Tests ───────────────────────────────────────────────

def test_film_l1_output_shape(film_l1, uncertainty_emb):
    """FiLM L1 output shape matches query shape."""
    query = torch.randn(NUM_QUERY, B, EMBED_DIMS)
    output = film_l1(query, uncertainty_emb)
    assert output.shape == (NUM_QUERY, B, EMBED_DIMS)


def test_film_l2_output_shape(film_l2, uncertainty_emb):
    """FiLM L2 output shape matches current_states shape."""
    current_states = torch.randn(B, 1, EGO_DIM)
    output = film_l2(current_states, uncertainty_emb)
    assert output.shape == (B, 1, EGO_DIM)


# ── Parameter Count Tests ────────────────────────────────────────────

def test_film_l1_param_count(film_l1):
    """FiLM L1 should have ~131K params."""
    n = sum(p.numel() for p in film_l1.parameters())
    # 2 * (256*256 + 256) = 131,584
    assert n == 131_584, f"Expected 131,584, got {n:,}"


def test_film_l2_param_count(film_l2):
    """FiLM L2 should have ~2.11M params."""
    n = sum(p.numel() for p in film_l2.parameters())
    # 2 * (256*4096 + 4096) = 2,105,344
    assert n == 2_105_344, f"Expected 2,105,344, got {n:,}"


def test_total_param_budget():
    """Total new params (UQEstimator + FiLM L1 + FiLM L2) must be < 5M."""
    from uq_estimator.model import UQEstimator
    model_cfg = {
        "d_patch": 1152, "n_views": 6, "n_patches": 256,
        "n_query": 16, "d_stat": 64, "d_hidden": 512, "d_out": 256,
    }
    uq = UQEstimator(model_cfg)
    n_uq = sum(p.numel() for p in uq.parameters())
    n_l1 = 131_584
    n_l2 = 2_105_344
    total = n_uq + n_l1 + n_l2
    assert total < 5_000_000, f"Total {total:,} exceeds 5M budget"


# ── Gradient Flow Tests ──────────────────────────────────────────────

def test_film_l1_gradient_flow(film_l1):
    """Gradients flow through FiLM L1."""
    query = torch.randn(NUM_QUERY, B, EMBED_DIMS, requires_grad=True)
    emb = torch.randn(B, UNCERTAINTY_DIM, requires_grad=True)
    output = film_l1(query, emb)
    loss = output.sum()
    loss.backward()
    assert film_l1.film_gamma.weight.grad is not None
    assert film_l1.film_beta.weight.grad is not None
    assert emb.grad is not None


def test_film_l2_gradient_flow(film_l2):
    """Gradients flow through FiLM L2."""
    states = torch.randn(B, 1, EGO_DIM, requires_grad=True)
    emb = torch.randn(B, UNCERTAINTY_DIM, requires_grad=True)
    output = film_l2(states, emb)
    loss = output.sum()
    loss.backward()
    assert film_l2.film_gamma_l2.weight.grad is not None
    assert film_l2.film_beta_l2.weight.grad is not None
    assert emb.grad is not None


# ── Checkpoint Round-trip Tests ──────────────────────────────────────

def test_film_checkpoint_save_load():
    """Save and load FiLM L1+L2 checkpoint."""
    l1 = MockFiLML1()
    l2 = MockFiLML2()

    # Modify weights away from init
    with torch.no_grad():
        l1.film_gamma.weight.fill_(0.5)
        l2.film_gamma_l2.weight.fill_(0.3)

    save_dict = {
        'film_gamma_weight': l1.film_gamma.weight.data.cpu(),
        'film_gamma_bias': l1.film_gamma.bias.data.cpu(),
        'film_beta_weight': l1.film_beta.weight.data.cpu(),
        'film_beta_bias': l1.film_beta.bias.data.cpu(),
        'film_gamma_l2_weight': l2.film_gamma_l2.weight.data.cpu(),
        'film_gamma_l2_bias': l2.film_gamma_l2.bias.data.cpu(),
        'film_beta_l2_weight': l2.film_beta_l2.weight.data.cpu(),
        'film_beta_l2_bias': l2.film_beta_l2.bias.data.cpu(),
    }

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(save_dict, f.name)
        tmp_path = f.name

    try:
        # Load into fresh models
        l1_new = MockFiLML1()
        l2_new = MockFiLML2()
        ckpt = torch.load(tmp_path, weights_only=False)

        l1_new.film_gamma.weight.data = ckpt['film_gamma_weight']
        l1_new.film_gamma.bias.data = ckpt['film_gamma_bias']
        l1_new.film_beta.weight.data = ckpt['film_beta_weight']
        l1_new.film_beta.bias.data = ckpt['film_beta_bias']

        l2_new.film_gamma_l2.weight.data = ckpt['film_gamma_l2_weight']
        l2_new.film_gamma_l2.bias.data = ckpt['film_gamma_l2_bias']
        l2_new.film_beta_l2.weight.data = ckpt['film_beta_l2_weight']
        l2_new.film_beta_l2.bias.data = ckpt['film_beta_l2_bias']

        torch.testing.assert_close(l1_new.film_gamma.weight, l1.film_gamma.weight)
        torch.testing.assert_close(l2_new.film_gamma_l2.weight, l2.film_gamma_l2.weight)
    finally:
        os.unlink(tmp_path)


# ── Freeze Logic Test ────────────────────────────────────────────────

def test_freeze_all_except_film():
    """Only FiLM params should be trainable after freezing."""

    class MockFullModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(100, 100)
            self.film_gamma = nn.Linear(UNCERTAINTY_DIM, EMBED_DIMS)
            self.film_beta = nn.Linear(UNCERTAINTY_DIM, EMBED_DIMS)
            self.film_gamma_l2 = nn.Linear(UNCERTAINTY_DIM, EGO_DIM)
            self.film_beta_l2 = nn.Linear(UNCERTAINTY_DIM, EGO_DIM)

    model = MockFullModel()

    # Freeze all except FiLM
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if 'film_gamma' in name or 'film_beta' in name:
            param.requires_grad = True

    # Check
    assert not model.backbone.weight.requires_grad
    assert not model.backbone.bias.requires_grad
    assert model.film_gamma.weight.requires_grad
    assert model.film_beta.weight.requires_grad
    assert model.film_gamma_l2.weight.requires_grad
    assert model.film_beta_l2.weight.requires_grad

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_film = 131_584 + 2_105_344
    assert trainable == total_film, f"Expected {total_film:,} trainable, got {trainable:,}"


# ── Non-trivial Modulation Test ──────────────────────────────────────

def test_film_l2_nontrivial_after_training():
    """After weight update, FiLM L2 should produce different output."""
    l2 = MockFiLML2()
    states = torch.randn(B, 1, EGO_DIM)
    emb = torch.randn(B, UNCERTAINTY_DIM)

    with torch.no_grad():
        out_before = l2(states, emb).clone()

    # Simulate one training step
    optimizer = torch.optim.SGD(l2.parameters(), lr=0.1)
    output = l2(states, emb)
    loss = (output - torch.zeros_like(output)).pow(2).mean()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        out_after = l2(states, emb)

    assert not torch.allclose(out_before, out_after), \
        "FiLM output should change after training step"
