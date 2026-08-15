"""Unit tests for numerical stability of weight normalization (Section E).

Tests:
  - Very small weights (e.g. 1e-6 and extreme 1e-11)
  - Very large weights (e.g. 1e6)
  - Single dominant weight with all others near zero
  - All equal weights
  - Batch size = 1
  - Last incomplete batch of arbitrary size
  - Handling of non-finite or degenerate inputs
"""

import pytest
import torch

from src.loss.ccr_loss import compute_batch_normalized_weights


def test_equal_weights():
    """All equal weights should normalize to 1.0 everywhere."""
    weights = torch.full((16,), 3.5)
    norm_w, telemetry = compute_batch_normalized_weights(weights)
    assert torch.allclose(norm_w, torch.ones_like(norm_w), atol=1e-5)
    assert abs(norm_w.mean().item() - 1.0) < 1e-6
    assert abs(telemetry["S_over_B"] - 3.5) < 1e-5


def test_single_dominant_weight():
    """Single dominant weight must not explode or cause NaNs."""
    weights = torch.zeros(32)
    weights[0] = 100.0
    norm_w, telemetry = compute_batch_normalized_weights(weights)
    assert torch.isfinite(norm_w).all()
    assert abs(norm_w.mean().item() - 1.0) < 1e-5
    # The dominant element gets 32.0, others get 0.0
    assert abs(norm_w[0].item() - 32.0) < 1e-4
    assert torch.all(norm_w[1:] == 0.0)


def test_small_weights_above_eps():
    """Small weights above epsilon should normalize to approximately 1.0."""
    weights = torch.full((64,), 1e-5)
    norm_w, telemetry = compute_batch_normalized_weights(weights, eps=1e-8)
    assert torch.isfinite(norm_w).all()
    assert abs(norm_w.mean().item() - 1.0) < 1e-2


def test_extreme_sub_eps_weights_finite_damping():
    """Extremely small weights below epsilon must remain finite and safely damped without NaN."""
    weights = torch.full((64,), 1e-11)
    norm_w, telemetry = compute_batch_normalized_weights(weights, eps=1e-8)
    assert torch.isfinite(norm_w).all()
    assert norm_w.min().item() >= 0.0
    assert norm_w.max().item() <= 1.0


def test_very_large_weights():
    """Very large weights must normalize without overflow."""
    weights = torch.rand(128) * 1e7 + 1e5
    norm_w, telemetry = compute_batch_normalized_weights(weights)
    assert torch.isfinite(norm_w).all()
    assert abs(norm_w.mean().item() - 1.0) < 1e-5


def test_batch_size_one():
    """Batch size 1 should normalize to 1.0."""
    weights = torch.tensor([5.7])
    norm_w, telemetry = compute_batch_normalized_weights(weights)
    assert torch.isfinite(norm_w).all()
    assert abs(norm_w[0].item() - 1.0) < 1e-5
    assert telemetry["S_over_B"] == pytest.approx(5.7, rel=1e-4)


def test_incomplete_batch_arbitrary_sizes():
    """Arbitrary batch sizes (3, 7, 13, 127) must have normalized mean = 1.0."""
    for B in [3, 7, 13, 127]:
        weights = torch.rand(B) * 5.0 + 0.1
        norm_w, telemetry = compute_batch_normalized_weights(weights)
        assert torch.isfinite(norm_w).all()
        assert abs(norm_w.mean().item() - 1.0) < 1e-5
        assert len(norm_w) == B


def test_all_zero_weights_safe_fallback():
    """All-zero weights should safely fallback to uniform unit weights."""
    weights = torch.zeros(16)
    norm_w, telemetry = compute_batch_normalized_weights(weights)
    assert torch.isfinite(norm_w).all()
    assert torch.allclose(norm_w, torch.ones_like(norm_w))
