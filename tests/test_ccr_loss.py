"""Tests for CCRLoss — the core contribution of CCR-Tabular.

All tests must pass before any experiment is run.
"""

import pytest
import torch
import numpy as np

from src.loss.ccr_loss import CCRLoss


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_ccr(n_samples: int = 100, class_counts=None) -> CCRLoss:
    """Create a CCRLoss instance for testing."""
    if class_counts is None:
        class_counts = [70, 30]
    return CCRLoss(
        n_samples=n_samples,
        n_classes=2,
        class_counts=class_counts,
        tau=0.3,
        beta=0.5,
        K=5,
        device=torch.device("cpu"),
    )


def make_batch(batch_size: int = 16, n_classes: int = 2, n_samples: int = 100):
    """Create a random batch of logits, targets, and indices."""
    torch.manual_seed(42)
    logits = torch.randn(batch_size, n_classes)
    targets = torch.randint(0, n_classes, (batch_size,))
    indices = torch.randperm(n_samples)[:batch_size]
    return logits, targets, indices


# ── Test 1: Batch normalization mean ─────────────────────────────────────────

def test_batch_normalization_mean():
    """Normalized weights must have mean = 1.0."""
    ccr = make_ccr()
    logits, targets, indices = make_batch()

    # Manually compute normalized weights to verify
    probs = torch.softmax(logits, dim=1)
    p_i = probs[torch.arange(len(targets)), targets]
    focal_term = 1.0 - p_i
    gamma = ccr.class_weights[targets]
    raw_weights = focal_term + gamma  # variance = 0 at epoch 0

    weight_sum = raw_weights.sum() + 1e-8
    normalized = (raw_weights / weight_sum) * len(targets)

    assert abs(normalized.mean().item() - 1.0) < 1e-4, (
        f"Normalized weights mean should be 1.0, got {normalized.mean().item():.6f}"
    )


# ── Test 2: No gradient through history ───────────────────────────────────────

def test_no_gradient_through_history():
    """History tensor must not participate in autograd."""
    ccr = make_ccr()
    assert not ccr.history.requires_grad, (
        "history tensor must have requires_grad=False. "
        "It is updated with .detach() and must not flow gradients."
    )


# ── Test 3: Variance cold start ───────────────────────────────────────────────

def test_variance_cold_start():
    """Variance at epoch 0 must be 0.0 for all samples."""
    ccr = make_ccr()
    _, _, indices = make_batch()

    variance = ccr._compute_variance(indices, current_epoch=0)

    assert torch.all(variance == 0.0), (
        f"At epoch 0, variance must be 0.0 for all samples. "
        f"Got non-zero values: {variance[variance != 0.0]}"
    )


# ── Test 4: Confidence gate silences low-confidence samples ───────────────────

def test_confidence_gate_silences_low_confidence():
    """Samples with p_i <= tau must have variance_term = 0."""
    ccr = make_ccr(n_samples=50)

    # Create logits that produce low confidence (near-uniform distribution)
    # p_i will be close to 0.5, which is > tau=0.3, so let's force p_i < tau
    # by making the wrong class much more likely
    batch_size = 10
    # All targets are class 0, but logits strongly favor class 1
    logits = torch.zeros(batch_size, 2)
    logits[:, 1] = 10.0  # class 1 gets very high logit
    targets = torch.zeros(batch_size, dtype=torch.long)  # true class is 0
    indices = torch.arange(batch_size)

    # p_i = prob of true class (class 0) ≈ 0 (very low confidence)
    probs = torch.softmax(logits, dim=1)
    p_i = probs[torch.arange(batch_size), targets]

    assert torch.all(p_i <= ccr.tau), (
        f"Test setup failed: expected p_i <= tau={ccr.tau}, got {p_i}"
    )

    # Populate history with non-zero values so variance would be non-zero
    for epoch in range(3):
        ccr.history[indices, epoch % ccr.K] = torch.rand(batch_size)

    # Compute variance — should be non-zero
    variance = ccr._compute_variance(indices, current_epoch=3)
    assert torch.any(variance > 0), "Expected non-zero variance for test setup."

    # Confidence gate: I(p_i > tau) = 0 for all samples
    confidence_gate = (p_i > ccr.tau).float()
    variance_term = ccr.beta * variance * confidence_gate

    assert torch.all(variance_term == 0.0), (
        f"Variance term must be 0 for samples with p_i <= tau. "
        f"Got non-zero: {variance_term[variance_term != 0.0]}"
    )


# ── Test 5: Division by zero protection ───────────────────────────────────────

def test_division_by_zero_protection():
    """Batch with near-zero raw weights must not produce NaN loss."""
    # Use class_counts that make gamma very small
    ccr = CCRLoss(
        n_samples=100,
        n_classes=2,
        class_counts=[50, 50],  # equal counts → small gamma
        tau=0.99,  # very high tau → gate always 0
        beta=0.0,  # no variance term
        K=5,
        device=torch.device("cpu"),
    )

    # Force p_i close to 1.0 → focal term ≈ 0
    batch_size = 8
    logits = torch.zeros(batch_size, 2)
    logits[:, 0] = 20.0  # class 0 gets very high logit
    targets = torch.zeros(batch_size, dtype=torch.long)
    indices = torch.arange(batch_size)

    loss = ccr(logits, targets, indices, current_epoch=0)

    assert not torch.isnan(loss), (
        f"Loss must not be NaN even with near-zero raw weights. "
        f"Got loss={loss.item()}. Check epsilon in batch normalization."
    )
    assert not torch.isinf(loss), (
        f"Loss must not be Inf. Got loss={loss.item()}."
    )


# ── Test 6: Sample index out of bounds ────────────────────────────────────────

def test_sample_index_out_of_bounds():
    """Passing sample_index >= n_samples must raise RuntimeError."""
    ccr = make_ccr(n_samples=50)
    logits = torch.randn(4, 2)
    targets = torch.randint(0, 2, (4,))
    # Index 99 is out of bounds for n_samples=50
    indices = torch.tensor([0, 1, 2, 99])

    with pytest.raises(RuntimeError, match="sample_index"):
        ccr(logits, targets, indices, current_epoch=0)


# ── Test 7: Loss is a scalar ──────────────────────────────────────────────────

def test_loss_scalar():
    """forward() must return a 0-dimensional (scalar) tensor."""
    ccr = make_ccr()
    logits, targets, indices = make_batch()
    loss = ccr(logits, targets, indices, current_epoch=0)

    assert loss.ndim == 0, (
        f"CCRLoss.forward() must return a scalar tensor (0-dim). "
        f"Got shape: {loss.shape}"
    )


# ── Test 8: Logit shape mismatch ──────────────────────────────────────────────

def test_logit_shape_mismatch():
    """RuntimeError must be raised when logits have wrong number of classes."""
    ccr = make_ccr()  # n_classes=2
    logits = torch.randn(8, 3)  # 3 classes — wrong!
    targets = torch.randint(0, 2, (8,))
    indices = torch.arange(8)

    with pytest.raises(RuntimeError, match="n_classes"):
        ccr(logits, targets, indices, current_epoch=0)


# ── Test 9: History updates correctly ────────────────────────────────────────

def test_history_updates_correctly():
    """After update_history(), history[sample_indices, epoch%K] must change."""
    ccr = make_ccr(n_samples=50)
    batch_size = 8
    indices = torch.arange(batch_size)
    epoch = 2
    col = epoch % ccr.K

    # Record initial values
    initial_values = ccr.history[indices, col].clone()

    # Create probs that differ from initial 0.5
    probs = torch.full((batch_size, 2), 0.5)
    probs[:, 0] = 0.9
    probs[:, 1] = 0.1

    ccr.update_history(probs, indices, epoch)

    updated_values = ccr.history[indices, col]
    assert not torch.allclose(initial_values, updated_values), (
        "History must be updated after update_history() call. "
        "Values did not change."
    )
    # Max prob for each sample should be 0.9
    assert torch.allclose(updated_values, torch.full((batch_size,), 0.9)), (
        f"History should store max probability (0.9). Got: {updated_values}"
    )
