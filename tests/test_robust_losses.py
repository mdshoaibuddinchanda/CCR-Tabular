"""Unit tests for robust loss functions and normalized loss variants."""

import pytest
import torch

from src.loss.robust_losses import (
    DynamicCELoss,
    EarlyLearningRegularizationLoss,
    FocalLoss,
    GeneralizedCrossEntropyLoss,
    NormalizedDynamicCELoss,
    NormalizedFocalLoss,
    NormalizedGCELoss,
    NormalizedLossWrapper,
    NormalizedSCELoss,
    NormalizedWeightedCELoss,
    SymmetricCrossEntropyLoss,
    build_loss,
)


def test_gce_loss_forward_backward():
    """GCE loss computes finite value and gradients flow to logits."""
    B, C = 16, 2
    logits = torch.randn(B, C, requires_grad=True)
    targets = torch.randint(0, C, (B,))

    loss_fn = GeneralizedCrossEntropyLoss(q=0.7)
    loss = loss_fn(logits, targets)

    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_sce_loss_forward_backward():
    """SCE loss computes finite value and gradients flow to logits."""
    B, C = 16, 3
    logits = torch.randn(B, C, requires_grad=True)
    targets = torch.randint(0, C, (B,))

    loss_fn = SymmetricCrossEntropyLoss(alpha=0.1, beta=1.0, n_classes=C)
    loss = loss_fn(logits, targets)

    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_elr_loss_forward_backward():
    """ELR loss updates momentum buffer and computes finite loss."""
    B, C, N = 8, 2, 50
    logits = torch.randn(B, C, requires_grad=True)
    targets = torch.randint(0, C, (B,))
    indices = torch.arange(B)

    loss_fn = EarlyLearningRegularizationLoss(n_samples=N, n_classes=C, lambda_elr=3.0)
    loss = loss_fn(logits, targets, sample_indices=indices, current_epoch=0)

    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    # Check that historical buffer got updated
    assert torch.any(loss_fn.target_history[indices] > 0)


def test_normalized_loss_variants_batch_mean():
    """All normalized loss variants must record raw telemetry and finite loss."""
    B, C = 16, 2
    logits = torch.randn(B, C, requires_grad=True)
    targets = torch.randint(0, C, (B,))

    norm_losses = [
        NormalizedFocalLoss(),
        NormalizedWeightedCELoss(class_counts=[10, 6]),
        NormalizedGCELoss(q=0.7),
        NormalizedSCELoss(alpha=0.1, beta=1.0, n_classes=C),
        DynamicCELoss(),
        NormalizedDynamicCELoss(),
    ]

    for loss_fn in norm_losses:
        logits.grad = None
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss)
        assert hasattr(loss_fn, "last_telemetry")
        assert "S_over_B" in loss_fn.last_telemetry
        loss.backward()
        assert logits.grad is not None


def test_build_loss_factory():
    """build_loss factory instantiates all registered losses."""
    for loss_name in [
        "ce", "wce", "focal", "gce", "sce", "elr",
        "norm_wce", "norm_focal", "norm_gce", "norm_sce",
        "dynamic_ce", "norm_dynamic_ce",
        "ccr", "ccr_no_norm", "ccr_no_gate", "ccr_no_variance",
    ]:
        loss_mod = build_loss(
            loss_name=loss_name,
            n_samples=100,
            n_classes=2,
            class_counts=[70, 30],
        )
        assert isinstance(loss_mod, torch.nn.Module)
