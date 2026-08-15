"""Unit test validating mathematical reconciliation of CCR loss gradients.

Reviewer Problem Addressed:
    Reviewer 4 noted that Proposition 1 treats weighting coefficients as fixed when
    analyzing the weighted gradient scale.
    This test verifies that:
      1. When weights are detached (w = w.detach()), PyTorch autograd computes
         EXACTLY the fixed-coefficient weighted gradient:
             grad_{logits} L = (1/B) * w_hat_i * (p_i - y_one_hot)
      2. The autograd gradient matches the closed-form analytical gradient to machine precision.
      3. The autograd gradient matches the numerical finite-difference gradient.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.loss.ccr_loss import CCRLoss, compute_batch_normalized_weights


def test_analytical_vs_autograd_detached_gradient():
    """Verify that autograd of CCR loss equals the theoretical weighted gradient."""
    torch.manual_seed(42)
    B, C = 8, 3
    logits = torch.randn(B, C, requires_grad=True, dtype=torch.float64)
    targets = torch.randint(0, C, (B,))
    indices = torch.arange(B)

    # Initialize CCR loss
    ccr = CCRLoss(
        n_samples=B,
        n_classes=C,
        class_counts=[4, 2, 2],
        tau=0.3,
        beta=0.5,
        K=3,
        eps=1e-8,
        device=torch.device("cpu"),
    )

    # Forward pass
    loss = ccr(logits, targets, indices, current_epoch=0)
    loss.backward()
    autograd_grad = logits.grad.clone()

    # Theoretical Analytical Gradient Computation:
    # L = (1/B) sum_i w_hat_i * CE(logits_i, y_i)
    # where w_hat_i is treated as fixed constants (detached).
    # d(CE_i)/d(logits_ik) = probs_ik - 1(y_i == k)
    # dL/d(logits_ik) = (1/B) * w_hat_i * (probs_ik - 1(y_i == k))
    with torch.no_grad():
        probs = F.softmax(logits, dim=1)
        p_true = probs[torch.arange(B), targets]
        focal_term = 1.0 - p_true
        gamma = ccr.class_weights[targets]
        raw_w = focal_term + gamma  # variance is 0 at epoch 0
        norm_w, _ = compute_batch_normalized_weights(raw_w, eps=1e-8)

        y_one_hot = F.one_hot(targets, num_classes=C).double()
        analytical_grad = (norm_w.unsqueeze(1) * (probs - y_one_hot)) / B

    # Assert exact agreement between autograd and analytical formulation
    max_diff = (autograd_grad - analytical_grad).abs().max().item()
    assert max_diff < 1e-10, (
        f"Autograd gradient diverged from theoretical weighted gradient! "
        f"Max absolute difference: {max_diff}"
    )


def test_autograd_vs_finite_difference_gradient():
    """Verify autograd gradient against finite difference on a toy optimization problem."""
    torch.manual_seed(123)
    B, C = 4, 2
    logits = torch.randn(B, C, requires_grad=True, dtype=torch.float64)
    targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    indices = torch.arange(B)

    ccr = CCRLoss(
        n_samples=B,
        n_classes=C,
        class_counts=[2, 2],
        tau=0.3,
        beta=0.5,
        K=2,
        eps=1e-8,
        device=torch.device("cpu"),
    )

    # Compute autograd gradient
    loss = ccr(logits, targets, indices, current_epoch=0)
    loss.backward()
    autograd_grad = logits.grad.clone()

    # Compute numerical gradient via central finite difference
    eps = 1e-6
    numerical_grad = torch.zeros_like(logits)

    with torch.no_grad():
        # Notice: In the detached-weight objective, weights are evaluated at logits_0
        # and held fixed as coefficients during differentiation.
        probs = F.softmax(logits, dim=1)
        p_true = probs[torch.arange(B), targets]
        raw_w = (1.0 - p_true) + ccr.class_weights[targets]
        norm_w, _ = compute_batch_normalized_weights(raw_w, eps=1e-8)

        for i in range(B):
            for c in range(C):
                logits_pos = logits.clone()
                logits_pos[i, c] += eps
                ce_pos = F.cross_entropy(logits_pos, targets, reduction="none")
                loss_pos = (norm_w * ce_pos).mean()

                logits_neg = logits.clone()
                logits_neg[i, c] -= eps
                ce_neg = F.cross_entropy(logits_neg, targets, reduction="none")
                loss_neg = (norm_w * ce_neg).mean()

                numerical_grad[i, c] = (loss_pos - loss_neg) / (2 * eps)

    rel_error = (autograd_grad - numerical_grad).abs() / (numerical_grad.abs() + 1e-8)
    max_rel_error = rel_error.max().item()
    assert max_rel_error < 1e-5, (
        f"Autograd gradient diverged from finite difference! Max relative error: {max_rel_error}"
    )
