"""Confidence-Calibrated Reweighting (CCR) Loss.

This is the core contribution of the CCR-Tabular paper. Implements a dynamic
loss function that simultaneously handles class imbalance, asymmetric label
noise, and feature-correlated label noise.

Mathematical Formulation & Autograd Alignment:
    In accordance with Proposition 1, the sample weights are computed from model
    predictions and historical variance under torch.no_grad() / .detach() to decouple
    sample importance weighting from the backpropagation chain:

    Step 1 — Raw dynamic weight:
        w_i = (1 - p_i) + beta * Var_K(p_i) * I(p_i > tau) + gamma_{y_i}
        where:
          - p_i = P(y = y_i | x_i) is the true-class probability
          - Var_K(p_i) is the rolling variance of confidence over the last K epochs
          - I(p_i > tau) is the confidence gate
          - gamma_{y_i} is the inverse class frequency weight

    Step 2 — Batch-level normalization:
        w_hat_i = (w_i / (sum_{j=1}^B w_j + eps)) * B
        Enforces mean(w_hat) = 1.0, controlling the expected gradient scale.

    Step 3 — Final weighted objective:
        L_CCR = (1 / B) * sum_{i=1}^B w_hat_i * CE(logits_i, y_i)
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config import BETA, K, TAU

logger = logging.getLogger(__name__)


def compute_batch_normalized_weights(
    raw_weights: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Normalize raw weights such that batch mean equals 1.0.

    w_hat_i = (w_i / (sum(w) + eps)) * B

    Args:
        raw_weights: 1D tensor of raw non-negative weights [B].
        eps: Small constant for numerical stability.

    Returns:
        Tuple of (normalized_weights, telemetry_dict).
    """
    batch_size = raw_weights.shape[0]
    if batch_size == 0:
        return raw_weights, {"raw_mean": 0.0, "raw_sum": 0.0, "S_over_B": 0.0}

    # Ensure detached weights for scale calculation
    w_detached = raw_weights.detach()
    raw_sum = w_detached.sum()
    s_over_b = (raw_sum / batch_size).item()

    # Guard against NaN/Inf or all-zero weights
    if not torch.isfinite(raw_sum) or raw_sum.item() <= 0.0:
        norm_w = torch.ones_like(raw_weights)
    else:
        norm_w = (raw_weights / (raw_sum + eps)) * batch_size

    telemetry = {
        "raw_sum": raw_sum.item(),
        "raw_mean": s_over_b,
        "S_over_B": s_over_b,
        "max_weight": w_detached.max().item() if batch_size > 0 else 0.0,
        "min_weight": w_detached.min().item() if batch_size > 0 else 0.0,
        "std_weight": w_detached.std().item() if batch_size > 1 else 0.0,
    }
    return norm_w, telemetry


class CCRLoss(nn.Module):
    """Confidence-Calibrated Reweighting Loss with detached-weight scale control.

    Args:
        n_samples: Total number of training samples (for history buffer).
        n_classes: Number of classes (>= 2).
        class_counts: List of sample counts per class.
        tau: Confidence gate threshold. Default 0.3.
        beta: Variance scaling factor. Default 0.5.
        K: History window length in epochs. Default 5.
        eps: Numerical stability constant. Default 1e-8.
        device: Torch device.
    """

    def __init__(
        self,
        n_samples: int,
        n_classes: int = 2,
        class_counts: Optional[List[int]] = None,
        tau: float = TAU,
        beta: float = BETA,
        K: int = K,
        eps: float = 1e-8,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}.")
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}.")
        if class_counts is not None:
            if len(class_counts) != n_classes:
                raise ValueError(
                    f"class_counts length ({len(class_counts)}) must equal "
                    f"n_classes ({n_classes})."
                )
            if any(c <= 0 for c in class_counts):
                raise ValueError(
                    f"All class_counts must be positive. Got: {class_counts}."
                )
        if not (0.0 <= tau <= 1.0):
            raise ValueError(f"tau must be in [0, 1], got {tau}.")
        if beta < 0:
            raise ValueError(f"beta must be non-negative, got {beta}.")
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}.")

        self.n_samples = n_samples
        self.n_classes = n_classes
        self.tau = tau
        self.beta = beta
        self.K = K
        self.eps = eps
        self.device = device or torch.device("cpu")

        # Confidence history buffer: [n_samples, K], detached, no autograd
        self.register_buffer(
            "history",
            torch.full((n_samples, K), 0.5, dtype=torch.float32),
        )

        # Class weights (gamma)
        if class_counts is not None:
            class_weights = self._compute_class_weights(class_counts)
        else:
            class_weights = torch.full((n_classes,), 1.0 / n_classes, dtype=torch.float32)
        self.register_buffer("class_weights", class_weights)

        # Telemetry from last forward pass
        self.last_telemetry: Dict[str, float] = {}

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_indices: torch.Tensor,
        current_epoch: int = 0,
    ) -> torch.Tensor:
        """Compute CCR loss with detached weights.

        Args:
            logits: Model predictions before softmax, shape [B, C].
            targets: True class indices, shape [B], dtype long.
            sample_indices: Dataset sample indices, shape [B].
            current_epoch: Current training epoch (0-indexed).

        Returns:
            Scalar loss tensor ready for backward().
        """
        if logits.shape[1] != self.n_classes:
            raise RuntimeError(
                f"logits has {logits.shape[1]} classes but CCRLoss was initialized "
                f"with n_classes={self.n_classes}."
            )
        if sample_indices.numel() > 0 and sample_indices.max().item() >= self.n_samples:
            raise RuntimeError(
                f"sample_index {sample_indices.max().item()} >= n_samples "
                f"({self.n_samples}). Ensure sample_indices are global dataset indices."
            )

        batch_size = logits.shape[0]
        if batch_size == 0:
            return logits.sum() * 0.0

        # Compute dynamic weights under torch.no_grad()
        with torch.no_grad():
            probs = F.softmax(logits.detach(), dim=1)  # [B, C]
            # p_i = probability assigned to the TRUE class
            p_i = probs[torch.arange(batch_size, device=logits.device), targets]  # [B]

            # Component 1: Focal penalty (1 - p_i)
            focal_term = 1.0 - p_i  # [B]

            # Component 2: Variance gate beta * Var(p_i) * I(p_i > tau)
            variance = self._compute_variance(sample_indices, current_epoch)  # [B]
            confidence_gate = (p_i > self.tau).float()  # [B]
            variance_term = self.beta * variance * confidence_gate  # [B]

            # Component 3: Class imbalance weight
            gamma = self.class_weights[targets]  # [B]

            # Raw weight combination
            raw_weights = focal_term + variance_term + gamma  # [B]

            # Batch normalization: mean(normalized_weights) = 1.0
            normalized_weights, telemetry = compute_batch_normalized_weights(
                raw_weights, eps=self.eps
            )
            self.last_telemetry = telemetry

        # Per-sample cross-entropy (autograd flows through CE only, not the weights)
        per_sample_ce = F.cross_entropy(logits, targets, reduction="none")  # [B]

        # Final CCR loss
        loss = (normalized_weights * per_sample_ce).mean()
        return loss

    def update_history(
        self,
        probs: torch.Tensor,
        sample_indices: torch.Tensor,
        current_epoch: int,
    ) -> None:
        """Update confidence history tensor after optimizer step.

        Args:
            probs: Detached softmax probabilities [B, C].
            sample_indices: Sample indices [B].
            current_epoch: Current epoch index.
        """
        with torch.no_grad():
            col = current_epoch % self.K
            max_probs = probs.detach().max(dim=1).values  # [B]
            self.history[sample_indices, col] = max_probs

    def _compute_variance(
        self,
        sample_indices: torch.Tensor,
        current_epoch: int,
    ) -> torch.Tensor:
        """Compute rolling variance for samples over available epochs."""
        n_filled = min(current_epoch + 1, self.K)
        if n_filled <= 1:
            return torch.zeros(len(sample_indices), device=sample_indices.device)

        cols = [(current_epoch - i) % self.K for i in range(n_filled)]
        history_slice = self.history[sample_indices][:, cols]  # [B, n_filled]
        return history_slice.var(dim=1)  # [B]

    def _compute_class_weights(self, class_counts: List[int]) -> torch.Tensor:
        """Compute normalized inverse class frequency weights."""
        inv_counts = torch.tensor(
            [1.0 / c for c in class_counts], dtype=torch.float32
        )
        normalized = inv_counts / inv_counts.sum()
        return normalized


# ── Ablation Variants with Detached Formulation ───────────────────────────────

class CCRLossNoGate(CCRLoss):
    """CCR ablation: variance term active for all samples (no confidence gate)."""

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_indices: torch.Tensor,
        current_epoch: int = 0,
    ) -> torch.Tensor:
        if logits.shape[1] != self.n_classes:
            raise RuntimeError(f"logits class dimension mismatch.")
        batch_size = logits.shape[0]
        if batch_size == 0:
            return logits.sum() * 0.0

        with torch.no_grad():
            probs = F.softmax(logits.detach(), dim=1)
            p_i = probs[torch.arange(batch_size, device=logits.device), targets]
            focal_term = 1.0 - p_i
            variance = self._compute_variance(sample_indices, current_epoch)
            variance_term = self.beta * variance  # NO gate
            gamma = self.class_weights[targets]
            raw_weights = focal_term + variance_term + gamma
            normalized_weights, telemetry = compute_batch_normalized_weights(
                raw_weights, eps=self.eps
            )
            self.last_telemetry = telemetry

        per_sample_ce = F.cross_entropy(logits, targets, reduction="none")
        return (normalized_weights * per_sample_ce).mean()


class CCRLossNoVariance(CCRLoss):
    """CCR ablation: no variance term at all (focal + class weight only)."""

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_indices: torch.Tensor,
        current_epoch: int = 0,
    ) -> torch.Tensor:
        if logits.shape[1] != self.n_classes:
            raise RuntimeError(f"logits class dimension mismatch.")
        batch_size = logits.shape[0]
        if batch_size == 0:
            return logits.sum() * 0.0

        with torch.no_grad():
            probs = F.softmax(logits.detach(), dim=1)
            p_i = probs[torch.arange(batch_size, device=logits.device), targets]
            focal_term = 1.0 - p_i
            gamma = self.class_weights[targets]
            raw_weights = focal_term + gamma  # NO variance
            normalized_weights, telemetry = compute_batch_normalized_weights(
                raw_weights, eps=self.eps
            )
            self.last_telemetry = telemetry

        per_sample_ce = F.cross_entropy(logits, targets, reduction="none")
        return (normalized_weights * per_sample_ce).mean()


class CCRLossNoNormalization(CCRLoss):
    """CCR ablation: unnormalized raw dynamic weights."""

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_indices: torch.Tensor,
        current_epoch: int = 0,
    ) -> torch.Tensor:
        if logits.shape[1] != self.n_classes:
            raise RuntimeError(f"logits class dimension mismatch.")
        batch_size = logits.shape[0]
        if batch_size == 0:
            return logits.sum() * 0.0

        with torch.no_grad():
            probs = F.softmax(logits.detach(), dim=1)
            p_i = probs[torch.arange(batch_size, device=logits.device), targets]
            focal_term = 1.0 - p_i
            variance = self._compute_variance(sample_indices, current_epoch)
            confidence_gate = (p_i > self.tau).float()
            variance_term = self.beta * variance * confidence_gate
            gamma = self.class_weights[targets]
            raw_weights = focal_term + variance_term + gamma
            self.last_telemetry = {
                "raw_sum": raw_weights.sum().item(),
                "raw_mean": (raw_weights.sum() / batch_size).item(),
                "S_over_B": (raw_weights.sum() / batch_size).item(),
            }

        per_sample_ce = F.cross_entropy(logits, targets, reduction="none")
        # Raw weights applied directly without normalization
        return (raw_weights * per_sample_ce).mean()


def get_ccr_loss(
    variant: str,
    n_samples: int,
    n_classes: int,
    class_counts: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
    tau: float = TAU,
    beta: float = BETA,
    K: int = K,
) -> CCRLoss:
    """Factory for CCR loss variants."""
    kwargs = dict(
        n_samples=n_samples,
        n_classes=n_classes,
        class_counts=class_counts,
        tau=tau,
        beta=beta,
        K=K,
        device=device,
    )
    registry = {
        "ccr": CCRLoss,
        "ccr_no_gate": CCRLossNoGate,
        "ccr_no_variance": CCRLossNoVariance,
        "ccr_no_norm": CCRLossNoNormalization,
    }
    if variant not in registry:
        raise ValueError(
            f"Unknown CCR variant '{variant}'. Valid options: {list(registry.keys())}."
        )
    return registry[variant](**kwargs).to(device or torch.device("cpu"))
