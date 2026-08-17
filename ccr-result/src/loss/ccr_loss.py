"""Confidence-Calibrated Reweighting (CCR) Loss.

This is the core contribution of the CCR-Tabular paper. Implements a dynamic
loss function that simultaneously handles class imbalance, asymmetric label
noise, and feature-correlated label noise.

Loss formula:
    Step 1 — Raw weight:
        w_i = (1 - p_i) + beta * Var_K(p_i) * I(p_i > tau) + gamma_yi

    Step 2 — Batch normalization:
        w_hat_i = (w_i / sum(w_j)) * batch_size

    Step 3 — Final loss:
        L_CCR = (1/B) * sum(w_hat_i * CrossEntropy(logits_i, y_i))
"""

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config import BETA, K, TAU

logger = logging.getLogger(__name__)


class CCRLoss(nn.Module):
    """Confidence-Calibrated Reweighting Loss for robust tabular learning.

    Implements the three-component CCR loss:
        1. Focal penalty:   (1 - p_i)           — focus on hard samples
        2. Variance gate:   beta * Var_K(p_i)   — only for confident samples
                            * I(p_i > tau)       — silence noisy/low-conf samples
        3. Class weight:    gamma_yi             — handle class imbalance

    Then applies batch-level normalization so mean weight = 1.0.

    Args:
        n_samples: Total number of training samples (for history tensor).
        n_classes: Number of classes (2 for binary).
        class_counts: List of [n_majority, n_minority] counts.
        tau: Confidence gate threshold. Default 0.3.
        beta: Variance scaling factor. Default 0.5.
        K: History window length in epochs. Default 5.
        device: Torch device.

    Usage:
        ccr = CCRLoss(n_samples=48842, n_classes=2,
                      class_counts=[37155, 11687])
        loss = ccr(logits, targets, sample_indices, current_epoch)
        loss.backward()
        ccr.update_history(softmax_probs, sample_indices, current_epoch)

    Note:
        Call update_history() AFTER loss.backward() and optimizer.step().
        The history tensor is updated with .detach() — no gradients flow through it.
    """

    def __init__(
        self,
        n_samples: int,
        n_classes: int,
        class_counts: List[int],
        tau: float = TAU,
        beta: float = BETA,
        K: int = K,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}.")
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}.")
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
        self.device = device or torch.device("cpu")

        # ── Confidence history buffer ─────────────────────────────────────────
        # Shape: [n_samples, K], initialized to 0.5 (neutral confidence)
        # Stored on device. Updated with .detach() — NO gradient flow.
        self.register_buffer(
            "history",
            torch.full((n_samples, K), 0.5, dtype=torch.float32),
        )

        # ── Class weights (gamma) ─────────────────────────────────────────────
        class_weights = self._compute_class_weights(class_counts)
        self.register_buffer("class_weights", class_weights)

        logger.info(
            f"CCRLoss initialized: n_samples={n_samples}, n_classes={n_classes}, "
            f"tau={tau}, beta={beta}, K={K}, "
            f"class_weights={class_weights.tolist()}"
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_indices: torch.Tensor,
        current_epoch: int,
    ) -> torch.Tensor:
        """Compute CCR loss for a batch.

        Args:
            logits: Raw model outputs, shape [B, C].
            targets: Ground truth class indices, shape [B], dtype long.
            sample_indices: Global indices into the training dataset, shape [B].
            current_epoch: Current epoch number (0-indexed).

        Returns:
            Scalar loss tensor (ready for .backward()).

        Raises:
            RuntimeError: If any sample_index >= n_samples.
            RuntimeError: If logits shape doesn't match n_classes.
        """
        # ── Shape validation ──────────────────────────────────────────────────
        if logits.shape[1] != self.n_classes:
            raise RuntimeError(
                f"logits has {logits.shape[1]} classes but CCRLoss was initialized "
                f"with n_classes={self.n_classes}. Check model output dimension."
            )
        if sample_indices.max().item() >= self.n_samples:
            raise RuntimeError(
                f"sample_index {sample_indices.max().item()} >= n_samples "
                f"({self.n_samples}). Ensure sample_indices are global training "
                f"indices, not batch-local indices."
            )

        batch_size = logits.shape[0]

        # ── Softmax probabilities ─────────────────────────────────────────────
        probs = F.softmax(logits, dim=1)  # [B, C]

        # p_i = probability of the TRUE class for each sample
        p_i = probs[torch.arange(batch_size, device=logits.device), targets]  # [B]

        # ── Component 1: Focal penalty ────────────────────────────────────────
        focal_term = 1.0 - p_i  # [B]

        # ── Component 2: Variance gate ────────────────────────────────────────
        variance = self._compute_variance(sample_indices, current_epoch)  # [B]
        confidence_gate = (p_i > self.tau).float()  # I(p_i > tau), [B]
        variance_term = self.beta * variance * confidence_gate  # [B]

        # ── Component 3: Class weight ─────────────────────────────────────────
        gamma = self.class_weights[targets]  # [B]

        # ── Raw weights ───────────────────────────────────────────────────────
        raw_weights = focal_term + variance_term + gamma  # [B]

        # ── Batch normalization (epsilon prevents division by zero) ───────────
        weight_sum = raw_weights.sum() + 1e-8
        normalized_weights = (raw_weights / weight_sum) * batch_size  # [B]

        # ── Per-sample cross-entropy loss ─────────────────────────────────────
        per_sample_ce = F.cross_entropy(logits, targets, reduction="none")  # [B]

        # ── Final CCR loss ────────────────────────────────────────────────────
        loss = (normalized_weights * per_sample_ce).mean()

        return loss

    def update_history(
        self,
        probs: torch.Tensor,
        sample_indices: torch.Tensor,
        current_epoch: int,
    ) -> None:
        """Update confidence history tensor. Call AFTER optimizer step.

        Args:
            probs: Softmax probabilities, detached. Shape [B, C].
            sample_indices: Global sample indices. Shape [B].
            current_epoch: Current epoch number (0-indexed).

        Note:
            History tensor shape: [n_samples, K].
            Circular buffer: epoch t writes to column (t % K).
        """
        with torch.no_grad():
            col = current_epoch % self.K
            # Store max probability (confidence) for each sample
            max_probs = probs.detach().max(dim=1).values  # [B]
            self.history[sample_indices, col] = max_probs

    def _compute_variance(
        self,
        sample_indices: torch.Tensor,
        current_epoch: int,
    ) -> torch.Tensor:
        """Compute rolling variance for samples at given indices.

        Critical edge case: For epochs < K, variance is computed over
        available history only (not full K window). Uses the actual
        number of filled columns, not K.

        Args:
            sample_indices: Global sample indices, shape [B].
            current_epoch: Current epoch number (0-indexed).

        Returns:
            Variance tensor, shape [B]. Values are 0.0 for epoch 0.
        """
        n_filled = min(current_epoch + 1, self.K)

        if n_filled <= 1:
            return torch.zeros(len(sample_indices), device=sample_indices.device)

        # Identify which columns contain valid history
        cols = [(current_epoch - i) % self.K for i in range(n_filled)]
        history_slice = self.history[sample_indices][:, cols]  # [B, n_filled]

        return history_slice.var(dim=1)  # [B]

    def _compute_class_weights(self, class_counts: List[int]) -> torch.Tensor:
        """Compute normalized inverse class-frequency weights.

        gamma_c = (1 / count_c) / sum(1 / count_j for all j)

        Args:
            class_counts: List of sample counts per class.

        Returns:
            Normalized weights tensor, shape [n_classes].
            Weights sum to 1.0.
        """
        inv_counts = torch.tensor(
            [1.0 / c for c in class_counts], dtype=torch.float32
        )
        normalized = inv_counts / inv_counts.sum()
        return normalized


# ── Ablation variants ─────────────────────────────────────────────────────────

class CCRLossNoGate(CCRLoss):
    """CCR ablation: variance term active for ALL samples (no confidence gate).

    Removes the I(p_i > tau) indicator. Used to measure the contribution
    of the confidence gate to noise suppression.
    """

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_indices: torch.Tensor,
        current_epoch: int,
    ) -> torch.Tensor:
        """CCR forward without confidence gate (ablation A1)."""
        if logits.shape[1] != self.n_classes:
            raise RuntimeError(
                f"logits has {logits.shape[1]} classes but CCRLoss was initialized "
                f"with n_classes={self.n_classes}."
            )
        if sample_indices.max().item() >= self.n_samples:
            raise RuntimeError(
                f"sample_index {sample_indices.max().item()} >= n_samples ({self.n_samples})."
            )

        batch_size = logits.shape[0]
        probs = F.softmax(logits, dim=1)
        p_i = probs[torch.arange(batch_size, device=logits.device), targets]

        focal_term = 1.0 - p_i
        variance = self._compute_variance(sample_indices, current_epoch)
        # NO gate — variance applied to all samples regardless of confidence
        variance_term = self.beta * variance
        gamma = self.class_weights[targets]

        raw_weights = focal_term + variance_term + gamma
        weight_sum = raw_weights.sum() + 1e-8
        normalized_weights = (raw_weights / weight_sum) * batch_size
        per_sample_ce = F.cross_entropy(logits, targets, reduction="none")
        return (normalized_weights * per_sample_ce).mean()


class CCRLossNoVariance(CCRLoss):
    """CCR ablation: no variance term at all (focal + class weight only).

    Removes beta * Var_K(p_i) * I(p_i > tau) entirely. Used to measure
    the contribution of the rolling variance mechanism.
    """

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_indices: torch.Tensor,
        current_epoch: int,
    ) -> torch.Tensor:
        """CCR forward without variance term (ablation A2)."""
        if logits.shape[1] != self.n_classes:
            raise RuntimeError(
                f"logits has {logits.shape[1]} classes but CCRLoss was initialized "
                f"with n_classes={self.n_classes}."
            )
        if sample_indices.max().item() >= self.n_samples:
            raise RuntimeError(
                f"sample_index {sample_indices.max().item()} >= n_samples ({self.n_samples})."
            )

        batch_size = logits.shape[0]
        probs = F.softmax(logits, dim=1)
        p_i = probs[torch.arange(batch_size, device=logits.device), targets]

        focal_term = 1.0 - p_i
        gamma = self.class_weights[targets]

        # NO variance term
        raw_weights = focal_term + gamma
        weight_sum = raw_weights.sum() + 1e-8
        normalized_weights = (raw_weights / weight_sum) * batch_size
        per_sample_ce = F.cross_entropy(logits, targets, reduction="none")
        return (normalized_weights * per_sample_ce).mean()


class CCRLossNoNormalization(CCRLoss):
    """CCR ablation: no batch-level weight normalization.

    Removes the w_hat = (w / sum(w)) * B step. Used to demonstrate
    the training stability contribution of the normalization patch.
    """

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_indices: torch.Tensor,
        current_epoch: int,
    ) -> torch.Tensor:
        """CCR forward without batch normalization (ablation A3)."""
        if logits.shape[1] != self.n_classes:
            raise RuntimeError(
                f"logits has {logits.shape[1]} classes but CCRLoss was initialized "
                f"with n_classes={self.n_classes}."
            )
        if sample_indices.max().item() >= self.n_samples:
            raise RuntimeError(
                f"sample_index {sample_indices.max().item()} >= n_samples ({self.n_samples})."
            )

        batch_size = logits.shape[0]
        probs = F.softmax(logits, dim=1)
        p_i = probs[torch.arange(batch_size, device=logits.device), targets]

        focal_term = 1.0 - p_i
        variance = self._compute_variance(sample_indices, current_epoch)
        confidence_gate = (p_i > self.tau).float()
        variance_term = self.beta * variance * confidence_gate
        gamma = self.class_weights[targets]

        # Raw weights used directly — NO normalization
        raw_weights = focal_term + variance_term + gamma
        per_sample_ce = F.cross_entropy(logits, targets, reduction="none")
        return (raw_weights * per_sample_ce).mean()


def get_ccr_loss(
    variant: str,
    n_samples: int,
    n_classes: int,
    class_counts: list,
    device: torch.device,
) -> CCRLoss:
    """Factory for CCR loss variants including ablations.

    Args:
        variant: One of 'ccr', 'ccr_no_gate', 'ccr_no_variance', 'ccr_no_norm'.
        n_samples: Total training samples.
        n_classes: Number of classes.
        class_counts: Per-class sample counts.
        device: Torch device.

    Returns:
        Instantiated CCRLoss variant.

    Raises:
        ValueError: If variant is not recognized.
    """
    kwargs = dict(
        n_samples=n_samples,
        n_classes=n_classes,
        class_counts=class_counts,
        device=device,
    )
    registry = {
        "ccr":             CCRLoss,
        "ccr_no_gate":     CCRLossNoGate,
        "ccr_no_variance": CCRLossNoVariance,
        "ccr_no_norm":     CCRLossNoNormalization,
    }
    if variant not in registry:
        raise ValueError(
            f"Unknown CCR variant '{variant}'. "
            f"Valid options: {list(registry.keys())}."
        )
    return registry[variant](**kwargs).to(device)
