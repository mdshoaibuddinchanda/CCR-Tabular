"""Robust loss functions and normalized loss wrappers for noisy & imbalanced tabular learning.

Implements baseline robust losses requested by reviewers:
  1. Generalized Cross Entropy (GCE) — Zhang & Sabuncu, NeurIPS 2018
  2. Symmetric Cross Entropy (SCE) — Wang et al., ICCV 2019
  3. Early-Learning Regularization (ELR) — Liu et al., NeurIPS 2020
  4. NormalizedLossWrapper and normalized variants:
     - Norm-WCE (Normalized Class-Weighted Cross Entropy)
     - Norm-Focal (Normalized Focal Loss)
     - Norm-GCE (Normalized Generalized Cross Entropy)
     - Norm-SCE (Normalized Symmetric Cross Entropy)
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loss.ccr_loss import compute_batch_normalized_weights

logger = logging.getLogger(__name__)


# ── 1. Generalized Cross Entropy (GCE) ────────────────────────────────────────

class GeneralizedCrossEntropyLoss(nn.Module):
    """Generalized Cross Entropy (GCE) loss for noisy labels.

    L_GCE = (1 - p_t^q) / q, where p_t is the model probability for the true class.
    Interpolates between CE (as q -> 0) and MAE (q = 1). Default q = 0.7.

    Reference: Zhang & Sabuncu, "Generalized Cross Entropy Loss for Training Deep
    Neural Networks with Noisy Labels", NeurIPS 2018.
    """

    def __init__(self, q: float = 0.7, eps: float = 1e-8) -> None:
        super().__init__()
        if not (0.0 < q <= 1.0):
            raise ValueError(f"q must be in (0, 1], got {q}.")
        self.q = q
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_indices: Optional[torch.Tensor] = None,
        current_epoch: int = 0,
    ) -> torch.Tensor:
        """Compute GCE loss."""
        probs = F.softmax(logits, dim=1)
        batch_size = logits.shape[0]
        if batch_size == 0:
            return logits.sum() * 0.0

        p_t = probs[torch.arange(batch_size, device=logits.device), targets]
        p_t = torch.clamp(p_t, min=self.eps, max=1.0)
        loss = (1.0 - torch.pow(p_t, self.q)) / self.q
        return loss.mean()


# ── 2. Symmetric Cross Entropy (SCE) ──────────────────────────────────────────

class SymmetricCrossEntropyLoss(nn.Module):
    """Symmetric Cross Entropy (SCE) loss combining CE and Reverse CE.

    L_SCE = alpha * CE(p, y) + beta * RCE(p, y)
    where RCE(p, y) = - sum_{k} p_k * log(y_k + eps) with one-hot y.

    Reference: Wang et al., "Symmetric Cross Entropy for Robust Learning with
    Noisy Labels", ICCV 2019.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 1.0,
        n_classes: int = 2,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.n_classes = n_classes
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_indices: Optional[torch.Tensor] = None,
        current_epoch: int = 0,
    ) -> torch.Tensor:
        """Compute SCE loss."""
        batch_size, n_classes = logits.shape
        if batch_size == 0:
            return logits.sum() * 0.0

        probs = F.softmax(logits, dim=1)
        probs = torch.clamp(probs, min=self.eps, max=1.0)

        # Standard Cross Entropy
        ce = F.cross_entropy(logits, targets, reduction="none")

        # Reverse Cross Entropy
        y_one_hot = F.one_hot(targets, num_classes=n_classes).float()
        y_one_hot = torch.clamp(y_one_hot, min=self.eps, max=1.0)
        rce = -torch.sum(probs * torch.log(y_one_hot), dim=1)

        loss = self.alpha * ce + self.beta * rce
        return loss.mean()


# ── 3. Early-Learning Regularization (ELR) ────────────────────────────────────

class EarlyLearningRegularizationLoss(nn.Module):
    """Early-Learning Regularization (ELR) loss.

    Prevents memorization of noisy labels by penalizing deviations from temporal
    moving average targets:
      L_ELR = CE(logits, targets) + (lambda_elr / B) * sum_i log(1 - <p_i, e_i> + eps)
    where e_i is the momentum target for sample i.

    Reference: Liu et al., "Early-Learning Regularization Prevents Memorization
    of Noisy Labels", NeurIPS 2020.
    """

    def __init__(
        self,
        n_samples: int,
        n_classes: int = 2,
        lambda_elr: float = 3.0,
        beta_momentum: float = 0.7,
        eps: float = 1e-7,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.lambda_elr = lambda_elr
        self.beta_momentum = beta_momentum
        self.eps = eps
        self.device = device or torch.device("cpu")

        # Temporal targets buffer [N, C]
        self.register_buffer(
            "target_history",
            torch.zeros((n_samples, n_classes), dtype=torch.float32),
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_indices: Optional[torch.Tensor] = None,
        current_epoch: int = 0,
    ) -> torch.Tensor:
        """Compute ELR loss and update momentum targets."""
        batch_size = logits.shape[0]
        if batch_size == 0:
            return logits.sum() * 0.0

        ce_loss = F.cross_entropy(logits, targets)
        probs = F.softmax(logits, dim=1)

        if sample_indices is None:
            return ce_loss

        # Get historical predictions
        with torch.no_grad():
            hist_preds = self.target_history[sample_indices].to(logits.device)
            # Update temporal buffer
            new_hist = self.beta_momentum * hist_preds + (1.0 - self.beta_momentum) * probs.detach()
            self.target_history[sample_indices] = new_hist.to(dtype=self.target_history.dtype, device=self.target_history.device)

        # Regularization term: log(1 - <p, e>)
        reg = torch.sum(probs * hist_preds, dim=1)
        reg = torch.clamp(reg, min=0.0, max=1.0 - self.eps)
        reg_loss = -torch.log(1.0 - reg + self.eps).mean()

        return ce_loss + self.lambda_elr * reg_loss


# ── 4. Normalized Loss Framework ──────────────────────────────────────────────

class NormalizedLossWrapper(nn.Module):
    """Wraps any base loss function with per-batch detached weight normalization.

    Given a base loss that assigns per-sample scalar weights or losses, this wrapper:
      1. Detaches the sample weights w_i.
      2. Normalizes: w_hat_i = (w_i / (sum w_j + eps)) * B.
      3. Computes the normalized loss: (1/B) sum_i w_hat_i * CE(logits_i, y_i).

    This enables testing the normalization mechanism across diverse loss formulations
    (Normalized WCE, Normalized Focal, Normalized GCE, Normalized SCE).
    """

    def __init__(
        self,
        base_weight_fn: nn.Module,
        eps: float = 1e-8,
        name: str = "norm_loss",
    ) -> None:
        super().__init__()
        self.base_weight_fn = base_weight_fn
        self.eps = eps
        self.name = name
        self.last_telemetry: Dict[str, float] = {}

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_indices: Optional[torch.Tensor] = None,
        current_epoch: int = 0,
    ) -> torch.Tensor:
        """Compute normalized loss."""
        batch_size = logits.shape[0]
        if batch_size == 0:
            return logits.sum() * 0.0

        with torch.no_grad():
            # Get raw sample weights from base function
            raw_weights = self.base_weight_fn(logits.detach(), targets)
            if raw_weights.ndim > 1:
                raw_weights = raw_weights.squeeze()

            # Ensure non-negative
            raw_weights = torch.clamp(raw_weights, min=0.0)

            # Apply batch normalization
            norm_weights, telemetry = compute_batch_normalized_weights(
                raw_weights, eps=self.eps
            )
            self.last_telemetry = telemetry

        per_sample_ce = F.cross_entropy(logits, targets, reduction="none")
        return (norm_weights * per_sample_ce).mean()


# ── 5. Standard and Normalized Baseline Losses ─────────────────────────────────

class FocalLoss(nn.Module):
    """Standard Focal Loss: (1 - p_t)^gamma * log(p_t)."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_indices: Optional[torch.Tensor] = None,
        current_epoch: int = 0,
    ) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        batch_size = logits.shape[0]
        if batch_size == 0:
            return logits.sum() * 0.0

        p_t = probs[torch.arange(batch_size, device=logits.device), targets]
        focal_weight = self.alpha * (1.0 - p_t) ** self.gamma
        loss = -focal_weight * torch.log(torch.clamp(p_t, min=self.eps))
        return loss.mean()


class NormalizedFocalLoss(nn.Module):
    """Normalized Focal Loss: focal weights normalized per batch."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.last_telemetry: Dict[str, float] = {}

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_indices: Optional[torch.Tensor] = None,
        current_epoch: int = 0,
    ) -> torch.Tensor:
        batch_size = logits.shape[0]
        if batch_size == 0:
            return logits.sum() * 0.0

        with torch.no_grad():
            probs = F.softmax(logits.detach(), dim=1)
            p_t = probs[torch.arange(batch_size, device=logits.device), targets]
            raw_weights = self.alpha * (1.0 - p_t) ** self.gamma
            norm_weights, telemetry = compute_batch_normalized_weights(
                raw_weights, eps=self.eps
            )
            self.last_telemetry = telemetry

        per_sample_ce = F.cross_entropy(logits, targets, reduction="none")
        return (norm_weights * per_sample_ce).mean()


class NormalizedWeightedCELoss(nn.Module):
    """Normalized Class-Weighted Cross Entropy."""

    def __init__(
        self,
        class_counts: List[int],
        eps: float = 1e-8,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        inv_counts = torch.tensor([1.0 / c for c in class_counts], dtype=torch.float32)
        class_weights = inv_counts / inv_counts.sum()
        self.register_buffer("class_weights", class_weights.to(device or torch.device("cpu")))
        self.eps = eps
        self.last_telemetry: Dict[str, float] = {}

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_indices: Optional[torch.Tensor] = None,
        current_epoch: int = 0,
    ) -> torch.Tensor:
        batch_size = logits.shape[0]
        if batch_size == 0:
            return logits.sum() * 0.0

        with torch.no_grad():
            raw_weights = self.class_weights[targets]
            norm_weights, telemetry = compute_batch_normalized_weights(
                raw_weights, eps=self.eps
            )
            self.last_telemetry = telemetry

        per_sample_ce = F.cross_entropy(logits, targets, reduction="none")
        return (norm_weights * per_sample_ce).mean()


class NormalizedGCELoss(nn.Module):
    """Normalized Generalized Cross Entropy: (1 - p_t^q)/q normalized per batch."""

    def __init__(self, q: float = 0.7, eps: float = 1e-8) -> None:
        super().__init__()
        self.q = q
        self.eps = eps
        self.last_telemetry: Dict[str, float] = {}

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_indices: Optional[torch.Tensor] = None,
        current_epoch: int = 0,
    ) -> torch.Tensor:
        batch_size = logits.shape[0]
        if batch_size == 0:
            return logits.sum() * 0.0

        with torch.no_grad():
            probs = F.softmax(logits.detach(), dim=1)
            p_t = probs[torch.arange(batch_size, device=logits.device), targets]
            p_t = torch.clamp(p_t, min=self.eps, max=1.0)
            raw_weights = (1.0 - torch.pow(p_t, self.q)) / self.q
            norm_weights, telemetry = compute_batch_normalized_weights(
                raw_weights, eps=self.eps
            )
            self.last_telemetry = telemetry

        per_sample_ce = F.cross_entropy(logits, targets, reduction="none")
        return (norm_weights * per_sample_ce).mean()


class NormalizedSCELoss(nn.Module):
    """Normalized Symmetric Cross Entropy."""

    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 1.0,
        n_classes: int = 2,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.n_classes = n_classes
        self.eps = eps
        self.last_telemetry: Dict[str, float] = {}

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_indices: Optional[torch.Tensor] = None,
        current_epoch: int = 0,
    ) -> torch.Tensor:
        batch_size, n_classes = logits.shape
        if batch_size == 0:
            return logits.sum() * 0.0

        with torch.no_grad():
            probs = F.softmax(logits.detach(), dim=1)
            probs = torch.clamp(probs, min=self.eps, max=1.0)
            ce_weights = F.cross_entropy(logits.detach(), targets, reduction="none")
            y_one_hot = F.one_hot(targets, num_classes=n_classes).float()
            y_one_hot = torch.clamp(y_one_hot, min=self.eps, max=1.0)
            rce_weights = -torch.sum(probs * torch.log(y_one_hot), dim=1)
            raw_weights = self.alpha * ce_weights + self.beta * rce_weights
            norm_weights, telemetry = compute_batch_normalized_weights(
                raw_weights, eps=self.eps
            )
            self.last_telemetry = telemetry

        per_sample_ce = F.cross_entropy(logits, targets, reduction="none")
        return (norm_weights * per_sample_ce).mean()


class DynamicCELoss(nn.Module):
    """Plain Dynamic Weighted CE: w_i = (1 - p_i) without batch normalization."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.last_telemetry: Dict[str, float] = {}

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_indices: Optional[torch.Tensor] = None,
        current_epoch: int = 0,
    ) -> torch.Tensor:
        batch_size = logits.shape[0]
        if batch_size == 0:
            return logits.sum() * 0.0

        with torch.no_grad():
            probs = F.softmax(logits.detach(), dim=1)
            p_t = probs[torch.arange(batch_size, device=logits.device), targets]
            raw_weights = 1.0 - p_t
            self.last_telemetry = {"S_over_B": raw_weights.mean().item()}

        per_sample_ce = F.cross_entropy(logits, targets, reduction="none")
        return (raw_weights * per_sample_ce).mean()


class NormalizedDynamicCELoss(nn.Module):
    """Normalized Dynamic Weighted CE: w_hat_i = ((1 - p_i) / sum(1 - p_j)) * B."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.last_telemetry: Dict[str, float] = {}

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_indices: Optional[torch.Tensor] = None,
        current_epoch: int = 0,
    ) -> torch.Tensor:
        batch_size = logits.shape[0]
        if batch_size == 0:
            return logits.sum() * 0.0

        with torch.no_grad():
            probs = F.softmax(logits.detach(), dim=1)
            p_t = probs[torch.arange(batch_size, device=logits.device), targets]
            raw_weights = 1.0 - p_t
            norm_weights, telemetry = compute_batch_normalized_weights(
                raw_weights, eps=self.eps
            )
            self.last_telemetry = telemetry

        per_sample_ce = F.cross_entropy(logits, targets, reduction="none")
        return (norm_weights * per_sample_ce).mean()


# ── Loss Function Registry & Factory ──────────────────────────────────────────

LOSS_REGISTRY = {
    # Unnormalized losses
    "ce": "CrossEntropyLoss",
    "wce": "ClassWeightedCrossEntropy",
    "focal": "FocalLoss",
    "gce": "GeneralizedCrossEntropyLoss",
    "sce": "SymmetricCrossEntropyLoss",
    "elr": "EarlyLearningRegularizationLoss",
    "ccr_no_norm": "CCRLossNoNormalization",
    "ccr_no_gate": "CCRLossNoGate",
    "ccr_no_variance": "CCRLossNoVariance",
    "dynamic_ce": "DynamicCELoss",
    # Normalized losses
    "norm_wce": "NormalizedWeightedCELoss",
    "norm_focal": "NormalizedFocalLoss",
    "norm_gce": "NormalizedGCELoss",
    "norm_sce": "NormalizedSCELoss",
    "norm_dynamic_ce": "NormalizedDynamicCELoss",
    "ccr": "CCRLoss",
}


def build_loss(
    loss_name: str,
    n_samples: int,
    n_classes: int = 2,
    class_counts: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
    tau: float = 0.3,
    beta: float = 0.5,
    K: int = 5,
    q_gce: float = 0.7,
    alpha_sce: float = 0.1,
    beta_sce: float = 1.0,
    lambda_elr: float = 3.0,
) -> nn.Module:
    """Build a loss function by name.

    Args:
        loss_name: Key from LOSS_REGISTRY (e.g. 'ce', 'wce', 'focal', 'gce', 'sce', 'elr', 'ccr', etc.)
        n_samples: Number of training samples.
        n_classes: Number of target classes.
        class_counts: Sample counts per class.
        device: Device tensor placement.
        ... other loss-specific hyperparameters.

    Returns:
        Configured nn.Module loss instance.
    """
    from src.loss.ccr_loss import (
        CCRLoss,
        CCRLossNoGate,
        CCRLossNoNormalization,
        CCRLossNoVariance,
    )

    loss_name = loss_name.lower().strip()
    dev = device or torch.device("cpu")

    if loss_name in ("ce", "cross_entropy"):
        return nn.CrossEntropyLoss().to(dev)

    elif loss_name in ("wce", "weighted_ce"):
        if class_counts is None:
            weights = torch.ones(n_classes, dtype=torch.float32)
        else:
            inv = torch.tensor([1.0 / c for c in class_counts], dtype=torch.float32)
            weights = inv / inv.sum()
        return nn.CrossEntropyLoss(weight=weights.to(dev)).to(dev)

    elif loss_name == "focal":
        return FocalLoss(alpha=0.25, gamma=2.0).to(dev)

    elif loss_name == "gce":
        return GeneralizedCrossEntropyLoss(q=q_gce).to(dev)

    elif loss_name == "sce":
        return SymmetricCrossEntropyLoss(alpha=alpha_sce, beta=beta_sce, n_classes=n_classes).to(dev)

    elif loss_name == "elr":
        return EarlyLearningRegularizationLoss(
            n_samples=n_samples, n_classes=n_classes, lambda_elr=lambda_elr, device=dev
        ).to(dev)

    elif loss_name == "norm_wce":
        counts = class_counts or [n_samples // n_classes] * n_classes
        return NormalizedWeightedCELoss(class_counts=counts, device=dev).to(dev)

    elif loss_name == "norm_focal":
        return NormalizedFocalLoss(alpha=0.25, gamma=2.0).to(dev)

    elif loss_name == "norm_gce":
        return NormalizedGCELoss(q=q_gce).to(dev)

    elif loss_name == "norm_sce":
        return NormalizedSCELoss(alpha=alpha_sce, beta=beta_sce, n_classes=n_classes).to(dev)

    elif loss_name == "dynamic_ce":
        return DynamicCELoss().to(dev)

    elif loss_name == "norm_dynamic_ce":
        return NormalizedDynamicCELoss().to(dev)

    elif loss_name in ("ccr", "mlp_ccr"):
        return CCRLoss(
            n_samples=n_samples,
            n_classes=n_classes,
            class_counts=class_counts,
            tau=tau,
            beta=beta,
            K=K,
            device=dev,
        ).to(dev)

    elif loss_name == "ccr_no_norm":
        return CCRLossNoNormalization(
            n_samples=n_samples,
            n_classes=n_classes,
            class_counts=class_counts,
            tau=tau,
            beta=beta,
            K=K,
            device=dev,
        ).to(dev)

    elif loss_name == "ccr_no_gate":
        return CCRLossNoGate(
            n_samples=n_samples,
            n_classes=n_classes,
            class_counts=class_counts,
            tau=tau,
            beta=beta,
            K=K,
            device=dev,
        ).to(dev)

    elif loss_name == "ccr_no_variance":
        return CCRLossNoVariance(
            n_samples=n_samples,
            n_classes=n_classes,
            class_counts=class_counts,
            tau=tau,
            beta=beta,
            K=K,
            device=dev,
        ).to(dev)

    else:
        raise ValueError(
            f"Unknown loss_name '{loss_name}'. Valid options: {list(LOSS_REGISTRY.keys())}."
        )
