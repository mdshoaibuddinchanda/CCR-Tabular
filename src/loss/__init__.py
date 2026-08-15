"""Loss functions for CCR-Tabular."""

from src.loss.ccr_loss import (
    CCRLoss,
    CCRLossNoGate,
    CCRLossNoNormalization,
    CCRLossNoVariance,
    compute_batch_normalized_weights,
    get_ccr_loss,
)
from src.loss.robust_losses import (
    EarlyLearningRegularizationLoss,
    FocalLoss,
    GeneralizedCrossEntropyLoss,
    NormalizedFocalLoss,
    NormalizedGCELoss,
    NormalizedLossWrapper,
    NormalizedSCELoss,
    NormalizedWeightedCELoss,
    SymmetricCrossEntropyLoss,
    build_loss,
    LOSS_REGISTRY,
)

__all__ = [
    "CCRLoss",
    "CCRLossNoGate",
    "CCRLossNoVariance",
    "CCRLossNoNormalization",
    "compute_batch_normalized_weights",
    "get_ccr_loss",
    "GeneralizedCrossEntropyLoss",
    "SymmetricCrossEntropyLoss",
    "EarlyLearningRegularizationLoss",
    "FocalLoss",
    "NormalizedFocalLoss",
    "NormalizedWeightedCELoss",
    "NormalizedGCELoss",
    "NormalizedSCELoss",
    "NormalizedLossWrapper",
    "build_loss",
    "LOSS_REGISTRY",
]
