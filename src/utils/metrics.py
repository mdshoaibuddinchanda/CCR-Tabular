"""All metric computation for CCR-Tabular evaluation.

Imported by evaluate.py only. All metrics are computed here in one place
to ensure consistency across all models and baselines.
"""

import logging
import warnings
from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    minority_class: int = 1,
) -> Dict[str, float]:
    """Compute all evaluation metrics for binary classification.

    Args:
        y_true: Ground truth labels, shape [N].
        y_pred: Hard predictions (argmax), shape [N].
        y_prob: Predicted probabilities for the positive (minority) class, shape [N].
        minority_class: Which class index is the minority (default 1).

    Returns:
        Dict with keys: accuracy, macro_f1, minority_recall, auc_roc, auc_pr.

    Raises:
        ValueError: If y_prob contains values outside [0, 1].
        ValueError: If y_true has fewer than 2 unique classes.

    Note:
        - AUC metrics MUST use y_prob (probabilities), not y_pred.
        - macro_f1 uses average='macro', NOT 'binary'.
        - minority_recall is recall_score(..., pos_label=minority_class).
    """
    # ── Input validation ──────────────────────────────────────────────────────
    if np.any(y_prob < 0.0) or np.any(y_prob > 1.0):
        raise ValueError(
            f"y_prob must contain values in [0, 1]. "
            f"Got min={y_prob.min():.4f}, max={y_prob.max():.4f}. "
            f"Pass softmax probabilities, not raw logits."
        )

    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        raise ValueError(
            f"y_true must have at least 2 unique classes for binary evaluation. "
            f"Got classes: {unique_classes.tolist()}. "
            f"Check that the test fold contains both majority and minority samples."
        )

    # ── Core metrics ──────────────────────────────────────────────────────────
    accuracy = float(accuracy_score(y_true, y_pred))

    # macro_f1: unweighted mean of per-class F1 — NOT binary F1
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # minority_recall: recall specifically for the minority class
    minority_recall = float(
        recall_score(y_true, y_pred, pos_label=minority_class, zero_division=0)
    )

    # ── AUC metrics (use probabilities, not hard predictions) ─────────────────
    auc_roc = _safe_auc_roc(y_true, y_prob)
    auc_pr = _safe_auc_pr(y_true, y_prob)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "minority_recall": minority_recall,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
    }


def _safe_auc_roc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute AUC-ROC, returning NaN if only one class is present.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities for the positive class.

    Returns:
        AUC-ROC score, or float('nan') if computation is not possible.
    """
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError as exc:
        logger.warning(
            f"AUC-ROC could not be computed (likely only one class in y_true): {exc}. "
            f"Returning NaN."
        )
        return float("nan")


def _safe_auc_pr(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute AUC-PR (average precision), returning NaN on failure.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities for the positive class.

    Returns:
        AUC-PR score, or float('nan') if computation is not possible.
    """
    try:
        return float(average_precision_score(y_true, y_prob))
    except ValueError as exc:
        logger.warning(
            f"AUC-PR could not be computed: {exc}. Returning NaN."
        )
        return float("nan")
