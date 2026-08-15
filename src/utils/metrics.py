"""All metric computation for CCR-Tabular evaluation.

Computes:
  1. Discrimination & classification metrics:
     - Accuracy
     - Macro F1 (unweighted per-class average)
     - Minority Recall (for binary) / Macro Recall (for multiclass)
     - AUC-ROC (One-vs-Rest for multiclass)
     - AUC-PR (Average Precision)
  2. Probability calibration metrics (Section N):
     - Expected Calibration Error (ECE, 10 bins)
     - Brier Score
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
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
    n_bins: int = 10,
) -> Dict[str, float]:
    """Compute comprehensive classification and calibration metrics.

    Args:
        y_true: Ground truth integer class labels [N].
        y_pred: Hard predicted class indices [N].
        y_prob: Predicted probability array [N] for binary (positive class) or [N, C] for multiclass.
        minority_class: Class index for minority recall (default 1).
        n_bins: Number of bins for ECE calculation (default 10).

    Returns:
        Dict with keys: accuracy, macro_f1, minority_recall, auc_roc, auc_pr, ece, brier_score.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    # ── Input validation ──────────────────────────────────────────────────────
    if np.any(np.isnan(y_prob)) or np.any(np.isinf(y_prob)):
        raise ValueError("y_prob contains NaN or Inf values.")

    if np.any(y_prob < 0.0) or np.any(y_prob > 1.0):
        raise ValueError(
            f"y_prob must contain values in [0, 1]. "
            f"Got min={y_prob.min():.4f}, max={y_prob.max():.4f}. "
            f"Pass softmax probabilities, not raw logits."
        )

    unique_classes = np.unique(y_true)
    n_classes = len(unique_classes)
    if n_classes < 2:
        raise ValueError(
            f"y_true must have at least 2 unique classes for binary evaluation. "
            f"Got classes: {unique_classes.tolist()}. "
            f"Check that the test fold contains both majority and minority samples."
        )

    # ── Classification Metrics ────────────────────────────────────────────────
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    if n_classes == 2:
        # Binary minority recall
        try:
            minority_recall = float(
                recall_score(y_true, y_pred, pos_label=minority_class, zero_division=0)
            )
        except Exception:
            minority_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    else:
        # Multiclass macro recall
        minority_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # ── Discrimination Metrics (AUC-ROC, AUC-PR) ──────────────────────────────
    auc_roc = _compute_auc_roc(y_true, y_prob, n_classes)
    auc_pr = _compute_auc_pr(y_true, y_prob, n_classes)

    # ── Calibration Metrics (ECE, Brier Score) ────────────────────────────────
    ece = compute_ece(y_true, y_prob, n_bins=n_bins)
    brier = compute_brier_score(y_true, y_prob)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "minority_recall": minority_recall,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "ece": ece,
        "brier_score": brier,
    }


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n_samples = len(y_true)
    if n_samples == 0:
        return 0.0

    if y_prob.ndim == 1 or (y_prob.ndim == 2 and y_prob.shape[1] == 1):
        # Binary: positive class probability
        probs = y_prob.flatten()
        preds = (probs >= 0.5).astype(np.int64)
        confidences = np.maximum(probs, 1.0 - probs)
        accuracies = (preds == y_true).astype(np.float64)
    else:
        # Multiclass: max predicted probability
        preds = np.argmax(y_prob, axis=1)
        confidences = np.max(y_prob, axis=1)
        accuracies = (preds == y_true).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier Score."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    if y_prob.ndim == 1:
        return float(brier_score_loss(y_true, y_prob))
    elif y_prob.ndim == 2 and y_prob.shape[1] == 2:
        return float(brier_score_loss(y_true, y_prob[:, 1]))
    else:
        n_classes = y_prob.shape[1]
        y_one_hot = np.eye(n_classes)[y_true]
        return float(np.mean(np.sum((y_prob - y_one_hot) ** 2, axis=1)))


def _compute_auc_roc(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> float:
    """Compute AUC-ROC with graceful fallback on single-class batches."""
    try:
        if n_classes == 2:
            prob_pos = y_prob[:, 1] if (y_prob.ndim == 2 and y_prob.shape[1] == 2) else y_prob
            return float(roc_auc_score(y_true, prob_pos))
        else:
            return float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError as exc:
        logger.warning(f"AUC-ROC calculation failed: {exc}. Returning NaN.")
        return float("nan")


def _compute_auc_pr(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> float:
    """Compute AUC-PR (Average Precision)."""
    try:
        if n_classes == 2:
            prob_pos = y_prob[:, 1] if (y_prob.ndim == 2 and y_prob.shape[1] == 2) else y_prob
            return float(average_precision_score(y_true, prob_pos))
        else:
            y_one_hot = np.eye(n_classes)[y_true]
            return float(average_precision_score(y_one_hot, y_prob, average="macro"))
    except ValueError as exc:
        logger.warning(f"AUC-PR calculation failed: {exc}. Returning NaN.")
        return float("nan")
