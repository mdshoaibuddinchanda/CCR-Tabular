"""Tests for metric computation functions.

Verifies correctness of all evaluation metrics, especially that:
- macro_f1 != binary_f1 on imbalanced data
- AUC uses probabilities, not hard predictions
- minority_recall uses pos_label=1
"""

import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.utils.metrics import compute_all_metrics


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_imbalanced_data(n_majority: int = 90, n_minority: int = 10, seed: int = 42):
    """Create imbalanced ground truth and predictions."""
    rng = np.random.default_rng(seed)
    y_true = np.array([0] * n_majority + [1] * n_minority)

    # Predictions: mostly correct for majority, some errors on minority
    y_pred = y_true.copy()
    # Flip 5 minority predictions to 0 (false negatives)
    minority_idx = np.where(y_true == 1)[0]
    y_pred[minority_idx[:5]] = 0

    # Probabilities: high for majority, lower for minority
    y_prob = np.where(y_true == 0, 0.8, 0.6)
    y_prob[minority_idx[:5]] = 0.3  # low prob for misclassified minority

    return y_true, y_pred, y_prob


# ── Test 1: macro_f1 != binary_f1 on imbalanced data ─────────────────────────

def test_macro_f1_not_equal_binary_f1_on_imbalanced_data():
    """macro_f1 must differ from binary_f1 on imbalanced data.

    This is the core validation that we're using the correct averaging strategy.
    """
    y_true, y_pred, y_prob = make_imbalanced_data()

    metrics = compute_all_metrics(y_true, y_pred, y_prob)
    macro_f1 = metrics["macro_f1"]

    # Compute binary F1 directly for comparison
    binary_f1 = float(f1_score(y_true, y_pred, average="binary", pos_label=1))

    assert macro_f1 != binary_f1, (
        f"macro_f1 ({macro_f1:.4f}) should NOT equal binary_f1 ({binary_f1:.4f}) "
        f"on imbalanced data. Check that average='macro' is used, not 'binary'."
    )

    # Also verify macro_f1 is the mean of per-class F1 scores
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    expected_macro = float(per_class_f1.mean())
    assert abs(macro_f1 - expected_macro) < 1e-6, (
        f"macro_f1 ({macro_f1:.6f}) should equal mean of per-class F1 "
        f"({expected_macro:.6f})."
    )


# ── Test 2: AUC uses probabilities, not hard predictions ─────────────────────

def test_auc_uses_probabilities_not_predictions():
    """AUC-ROC must differ between probability inputs and one-hot inputs."""
    y_true, y_pred, y_prob = make_imbalanced_data()

    # Metrics with real probabilities
    metrics_prob = compute_all_metrics(y_true, y_pred, y_prob)

    # Metrics with hard predictions cast to float (0.0 or 1.0)
    y_pred_as_prob = y_pred.astype(float)
    metrics_hard = compute_all_metrics(y_true, y_pred, y_pred_as_prob)

    assert metrics_prob["auc_roc"] != metrics_hard["auc_roc"], (
        f"AUC-ROC should differ between probability inputs ({metrics_prob['auc_roc']:.4f}) "
        f"and hard prediction inputs ({metrics_hard['auc_roc']:.4f}). "
        f"Ensure AUC uses y_prob, not y_pred."
    )


# ── Test 3: minority_recall uses pos_label=1 ─────────────────────────────────

def test_minority_recall_correct_class():
    """minority_recall must specifically measure recall for class 1 (minority)."""
    # Perfect recall for majority (class 0), zero recall for minority (class 1)
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # predict all as majority
    y_prob = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.3, 0.3, 0.3, 0.3, 0.3])

    metrics = compute_all_metrics(y_true, y_pred, y_prob)

    assert metrics["minority_recall"] == 0.0, (
        f"minority_recall should be 0.0 when all minority samples are misclassified. "
        f"Got {metrics['minority_recall']:.4f}. "
        f"Check that pos_label=1 is used in recall_score."
    )

    # Now predict all as minority — minority recall should be 1.0
    y_pred_all_minority = np.ones(10, dtype=int)
    metrics2 = compute_all_metrics(y_true, y_pred_all_minority, y_prob)
    assert metrics2["minority_recall"] == 1.0, (
        f"minority_recall should be 1.0 when all minority samples are correctly predicted. "
        f"Got {metrics2['minority_recall']:.4f}."
    )


# ── Test 4: Invalid probability raises ValueError ─────────────────────────────

def test_invalid_prob_raises():
    """y_prob values > 1.0 or < 0.0 must raise ValueError."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])

    # Values > 1.0
    y_prob_bad = np.array([0.8, 0.7, 1.5, 0.9])
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        compute_all_metrics(y_true, y_pred, y_prob_bad)

    # Values < 0.0
    y_prob_neg = np.array([-0.1, 0.7, 0.9, 0.8])
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        compute_all_metrics(y_true, y_pred, y_prob_neg)


# ── Test 5: Single class in y_true raises ValueError ─────────────────────────

def test_single_class_raises():
    """y_true with only one unique class must raise ValueError."""
    y_true = np.zeros(10, dtype=int)  # only class 0
    y_pred = np.zeros(10, dtype=int)
    y_prob = np.zeros(10, dtype=float)

    with pytest.raises(ValueError, match="2 unique classes"):
        compute_all_metrics(y_true, y_pred, y_prob)


# ── Test 6: All metrics are in valid ranges ───────────────────────────────────

def test_all_metrics_valid_ranges():
    """All returned metrics must be in their expected ranges."""
    y_true, y_pred, y_prob = make_imbalanced_data()
    metrics = compute_all_metrics(y_true, y_pred, y_prob)

    assert 0.0 <= metrics["accuracy"] <= 1.0, f"accuracy out of range: {metrics['accuracy']}"
    assert 0.0 <= metrics["macro_f1"] <= 1.0, f"macro_f1 out of range: {metrics['macro_f1']}"
    assert 0.0 <= metrics["minority_recall"] <= 1.0, (
        f"minority_recall out of range: {metrics['minority_recall']}"
    )
    # AUC can be NaN if computation fails, but if not NaN must be in [0, 1]
    if not np.isnan(metrics["auc_roc"]):
        assert 0.0 <= metrics["auc_roc"] <= 1.0, (
            f"auc_roc out of range: {metrics['auc_roc']}"
        )
    if not np.isnan(metrics["auc_pr"]):
        assert 0.0 <= metrics["auc_pr"] <= 1.0, (
            f"auc_pr out of range: {metrics['auc_pr']}"
        )
