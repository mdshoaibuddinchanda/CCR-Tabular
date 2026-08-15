"""Unit tests for calibration metrics (ECE and Brier score)."""

import numpy as np
import pytest

from src.utils.metrics import compute_all_metrics, compute_brier_score, compute_ece


def test_perfect_calibration():
    """Perfect predictions should yield 0.0 ECE and 0.0 Brier score."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

    ece = compute_ece(y_true, y_prob, n_bins=10)
    brier = compute_brier_score(y_true, y_prob)

    assert ece == pytest.approx(0.0, abs=1e-5)
    assert brier == pytest.approx(0.0, abs=1e-5)


def test_completely_miscalibrated():
    """Opposite predictions should yield high ECE and Brier score near 1.0."""
    y_true = np.array([0, 0, 0, 0])
    y_prob = np.array([0.99, 0.99, 0.99, 0.99])

    ece = compute_ece(y_true, y_prob, n_bins=10)
    brier = compute_brier_score(y_true, y_prob)

    assert ece > 0.9
    assert brier > 0.9


def test_multiclass_calibration():
    """Multiclass ECE and Brier score work on 2D probability matrices."""
    y_true = np.array([0, 1, 2, 0])
    y_prob = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.2, 0.7, 0.1],
    ])
    y_pred = np.argmax(y_prob, axis=1)

    metrics = compute_all_metrics(y_true, y_pred, y_prob)
    assert "ece" in metrics
    assert "brier_score" in metrics
    assert 0.0 <= metrics["ece"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 2.0
