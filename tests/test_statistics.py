"""Unit tests for statistical significance, FDR correction, and effect sizes."""

import numpy as np
import pandas as pd
import pytest

from src.utils.statistics import (
    analyze_dataset_level_significance,
    benjamini_hochberg_correction,
    compute_cliffs_delta,
    compute_cohens_d,
    compute_confidence_interval,
)


def test_confidence_interval():
    """CI should contain sample mean and span symmetrically."""
    data = np.array([0.80, 0.82, 0.81, 0.83, 0.79])
    mean, lower, upper = compute_confidence_interval(data, confidence=0.95)
    assert lower < mean < upper
    assert mean == pytest.approx(0.81, abs=1e-4)


def test_cohens_d_and_cliffs_delta():
    """Cohen's d and Cliff's delta should be positive when treatment dominates."""
    treatment = np.array([0.85, 0.86, 0.84, 0.87, 0.85])
    baseline = np.array([0.75, 0.76, 0.74, 0.77, 0.75])

    d = compute_cohens_d(treatment, baseline, paired=True)
    cliff = compute_cliffs_delta(treatment, baseline)

    assert d > 2.0  # Very large positive effect size
    assert cliff == pytest.approx(1.0)  # All treatment > baseline


def test_benjamini_hochberg_monotonicity():
    """Adjusted p-values must be >= raw p-values and <= 1.0."""
    raw_p = [0.001, 0.01, 0.03, 0.04, 0.20, 0.80]
    adj_p, rejected = benjamini_hochberg_correction(raw_p, alpha=0.05)

    assert len(adj_p) == len(raw_p)
    for p, q in zip(raw_p, adj_p):
        assert q >= p - 1e-9
        assert q <= 1.0

    # Smallest p-value (0.001) with m=6 becomes 0.001 * 6 = 0.006 <= 0.05 (rejected)
    assert rejected[0] is True
    # Largest p-value (0.80) should not be rejected
    assert rejected[-1] is False


def test_analyze_dataset_level_significance():
    """Significance analysis dataframe includes per-dataset CIs and FDR p-values."""
    df_runs = pd.DataFrame([
        {"dataset": "adult", "model": "mlp_ccr", "fold": i, "seed": 42, "macro_f1": 0.80 + 0.01 * i}
        for i in range(5)
    ] + [
        {"dataset": "adult", "model": "mlp_standard", "fold": i, "seed": 42, "macro_f1": 0.75 + 0.01 * i}
        for i in range(5)
    ])

    results = analyze_dataset_level_significance(
        df_runs, primary_model="mlp_ccr", baseline_models=["mlp_standard"], metrics=["macro_f1"]
    )

    assert len(results) == 1
    row = results.iloc[0]
    assert row["dataset"] == "adult"
    assert row["abs_delta"] == pytest.approx(0.05, abs=1e-3)
    assert "p_fdr_wilcoxon" in row
    assert "cohens_d" in row
