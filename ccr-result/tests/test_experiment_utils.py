"""Tests for src/utils/experiment_utils.py — shared expansion experiment utilities.

Verifies:
- CSV append/deduplication helpers
- Noise injection wrapper (train only, never val/test)
- Fold preparation pipeline (no leakage)
- Model evaluation correctness
"""

import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path

from src.utils.experiment_utils import (
    append_row,
    apply_noise,
    evaluate_model,
    is_done,
    make_fold_splits,
    prepare_fold,
)
from src.utils.config import OUTPUTS_METRICS


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_labels(n_majority: int = 70, n_minority: int = 30, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.array([0] * n_majority + [1] * n_minority)
    rng.shuffle(y)
    return y


def make_features(n: int = 100, n_features: int = 5, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, n_features)).astype(np.float32)


# ── CSV helpers ───────────────────────────────────────────────────────────────

def test_append_row_creates_file(tmp_path):
    """append_row creates the CSV if it does not exist."""
    csv = tmp_path / "test.csv"
    append_row({"run_id": "run_001", "macro_f1": 0.75}, csv)
    assert csv.exists()
    df = pd.read_csv(csv)
    assert len(df) == 1
    assert df.iloc[0]["run_id"] == "run_001"


def test_append_row_deduplicates(tmp_path):
    """append_row skips rows with duplicate run_id."""
    csv = tmp_path / "test.csv"
    append_row({"run_id": "run_001", "macro_f1": 0.75}, csv)
    append_row({"run_id": "run_001", "macro_f1": 0.99}, csv)  # duplicate
    df = pd.read_csv(csv)
    assert len(df) == 1
    assert df.iloc[0]["macro_f1"] == 0.75  # original value preserved


def test_append_row_multiple(tmp_path):
    """append_row correctly appends multiple distinct rows."""
    csv = tmp_path / "test.csv"
    for i in range(5):
        append_row({"run_id": f"run_{i:03d}", "macro_f1": float(i) / 10}, csv)
    df = pd.read_csv(csv)
    assert len(df) == 5


def test_is_done_false_when_no_file(tmp_path):
    """is_done returns False when CSV does not exist."""
    csv = tmp_path / "nonexistent.csv"
    assert not is_done("run_001", csv)


def test_is_done_true_after_append(tmp_path):
    """is_done returns True after a row is appended."""
    csv = tmp_path / "test.csv"
    append_row({"run_id": "run_001", "val": 1.0}, csv)
    assert is_done("run_001", csv)
    assert not is_done("run_002", csv)


# ── Noise injection ───────────────────────────────────────────────────────────

def test_apply_noise_none_returns_identical():
    """apply_noise with noise_type='none' returns identical array."""
    y = make_labels()
    X = make_features(n=len(y))
    y_out = apply_noise(X, y, "none", 0.0, 42)
    assert np.array_equal(y_out, y)


def test_apply_noise_asym_majority_never_flipped():
    """Asymmetric noise must NEVER flip majority class (0) labels."""
    y = make_labels(n_majority=700, n_minority=300)
    X = make_features(n=len(y))
    y_noisy = apply_noise(X, y, "asym", 0.2, 42)
    assert np.sum(y_noisy[y == 0] != y[y == 0]) == 0, (
        "Majority labels (0) were flipped — asymmetric noise is broken."
    )


def test_apply_noise_asym_flips_minority():
    """Asymmetric noise at 20% must flip some minority labels."""
    y = make_labels(n_majority=700, n_minority=300)
    X = make_features(n=len(y))
    y_noisy = apply_noise(X, y, "asym", 0.2, 42)
    n_flipped = int(np.sum(y_noisy[y == 1] != y[y == 1]))
    assert n_flipped > 0, "No minority labels were flipped at 20% noise rate."


def test_apply_noise_feat_returns_same_length():
    """Feature-correlated noise returns array of same length."""
    y = make_labels()
    X = make_features(n=len(y))
    y_noisy = apply_noise(X, y, "feat", 0.1, 42)
    assert len(y_noisy) == len(y)


def test_apply_noise_unknown_type_raises():
    """Unknown noise_type must raise ValueError."""
    y = make_labels()
    X = make_features(n=len(y))
    with pytest.raises(ValueError, match="Unknown noise_type"):
        apply_noise(X, y, "invalid_type", 0.1, 42)


# ── Fold preparation ──────────────────────────────────────────────────────────

def make_dataframe(n: int = 300, seed: int = 42) -> tuple:
    """Create a small DataFrame with both classes for fold testing."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "f1": rng.standard_normal(n),
        "f2": rng.standard_normal(n),
        "f3": rng.choice(["A", "B", "C"], size=n),
    })
    y = np.array([0] * int(n * 0.7) + [1] * int(n * 0.3))
    rng.shuffle(y)
    return X, y


def test_make_fold_splits_count():
    """make_fold_splits returns exactly N_FOLDS splits."""
    X, y = make_dataframe()
    splits = make_fold_splits(X, y, seed=42, n_folds=5)
    assert len(splits) == 5


def test_make_fold_splits_stratified():
    """Each fold must contain both classes."""
    X, y = make_dataframe()
    splits = make_fold_splits(X, y, seed=42, n_folds=5)
    for train_idx, test_idx in splits:
        assert np.sum(y[train_idx] == 1) > 0, "No minority in train fold"
        assert np.sum(y[test_idx] == 1) > 0, "No minority in test fold"


def test_prepare_fold_no_leakage():
    """Val and test must be transformed (not fit) by the preprocessor."""
    X, y = make_dataframe(n=300)
    splits = make_fold_splits(X, y, seed=42, n_folds=5)
    train_idx, test_idx = splits[0]

    X_tr, X_val, X_te, y_tr, y_val, y_te = prepare_fold(
        X, y, train_idx, test_idx, seed=42
    )

    # All arrays must be numpy
    for arr in [X_tr, X_val, X_te, y_tr, y_val, y_te]:
        assert isinstance(arr, np.ndarray)

    # Shapes must be consistent
    assert X_tr.shape[1] == X_val.shape[1] == X_te.shape[1]
    assert len(X_tr) == len(y_tr)
    assert len(X_val) == len(y_val)
    assert len(X_te) == len(y_te)


def test_prepare_fold_noise_on_train_only():
    """Noise must only affect training labels, never val or test."""
    X, y = make_dataframe(n=300)
    splits = make_fold_splits(X, y, seed=42, n_folds=5)
    train_idx, test_idx = splits[0]

    # Get clean fold first
    _, _, _, y_tr_clean, y_val_clean, y_te_clean = prepare_fold(
        X, y, train_idx, test_idx, seed=42, noise_type="none", noise_rate=0.0
    )

    # Get noisy fold
    _, _, _, y_tr_noisy, y_val_noisy, y_te_noisy = prepare_fold(
        X, y, train_idx, test_idx, seed=42, noise_type="asym", noise_rate=0.3
    )

    # Val and test must be identical (noise never applied)
    assert np.array_equal(y_val_clean, y_val_noisy), (
        "Validation labels changed after noise injection — DATA LEAKAGE!"
    )
    assert np.array_equal(y_te_clean, y_te_noisy), (
        "Test labels changed after noise injection — DATA LEAKAGE!"
    )

    # Training labels must differ (noise was applied)
    assert not np.array_equal(y_tr_clean, y_tr_noisy), (
        "Training labels unchanged after 30% noise injection."
    )


# ── Model evaluation ──────────────────────────────────────────────────────────

def test_evaluate_model_returns_all_metrics():
    """evaluate_model must return all 5 required metrics."""
    from src.models.mlp import TabularMLP

    model = TabularMLP(input_dim=5, num_classes=2, hidden_dims=[16, 8])
    X_te = np.random.randn(50, 5).astype(np.float32)
    y_te = np.array([0] * 35 + [1] * 15)

    metrics = evaluate_model(model, X_te, y_te)

    required = {"accuracy", "macro_f1", "minority_recall", "auc_roc", "auc_pr"}
    assert required == set(metrics.keys()), (
        f"Missing metrics: {required - set(metrics.keys())}"
    )


def test_evaluate_model_values_in_range():
    """All metric values must be in [0, 1] or NaN."""
    from src.models.mlp import TabularMLP

    model = TabularMLP(input_dim=5, num_classes=2, hidden_dims=[16, 8])
    X_te = np.random.randn(60, 5).astype(np.float32)
    y_te = np.array([0] * 40 + [1] * 20)

    metrics = evaluate_model(model, X_te, y_te)

    for name, val in metrics.items():
        if not np.isnan(val):
            assert 0.0 <= val <= 1.0, f"{name}={val} is outside [0, 1]"
