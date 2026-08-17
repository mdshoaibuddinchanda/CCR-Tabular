"""Tests for data leakage prevention.

Verifies that:
- Preprocessor is fit only on training data
- Column mismatches between splits are caught
- Noise injection is never applied to the full dataset
"""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import build_preprocessor, preprocess_split
from src.data.noise_injection import inject_asymmetric_noise


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_dataframes(n_train: int = 200, n_val: int = 50, n_test: int = 50, seed: int = 42):
    """Create train/val/test DataFrames with matching columns and both classes in every split."""
    rng = np.random.default_rng(seed)
    n_total = n_train + n_val + n_test

    df = pd.DataFrame({
        "num_feat_1": rng.standard_normal(n_total),
        "num_feat_2": rng.standard_normal(n_total),
        "cat_feat_1": rng.choice(["A", "B", "C"], size=n_total),
    })

    # Build labels with both classes guaranteed in every split
    # 70% majority (0), 30% minority (1) — shuffled so both classes appear in each slice
    y_arr = np.array([0] * int(n_total * 0.7) + [1] * int(n_total * 0.3))
    # Pad/trim to exact n_total
    y_arr = np.resize(y_arr, n_total)
    rng.shuffle(y_arr)
    y = pd.Series(y_arr.astype(np.int32))

    X_train = df.iloc[:n_train].reset_index(drop=True)
    X_val = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    X_test = df.iloc[n_train + n_val:].reset_index(drop=True)
    y_train = y.iloc[:n_train].reset_index(drop=True)
    y_val = y.iloc[n_train:n_train + n_val].reset_index(drop=True)
    y_test = y.iloc[n_train + n_val:].reset_index(drop=True)

    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Test 1: Scaler fit only on training data ──────────────────────────────────

def test_scaler_fit_only_on_train():
    """Preprocessor's n_samples_seen_ must match len(X_train) only."""
    X_train, X_val, X_test, y_train, y_val, y_test = make_dataframes()

    preprocessor = build_preprocessor(X_train)

    # Check that the preprocessor was fit on exactly n_train samples
    # StandardScaler stores n_samples_seen_ after fitting
    for name, transformer, cols in preprocessor.transformers_:
        if name == "numerical":
            scaler = transformer.named_steps["scaler"]
            assert scaler.n_samples_seen_ == len(X_train), (
                f"Scaler was fit on {scaler.n_samples_seen_} samples, "
                f"but X_train has {len(X_train)} samples. "
                f"Scaler must be fit on training data only."
            )

    # Also verify the custom attribute we set
    assert preprocessor._train_n_samples == len(X_train), (
        f"_train_n_samples={preprocessor._train_n_samples} != "
        f"len(X_train)={len(X_train)}."
    )


# ── Test 2: Column mismatch raises AssertionError ─────────────────────────────

def test_column_mismatch_raises():
    """Mismatched val/train columns must raise AssertionError."""
    X_train, X_val, X_test, y_train, y_val, y_test = make_dataframes()

    # Introduce a column mismatch in X_val
    X_val_bad = X_val.rename(columns={"num_feat_1": "WRONG_COLUMN"})

    with pytest.raises(AssertionError, match="Val and train columns must match"):
        preprocess_split(
            X_train, X_val_bad, X_test,
            y_train, y_val, y_test,
        )


def test_test_column_mismatch_raises():
    """Mismatched test/train columns must raise AssertionError."""
    X_train, X_val, X_test, y_train, y_val, y_test = make_dataframes()

    # Introduce a column mismatch in X_test
    X_test_bad = X_test.rename(columns={"num_feat_2": "EXTRA_COLUMN"})

    with pytest.raises(AssertionError, match="Test and train columns must match"):
        preprocess_split(
            X_train, X_val, X_test_bad,
            y_train, y_val, y_test,
        )


# ── Test 3: Noise not applied to full dataset ─────────────────────────────────

def test_noise_not_on_test():
    """Passing the full dataset to noise injection must raise ValueError."""
    # Create a dataset larger than the safety threshold (60,000 samples)
    y_full = np.array([0] * 40000 + [1] * 25000)  # 65,000 samples

    with pytest.raises(ValueError, match="training fold"):
        inject_asymmetric_noise(y_full, noise_rate=0.1, seed=42)


# ── Test 4: Preprocessor transforms val/test without refitting ────────────────

def test_preprocessor_transform_only_on_val_test():
    """Val and test must be transformed (not fit) by the preprocessor."""
    X_train, X_val, X_test, y_train, y_val, y_test = make_dataframes()

    (
        X_tr_np, X_val_np, X_test_np,
        y_tr_np, y_val_np, y_test_np,
        preprocessor,
    ) = preprocess_split(X_train, X_val, X_test, y_train, y_val, y_test)

    # Verify output shapes are correct
    assert X_tr_np.shape[0] == len(X_train), (
        f"X_train_np has {X_tr_np.shape[0]} rows, expected {len(X_train)}."
    )
    assert X_val_np.shape[0] == len(X_val), (
        f"X_val_np has {X_val_np.shape[0]} rows, expected {len(X_val)}."
    )
    assert X_test_np.shape[0] == len(X_test), (
        f"X_test_np has {X_test_np.shape[0]} rows, expected {len(X_test)}."
    )

    # All splits must have the same number of features
    assert X_tr_np.shape[1] == X_val_np.shape[1] == X_test_np.shape[1], (
        f"Feature dimensions must match across splits: "
        f"train={X_tr_np.shape[1]}, val={X_val_np.shape[1]}, "
        f"test={X_test_np.shape[1]}."
    )

    # Verify preprocessor was fit on training data only
    assert preprocessor._train_n_samples == len(X_train), (
        "Preprocessor must be fit on training data only."
    )


# ── Test 5: Binary target assertion ──────────────────────────────────────────

def test_non_binary_target_raises():
    """y_train with more than 2 classes must raise AssertionError."""
    X_train, X_val, X_test, _, y_val, y_test = make_dataframes()

    # Create a multi-class target
    y_train_multiclass = pd.Series(np.array([0, 1, 2] * (len(X_train) // 3 + 1))[:len(X_train)])

    with pytest.raises(AssertionError, match="Binary classification only"):
        preprocess_split(
            X_train, X_val, X_test,
            y_train_multiclass, y_val, y_test,
        )
