"""Tests for noise injection functions.

Verifies correctness, safety, and reproducibility of label noise injection.
"""

import numpy as np
import pytest

from src.data.noise_injection import inject_asymmetric_noise, inject_feature_correlated_noise


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_labels(n_majority: int = 700, n_minority: int = 300, seed: int = 0) -> np.ndarray:
    """Create a binary label array with known class distribution."""
    rng = np.random.default_rng(seed)
    y = np.array([0] * n_majority + [1] * n_minority)
    rng.shuffle(y)
    return y


def make_features(n: int = 1000, n_features: int = 10, seed: int = 0) -> np.ndarray:
    """Create a random feature array."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, n_features)).astype(np.float32)


# ── Asymmetric noise tests ────────────────────────────────────────────────────

def test_asymmetric_majority_never_flipped():
    """Asymmetric noise must NEVER flip majority class (0) labels."""
    y = make_labels()
    y_noisy, stats = inject_asymmetric_noise(y, noise_rate=0.20, seed=42)

    majority_mask = (y == 0)
    assert np.all(y_noisy[majority_mask] == y[majority_mask]), (
        "Asymmetric noise flipped majority class (0) labels. "
        "Only minority class (1) labels should be flipped."
    )


def test_asymmetric_noise_rate_within_tolerance():
    """Actual noise rate must match target rate within ±2%."""
    y = make_labels(n_majority=700, n_minority=300)
    target_rate = 0.20
    y_noisy, stats = inject_asymmetric_noise(y, noise_rate=target_rate, seed=42)

    actual_rate = stats["actual_noise_rate"]
    assert abs(actual_rate - target_rate) <= 0.02, (
        f"Actual noise rate {actual_rate:.4f} deviates from target "
        f"{target_rate:.4f} by more than ±2%."
    )


def test_asymmetric_output_same_length():
    """Noisy array must have the same length as input."""
    y = make_labels()
    y_noisy, _ = inject_asymmetric_noise(y, noise_rate=0.15, seed=42)
    assert len(y_noisy) == len(y), (
        f"Output length {len(y_noisy)} != input length {len(y)}."
    )


def test_asymmetric_zero_noise_returns_identical():
    """noise_rate=0.0 must return an array identical to the input."""
    y = make_labels()
    y_noisy, stats = inject_asymmetric_noise(y, noise_rate=0.0, seed=42)

    assert np.array_equal(y_noisy, y), (
        "With noise_rate=0.0, output must be identical to input."
    )
    assert stats["n_flipped"] == 0, (
        f"n_flipped must be 0 for noise_rate=0.0. Got {stats['n_flipped']}."
    )


def test_asymmetric_invalid_noise_rate_raises():
    """noise_rate outside [0, 1] must raise ValueError."""
    y = make_labels()
    with pytest.raises(ValueError, match="noise_rate"):
        inject_asymmetric_noise(y, noise_rate=1.5, seed=42)

    with pytest.raises(ValueError, match="noise_rate"):
        inject_asymmetric_noise(y, noise_rate=-0.1, seed=42)


def test_asymmetric_missing_class_raises():
    """y_train with only one class must raise ValueError."""
    y_all_zeros = np.zeros(100, dtype=int)
    with pytest.raises(ValueError, match="class 0.*class 1|both"):
        inject_asymmetric_noise(y_all_zeros, noise_rate=0.1, seed=42)


def test_asymmetric_large_dataset_raises():
    """Passing a dataset larger than the safety threshold must raise ValueError."""
    y_large = np.array([0] * 40000 + [1] * 25000)  # 65000 > 60000 threshold
    with pytest.raises(ValueError, match="training fold"):
        inject_asymmetric_noise(y_large, noise_rate=0.1, seed=42)


def test_asymmetric_reproducibility():
    """Same seed must produce identical results."""
    y = make_labels()
    y_noisy_1, _ = inject_asymmetric_noise(y, noise_rate=0.20, seed=99)
    y_noisy_2, _ = inject_asymmetric_noise(y, noise_rate=0.20, seed=99)
    assert np.array_equal(y_noisy_1, y_noisy_2), (
        "Same seed must produce identical noisy labels."
    )


# ── Feature-correlated noise tests ───────────────────────────────────────────

def test_feature_correlated_output_same_length():
    """Noisy array must have the same length as input."""
    y = make_labels()
    X = make_features(n=len(y))
    y_noisy, _ = inject_feature_correlated_noise(X, y, noise_rate=0.10, seed=42)
    assert len(y_noisy) == len(y), (
        f"Output length {len(y_noisy)} != input length {len(y)}."
    )


def test_feature_correlated_zero_noise_returns_identical():
    """noise_rate=0.0 must return an array identical to the input."""
    y = make_labels()
    X = make_features(n=len(y))
    y_noisy, stats = inject_feature_correlated_noise(X, y, noise_rate=0.0, seed=42)
    assert np.array_equal(y_noisy, y), (
        "With noise_rate=0.0, output must be identical to input."
    )


def test_feature_correlated_large_dataset_raises():
    """Passing a dataset larger than the safety threshold must raise ValueError."""
    y_large = np.array([0] * 40000 + [1] * 25000)
    X_large = make_features(n=len(y_large))
    with pytest.raises(ValueError, match="training fold"):
        inject_feature_correlated_noise(X_large, y_large, noise_rate=0.1, seed=42)


def test_feature_correlated_with_confidences():
    """With model_confidences provided, low-confidence samples are targeted."""
    y = make_labels(n_majority=700, n_minority=300)
    X = make_features(n=len(y))

    # All samples have low confidence → all are candidates
    confidences = np.full(len(y), 0.4)  # all < 0.6 threshold
    y_noisy, stats = inject_feature_correlated_noise(
        X, y, noise_rate=0.10, seed=42, model_confidences=confidences
    )

    assert stats["n_candidates"] == len(y), (
        f"All samples should be candidates when confidence < 0.6. "
        f"Got n_candidates={stats['n_candidates']}, expected {len(y)}."
    )
    assert stats["n_flipped"] > 0, "Expected some labels to be flipped."
