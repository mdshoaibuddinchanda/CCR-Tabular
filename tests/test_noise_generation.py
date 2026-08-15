"""Unit tests for the expanded noise injection suite (Symmetric, Asymmetric, Feature, IDN)."""

import numpy as np
import pytest

from src.data.noise_injection import (
    generate_noise,
    inject_asymmetric_noise,
    inject_feature_correlated_noise,
    inject_instance_dependent_noise,
    inject_symmetric_noise,
)


def test_asymmetric_noise_exact_and_safe():
    """Asymmetric noise flips minority only and respects rate."""
    y = np.array([0] * 80 + [1] * 20)
    y_noisy, stats = inject_asymmetric_noise(y, noise_rate=0.30, seed=42)

    # 20 * 0.30 = 6 minority samples flipped
    assert stats["n_flipped"] == 6
    assert np.sum(y_noisy == 1) == 14
    # Majority must never be flipped
    assert np.sum(y_noisy[y == 0] != 0) == 0


def test_symmetric_noise_exact_rate():
    """Symmetric noise flips across classes."""
    y = np.array([0] * 100 + [1] * 100)
    y_noisy, stats = inject_symmetric_noise(y, noise_rate=0.20, seed=42, n_classes=2)

    # 200 * 0.20 = 40 samples flipped
    assert stats["n_flipped"] == 40
    assert len(y_noisy) == 200


def test_feature_correlated_noise_margin_candidates():
    """Feature-correlated noise trains fold-local model and flips low margin samples."""
    np.random.seed(42)
    X = np.random.randn(100, 4)
    # Linearly separable with noise
    y = (X[:, 0] > 0).astype(int)

    y_noisy, stats = inject_feature_correlated_noise(
        X, y, noise_rate=0.15, seed=42, candidate_fraction=0.40
    )

    # Total corruptions = floor(0.15 * 100) = 15
    assert stats["n_flipped"] == 15
    assert stats["n_candidates"] == 40
    assert len(y_noisy) == 100


def test_instance_dependent_noise():
    """Instance-dependent noise flips labels with feature projection."""
    np.random.seed(42)
    X = np.random.randn(150, 5)
    y = np.random.choice([0, 1], size=150)

    y_noisy, stats = inject_instance_dependent_noise(X, y, noise_rate=0.20, seed=42)
    assert stats["n_flipped"] > 0
    assert len(y_noisy) == 150


def test_master_generate_noise_routing():
    """generate_noise routes cleanly to all options."""
    X = np.random.randn(50, 3)
    y = np.array([0] * 35 + [1] * 15)

    for ntype in ["none", "asym", "sym", "feat", "idn"]:
        y_noisy, stats = generate_noise(X, y, noise_type=ntype, noise_rate=0.20, seed=42)
        assert len(y_noisy) == 50
        assert stats["noise_type"] == ntype
