"""Label noise injection engine for CCR-Tabular experiments.

Implements standalone, fold-local noise generation for all requested noise regimes:
  1. Asymmetric Noise: Minority (1) -> Majority (0) conditional flips.
     Reports both conditional minority flip rate and overall dataset corruption rate.
  2. Symmetric Noise: Uniform random class flips.
  3. Feature-Correlated Noise: Fold-local boundary ranking via margin M(x) = |P(y=1) - P(y=0)|.
     Candidates chosen from lowest 40% margin with exact corruption count floor(eps * N_train).
  4. Instance-Dependent Noise (IDN): Feature-dependent corruption probability P(y_tilde != y | x).

CRITICAL: All reference models and noise generation logic operate STRICTLY on the
training fold (X_train, y_train). Never pass validation or test splits.
"""

import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

_MAX_SAFE_TRAIN_SIZE = 60_000


def inject_asymmetric_noise(
    y_train: np.ndarray,
    noise_rate: float,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Flip minority class (1) labels to majority class (0).

    Args:
        y_train: Binary labels {0, 1}, shape [N].
        noise_rate: Fraction of minority-class labels to flip (0.0 to 1.0).
        seed: Random seed.

    Returns:
        Tuple of (y_noisy, stats_dict).
    """
    if len(y_train) > _MAX_SAFE_TRAIN_SIZE:
        raise ValueError(
            f"y_train has {len(y_train)} samples. If this is the full dataset "
            f"(not just the training fold), noise is being applied to test data. "
            f"Pass only the training fold. Maximum safe training size: {_MAX_SAFE_TRAIN_SIZE}."
        )
    if not (0.0 <= noise_rate <= 1.0):
        raise ValueError(f"noise_rate must be in [0.0, 1.0], got {noise_rate}.")

    unique_vals = np.unique(y_train)
    if not (0 in unique_vals and 1 in unique_vals):
        raise ValueError(f"y_train must contain both class 0 and 1. Got: {unique_vals}.")

    y_noisy = y_train.copy()
    n_total = len(y_train)
    minority_indices = np.where(y_train == 1)[0]
    n_minority_before = len(minority_indices)

    if noise_rate == 0.0 or n_minority_before == 0:
        return y_noisy, {
            "noise_type": "asym",
            "n_flipped": 0,
            "n_minority_before": n_minority_before,
            "n_minority_after": n_minority_before,
            "target_conditional_noise_rate": 0.0,
            "actual_conditional_noise_rate": 0.0,
            "target_overall_noise_rate": 0.0,
            "actual_overall_noise_rate": 0.0,
            "actual_noise_rate": 0.0,
        }

    rng = np.random.default_rng(seed)
    n_to_flip = int(np.floor(noise_rate * n_minority_before))
    if n_to_flip > 0:
        flip_indices = rng.choice(minority_indices, size=n_to_flip, replace=False)
        y_noisy[flip_indices] = 0

    n_minority_after = int(np.sum(y_noisy == 1))
    actual_cond_rate = n_to_flip / n_minority_before if n_minority_before > 0 else 0.0
    actual_overall_rate = n_to_flip / n_total if n_total > 0 else 0.0
    target_overall_rate = (noise_rate * n_minority_before) / n_total if n_total > 0 else 0.0

    # Assertion: majority labels must never be modified
    assert np.sum(y_noisy[y_train == 0] != y_train[y_train == 0]) == 0, (
        "Asymmetric noise must NEVER flip majority class (0) labels."
    )

    stats = {
        "noise_type": "asym",
        "n_flipped": n_to_flip,
        "n_minority_before": n_minority_before,
        "n_minority_after": n_minority_after,
        "target_conditional_noise_rate": noise_rate,
        "actual_conditional_noise_rate": actual_cond_rate,
        "target_overall_noise_rate": target_overall_rate,
        "actual_overall_noise_rate": actual_overall_rate,
        "actual_noise_rate": actual_cond_rate,  # Backward compatible
    }
    return y_noisy, stats


def inject_symmetric_noise(
    y_train: np.ndarray,
    noise_rate: float,
    seed: int,
    n_classes: int = 2,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Flip labels uniformly across all classes with rate eps."""
    if len(y_train) > _MAX_SAFE_TRAIN_SIZE:
        raise ValueError(
            f"y_train has {len(y_train)} samples. Pass only the training fold. "
            f"Maximum safe training size: {_MAX_SAFE_TRAIN_SIZE}."
        )
    if not (0.0 <= noise_rate <= 1.0):
        raise ValueError(f"noise_rate must be in [0.0, 1.0], got {noise_rate}.")

    y_noisy = y_train.copy()
    n_total = len(y_train)

    if noise_rate == 0.0 or n_total == 0:
        return y_noisy, {
            "noise_type": "sym",
            "n_flipped": 0,
            "target_conditional_noise_rate": 0.0,
            "actual_conditional_noise_rate": 0.0,
            "target_overall_noise_rate": 0.0,
            "actual_overall_noise_rate": 0.0,
            "actual_noise_rate": 0.0,
        }

    rng = np.random.default_rng(seed)
    n_to_flip = int(np.floor(noise_rate * n_total))
    flip_indices = rng.choice(n_total, size=n_to_flip, replace=False)

    for idx in flip_indices:
        curr_label = y_noisy[idx]
        other_classes = [c for c in range(n_classes) if c != curr_label]
        if other_classes:
            y_noisy[idx] = rng.choice(other_classes)
        else:
            y_noisy[idx] = 1 - curr_label

    actual_rate = n_to_flip / n_total if n_total > 0 else 0.0
    stats = {
        "noise_type": "sym",
        "n_flipped": n_to_flip,
        "target_conditional_noise_rate": noise_rate,
        "actual_conditional_noise_rate": actual_rate,
        "target_overall_noise_rate": noise_rate,
        "actual_overall_noise_rate": actual_rate,
        "actual_noise_rate": actual_rate,
    }
    return y_noisy, stats


def inject_feature_correlated_noise(
    X_train: np.ndarray,
    y_train: np.ndarray,
    noise_rate: float,
    seed: int,
    candidate_fraction: float = 0.40,
    reference_model: Optional[Any] = None,
    model_confidences: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Inject feature-correlated noise by corrupting boundary-adjacent samples."""
    if len(y_train) > _MAX_SAFE_TRAIN_SIZE:
        raise ValueError(
            f"y_train has {len(y_train)} samples. Pass only the training fold. "
            f"Maximum safe training size: {_MAX_SAFE_TRAIN_SIZE}."
        )
    if not (0.0 <= noise_rate <= 1.0):
        raise ValueError(f"noise_rate must be in [0.0, 1.0], got {noise_rate}.")

    y_noisy = y_train.copy()
    n_total = len(y_train)

    if noise_rate == 0.0 or n_total == 0:
        return y_noisy, {
            "noise_type": "feat",
            "n_flipped": 0,
            "n_candidates": 0,
            "target_conditional_noise_rate": 0.0,
            "actual_conditional_noise_rate": 0.0,
            "target_overall_noise_rate": 0.0,
            "actual_overall_noise_rate": 0.0,
            "actual_noise_rate": 0.0,
        }

    rng = np.random.default_rng(seed)

    if model_confidences is not None:
        candidate_mask = model_confidences < 0.6
        candidate_indices = np.where(candidate_mask)[0]
    else:
        # 1. Fit fold-local reference model
        if reference_model is None:
            clf = LogisticRegression(max_iter=500, random_state=seed)
            clf.fit(X_train, y_train)
        else:
            clf = reference_model

        # 2. Compute margin
        probs = clf.predict_proba(X_train)
        if probs.shape[1] == 2:
            margins = np.abs(probs[:, 1] - probs[:, 0])
        else:
            sorted_probs = np.sort(probs, axis=1)
            margins = sorted_probs[:, -1] - sorted_probs[:, -2]

        sorted_indices = np.argsort(margins)
        n_candidates = max(1, int(np.ceil(candidate_fraction * n_total)))
        candidate_indices = sorted_indices[:n_candidates]

    if len(candidate_indices) == 0:
        return y_noisy, {
            "noise_type": "feat", "n_flipped": 0, "n_candidates": 0,
            "target_conditional_noise_rate": 0.0, "actual_conditional_noise_rate": 0.0,
            "target_overall_noise_rate": 0.0, "actual_overall_noise_rate": 0.0,
            "actual_noise_rate": 0.0
        }

    n_to_flip = min(int(np.floor(noise_rate * n_total)), len(candidate_indices))
    if n_to_flip == 0 and noise_rate > 0.0:
        n_to_flip = min(1, len(candidate_indices))

    flip_indices = rng.choice(candidate_indices, size=n_to_flip, replace=False)

    n_classes = len(np.unique(y_train))
    for idx in flip_indices:
        curr_label = y_noisy[idx]
        other_classes = [c for c in range(n_classes) if c != curr_label]
        if other_classes:
            y_noisy[idx] = rng.choice(other_classes)
        else:
            y_noisy[idx] = 1 - curr_label

    actual_rate = n_to_flip / n_total if n_total > 0 else 0.0
    stats = {
        "noise_type": "feat",
        "n_flipped": n_to_flip,
        "n_candidates": len(candidate_indices),
        "target_conditional_noise_rate": noise_rate,
        "actual_conditional_noise_rate": actual_rate,
        "target_overall_noise_rate": noise_rate,
        "actual_overall_noise_rate": actual_rate,
        "actual_noise_rate": actual_rate,
    }
    return y_noisy, stats


def inject_instance_dependent_noise(
    X_train: np.ndarray,
    y_train: np.ndarray,
    noise_rate: float,
    seed: int,
    n_classes: int = 2,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Inject Instance-Dependent Noise (IDN)."""
    if len(y_train) > _MAX_SAFE_TRAIN_SIZE:
        raise ValueError(
            f"y_train has {len(y_train)} samples. Pass only the training fold. "
            f"Maximum safe training size: {_MAX_SAFE_TRAIN_SIZE}."
        )
    if not (0.0 <= noise_rate <= 1.0):
        raise ValueError(f"noise_rate must be in [0.0, 1.0], got {noise_rate}.")

    y_noisy = y_train.copy()
    n_total, n_features = X_train.shape

    if noise_rate == 0.0 or n_total == 0:
        return y_noisy, {
            "noise_type": "idn",
            "n_flipped": 0,
            "target_conditional_noise_rate": 0.0,
            "actual_conditional_noise_rate": 0.0,
            "target_overall_noise_rate": 0.0,
            "actual_overall_noise_rate": 0.0,
            "actual_noise_rate": 0.0,
        }

    rng = np.random.default_rng(seed)
    w = rng.standard_normal(size=(n_features, 1))
    projections = (X_train @ w).flatten()
    std_proj = np.std(projections) + 1e-8
    mean_proj = np.mean(projections)
    norm_proj = (projections - mean_proj) / std_proj

    flip_probs = noise_rate * (1.0 / (1.0 + np.exp(-norm_proj)))
    flip_probs = np.clip(flip_probs, 0.0, 0.95)

    random_draws = rng.uniform(0.0, 1.0, size=n_total)
    flip_mask = random_draws < flip_probs
    flip_indices = np.where(flip_mask)[0]

    for idx in flip_indices:
        curr_label = y_noisy[idx]
        other_classes = [c for c in range(n_classes) if c != curr_label]
        if other_classes:
            y_noisy[idx] = rng.choice(other_classes)
        else:
            y_noisy[idx] = 1 - curr_label

    n_to_flip = len(flip_indices)
    actual_rate = n_to_flip / n_total if n_total > 0 else 0.0

    stats = {
        "noise_type": "idn",
        "n_flipped": n_to_flip,
        "target_conditional_noise_rate": noise_rate,
        "actual_conditional_noise_rate": actual_rate,
        "target_overall_noise_rate": noise_rate,
        "actual_overall_noise_rate": actual_rate,
        "actual_noise_rate": actual_rate,
    }
    return y_noisy, stats


def generate_noise(
    X_train: np.ndarray,
    y_train: np.ndarray,
    noise_type: str,
    noise_rate: float,
    seed: int,
    n_classes: int = 2,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Master noise generator routing to specific noise models."""
    noise_type = noise_type.lower().strip()
    if noise_type in ("none", "clean") or noise_rate == 0.0:
        return y_train.copy(), {
            "noise_type": "none",
            "n_flipped": 0,
            "target_conditional_noise_rate": 0.0,
            "actual_conditional_noise_rate": 0.0,
            "target_overall_noise_rate": 0.0,
            "actual_overall_noise_rate": 0.0,
            "actual_noise_rate": 0.0,
        }

    if noise_type in ("asym", "asymmetric"):
        return inject_asymmetric_noise(y_train, noise_rate, seed)
    elif noise_type in ("sym", "symmetric"):
        return inject_symmetric_noise(y_train, noise_rate, seed, n_classes=n_classes)
    elif noise_type in ("feat", "feature_correlated"):
        return inject_feature_correlated_noise(X_train, y_train, noise_rate, seed)
    elif noise_type in ("idn", "instance_dependent"):
        return inject_instance_dependent_noise(X_train, y_train, noise_rate, seed, n_classes=n_classes)
    else:
        raise ValueError(
            f"Unknown noise_type '{noise_type}'. Valid options: ['none', 'asym', 'sym', 'feat', 'idn']."
        )
