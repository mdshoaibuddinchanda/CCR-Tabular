"""Label noise injection for CCR-Tabular experiments.

CRITICAL: Noise injection touches the TRAINING SPLIT ONLY. Never call these
functions on validation or test data. A size-based guardrail is included.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Safety threshold: if y_train has more samples than this, it's likely the
# full dataset (not just a training fold), which would mean noise is being
# applied to test data.
_MAX_SAFE_TRAIN_SIZE = 60_000


def inject_asymmetric_noise(
    y_train: np.ndarray,
    noise_rate: float,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Flip minority class (1) labels to majority class (0).

    This simulates the most damaging real-world noise for imbalanced learning:
    the minority signal is corrupted by majority mislabeling.

    Args:
        y_train: Binary labels {0, 1}. Shape [N].
        noise_rate: Fraction of minority-class labels to flip (0.0 to 1.0).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (y_noisy, stats_dict) where stats_dict contains:
            - n_flipped: Number of labels actually flipped.
            - n_minority_before: Count of minority samples before noise.
            - n_minority_after: Count of minority samples after noise.
            - actual_noise_rate: Actual fraction flipped.

    Raises:
        ValueError: If noise_rate not in [0.0, 1.0].
        ValueError: If y_train does not contain both 0 and 1.
        ValueError: If y_train is suspiciously large (likely full dataset).
    """
    # ── Safety guardrail: prevent accidental noise on test data ───────────────
    if len(y_train) > _MAX_SAFE_TRAIN_SIZE:
        raise ValueError(
            f"y_train has {len(y_train)} samples. If this is the full dataset "
            f"(not just the training fold), noise is being applied to test data. "
            f"Pass only the training fold. "
            f"Maximum safe training size: {_MAX_SAFE_TRAIN_SIZE}."
        )

    # ── Input validation ──────────────────────────────────────────────────────
    if not (0.0 <= noise_rate <= 1.0):
        raise ValueError(
            f"noise_rate must be in [0.0, 1.0], got {noise_rate}. "
            f"If you meant 20%, pass noise_rate=0.20, not noise_rate=20."
        )

    unique_vals = np.unique(y_train)
    if not (0 in unique_vals and 1 in unique_vals):
        raise ValueError(
            f"y_train must contain both class 0 (majority) and class 1 (minority). "
            f"Got unique values: {unique_vals.tolist()}."
        )

    y_noisy = y_train.copy()

    if noise_rate == 0.0:
        stats = {
            "n_flipped": 0,
            "n_minority_before": int(np.sum(y_train == 1)),
            "n_minority_after": int(np.sum(y_noisy == 1)),
            "actual_noise_rate": 0.0,
        }
        return y_noisy, stats

    # ── Flip minority labels ──────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    minority_indices = np.where(y_train == 1)[0]
    n_minority_before = len(minority_indices)

    n_to_flip = int(np.round(noise_rate * n_minority_before))
    flip_indices = rng.choice(minority_indices, size=n_to_flip, replace=False)
    y_noisy[flip_indices] = 0

    n_minority_after = int(np.sum(y_noisy == 1))
    actual_rate = n_to_flip / n_minority_before if n_minority_before > 0 else 0.0

    # ── CRITICAL ASSERTION: majority labels must never be flipped ─────────────
    assert np.sum(y_noisy[y_train == 0] != y_train[y_train == 0]) == 0, (
        "Asymmetric noise must NEVER flip majority class (0) labels. "
        "This is a bug in inject_asymmetric_noise."
    )

    stats = {
        "n_flipped": n_to_flip,
        "n_minority_before": n_minority_before,
        "n_minority_after": n_minority_after,
        "actual_noise_rate": actual_rate,
    }

    logger.info(
        f"Asymmetric noise injected: flipped {n_to_flip}/{n_minority_before} "
        f"minority labels (target={noise_rate:.0%}, actual={actual_rate:.2%}). "
        f"Minority count: {n_minority_before} → {n_minority_after}."
    )

    return y_noisy, stats


def inject_feature_correlated_noise(
    X_train: np.ndarray,
    y_train: np.ndarray,
    noise_rate: float,
    seed: int,
    model_confidences: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Flip labels of low-confidence samples (near decision boundary).

    If model_confidences is None, uses random sampling as a proxy.
    In the full pipeline, this is called after epoch 5 using the current
    model's predicted probabilities as confidences.

    Args:
        X_train: Training features. Shape [N, F].
        y_train: Training labels. Shape [N].
        noise_rate: Fraction of boundary samples to corrupt.
        seed: Random seed.
        model_confidences: Optional array of max softmax probabilities.
            If provided, samples with confidence < 0.6 are candidates.
            If None, candidates are selected randomly.

    Returns:
        Tuple of (y_noisy, stats_dict) where stats_dict contains:
            - n_flipped: Number of labels flipped.
            - n_candidates: Number of boundary candidates identified.
            - actual_noise_rate: Actual fraction of total samples flipped.

    Raises:
        ValueError: If noise_rate not in [0.0, 1.0].
        ValueError: If y_train is suspiciously large (likely full dataset).
    """
    # ── Safety guardrail ──────────────────────────────────────────────────────
    if len(y_train) > _MAX_SAFE_TRAIN_SIZE:
        raise ValueError(
            f"y_train has {len(y_train)} samples. If this is the full dataset "
            f"(not just the training fold), noise is being applied to test data. "
            f"Pass only the training fold. "
            f"Maximum safe training size: {_MAX_SAFE_TRAIN_SIZE}."
        )

    if not (0.0 <= noise_rate <= 1.0):
        raise ValueError(
            f"noise_rate must be in [0.0, 1.0], got {noise_rate}. "
            f"If you meant 20%, pass noise_rate=0.20, not noise_rate=20."
        )

    y_noisy = y_train.copy()

    if noise_rate == 0.0:
        return y_noisy, {"n_flipped": 0, "n_candidates": 0, "actual_noise_rate": 0.0}

    rng = np.random.default_rng(seed)
    n_total = len(y_train)

    # ── Identify boundary candidates ──────────────────────────────────────────
    _CONFIDENCE_THRESHOLD = 0.6

    if model_confidences is not None:
        if len(model_confidences) != n_total:
            raise ValueError(
                f"model_confidences length ({len(model_confidences)}) must match "
                f"y_train length ({n_total})."
            )
        candidate_mask = model_confidences < _CONFIDENCE_THRESHOLD
        candidate_indices = np.where(candidate_mask)[0]
    else:
        # Fallback: random selection as proxy for boundary samples
        logger.warning(
            "model_confidences not provided for feature-correlated noise. "
            "Using random sampling as a proxy for boundary samples."
        )
        n_candidates = max(1, int(noise_rate * n_total * 2))  # oversample candidates
        candidate_indices = rng.choice(n_total, size=min(n_candidates, n_total), replace=False)

    n_candidates = len(candidate_indices)

    if n_candidates == 0:
        logger.warning(
            "No boundary candidates found for feature-correlated noise injection. "
            "All model confidences are >= threshold. Returning unmodified labels."
        )
        return y_noisy, {"n_flipped": 0, "n_candidates": 0, "actual_noise_rate": 0.0}

    # ── Flip labels of selected candidates ───────────────────────────────────
    n_to_flip = max(1, int(np.round(noise_rate * n_candidates)))
    n_to_flip = min(n_to_flip, n_candidates)

    flip_indices = rng.choice(candidate_indices, size=n_to_flip, replace=False)
    # Flip: 0 → 1, 1 → 0
    y_noisy[flip_indices] = 1 - y_noisy[flip_indices]

    actual_rate = n_to_flip / n_total

    stats = {
        "n_flipped": n_to_flip,
        "n_candidates": n_candidates,
        "actual_noise_rate": actual_rate,
    }

    logger.info(
        f"Feature-correlated noise injected: flipped {n_to_flip} labels "
        f"from {n_candidates} boundary candidates "
        f"(actual_rate={actual_rate:.2%})."
    )

    return y_noisy, stats
