"""Stratified K-Fold cross-validation orchestration for CCR-Tabular.

This is the main entry point for a single experiment configuration.
Runs 5-fold × 3-seed CV, handles preprocessing, noise injection,
training, and evaluation for each fold.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.data.load_data import load_dataset
from src.data.noise_injection import inject_asymmetric_noise, inject_feature_correlated_noise
from src.data.preprocess import preprocess_split
from src.training.evaluate import evaluate_model
from src.training.train import make_run_id, train_one_fold
from src.utils.config import (
    DATASETS,
    MODEL_NAMES,
    N_FOLDS,
    OUTPUTS_METRICS,
    SEEDS,
    VAL_SIZE,
)

logger = logging.getLogger(__name__)


def run_cross_validation(
    dataset_name: str,
    model_name: str,
    noise_type: str = "none",
    noise_rate: float = 0.0,
    seeds: Optional[List[int]] = None,
    n_folds: int = N_FOLDS,
) -> pd.DataFrame:
    """Run full 5-fold × 3-seed cross-validation.

    For each seed, for each fold:
        1. Split data with StratifiedKFold(random_state=seed).
        2. Preprocess (fit on train fold ONLY).
        3. Inject noise into train fold (if configured).
        4. Train model.
        5. Evaluate on test fold.
        6. Append results to outputs/metrics/results.csv.

    Args:
        dataset_name: One of config.DATASETS keys.
        model_name: One of config.MODEL_NAMES.
        noise_type: 'none', 'asym', or 'feat'.
        noise_rate: 0.0, 0.1, 0.2, or 0.3.
        seeds: List of random seeds. Default is config.SEEDS = [42, 123, 2024].
        n_folds: Number of CV folds. Default 5.

    Returns:
        DataFrame with all runs' metrics.
        Summary printed: Mean ± Std for each metric.

    Saves:
        Per-run rows to outputs/metrics/results.csv.
        Summary to outputs/metrics/cv_summary_<dataset>_<model>_<noise>.csv.

    Raises:
        ValueError: If dataset_name or model_name is not recognized.
        AssertionError: If any fold has zero minority samples.
    """
    # ── Input validation ──────────────────────────────────────────────────────
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Valid options: {list(DATASETS.keys())}."
        )
    if model_name not in MODEL_NAMES:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Valid options: {MODEL_NAMES}."
        )
    if noise_type not in ("none", "asym", "feat"):
        raise ValueError(
            f"noise_type must be 'none', 'asym', or 'feat'. Got '{noise_type}'."
        )
    if not (0.0 <= noise_rate <= 1.0):
        raise ValueError(
            f"noise_rate must be in [0.0, 1.0]. Got {noise_rate}."
        )

    if seeds is None:
        seeds = SEEDS

    # ── Load dataset ──────────────────────────────────────────────────────────
    logger.info(f"Loading dataset '{dataset_name}'...")
    df = load_dataset(dataset_name)

    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols]
    y = df["target"].values

    all_results: List[Dict] = []

    for seed in seeds:
        logger.info(
            f"\n{'='*60}\n"
            f"Seed {seed} | {dataset_name} | {model_name} | "
            f"noise={noise_type}@{noise_rate:.0%}\n"
            f"{'='*60}"
        )

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            fold = fold_idx + 1  # 1-indexed

            run_id = make_run_id(
                dataset_name, model_name, noise_type, noise_rate, seed, fold
            )

            # ── Check if already completed (resume support) ───────────────────
            if _is_already_completed(run_id):
                logger.info(f"[SKIP] {run_id} already in results.csv. Skipping.")
                # Load existing result for summary
                existing = _load_existing_result(run_id)
                if existing:
                    all_results.append(existing)
                continue

            # ── Split ─────────────────────────────────────────────────────────
            X_train_df = X.iloc[train_idx].reset_index(drop=True)
            X_test_df = X.iloc[test_idx].reset_index(drop=True)
            y_train_raw = y[train_idx]
            y_test_raw = y[test_idx]

            # ── Mandatory fold integrity assertions ───────────────────────────
            assert np.sum(y_train_raw == 1) > 0, (
                f"Fold {fold}: NO minority samples in training fold. "
                f"Use StratifiedKFold."
            )
            assert np.sum(y_test_raw == 1) > 0, (
                f"Fold {fold}: NO minority samples in test fold."
            )

            # ── Val split from training fold ──────────────────────────────────
            X_tr_df, X_val_df, y_tr, y_val = train_test_split(
                X_train_df, y_train_raw,
                test_size=VAL_SIZE,
                stratify=y_train_raw,
                random_state=seed,
            )

            # ── Preprocessing (fit on train ONLY) ─────────────────────────────
            (
                X_tr_np, X_val_np, X_test_np,
                y_tr_np, y_val_np, y_test_np,
                _preprocessor,
            ) = preprocess_split(
                X_tr_df, X_val_df, X_test_df,
                pd.Series(y_tr), pd.Series(y_val), pd.Series(y_test_raw),
            )

            # ── Noise injection (train split ONLY) ────────────────────────────
            y_tr_noisy = _inject_noise(
                X_tr_np, y_tr_np, noise_type, noise_rate, seed
            )

            # ── Train ─────────────────────────────────────────────────────────
            logger.info(
                f"Training fold {fold}/{n_folds} | seed={seed} | "
                f"train={len(y_tr_noisy)}, val={len(y_val_np)}, test={len(y_test_np)}"
            )

            model, val_metrics = train_one_fold(
                model_name=model_name,
                dataset_name=dataset_name,
                X_train=X_tr_np,
                y_train=y_tr_noisy,
                X_val=X_val_np,
                y_val=y_val_np,
                fold=fold,
                seed=seed,
                noise_type=noise_type,
                noise_rate=noise_rate,
                run_id=run_id,
            )

            # ── Evaluate on test fold ─────────────────────────────────────────
            test_metrics = evaluate_model(
                model_or_path=model,
                X_test=X_test_np,
                y_test=y_test_np,
                run_id=run_id,
                metadata={
                    "dataset": dataset_name,
                    "model": model_name,
                    "fold": fold,
                    "seed": seed,
                    "noise_type": noise_type,
                    "noise_rate": noise_rate,
                },
                train_time_s=val_metrics.get("train_time_s", 0.0),
                n_epochs=val_metrics.get("n_epochs", -1),
            )

            result = {
                "run_id": run_id,
                "dataset": dataset_name,
                "model": model_name,
                "fold": fold,
                "seed": seed,
                "noise_type": noise_type,
                "noise_rate": noise_rate,
                **test_metrics,
            }
            all_results.append(result)

            logger.info(
                f"Fold {fold} complete: "
                f"macro_f1={test_metrics['macro_f1']:.4f}, "
                f"minority_recall={test_metrics['minority_recall']:.4f}"
            )

    # ── Aggregate results ─────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)

    if len(results_df) > 0:
        _print_summary(results_df, dataset_name, model_name, noise_type, noise_rate)
        _save_cv_summary(results_df, dataset_name, model_name, noise_type, noise_rate)

    return results_df


def _inject_noise(
    X_train: np.ndarray,
    y_train: np.ndarray,
    noise_type: str,
    noise_rate: float,
    seed: int,
) -> np.ndarray:
    """Apply noise injection to training labels.

    Args:
        X_train: Training features.
        y_train: Training labels.
        noise_type: 'none', 'asym', or 'feat'.
        noise_rate: Fraction of labels to corrupt.
        seed: Random seed.

    Returns:
        Possibly-noisy training labels (same shape as y_train).
    """
    if noise_type == "none" or noise_rate == 0.0:
        return y_train

    if noise_type == "asym":
        y_noisy, stats = inject_asymmetric_noise(y_train, noise_rate, seed)
        logger.info(f"Asymmetric noise stats: {stats}")
        return y_noisy

    if noise_type == "feat":
        y_noisy, stats = inject_feature_correlated_noise(
            X_train, y_train, noise_rate, seed
        )
        logger.info(f"Feature-correlated noise stats: {stats}")
        return y_noisy

    raise ValueError(
        f"Unknown noise_type '{noise_type}'. Must be 'none', 'asym', or 'feat'."
    )


def _is_already_completed(run_id: str) -> bool:
    """Check if a run_id already exists in results.csv.

    Args:
        run_id: Run identifier to check.

    Returns:
        True if run_id is already in results.csv.
    """
    results_path = OUTPUTS_METRICS / "results.csv"
    if not results_path.exists():
        return False
    try:
        df = pd.read_csv(results_path)
        return run_id in df["run_id"].values
    except Exception:
        return False


def _load_existing_result(run_id: str) -> Optional[Dict]:
    """Load an existing result row from results.csv.

    Args:
        run_id: Run identifier to load.

    Returns:
        Dict with result row, or None if not found.
    """
    results_path = OUTPUTS_METRICS / "results.csv"
    if not results_path.exists():
        return None
    try:
        df = pd.read_csv(results_path)
        row = df[df["run_id"] == run_id]
        if len(row) > 0:
            return row.iloc[0].to_dict()
    except Exception:
        pass
    return None


def _print_summary(
    results_df: pd.DataFrame,
    dataset_name: str,
    model_name: str,
    noise_type: str,
    noise_rate: float,
) -> None:
    """Print mean ± std summary for all metrics.

    Args:
        results_df: DataFrame with all fold results.
        dataset_name: Dataset name for display.
        model_name: Model name for display.
        noise_type: Noise type for display.
        noise_rate: Noise rate for display.
    """
    metric_cols = ["accuracy", "macro_f1", "minority_recall", "auc_roc", "auc_pr"]
    available = [c for c in metric_cols if c in results_df.columns]

    print(f"\n{'='*60}")
    print(f"CV Summary: {dataset_name} | {model_name} | {noise_type}@{noise_rate:.0%}")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"{'-'*52}")

    for col in available:
        vals = results_df[col].dropna()
        if len(vals) > 0:
            print(
                f"{col:<20} {vals.mean():>8.4f} {vals.std():>8.4f} "
                f"{vals.min():>8.4f} {vals.max():>8.4f}"
            )

    print(f"{'='*60}\n")


def _save_cv_summary(
    results_df: pd.DataFrame,
    dataset_name: str,
    model_name: str,
    noise_type: str,
    noise_rate: float,
) -> None:
    """Save CV summary statistics to a CSV file.

    Args:
        results_df: DataFrame with all fold results.
        dataset_name: Dataset name.
        model_name: Model name.
        noise_type: Noise type.
        noise_rate: Noise rate.
    """
    rate_str = f"{int(noise_rate * 100):02d}"
    summary_path = (
        OUTPUTS_METRICS
        / f"cv_summary_{dataset_name}_{model_name}_{noise_type}_{rate_str}.csv"
    )

    metric_cols = ["accuracy", "macro_f1", "minority_recall", "auc_roc", "auc_pr"]
    available = [c for c in metric_cols if c in results_df.columns]

    summary_rows = []
    for col in available:
        vals = results_df[col].dropna()
        if len(vals) > 0:
            summary_rows.append({
                "metric": col,
                "mean": vals.mean(),
                "std": vals.std(),
                "min": vals.min(),
                "max": vals.max(),
                "n_runs": len(vals),
            })

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        logger.info(f"CV summary saved to {summary_path}")
