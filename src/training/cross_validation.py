"""Stratified K-Fold cross-validation orchestration for CCR-Tabular.

Handles CV fold generation, fold-local preprocessing, fold-local noise generation,
training, test evaluation, and metric persistence.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.data.load_data import load_dataset
from src.data.noise_injection import generate_noise
from src.data.preprocess import preprocess_split
from src.training.evaluate import evaluate_model
from src.training.train import make_run_id, train_one_fold
from src.utils.config import (
    DATASETS,
    N_FOLDS,
    OPTIMIZER,
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
    architecture: str = "mlp",
    optimizer_name: str = OPTIMIZER,
    seeds: Optional[List[int]] = None,
    n_folds: int = N_FOLDS,
    instrument_batch: bool = False,
    results_path: Optional[Path] = None,
    batch_size: int = 128,
    device: Optional[Any] = None,
    use_amp: Optional[bool] = None,
) -> pd.DataFrame:
    """Run full stratified cross-validation.

    Args:
        dataset_name: Dataset key from config.DATASETS.
        model_name: Model or loss name.
        noise_type: 'none', 'asym', 'sym', 'feat', or 'idn'.
        noise_rate: Corruption rate (0.0 to 0.4).
        architecture: 'mlp', 'resnet', or 'ft_transformer'.
        optimizer_name: 'AdamW', 'Adam', or 'SGD'.
        seeds: List of seeds (default [42, 123, 2024]).
        n_folds: Number of folds (default 5).
        instrument_batch: If True, logs batch-level optimization telemetry.
        results_path: Destination CSV path.

    Returns:
        DataFrame with results for all runs.
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_name}'.")

    if seeds is None:
        seeds = SEEDS

    df = load_dataset(dataset_name)
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols]
    y = df["target"].values

    all_results: List[Dict] = []
    target_csv = results_path or (OUTPUTS_METRICS / "results.csv")

    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            fold = fold_idx + 1
            run_id = make_run_id(
                dataset_name=dataset_name,
                model_name=model_name,
                noise_type=noise_type,
                noise_rate=noise_rate,
                seed=seed,
                fold=fold,
                architecture=architecture,
            )

            # Check if run exists (idempotent resume)
            if _is_already_completed(run_id, target_csv):
                logger.info(f"[SKIP] {run_id} already exists.")
                continue

            # Split
            X_train_df = X.iloc[train_idx].reset_index(drop=True)
            X_test_df = X.iloc[test_idx].reset_index(drop=True)
            y_train_raw = y[train_idx]
            y_test_raw = y[test_idx]

            # Validation split from training fold only
            X_tr_df, X_val_df, y_tr, y_val = train_test_split(
                X_train_df, y_train_raw,
                test_size=VAL_SIZE,
                stratify=y_train_raw,
                random_state=seed,
            )

            # Preprocessing (fit on train fold ONLY)
            n_classes = len(np.unique(y))
            (
                X_tr_np, X_val_np, X_test_np,
                y_tr_np, y_val_np, y_test_np,
                _preprocessor,
            ) = preprocess_split(
                X_tr_df, X_val_df, X_test_df,
                pd.Series(y_tr), pd.Series(y_val), pd.Series(y_test_raw),
                allow_multiclass=(n_classes > 2),
            )

            # Noise injection (training split ONLY)
            n_classes = len(np.unique(y))
            y_tr_noisy, noise_stats = generate_noise(
                X_train=X_tr_np,
                y_train=y_tr_np,
                noise_type=noise_type,
                noise_rate=noise_rate,
                seed=seed,
                n_classes=n_classes,
            )

            # Train
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
                architecture=architecture,
                optimizer_name=optimizer_name,
                instrument_batch=instrument_batch,
                run_id=run_id,
                clean_y_train=y_tr_np,
                batch_size=batch_size,
                device=device,
                use_amp=use_amp,
            )

            # Evaluate on untouched test fold
            test_metrics = evaluate_model(
                model_or_path=model,
                X_test=X_test_np,
                y_test=y_test_np,
                run_id=run_id,
                metadata={
                    "dataset": dataset_name,
                    "model": model_name,
                    "architecture": architecture,
                    "fold": fold,
                    "seed": seed,
                    "noise_type": noise_type,
                    "noise_rate": noise_rate,
                },
                train_time_s=val_metrics.get("train_time_s", 0.0),
                peak_vram_mb=val_metrics.get("peak_vram_mb", 0.0),
                n_epochs=val_metrics.get("best_epoch", -1),
                results_path=target_csv,
            )

            result = {
                "run_id": run_id,
                "dataset": dataset_name,
                "model": model_name,
                "architecture": architecture,
                "fold": fold,
                "seed": seed,
                "noise_type": noise_type,
                "noise_rate": noise_rate,
                **test_metrics,
            }
            all_results.append(result)

    return pd.DataFrame(all_results)


def _is_already_completed(run_id: str, results_path: Path) -> bool:
    """Check if a run_id already exists in results CSV."""
    if not results_path.exists():
        return False
    try:
        df = pd.read_csv(results_path)
        return "run_id" in df.columns and run_id in df["run_id"].values
    except Exception:
        return False
