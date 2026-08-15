"""Stratified K-Fold cross-validation orchestration for CCR-Tabular.

Handles CV fold generation, fold-local preprocessing, fold-local noise generation,
training, test evaluation, metric persistence, and in-memory preprocessing caching.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

# Global in-memory cache for preprocessed fold arrays
_SPLIT_CACHE: Dict[Tuple, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def _get_or_create_fold_split(
    dataset_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    noise_type: str,
    noise_rate: float,
    seed: int,
    fold: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Retrieve preprocessed fold splits from memory cache or compute and cache them."""
    cache_key = (dataset_name, noise_type, noise_rate, seed, fold)
    if cache_key in _SPLIT_CACHE:
        return _SPLIT_CACHE[cache_key]

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
    y_tr_noisy, _noise_stats = generate_noise(
        X_train=X_tr_np,
        y_train=y_tr_np,
        noise_type=noise_type,
        noise_rate=noise_rate,
        seed=seed,
        n_classes=n_classes,
    )

    split_data = (X_tr_np, y_tr_noisy, X_val_np, y_val_np, X_test_np, y_test_np, y_tr_np)
    _SPLIT_CACHE[cache_key] = split_data
    return split_data


def _is_already_completed(run_id: str, results_path: Path) -> bool:
    """Check if run_id is already completed in target CSV file."""
    if not results_path.exists():
        return False
    try:
        df_existing = pd.read_csv(results_path)
        if "run_id" in df_existing.columns and run_id in df_existing["run_id"].values:
            return True
    except Exception:
        pass
    return False


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
    """Run full stratified cross-validation with in-memory fold caching."""
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_name}'.")

    if seeds is None:
        seeds = SEEDS

    df = load_dataset(dataset_name)
    target_col = "target" if "target" in df.columns else df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col].values.astype(int)

    target_csv = results_path or (OUTPUTS_METRICS / "results.csv")
    target_csv.parent.mkdir(parents=True, exist_ok=True)

    results = []

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

            # In-memory cached split and preprocessing
            (
                X_tr_np, y_tr_noisy, X_val_np, y_val_np,
                X_test_np, y_test_np, y_tr_clean,
            ) = _get_or_create_fold_split(
                dataset_name=dataset_name,
                X=X,
                y=y,
                train_idx=train_idx,
                test_idx=test_idx,
                noise_type=noise_type,
                noise_rate=noise_rate,
                seed=seed,
                fold=fold,
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
                clean_y_train=y_tr_clean,
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
                    "train_time_s": val_metrics.get("train_time_s", 0.0),
                    "peak_vram_mb": val_metrics.get("peak_vram_mb", 0.0),
                    "n_epochs": val_metrics.get("n_epochs", 0),
                },
            )

            # Append to persistent CSV immediately
            df_single = pd.DataFrame([test_metrics])
            if target_csv.exists() and target_csv.stat().st_size > 0:
                df_single.to_csv(target_csv, mode="a", header=False, index=False)
            else:
                df_single.to_csv(target_csv, mode="w", header=True, index=False)

            results.append(test_metrics)

    if target_csv.exists():
        try:
            df_target = pd.read_csv(target_csv)
            sub = df_target[
                (df_target["dataset"] == dataset_name) &
                (df_target["model"] == model_name) &
                (df_target["noise_type"] == noise_type) &
                (df_target["noise_rate"] == noise_rate)
            ]
            if len(sub) > 0:
                return sub
        except Exception:
            pass

    return pd.DataFrame(results)
