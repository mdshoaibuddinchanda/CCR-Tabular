"""Dataset loading utilities for CCR-Tabular.

Downloads datasets from OpenML and caches them locally as CSV.
Standardizes binary targets as {0=majority, 1=minority} and multiclass targets as integers {0, ..., C-1}.
Automatically drops zero-variance/constant features with zero information loss.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.config import DATA_RAW, DATASETS

logger = logging.getLogger(__name__)

_DOWNLOAD_RETRIES = 3
_RETRY_DELAY_SECONDS = 5


def ensure_all_datasets_cached(datasets: Optional[List[str]] = None) -> None:
    """Verify all 14 study datasets exist in data/raw/, automatically downloading missing ones from OpenML."""
    target_datasets = datasets or list(DATASETS.keys())
    logger.info("=================================================================")
    logger.info("       CHECKING / DOWNLOADING ALL 14 STUDY DATASETS             ")
    logger.info("=================================================================")
    for ds_name in target_datasets:
        csv_path = DATA_RAW / f"{ds_name}.csv"
        if not csv_path.exists():
            logger.info(f"  [DOWNLOADING] '{ds_name}' from OpenML...")
        else:
            logger.info(f"  [READY] '{ds_name}' exists in local cache ({csv_path.name}).")
        load_dataset(ds_name)
    logger.info("=================================================================\n")


def load_dataset(name: str, force_download: bool = False) -> pd.DataFrame:
    """Load dataset from OpenML or local cache.

    Args:
        name: Dataset key from config.DATASETS.
        force_download: If True, re-download from OpenML.

    Returns:
        DataFrame with features and integer 'target' column.
    """
    if name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. Valid options: {list(DATASETS.keys())}."
        )

    cache_path = DATA_RAW / f"{name}.csv"

    if cache_path.exists() and not force_download:
        logger.info(f"Loading '{name}' from local cache: {cache_path}")
        df = pd.read_csv(cache_path)
        _log_imbalance(df, name)
        return df

    logger.info(f"Downloading '{name}' from OpenML (id={DATASETS[name]['openml_id']})...")
    df = _download_from_openml(name)
    df = _standardize(df, name)
    df.to_csv(cache_path, index=False)
    logger.info(f"Saved '{name}' to {cache_path}")
    _log_imbalance(df, name)
    return df


def _download_from_openml(name: str) -> pd.DataFrame:
    """Download dataset from OpenML with retry and sklearn fallback."""
    dataset_info = DATASETS[name]
    dataset_id = dataset_info["openml_id"]
    target_col = dataset_info.get("target", "class")
    last_exc: Optional[Exception] = None

    for attempt in range(1, _DOWNLOAD_RETRIES + 1):
        try:
            import openml
            dataset = openml.datasets.get_dataset(
                dataset_id,
                download_data=True,
                download_qualities=False,
                download_features_meta_data=False,
            )
            X, y, _, _ = dataset.get_data(
                target=dataset.default_target_attribute,
                dataset_format="dataframe",
            )
            if y is None:
                raise ValueError("Target attribute not found in OpenML dataset.")

            df = pd.DataFrame(X)
            df["target"] = y
            return df

        except Exception as exc:
            logger.warning(
                f"OpenML download attempt {attempt}/{_DOWNLOAD_RETRIES} failed for '{name}': {exc}"
            )
            last_exc = exc
            if attempt < _DOWNLOAD_RETRIES:
                time.sleep(_RETRY_DELAY_SECONDS)

    # Fallback to fetch_openml
    logger.info(f"Falling back to sklearn.datasets.fetch_openml for '{name}' (id={dataset_id})...")
    try:
        from sklearn.datasets import fetch_openml
        bunch = fetch_openml(data_id=dataset_id, as_frame=True, parser="auto")
        df = bunch.frame
        if "target" not in df.columns:
            target_candidate = bunch.target_names[0] if bunch.target_names else target_col
            if target_candidate in df.columns:
                df = df.rename(columns={target_candidate: "target"})
            else:
                df["target"] = bunch.target
        return df
    except Exception as sklearn_exc:
        raise RuntimeError(
            f"Failed to download '{name}' from OpenML after {_DOWNLOAD_RETRIES} attempts. "
            f"OpenML error: {last_exc}. Sklearn error: {sklearn_exc}."
        ) from sklearn_exc


def _standardize(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Clean column names, drop constant features, and map target labels to integers."""
    df = df.copy()

    # Find target column
    target_col = None
    target_candidates = ["target", "class", "Class", "CLASS", "Outcome", "churn", "diagnosis", "stroke", "Revenue"]
    for col in df.columns:
        if col.lower() in [tc.lower() for tc in target_candidates]:
            target_col = col
            break

    if target_col is None:
        target_col = df.columns[-1]

    if target_col != "target":
        df = df.rename(columns={target_col: "target"})

    # Drop ID-like columns
    id_like = [c for c in df.columns if c.lower() in ("id", "index", "row_id", "instance_id")]
    if id_like:
        df = df.drop(columns=id_like)

    # Drop constant features (zero variance)
    feature_cols = [c for c in df.columns if c != "target"]
    constant_cols = [c for c in feature_cols if df[c].nunique(dropna=False) <= 1]
    if constant_cols:
        logger.info(f"[{name}] Dropping {len(constant_cols)} constant feature(s): {constant_cols}")
        df = df.drop(columns=constant_cols)

    # Target standardization
    raw_target = df["target"]
    val_counts = raw_target.value_counts()
    n_classes = len(val_counts)

    if n_classes == 2:
        # Binary: 0 = Majority, 1 = Minority
        majority_label = val_counts.index[0]
        minority_label = val_counts.index[1]
        mapping = {majority_label: 0, minority_label: 1}
        df["target"] = raw_target.map(mapping).astype(int)
    else:
        # Multiclass: 0, ..., C-1 sorted by raw labels
        unique_labels = sorted(val_counts.index.tolist())
        mapping = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        df["target"] = raw_target.map(mapping).astype(int)

    return df


def _log_imbalance(df: pd.DataFrame, name: str) -> None:
    """Log dataset distribution statistics."""
    counts = df["target"].value_counts().to_dict()
    total = len(df)
    logger.info(f"[{name}] Class distribution: {counts}, total={total}")
