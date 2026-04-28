"""Dataset loading utilities for CCR-Tabular.

Downloads datasets from OpenML and caches them locally. Standardizes all
datasets to a common format with binary target encoded as {0=majority, 1=minority}.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.config import DATA_RAW, DATASETS

logger = logging.getLogger(__name__)

_DOWNLOAD_RETRIES = 3
_RETRY_DELAY_SECONDS = 5


def load_dataset(name: str, force_download: bool = False) -> pd.DataFrame:
    """Load dataset from OpenML or local cache.

    Args:
        name: Dataset key from config.DATASETS (e.g., 'adult', 'bank').
        force_download: If True, re-download even if local file exists.

    Returns:
        DataFrame with features + target column. Target column is always
        named 'target'. Binary target is always encoded as {0, 1}.
        Class 1 is ALWAYS the minority class.

    Raises:
        ValueError: If dataset name is not in config.DATASETS.
        ValueError: If dataset has more than 2 unique target values.
        RuntimeError: If OpenML download fails and no local cache exists.

    Output file:
        data/raw/<name>.csv
    """
    if name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Valid options: {list(DATASETS.keys())}. "
            f"Check config.DATASETS for the full registry."
        )

    cache_path = DATA_RAW / f"{name}.csv"

    if cache_path.exists() and not force_download:
        logger.info(f"Loading '{name}' from local cache: {cache_path}")
        df = pd.read_csv(cache_path)
        _log_imbalance(df, name)
        return df

    logger.info(f"Downloading '{name}' from OpenML (id={DATASETS[name]['openml_id']})...")
    print(f"  Downloading '{name}' from OpenML (id={DATASETS[name]['openml_id']})... ", end="", flush=True)
    df = _download_from_openml(name)
    df = _standardize(df, name)
    df.to_csv(cache_path, index=False)
    print("done.")
    logger.info(f"Saved '{name}' to {cache_path}")
    _log_imbalance(df, name)
    return df


def _download_from_openml(name: str) -> pd.DataFrame:
    """Download a dataset from OpenML with retry logic.

    Tries the openml library first, falls back to sklearn's fetch_openml
    which handles more dataset formats robustly.

    Args:
        name: Dataset key from config.DATASETS.

    Returns:
        Raw DataFrame with 'target_raw' column.

    Raises:
        RuntimeError: If all retries are exhausted on both methods.
    """
    dataset_id = DATASETS[name]["openml_id"]
    target_col = DATASETS[name]["target"]
    last_exc: Optional[Exception] = None

    # ── Method 1: openml library ──────────────────────────────────────────────
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
                dataset_format="dataframe",
                target=target_col,
            )
            df = X.copy()
            df["target_raw"] = y
            logger.info(f"[{name}] Downloaded via openml library.")
            return df
        except Exception as exc:
            last_exc = exc
            logger.warning(
                f"openml attempt {attempt}/{_DOWNLOAD_RETRIES} failed for "
                f"'{name}' (id={dataset_id}): {exc}. "
                f"Retrying in {_RETRY_DELAY_SECONDS}s..."
            )
            if attempt < _DOWNLOAD_RETRIES:
                time.sleep(_RETRY_DELAY_SECONDS)

    # ── Method 2: sklearn fetch_openml (more robust fallback) ─────────────────
    logger.warning(
        f"[{name}] openml library failed after {_DOWNLOAD_RETRIES} attempts. "
        f"Trying sklearn.datasets.fetch_openml as fallback..."
    )
    print(f"  [{name}] Trying sklearn fallback... ", end="", flush=True)
    try:
        from sklearn.datasets import fetch_openml
        bunch = fetch_openml(
            data_id=dataset_id,
            as_frame=True,
            parser="auto",
        )
        df = bunch.frame.copy()
        # sklearn puts target in the frame already; rename it
        if target_col in df.columns:
            df = df.rename(columns={target_col: "target_raw"})
        elif bunch.target_names and bunch.target_names[0] in df.columns:
            df = df.rename(columns={bunch.target_names[0]: "target_raw"})
        else:
            # target is separate
            df["target_raw"] = bunch.target
        logger.info(f"[{name}] Downloaded via sklearn.fetch_openml fallback.")
        print("done.")
        return df
    except Exception as exc2:
        raise RuntimeError(
            f"Failed to download dataset '{name}' (OpenML id={dataset_id}) via both "
            f"openml library and sklearn fallback. "
            f"openml error: {last_exc}. sklearn error: {exc2}. "
            f"Check your internet connection or place the file manually at "
            f"{DATA_RAW / name}.csv"
        ) from exc2


def _standardize(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Standardize a raw OpenML DataFrame to the CCR-Tabular format.

    Steps:
        1. Strip leading/trailing whitespace from all string columns.
        2. Drop rows with >50% missing values.
        3. Encode target as binary {0=majority, 1=minority}.
        4. Rename target column to 'target'.

    Args:
        df: Raw DataFrame with 'target_raw' column.
        name: Dataset name (for logging).

    Returns:
        Standardized DataFrame with 'target' column encoded as {0, 1}.

    Raises:
        ValueError: If target has more than 2 unique non-null values.
    """
    # Strip whitespace from all object columns (OpenML often has " >50K")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # Drop rows with >50% missing values
    threshold = len(df.columns) * 0.5
    n_before = len(df)
    df = df.dropna(thresh=int(threshold))
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning(
            f"[{name}] Dropped {n_dropped} rows with >50% missing values."
        )

    # Validate binary target
    target_col = "target_raw"
    unique_vals = df[target_col].dropna().unique()
    if len(unique_vals) > 2:
        raise ValueError(
            f"CCR-Tabular only supports binary classification. "
            f"Dataset '{name}' has {len(unique_vals)} unique target values: "
            f"{unique_vals.tolist()}. "
            f"Pre-process the dataset to binary before using this codebase."
        )
    if len(unique_vals) < 2:
        raise ValueError(
            f"Dataset '{name}' has only {len(unique_vals)} unique target value(s): "
            f"{unique_vals.tolist()}. Cannot perform binary classification."
        )

    # Encode: minority → 1, majority → 0
    # Minority = class with fewer samples
    value_counts = df[target_col].value_counts()
    majority_label = value_counts.index[0]  # most frequent
    minority_label = value_counts.index[1]  # least frequent

    logger.info(
        f"[{name}] Target encoding: '{minority_label}' → 1 (minority), "
        f"'{majority_label}' → 0 (majority)"
    )

    df["target"] = df[target_col].map({majority_label: 0, minority_label: 1})
    df = df.drop(columns=["target_raw"])

    # Drop rows where target could not be mapped (NaN)
    n_unmapped = df["target"].isna().sum()
    if n_unmapped > 0:
        logger.warning(f"[{name}] Dropping {n_unmapped} rows with unmapped target values.")
        df = df.dropna(subset=["target"])

    df["target"] = df["target"].astype(int)

    # ASSERT: minority is class 1
    counts = df["target"].value_counts()
    assert counts.get(1, 0) < counts.get(0, 0), (
        f"[{name}] After encoding, class 1 ({counts.get(1, 0)}) is NOT smaller than "
        f"class 0 ({counts.get(0, 0)}). Minority encoding failed."
    )

    return df.reset_index(drop=True)


def _log_imbalance(df: pd.DataFrame, name: str) -> None:
    """Log the actual imbalance ratio for a loaded dataset.

    Args:
        df: DataFrame with 'target' column encoded as {0, 1}.
        name: Dataset name for logging context.
    """
    if "target" not in df.columns:
        return
    counts = df["target"].value_counts()
    n_majority = counts.get(0, 0)
    n_minority = counts.get(1, 0)
    if n_minority > 0:
        ratio = n_majority / n_minority
        logger.info(
            f"[{name}] Imbalance ratio: {ratio:.2f}:1 "
            f"(majority={n_majority}, minority={n_minority}, total={len(df)})"
        )
    else:
        logger.warning(f"[{name}] No minority samples found after loading!")
