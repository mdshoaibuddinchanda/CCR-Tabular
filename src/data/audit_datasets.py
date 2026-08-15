"""Automated dataset audit and validation pipeline for CCR-Tabular.

Performs strict validation on all 14 datasets (10+2+2 design) prior to benchmark entry:
  1. Sample count and feature dimensionality
  2. Class distribution and exact Imbalance Ratio (IR)
  3. Missing value detection and strategy verification
  4. Categorical vs numerical column identification
  5. Duplicate sample identification
  6. Constant / zero-variance feature detection
  7. Target leakage / correlation checks
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from src.data.load_data import load_dataset
from src.utils.config import (
    CORE_10_DATASETS,
    DATASETS,
    MULTICLASS_DATASETS,
    NATURAL_NOISE_DATASETS,
    OUTPUTS_METRICS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def audit_single_dataset(name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Perform comprehensive audit on a single dataset."""
    logger.info(f"Auditing dataset '{name}'...")
    df = load_dataset(name)

    n_samples, n_total_cols = df.shape
    target_col = "target"
    if target_col not in df.columns:
        raise ValueError(f"Dataset '{name}' missing 'target' column after standardization.")

    feature_cols = [c for c in df.columns if c != target_col]
    n_features = len(feature_cols)

    # Class distribution
    target_series = df[target_col]
    class_counts = target_series.value_counts().to_dict()
    unique_classes = sorted(list(class_counts.keys()))
    n_classes = len(unique_classes)

    if n_classes == 2:
        maj_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = maj_count / (min_count + 1e-8)
    else:
        sorted_counts = sorted(list(class_counts.values()), reverse=True)
        imbalance_ratio = sorted_counts[0] / (sorted_counts[-1] + 1e-8)

    # Missing values
    missing_count = int(df[feature_cols].isna().sum().sum())
    missing_pct = (missing_count / (n_samples * n_features)) * 100.0

    # Feature types
    num_cols = df[feature_cols].select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df[feature_cols].select_dtypes(exclude=["number"]).columns.tolist()

    # Constant features
    constant_features = [c for c in feature_cols if df[c].nunique(dropna=False) <= 1]

    # Duplicate rows
    n_duplicates = int(df.duplicated(subset=feature_cols).sum())

    # Target correlation check (Leakage guard: no feature should have >0.999 correlation with target)
    max_target_corr = 0.0
    for col in num_cols:
        try:
            corr = abs(df[col].corr(target_series.astype(float)))
            if not np.isnan(corr) and corr > max_target_corr:
                max_target_corr = corr
        except Exception:
            pass

    leakage_flag = max_target_corr > 0.995

    # Status judgment
    passed = (
        n_samples >= 200
        and n_features >= 2
        and n_classes >= 2
        and not leakage_flag
        and len(constant_features) == 0
    )

    return {
        "dataset": name,
        "tier": "Core 10" if name in CORE_10_DATASETS else ("Multiclass" if name in MULTICLASS_DATASETS else "Real-World External"),
        "n_samples": n_samples,
        "n_features": n_features,
        "n_numerical": len(num_cols),
        "n_categorical": len(cat_cols),
        "n_classes": n_classes,
        "class_distribution": str(class_counts),
        "imbalance_ratio": round(imbalance_ratio, 2),
        "missing_pct": round(missing_pct, 2),
        "n_duplicates": n_duplicates,
        "constant_features": len(constant_features),
        "max_target_corr": round(max_target_corr, 4),
        "leakage_suspect": leakage_flag,
        "audit_status": "PASSED" if passed else "FAILED",
    }


def run_full_dataset_audit() -> pd.DataFrame:
    """Run audit across all 14 datasets."""
    all_datasets = CORE_10_DATASETS + MULTICLASS_DATASETS + NATURAL_NOISE_DATASETS
    records = []

    logger.info(f"Running automated audit on {len(all_datasets)} datasets...")
    for ds_name in all_datasets:
        cfg = DATASETS.get(ds_name, {})
        res = audit_single_dataset(ds_name, cfg)
        records.append(res)
        logger.info(
            f"  [{ds_name:14s}] N={res['n_samples']:6d} | D={res['n_features']:3d} | "
            f"IR={res['imbalance_ratio']:5.2f} | Audit: {res['audit_status']}"
        )

    df_audit = pd.DataFrame(records)
    out_csv = OUTPUTS_METRICS / "dataset_audit_report.csv"
    df_audit.to_csv(out_csv, index=False)
    logger.info(f"Saved complete dataset audit report to {out_csv}")

    all_passed = (df_audit["audit_status"] == "PASSED").all()
    if not all_passed:
        failed = df_audit[df_audit["audit_status"] != "PASSED"]["dataset"].tolist()
        raise RuntimeError(f"Dataset audit failed for: {failed}")

    logger.info("All 14 datasets successfully passed the audit quality gate.")
    return df_audit


if __name__ == "__main__":
    run_full_dataset_audit()
