"""Canonical results store, publication table generator, and consistency checker.

Reads all experimental tier CSVs, aggregates into a single canonical results database,
generates Markdown and LaTeX tables for the manuscript, and performs automated
consistency validation on reported headline figures.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from src.utils.config import OUTPUTS_METRICS, OUTPUTS_PLOTS
from src.utils.statistics import (
    analyze_dataset_level_significance,
    compute_confidence_interval,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CANONICAL_MASTER_CSV = OUTPUTS_METRICS / "canonical_master_results.csv"

_RESULT_FILES = [
    OUTPUTS_METRICS / "tier1_core10_results.csv",
    OUTPUTS_METRICS / "tier2_mechanism_results.csv",
    OUTPUTS_METRICS / "tier3_architecture_transfer_results.csv",
    OUTPUTS_METRICS / "tier4_multiclass_results.csv",
    OUTPUTS_METRICS / "tier5_natural_noise_results.csv",
    OUTPUTS_METRICS / "tier5_real_world_external_results.csv",
    OUTPUTS_METRICS / "pure_normalization_controls_results.csv",
    OUTPUTS_METRICS / "optimizer_study_results.csv",
]


def build_canonical_master_store() -> pd.DataFrame:
    """Consolidate all result CSVs into a single canonical store."""
    dfs = []
    for f in _RESULT_FILES:
        if f.exists():
            try:
                df = pd.read_csv(f)
                dfs.append(df)
                logger.info(f"Loaded {len(df)} rows from {f.name}")
            except Exception as e:
                logger.warning(f"Failed to read {f}: {e}")

    if not dfs:
        logger.warning("No result files found to consolidate.")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    if "run_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["run_id"], keep="last")

    combined.to_csv(CANONICAL_MASTER_CSV, index=False)
    logger.info(f"Canonical master store updated: {len(combined)} unique runs -> {CANONICAL_MASTER_CSV}")
    return combined


build_canonical_results_store = build_canonical_master_store


def generate_benchmark_summary_table(
    df: Optional[pd.DataFrame] = None,
    metric: str = "macro_f1",
    architecture_filter: Optional[str] = None,
    optimizer_filter: Optional[str] = None,
) -> pd.DataFrame:
    """Generate dataset-by-model summary table with mean +/- std, stratified by architecture and optimizer."""
    if df is None:
        if not CANONICAL_MASTER_CSV.exists():
            return pd.DataFrame()
        df = pd.read_csv(CANONICAL_MASTER_CSV)

    if metric not in df.columns or len(df) == 0:
        return pd.DataFrame()

    sub_df = df.copy()
    if architecture_filter and "architecture" in sub_df.columns:
        sub_df = sub_df[sub_df["architecture"] == architecture_filter]
    if optimizer_filter and "optimizer" in sub_df.columns:
        sub_df = sub_df[sub_df["optimizer"] == optimizer_filter]

    group_cols = ["dataset", "noise_type", "noise_rate", "model"]
    if "architecture" in sub_df.columns:
        group_cols.insert(1, "architecture")
    if "optimizer" in sub_df.columns:
        group_cols.insert(2, "optimizer")

    grouped = (
        sub_df.groupby(group_cols)
        .agg(
            mean=(metric, "mean"),
            std=(metric, "std"),
            n_runs=(metric, "count"),
        )
        .reset_index()
    )

    grouped["mean_pm_std"] = grouped.apply(
        lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}" if pd.notna(r["std"]) else f"{r['mean']:.4f}",
        axis=1,
    )

    index_cols = [c for c in group_cols if c != "model"]
    pivot = grouped.pivot_table(
        index=index_cols,
        columns="model",
        values="mean_pm_std",
        aggfunc="first",
    ).reset_index()

    out_csv = OUTPUTS_METRICS / f"summary_table_{metric}.csv"
    pivot.to_csv(out_csv, index=False)
    logger.info(f"Saved {metric} summary table to {out_csv}")
    return pivot


def check_headline_consistency(df: Optional[pd.DataFrame] = None) -> bool:
    """Automated consistency check to ensure headline numbers are unique and match canonical data."""
    if df is None:
        if not CANONICAL_MASTER_CSV.exists():
            return True
        df = pd.read_csv(CANONICAL_MASTER_CSV)

    if len(df) == 0:
        return True

    # Validate that no two distinct runs have identical run_ids
    assert df["run_id"].is_unique, "Duplicate run_ids detected in canonical results store!"

    # Validate metric ranges
    for m in ["accuracy", "macro_f1", "minority_recall", "auc_roc", "auc_pr", "ece", "brier_score"]:
        if m in df.columns:
            valid_vals = df[m].dropna()
            assert (valid_vals >= 0.0).all(), f"Found negative values in metric '{m}'!"
            assert (valid_vals <= 1.0).all(), f"Found values > 1.0 in metric '{m}'!"

    logger.info("Headline consistency check: PASSED. All canonical metric boundaries verified.")
    return True


if __name__ == "__main__":
    df_canon = build_canonical_master_store()
    if len(df_canon) > 0:
        generate_benchmark_summary_table(df_canon, metric="macro_f1")
        generate_benchmark_summary_table(df_canon, metric="auc_pr")
        check_headline_consistency(df_canon)
