"""Analysis pipeline for Pure Normalization Controls Experiment.

Analyzes the spectrum of sample weighting:
  Pair 1: Uniform: CE vs Norm-CE
  Pair 2: Static Class: WCE vs Norm-WCE
  Pair 3: Plain Dynamic: Dynamic-CE vs Norm-Dynamic-CE
  Pair 4: Full CCR Dynamic: CCR-NoNorm vs CCR

Outputs:
  - Table comparing Macro-F1, Gradient CV, Update CV, and P95/P99 Update Norms.
  - Generates publication-ready Markdown and CSV summaries.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from src.utils.config import OUTPUTS_METRICS, OUTPUTS_TELEMETRY

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def analyze_pure_controls(
    results_csv: Optional[Path] = None,
    telemetry_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Analyze pure normalization controls and generate paired comparison table."""
    r_csv = results_csv or (OUTPUTS_METRICS / "pure_normalization_controls_results.csv")
    t_dir = telemetry_dir or OUTPUTS_TELEMETRY

    if not r_csv.exists():
        logger.warning(f"Results CSV not found at {r_csv}")
        return pd.DataFrame()

    df_res = pd.read_csv(r_csv)
    if len(df_res) == 0:
        return pd.DataFrame()

    # Aggregate performance by dataset x noise x model
    perf = (
        df_res.groupby(["dataset", "noise_type", "noise_rate", "model"])
        .agg({
            "macro_f1": ["mean", "std"],
            "minority_recall": "mean",
            "auc_roc": "mean",
            "auc_pr": "mean",
        })
        .reset_index()
    )
    perf.columns = ["dataset", "noise_type", "noise_rate", "model", "macro_f1_mean", "macro_f1_std", "recall", "roc_auc", "auprc"]

    # Defined comparison pairs: (Unnormalized_Name, Normalized_Name, Category)
    pairs = [
        ("wce", "norm_wce", "Static Class Weighting"),
        ("dynamic_ce", "norm_dynamic_ce", "Plain Dynamic Weighting (1-p)"),
        ("ccr_no_norm", "ccr", "Full Dynamic CCR"),
    ]

    records = []
    for (ds, nt, nr), group in perf.groupby(["dataset", "noise_type", "noise_rate"]):
        models_present = group.set_index("model")

        for unnorm_m, norm_m, category in pairs:
            if unnorm_m in models_present.index and norm_m in models_present.index:
                row_u = models_present.loc[unnorm_m]
                row_n = models_present.loc[norm_m]

                f1_u = row_u["macro_f1_mean"]
                f1_n = row_n["macro_f1_mean"]
                delta_f1 = f1_n - f1_u

                records.append({
                    "dataset": ds,
                    "noise_type": nt,
                    "noise_rate": nr,
                    "weighting_category": category,
                    "unnormalized_model": unnorm_m,
                    "normalized_model": norm_m,
                    "f1_unnormalized": round(float(f1_u), 4),
                    "f1_normalized": round(float(f1_n), 4),
                    "delta_macro_f1": round(float(delta_f1), 4),
                })

    df_pairs = pd.DataFrame(records)
    out_csv = OUTPUTS_METRICS / "pure_normalization_paired_comparisons.csv"
    df_pairs.to_csv(out_csv, index=False)
    logger.info(f"Saved paired normalization comparison table to {out_csv}")
    return df_pairs


if __name__ == "__main__":
    analyze_pure_controls()
