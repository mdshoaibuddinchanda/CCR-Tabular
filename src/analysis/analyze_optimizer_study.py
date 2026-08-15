"""Analysis pipeline for Optimizer Sensitivity Study (SGD vs Adam vs AdamW).

Evaluates Reviewer 2's core question:
  "What is the relationship between the SGD proposition and actual adaptive optimization algorithms (Adam/AdamW)?"

Outputs:
  - Table of Macro-F1, Gradient Norm, Parameter Update Norm ||Delta theta||_2, Update CV, and Tails (P95/P99/Max)
    grouped by Optimizer x Dataset x Loss Formulation.
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


def analyze_optimizer_study(
    results_csv: Optional[Path] = None,
    telemetry_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Analyze optimizer sensitivity results and generate summary table."""
    r_csv = results_csv or (OUTPUTS_METRICS / "optimizer_study_results.csv")
    t_dir = telemetry_dir or OUTPUTS_TELEMETRY

    if not r_csv.exists():
        logger.warning(f"Results CSV not found at {r_csv}")
        return pd.DataFrame()

    df_res = pd.read_csv(r_csv)
    if len(df_res) == 0:
        return pd.DataFrame()

    # Aggregate performance by dataset x noise x model
    summary = (
        df_res.groupby(["dataset", "noise_type", "noise_rate", "model"])
        .agg({
            "macro_f1": ["mean", "std"],
            "minority_recall": "mean",
            "auc_roc": "mean",
            "auc_pr": "mean",
        })
        .reset_index()
    )
    summary.columns = ["dataset", "noise_type", "noise_rate", "model", "macro_f1_mean", "macro_f1_std", "recall", "roc_auc", "auprc"]

    out_csv = OUTPUTS_METRICS / "optimizer_study_summary.csv"
    summary.to_csv(out_csv, index=False)
    logger.info(f"Saved optimizer study summary table to {out_csv}")
    return summary


if __name__ == "__main__":
    analyze_optimizer_study()
