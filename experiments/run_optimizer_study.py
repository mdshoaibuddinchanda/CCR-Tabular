"""Optimizer comparison experiment: SGD vs Adam vs AdamW with & without normalization (Section D).

Reviewer 2 requested:
  "State the optimizer and explain the relationship between your SGD-based proposition
  and the actual optimization algorithm."

This script compares:
  - Optimizers: SGD (momentum=0.9), Adam, AdamW
  - Losses: CCR-NoNorm vs CCR (Normalized)
  - Noise conditions: Clean, 20% Asymmetric, 40% Asymmetric
  - Tracks: Final Macro F1, update scale stability, gradient norm variance.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.training.cross_validation import run_cross_validation
from src.utils.config import OUTPUTS_METRICS, OUTPUTS_PLOTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_optimizer_study(
    datasets: List[str] = ["credit_g", "spambase", "phoneme"],
    seeds: List[int] = [42, 123],
    n_folds: int = 3,
) -> pd.DataFrame:
    """Run cross-optimizer validation."""
    optimizers = ["SGD", "Adam", "AdamW"]
    models = ["ccr_no_norm", "ccr"]
    noise_configs = [("none", 0.0), ("asym", 0.20), ("asym", 0.40)]

    out_csv = OUTPUTS_METRICS / "optimizer_study_results.csv"
    all_runs = []

    for ds in datasets:
        for opt in optimizers:
            for model_name in models:
                for n_type, n_rate in noise_configs:
                    logger.info(
                        f"Running [{ds}] | Opt: {opt} | Loss: {model_name} | Noise: {n_type}@{n_rate:.0%}"
                    )
                    df_run = run_cross_validation(
                        dataset_name=ds,
                        model_name=model_name,
                        noise_type=n_type,
                        noise_rate=n_rate,
                        architecture="mlp",
                        optimizer_name=opt,
                        seeds=seeds,
                        n_folds=n_folds,
                        instrument_batch=True,
                        results_path=out_csv,
                    )
                    all_runs.append(df_run)

    combined_df = pd.read_csv(out_csv) if out_csv.exists() else pd.concat(all_runs, ignore_index=True)
    logger.info(f"Optimizer study complete. Results saved to {out_csv}")
    return combined_df


if __name__ == "__main__":
    run_optimizer_study()
