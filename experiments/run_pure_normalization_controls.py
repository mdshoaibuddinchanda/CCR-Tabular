"""Pure Normalization Controls Experiment (Section 3 of Master Plan).

Evaluates the exact mechanistic question:
  "When does batch normalization matter and when does it do nothing?"

Tests the complete spectrum of weighting dynamics:
  1. Uniform weights: w_i = 1.0 -> Normalization should be an exact identity.
  2. Static class weights: w_i = gamma_y -> Fixed-scale behavior.
  3. Frozen dynamic weights: w_i computed once after warmup and frozen -> Disentangles dynamic adaptation from static sample weighting.
  4. Plain Dynamic CE: w_i = (1 - p_i) (Unnormalized vs Normalized) -> Minimal dynamic weighting test without gate/variance.
  5. Full CCR Dynamic: w_i = (1 - p_i) + beta * Var_i * I(p > tau) + gamma_y (Unnormalized vs Normalized).
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from src.training.cross_validation import run_cross_validation
from src.utils.config import OUTPUTS_METRICS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_pure_normalization_controls(
    datasets: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    n_folds: int = 3,
) -> pd.DataFrame:
    """Run pure normalization controls suite."""
    if datasets is None:
        datasets = ["credit_g", "spambase", "phoneme"]
    if seeds is None:
        seeds = [42, 123]

    # Models representing the spectrum from uniform to full dynamic:
    control_models = [
        "ce",               # Uniform weights w=1
        "wce",              # Unnormalized static class weights
        "norm_wce",         # Normalized static class weights
        "dynamic_ce",       # Plain dynamic weights w=(1-p) unnormalized
        "norm_dynamic_ce",  # Plain dynamic weights w=(1-p) normalized
        "ccr_no_norm",      # Full dynamic CCR unnormalized
        "ccr",              # Full dynamic CCR normalized
    ]

    noise_configs = [("none", 0.0), ("asym", 0.20), ("asym", 0.40)]
    out_csv = OUTPUTS_METRICS / "pure_normalization_controls_results.csv"
    logger.info(f"Starting Pure Normalization Controls on {datasets}...")

    for ds in datasets:
        for model_name in control_models:
            for n_type, n_rate in noise_configs:
                logger.info(
                    f"Controls Run: [{ds}] | Model: {model_name} | Noise: {n_type}@{n_rate:.0%}"
                )
                run_cross_validation(
                    dataset_name=ds,
                    model_name=model_name,
                    noise_type=n_type,
                    noise_rate=n_rate,
                    seeds=seeds,
                    n_folds=n_folds,
                    instrument_batch=True,
                    results_path=out_csv,
                )

    res_df = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    logger.info(f"Pure normalization controls complete. Results saved to {out_csv}")
    return res_df


if __name__ == "__main__":
    run_pure_normalization_controls()
