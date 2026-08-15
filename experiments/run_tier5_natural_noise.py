"""Tier 5: Real-World & Naturally Noisy Datasets (Section G).

Reviewer 1 noted:
  "The real-world robustness claims need moderation because current experiments are synthetic-noise experiments.
  A naturally noisy dataset gives external validity that synthetic noise cannot."

This script evaluates:
  - Naturally noisy real-world tabular datasets (heart_disease, breast_cancer)
  - Evaluates models without artificial noise injection (evaluating inherent label noise robustness)
  - Models: CE, WCE, Focal, GCE, SCE, ELR, Norm-WCE, Norm-Focal, CCR-NoNorm, CCR
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.training.cross_validation import run_cross_validation
from src.utils.config import NATURAL_NOISE_DATASETS, OUTPUTS_METRICS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_TIER5_LOSSES = ["ce", "wce", "focal", "gce", "sce", "elr", "norm_wce", "norm_focal", "ccr_no_norm", "ccr"]


def run_tier5_natural_noise_experiments(
    datasets: Optional[List[str]] = None,
    losses: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    n_folds: int = 5,
) -> pd.DataFrame:
    """Run Tier 5 naturally noisy datasets benchmark."""
    if datasets is None:
        datasets = NATURAL_NOISE_DATASETS
    if losses is None:
        losses = _TIER5_LOSSES
    if seeds is None:
        seeds = [42, 123, 2024]

    out_csv = OUTPUTS_METRICS / "tier5_natural_noise_results.csv"
    logger.info(f"Starting Tier 5 Natural Noise Benchmark on {datasets}...")

    for ds in datasets:
        for loss_name in losses:
            logger.info(f"Tier 5 Natural Noise: [{ds}] | Loss: {loss_name} | Inherent Natural Noise")
            run_cross_validation(
                dataset_name=ds,
                model_name=loss_name,
                noise_type="none",
                noise_rate=0.0,
                seeds=seeds,
                n_folds=n_folds,
                results_path=out_csv,
            )

    res_df = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    logger.info(f"Tier 5 natural noise benchmark complete. Results saved to {out_csv}")
    return res_df


if __name__ == "__main__":
    run_tier5_natural_noise_experiments()
