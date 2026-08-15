"""Tier 5: Real-World Clinical External Validation.

Evaluates:
  - 2 Real-world clinical tabular datasets: Heart Disease (N=462) & Breast Cancer (N=286)
  - Inherent real-world annotation ambiguity without synthetic corruption
  - Canonical 10 Losses: CE, WCE, Norm-WCE, Focal, Norm-Focal, GCE, SCE, ELR, CCR-NoNorm, CCR
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
from src.utils.config import LOSS_NAMES, OUTPUTS_METRICS, REAL_WORLD_DATASETS, SEEDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Tier5RealWorld")


def run_tier5_natural_noise_experiments(
    datasets: Optional[List[str]] = None,
    losses: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    n_folds: int = 5,
) -> pd.DataFrame:
    """Run Tier 5 real-world external validation benchmark."""
    if datasets is None:
        datasets = REAL_WORLD_DATASETS
    if losses is None:
        losses = LOSS_NAMES
    if seeds is None:
        seeds = SEEDS

    out_csv = OUTPUTS_METRICS / "tier5_real_world_external_results.csv"
    logger.info(f"Starting Tier 5 Real-World External Validation on {datasets}...")

    for ds in datasets:
        for loss_name in losses:
            logger.info(f"Tier 5 Real-World: [{ds}] | Loss: {loss_name} | Inherent Annotation Ambiguity")
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
    return res_df


if __name__ == "__main__":
    run_tier5_natural_noise_experiments()
