"""Tier 4: Multiclass Validation Experiments (Section R).

Evaluates:
  - 2 Multiclass tabular datasets (segment, steel_faults)
  - Loss functions: CE, WCE, Focal, GCE, SCE, CCR-NoNorm, CCR
  - Multiclass noise injection (Symmetric noise 10-30%)
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
from src.utils.config import MULTICLASS_DATASETS, OUTPUTS_METRICS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_TIER4_LOSSES = ["ce", "wce", "focal", "gce", "sce", "ccr_no_norm", "ccr"]


def run_tier4_multiclass_experiments(
    datasets: Optional[List[str]] = None,
    losses: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    n_folds: int = 3,
) -> pd.DataFrame:
    """Run Tier 4 multiclass transfer benchmark."""
    if datasets is None:
        datasets = MULTICLASS_DATASETS
    if losses is None:
        losses = _TIER4_LOSSES
    if seeds is None:
        seeds = [42, 123]

    out_csv = OUTPUTS_METRICS / "tier4_multiclass_results.csv"
    logger.info(f"Starting Tier 4 Multiclass Benchmark on {datasets}...")

    for ds in datasets:
        for loss_name in losses:
            for n_type, n_rate in [("none", 0.0), ("sym", 0.10), ("sym", 0.20), ("sym", 0.30)]:
                logger.info(
                    f"Tier 4 Multiclass: [{ds}] | Loss: {loss_name} | Noise: {n_type}@{n_rate:.0%}"
                )
                run_cross_validation(
                    dataset_name=ds,
                    model_name=loss_name,
                    noise_type=n_type,
                    noise_rate=n_rate,
                    seeds=seeds,
                    n_folds=n_folds,
                    results_path=out_csv,
                )

    res_df = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    logger.info(f"Tier 4 multiclass benchmark complete. Results saved to {out_csv}")
    return res_df


if __name__ == "__main__":
    run_tier4_multiclass_experiments()
