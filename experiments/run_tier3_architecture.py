"""Tier 3: Architecture Transferability Experiments (Section J).

Compares:
  - Architectures: TabularMLP, TabularResNet, TabularFTTransformer
  - Losses: Standard CE, Normalized CE/WCE, CCR-NoNorm, CCR
  - Datasets: 5 representative datasets spanning small to large scale
  - Verifies whether batch normalization provides consistent stabilization across modern architectures.
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
from src.utils.config import OUTPUTS_METRICS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_TIER3_DATASETS = ["credit_g", "spambase", "phoneme", "magic", "adult"]
_TIER3_ARCHITECTURES = ["mlp", "resnet", "ft_transformer"]
_TIER3_LOSSES = ["ce", "wce", "norm_wce", "ccr_no_norm", "ccr"]


def run_tier3_architecture_experiments(
    datasets: Optional[List[str]] = None,
    architectures: Optional[List[str]] = None,
    losses: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    n_folds: int = 3,
) -> pd.DataFrame:
    """Run Tier 3 architecture transferability benchmark."""
    if datasets is None:
        datasets = _TIER3_DATASETS
    if architectures is None:
        architectures = _TIER3_ARCHITECTURES
    if losses is None:
        losses = _TIER3_LOSSES
    if seeds is None:
        seeds = [42, 123]

    out_csv = OUTPUTS_METRICS / "tier3_architecture_transfer_results.csv"
    logger.info("Starting Tier 3 Architecture Transferability Benchmark...")

    for ds in datasets:
        for arch in architectures:
            for loss_name in losses:
                for n_type, n_rate in [("none", 0.0), ("asym", 0.20), ("asym", 0.40)]:
                    logger.info(
                        f"Tier 3: [{ds}] | Arch: {arch} | Loss: {loss_name} | Noise: {n_type}@{n_rate:.0%}"
                    )
                    run_cross_validation(
                        dataset_name=ds,
                        model_name=loss_name,
                        architecture=arch,
                        noise_type=n_type,
                        noise_rate=n_rate,
                        seeds=seeds,
                        n_folds=n_folds,
                        results_path=out_csv,
                    )

    res_df = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    logger.info(f"Tier 3 architecture transfer complete. Saved to {out_csv}")
    return res_df


if __name__ == "__main__":
    run_tier3_architecture_experiments()
