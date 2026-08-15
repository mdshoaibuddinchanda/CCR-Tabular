"""Tier 1: Core 10-Dataset Master Benchmark across 10 Loss Functions (10+2+2 Design).

10 Core Binary Datasets:
  1. Adult (N=48,842, IR=3.17) — Large, mixed features, standard benchmark
  2. Bank (N=45,211, IR=7.55) — Large scale + strong imbalance
  3. MAGIC (N=19,020, IR=1.84) — Medium scale, continuous gamma physics
  4. Phoneme (N=5,404, IR=2.41) — Medium scale, acoustics / signal
  5. Spambase (N=4,601, IR=1.54) — Continuous text-derived features
  6. Credit-G (N=1,000, IR=2.33) — Small mixed tabular finance
  7. Churn (N=10,000, IR=3.90) — Customer behavior with moderate-to-high imbalance
  8. Electricity (N=45,312, IR=1.35) — Large-scale, energy domain
  9. WILT (N=4,839, IR=17.50) — Severe/extreme imbalance regime
  10. Ionosphere (N=351, IR=1.79) — Low-N stress test

Evaluates:
  - 10 Losses: CE, WCE, Focal, GCE, SCE, ELR, Norm-WCE, Norm-Focal, Norm-GCE, Norm-SCE, CCR-NoNorm, CCR
  - Noise conditions: Clean, Asymmetric (10-40%), Symmetric (10-40%), Feature-Correlated (10-30%), IDN (10-30%)
  - Full Benjamini-Hochberg FDR correction and Cohen's d effect sizes.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.training.cross_validation import run_cross_validation
from src.utils.config import (
    CORE_10_DATASETS,
    LOSS_NAMES,
    N_FOLDS,
    OUTPUTS_METRICS,
    SEEDS,
)
from src.utils.statistics import analyze_dataset_level_significance

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_tier1_benchmark(
    datasets: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    noise_configs: Optional[List[tuple]] = None,
    seeds: Optional[List[int]] = None,
    n_folds: int = 3,
) -> pd.DataFrame:
    """Run core 10-dataset master benchmark."""
    if datasets is None:
        datasets = CORE_10_DATASETS
    if models is None:
        models = [
            "ce", "wce", "focal", "gce", "sce", "elr",
            "ccr_no_norm", "ccr",
        ]
    if noise_configs is None:
        noise_configs = [
            ("none", 0.0),
            ("asym", 0.20),
            ("asym", 0.40),
            ("sym", 0.20),
        ]
    if seeds is None:
        seeds = [42, 123]

    out_csv = OUTPUTS_METRICS / "tier1_core10_results.csv"
    logger.info(f"Starting Tier 1 Master Benchmark on 10 Core Datasets: {datasets}...")

    for ds in datasets:
        for model_name in models:
            for n_type, n_rate in noise_configs:
                logger.info(
                    f"Tier 1: [{ds}] | Loss: {model_name} | Noise: {n_type}@{n_rate:.0%}"
                )
                run_cross_validation(
                    dataset_name=ds,
                    model_name=model_name,
                    noise_type=n_type,
                    noise_rate=n_rate,
                    seeds=seeds,
                    n_folds=n_folds,
                    results_path=out_csv,
                )

    res_df = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()

    if len(res_df) > 0:
        sig_df = analyze_dataset_level_significance(
            res_df,
            primary_model="ccr",
            baseline_models=[m for m in models if m != "ccr"],
        )
        sig_path = OUTPUTS_METRICS / "tier1_statistical_significance_fdr.csv"
        sig_df.to_csv(sig_path, index=False)
        logger.info(f"Saved FDR-corrected statistical analysis to {sig_path}")

    return res_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tier 1 Master Benchmark.")
    parser.add_argument("--dataset", type=str, default=None, help="Run single dataset.")
    parser.add_argument("--model", type=str, default=None, help="Run single model/loss.")
    parser.add_argument("--n_folds", type=int, default=N_FOLDS)
    args = parser.parse_args()

    ds_list = [args.dataset] if args.dataset else None
    model_list = [args.model] if args.model else None
    run_tier1_benchmark(datasets=ds_list, models=model_list, n_folds=args.n_folds)
