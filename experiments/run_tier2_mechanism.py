"""Tier 2: Direct Mechanism Verification & Gradient Instrumentation (Section B).

Directly measures the causal chain:
  noise concentration -> S/B inflation -> gradient-scale variation -> update norm variation

Compares:
  - Methods: CE, WCE, Focal, CCR-NoNorm, CCR (Normalized)
  - Noise conditions: Clean, Asym@10%, Asym@20%, Asym@30%, Asym@40%
  - Datasets: 6 representative datasets
  - Instruments every batch:
    * S/B (weight sum inflation)
    * grad_norm_weighted vs grad_norm_unweighted
    * grad_cosine_sim (gradient direction alignment)
    * param_update_norm ||Delta theta_t||_2
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.training.cross_validation import run_cross_validation
from src.utils.config import (
    OUTPUTS_METRICS,
    OUTPUTS_PLOTS,
    OUTPUTS_TELEMETRY,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_tier2_mechanism_experiments(
    datasets: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    noise_rates: Optional[List[float]] = None,
    seeds: Optional[List[int]] = None,
    n_folds: int = 3,
) -> pd.DataFrame:
    """Run Tier 2 mechanism experiments with full batch instrumentation."""
    if datasets is None:
        datasets = ["credit_g", "spambase", "phoneme", "magic", "adult", "bank"]
    if models is None:
        models = ["ce", "wce", "focal", "ccr_no_norm", "ccr"]
    if noise_rates is None:
        noise_rates = [0.0, 0.10, 0.20, 0.30, 0.40]
    if seeds is None:
        seeds = [42]

    out_csv = OUTPUTS_METRICS / "tier2_mechanism_results.csv"
    logger.info(f"Starting Tier 2 Mechanism Validation across {len(datasets)} datasets...")

    for ds in datasets:
        for model_name in models:
            for rate in noise_rates:
                noise_type = "none" if rate == 0.0 else "asym"
                logger.info(
                    f"Mechanism Run: [{ds}] | Model: {model_name} | Noise: {noise_type}@{rate:.0%}"
                )
                run_cross_validation(
                    dataset_name=ds,
                    model_name=model_name,
                    noise_type=noise_type,
                    noise_rate=rate,
                    seeds=seeds,
                    n_folds=n_folds,
                    instrument_batch=True,
                    results_path=out_csv,
                )

    res_df = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    logger.info(f"Tier 2 mechanism runs complete. Results saved to {out_csv}")
    return res_df


def aggregate_and_plot_mechanism_dynamics() -> None:
    """Load telemetry files and plot S/B and update norm dynamics across training."""
    telemetry_files = list(OUTPUTS_TELEMETRY.glob("*_telemetry.csv"))
    if not telemetry_files:
        logger.warning("No telemetry files found to plot.")
        return

    dfs = [pd.read_csv(f) for f in telemetry_files]
    all_telemetry = pd.concat(dfs, ignore_index=True)

    summary = (
        all_telemetry.groupby(["epoch"])
        .agg({
            "S_over_B": ["mean", "std", "max"],
            "param_update_norm": ["mean", "std"],
            "grad_norm_weighted": ["mean", "std"],
            "grad_cosine_sim": ["mean"],
        })
        .reset_index()
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(summary["epoch"], summary[("S_over_B", "mean")], "r-", label="Mean S/B")
    ax1.fill_between(
        summary["epoch"],
        summary[("S_over_B", "mean")] - summary[("S_over_B", "std")],
        summary[("S_over_B", "mean")] + summary[("S_over_B", "std")],
        color="r", alpha=0.2, label="+/- 1 std",
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Weight Sum Inflation (S / B)")
    ax1.set_title("Evolution of Weight-Sum Inflation (S/B) During Training")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(summary["epoch"], summary[("param_update_norm", "mean")], "b-", label="Mean ||Delta theta||_2")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Parameter Update Norm")
    ax2.set_title("Parameter Update Norm Under Normalized Optimization")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plot_path = OUTPUTS_PLOTS / "mechanism_dynamics_training.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved mechanism dynamics plot to {plot_path}")


if __name__ == "__main__":
    run_tier2_mechanism_experiments()
    aggregate_and_plot_mechanism_dynamics()
