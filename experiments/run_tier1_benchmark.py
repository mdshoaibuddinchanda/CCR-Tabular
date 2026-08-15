"""Tier 1: Core 10-Dataset Master Benchmark across 10 Loss Functions (10+2+2 Design).

10 Core Binary Datasets:
  1. Adult (N=48,842, IR=3.17) — Large, mixed features, standard census benchmark
  2. Bank (N=45,211, IR=7.55) — Large scale + strong marketing imbalance
  3. MAGIC (N=19,020, IR=1.84) — Medium scale, continuous gamma physics
  4. Phoneme (N=5,404, IR=2.41) — Medium scale, acoustics / speech signal
  5. Spambase (N=4,601, IR=1.54) — Continuous text-derived features
  6. Credit-G (N=1,000, IR=2.33) — Small mixed tabular finance
  7. Churn (N=5,000, IR=6.07) — Customer behavior with moderate-to-high imbalance
  8. Electricity (N=45,312, IR=1.36) — Large-scale, energy demand
  9. WILT (N=4,839, IR=17.54) — Severe/extreme imbalance regime
  10. Ionosphere (N=351, IR=1.79) — Low-N aerospace radar stress test

Evaluates:
  - 10 Canonical Losses: CE, WCE, Norm-WCE, Focal, Norm-Focal, GCE, SCE, ELR, CCR-NoNorm, CCR
  - 4 Primary Noise Conditions: Clean (0%), Asymmetric 20%, Asymmetric 40%, Symmetric 20%
  - Full Benjamini-Hochberg FDR correction and Cohen's d effect sizes with Dataset as the primary observational unit.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.utils.config import (
    CORE_10_DATASETS,
    LOSS_NAMES,
    N_FOLDS,
    OUTPUTS_FINAL_MASTER,
    OUTPUTS_METRICS,
    SEEDS,
)
from src.utils.manifest import generate_experiment_manifest
from src.utils.statistics import analyze_dataset_level_significance

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Tier1Benchmark")


def run_tier1_benchmark(
    datasets: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    noise_configs: Optional[List[tuple]] = None,
    seeds: Optional[List[int]] = None,
    n_folds: int = N_FOLDS,
    fast_mode: bool = True,
    device_override: str = "auto",
    results_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Run core 10-dataset master benchmark using heterogeneous scheduling into final_master namespace."""
    from main import HeterogeneousJobScheduler, JobDescriptor

    if datasets is None:
        datasets = CORE_10_DATASETS
    if models is None:
        models = LOSS_NAMES  # Canonical 10 losses
    if noise_configs is None:
        noise_configs = [
            ("none", 0.0),
            ("asym", 0.20),
            ("asym", 0.40),
            ("sym", 0.20),
        ]
    if seeds is None:
        seeds = SEEDS

    # Ensure clean final_master output namespace and write provenance manifest
    out_csv = Path(results_path) if results_path else (OUTPUTS_FINAL_MASTER / "metrics" / "tier1_core10_results.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    generate_experiment_manifest(OUTPUTS_FINAL_MASTER)

    from src.data.load_data import ensure_all_datasets_cached
    ensure_all_datasets_cached(datasets)

    logger.info(f"Starting Tier 1 Master Benchmark on 10 Core Datasets: {datasets}...")
    logger.info(f"Target Output CSV: {out_csv}")

    scheduler = HeterogeneousJobScheduler(device_override=device_override, fast_mode=fast_mode)
    jobs: List[JobDescriptor] = []

    for ds in datasets:
        for model_name in models:
            for n_type, n_rate in noise_configs:
                jobs.append(
                    JobDescriptor(
                        dataset=ds,
                        model=model_name,
                        noise_type=n_type,
                        noise_rate=n_rate,
                        architecture="mlp",
                        seeds=seeds,
                        n_folds=n_folds,
                        instrument_batch=False,
                        results_path=str(out_csv),
                        tier_name="tier1",
                    )
                )

    scheduler.run_jobs_heterogeneous(jobs)
    scheduler.print_final_summary()

    res_df = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()

    if len(res_df) > 0:
        sig_df = analyze_dataset_level_significance(
            res_df,
            primary_model="ccr",
            baseline_models=[m for m in models if m != "ccr"],
        )
        sig_path = out_csv.parent / "tier1_statistical_significance_fdr.csv"
        sig_df.to_csv(sig_path, index=False)
        logger.info(f"Saved FDR-corrected statistical analysis to {sig_path}")

        # Automatically consolidate canonical store and generate all paper figures
        try:
            from src.analysis.generate_canonical_results import build_canonical_results_store
            from src.analysis.generate_paper_figures import generate_all_figures
            build_canonical_results_store()
            generate_all_figures()
            logger.info("[FIGURES & TABLES READY] All publication figures and canonical tables generated automatically.")
        except Exception as e_fig:
            logger.warning(f"Could not auto-generate figures: {e_fig}")

    return res_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tier 1 Master Benchmark.")
    parser.add_argument("--dataset", type=str, default=None, help="Run single dataset.")
    parser.add_argument("--model", type=str, default=None, help="Run single model/loss.")
    parser.add_argument("--n_folds", type=int, default=N_FOLDS)
    parser.add_argument("--device", type=str, default="auto", help="Execution device.")
    parser.add_argument("--safe", action="store_true", help="Run in safe debugging mode.")
    args = parser.parse_args()

    ds_list = [args.dataset] if args.dataset else None
    model_list = [args.model] if args.model else None
    run_tier1_benchmark(
        datasets=ds_list,
        models=model_list,
        n_folds=args.n_folds,
        fast_mode=not args.safe,
        device_override=args.device,
    )
