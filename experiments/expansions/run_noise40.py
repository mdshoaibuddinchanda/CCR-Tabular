"""
run_noise40.py â€” Noise@40% Extension Experiment
================================================
Runs ALL 8 models on ALL 6 datasets with asym@40% noise.
Extends the main experiment to a harder noise level.
5 folds x 3 seeds = 15 runs per (dataset, model).
Total: 8 models x 6 datasets x 15 runs = 720 runs.

Output: Appended to outputs/metrics/results.csv (same schema as main results).
Usage:  python run_noise40.py
"""

import gc
import logging
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.cross_validation import run_cross_validation
from src.utils.config import DATASETS, MODEL_NAMES, OUTPUTS_METRICS, SEEDS
from src.utils.logger import setup_logging

setup_logging(level=logging.WARNING)
logger = logging.getLogger(__name__)

NOISE_TYPE = "asym"
NOISE_RATE = 0.4
RESULTS_CSV = OUTPUTS_METRICS / "results.csv"


def all_runs_done(dataset_name: str, model_name: str) -> bool:
    """Check if all 15 runs for this dataset/model combo are already done."""
    if not RESULTS_CSV.exists():
        return False
    try:
        df = pd.read_csv(RESULTS_CSV)
        mask = (
            (df["dataset"] == dataset_name) &
            (df["model"] == model_name) &
            (df["noise_type"] == NOISE_TYPE) &
            (df["noise_rate"].round(2) == round(NOISE_RATE, 2))
        )
        return int(mask.sum()) >= len(SEEDS) * 5  # 3 seeds x 5 folds
    except Exception:
        return False


def run_noise40():
    all_datasets = list(DATASETS.keys())
    total_combos = len(all_datasets) * len(MODEL_NAMES)
    done = 0

    print("=" * 65, flush=True)
    print("  Noise@40% Extension Experiment", flush=True)
    print(f"  Datasets:     {all_datasets}", flush=True)
    print(f"  Models:       {MODEL_NAMES}", flush=True)
    print(f"  Noise:        asym@40%", flush=True)
    print(f"  Total combos: {total_combos} (each = 15 runs)", flush=True)
    print("=" * 65, flush=True)

    for dataset_name in all_datasets:
        for model_name in MODEL_NAMES:
            done += 1

            if all_runs_done(dataset_name, model_name):
                print(
                    f"  [{done}/{total_combos}] SKIP {dataset_name} | {model_name} (done)",
                    flush=True,
                )
                continue

            print(
                f"  [{done}/{total_combos}] {dataset_name} | {model_name} | asym@40%",
                flush=True,
            )

            try:
                results = run_cross_validation(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    noise_type=NOISE_TYPE,
                    noise_rate=NOISE_RATE,
                    seeds=SEEDS,
                    n_folds=5,
                )
                if len(results) > 0:
                    print(
                        f"    Done. macro_f1={results['macro_f1'].mean():.4f} "
                        f"({len(results)} runs)",
                        flush=True,
                    )

            except Exception as exc:
                logger.error(f"FAILED {dataset_name}/{model_name}: {exc}", exc_info=True)
                print(f"  ERROR: {dataset_name}/{model_name}: {exc}", flush=True)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    print(f"\n  Noise@40% complete. Results in {RESULTS_CSV}", flush=True)


if __name__ == "__main__":
    run_noise40()
