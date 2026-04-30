"""
run_noise40.py — Noise@40% Extension Experiment
================================================
Run ALL 8 models (CCR + 7 baselines) on ALL 6 datasets with asym@40% noise.
Extends the main experiment to a harder noise level.
5 folds × 3 seeds = 15 runs per (dataset, model).
Total: 8 models × 6 datasets × 15 runs = 720 runs.

Output: Appends to outputs/metrics/results.csv (same format as main results).
        noise_rate=0.4 in all rows.

Usage:
    python run_noise40.py
"""

import gc
import logging
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.training.cross_validation import run_cross_validation
from src.utils.config import DATASETS, MODEL_NAMES, OUTPUTS_METRICS, SEEDS
from src.utils.logger import setup_logging

setup_logging(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
NOISE_TYPE = "asym"
NOISE_RATE = 0.4
RESULTS_CSV = OUTPUTS_METRICS / "results.csv"


def is_done(run_id: str) -> bool:
    """Check if run_id already exists in results.csv."""
    if not RESULTS_CSV.exists():
        return False
    try:
        df = pd.read_csv(RESULTS_CSV)
        return run_id in df["run_id"].values
    except Exception:
        return False


def make_run_id(dataset_name, model_name, noise_type, noise_rate, seed, fold):
    rate_str = f"{int(noise_rate * 100):02d}"
    return f"{dataset_name}_{model_name}_{noise_type}_{rate_str}_seed{seed}_fold{fold}"


def run_noise40():
    total_combos = len(list(DATASETS.keys())) * len(MODEL_NAMES)
    done = 0

    print("=" * 65, flush=True)
    print(f"  Noise@40% Extension Experiment", flush=True)
    print(f"  Models:     {MODEL_NAMES}", flush=True)
    print(f"  Datasets:   {list(DATASETS.keys())}", flush=True)
    print(f"  Noise:      asym@40%", flush=True)
    print(f"  Total combos: {total_combos} (each = 15 runs)", flush=True)
    print("=" * 65, flush=True)

    for dataset_name in DATASETS.keys():
        for model_name in MODEL_NAMES:
            done += 1

            # Check if all 15 runs for this combo are already done
            all_done = True
            for seed in SEEDS:
                for fold in range(1, 6):
                    run_id = make_run_id(dataset_name, model_name, NOISE_TYPE, NOISE_RATE, seed, fold)
                    if not is_done(run_id):
                        all_done = False
                        break
                if not all_done:
                    break

            if all_done:
                print(
                    f"  [{done}/{total_combos}] SKIP {dataset_name} | {model_name} "
                    f"(all 15 runs done)",
                    flush=True,
                )
                continue

            print(
                f"  [{done}/{total_combos}] Running {dataset_name} | {model_name} | asym@40%",
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
                    mean_f1 = results["macro_f1"].mean()
                    print(
                        f"    Done. mean_macro_f1={mean_f1:.4f} "
                        f"({len(results)} runs)",
                        flush=True,
                    )
                else:
                    print(f"    Done (0 new runs — all already existed).", flush=True)

            except Exception as exc:
                logger.error(
                    f"FAILED {dataset_name}/{model_name}/asym@40%: {exc}",
                    exc_info=True,
                )
                print(f"  ERROR: {dataset_name}/{model_name}: {exc}", flush=True)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    print("\n" + "=" * 65, flush=True)
    print(f"  Noise@40% complete. Results appended to {RESULTS_CSV}", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    run_noise40()
