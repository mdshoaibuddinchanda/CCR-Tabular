"""
run_ablation.py — CCR Ablation Study
=====================================
Runs 4 CCR variants on ALL 6 datasets across all 7 noise configs.
5 folds x 3 seeds = 15 runs per (dataset, variant, condition).
Total: 4 variants x 6 datasets x 7 configs x 15 runs = 2,520 runs.

Variants:
    ccr_full    — Full CCR (focal + variance gate + batch norm)
    ccr_no_gate — No confidence gate (variance applied to all samples)
    ccr_no_var  — No variance term (focal + class weight only)
    ccr_no_norm — No batch normalization (raw weights used directly)

Output: outputs/metrics/results_ablation.csv
Usage:  python run_ablation.py
"""

import gc
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.data.load_data import load_dataset
from src.loss.ccr_loss import get_ccr_loss
from src.utils.config import (
    DATASETS, N_FOLDS, OUTPUTS_METRICS, SEEDS,
)
from src.utils.experiment_utils import (
    append_row, evaluate_model, is_done,
    make_fold_splits, prepare_fold, train_ccr_fold,
)
from src.utils.logger import setup_logging
from src.utils.reproducibility import get_device

setup_logging(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
ABLATION_DATASETS = list(DATASETS.keys())  # all 6
ABLATION_VARIANTS = ["ccr_full", "ccr_no_gate", "ccr_no_var", "ccr_no_norm"]
VARIANT_MAP = {
    "ccr_full":    "ccr",
    "ccr_no_gate": "ccr_no_gate",
    "ccr_no_var":  "ccr_no_variance",
    "ccr_no_norm": "ccr_no_norm",
}
NOISE_CONFIGS = [
    ("none", 0.0),
    ("asym", 0.1), ("asym", 0.2), ("asym", 0.3),
    ("feat", 0.1), ("feat", 0.2), ("feat", 0.3),
]
OUTPUT_CSV = OUTPUTS_METRICS / "results_ablation.csv"


def make_run_id(dataset, variant, noise_type, noise_rate, seed, fold):
    rate_str = f"{int(noise_rate * 100):02d}"
    return f"abl_{dataset}_{variant}_{noise_type}_{rate_str}_seed{seed}_fold{fold}"


def run_ablation():
    device = get_device()
    total = len(ABLATION_DATASETS) * len(ABLATION_VARIANTS) * len(NOISE_CONFIGS) * N_FOLDS * len(SEEDS)
    done = 0

    print("=" * 65, flush=True)
    print("  CCR Ablation Study", flush=True)
    print(f"  Datasets:   {ABLATION_DATASETS}", flush=True)
    print(f"  Variants:   {ABLATION_VARIANTS}", flush=True)
    print(f"  Total runs: {total}", flush=True)
    print(f"  Device:     {device}", flush=True)
    print("=" * 65, flush=True)

    for dataset_name in ABLATION_DATASETS:
        print(f"\nLoading {dataset_name}...", flush=True)
        df_data = load_dataset(dataset_name)
        feature_cols = [c for c in df_data.columns if c != "target"]
        X = df_data[feature_cols]
        y = df_data["target"].values

        for noise_type, noise_rate in NOISE_CONFIGS:
            for seed in SEEDS:
                fold_splits = make_fold_splits(X, y, seed, N_FOLDS)

                for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
                    fold = fold_idx + 1

                    X_tr, X_val, X_te, y_tr, y_val, y_te = prepare_fold(
                        X, y, train_idx, test_idx, seed, noise_type, noise_rate
                    )

                    n_majority = int(np.sum(y_tr == 0))
                    n_minority = int(np.sum(y_tr == 1))

                    for variant in ABLATION_VARIANTS:
                        run_id = make_run_id(dataset_name, variant, noise_type, noise_rate, seed, fold)

                        if is_done(run_id, OUTPUT_CSV):
                            done += 1
                            continue

                        print(
                            f"  [{done + 1}/{total}] {dataset_name} | {variant} | "
                            f"{noise_type}@{noise_rate:.0%} | seed={seed} fold={fold}",
                            flush=True,
                        )

                        try:
                            criterion = get_ccr_loss(
                                variant=VARIANT_MAP[variant],
                                n_samples=len(y_tr),
                                n_classes=2,
                                class_counts=[n_majority, n_minority],
                                device=device,
                            )

                            model, train_time, _ = train_ccr_fold(
                                dataset_name, X_tr, y_tr, X_val, y_val, seed, criterion
                            )
                            metrics = evaluate_model(model, X_te, y_te)

                            append_row({
                                "run_id":          run_id,
                                "dataset":         dataset_name,
                                "variant":         variant,
                                "noise_type":      noise_type,
                                "noise_rate":      noise_rate,
                                "fold":            fold,
                                "seed":            seed,
                                "train_time_s":    round(train_time, 3),
                                **metrics,
                            }, OUTPUT_CSV)

                            print(
                                f"    macro_f1={metrics['macro_f1']:.4f}  "
                                f"recall={metrics['minority_recall']:.4f}  "
                                f"time={train_time:.1f}s",
                                flush=True,
                            )

                        except Exception as exc:
                            logger.error(f"FAILED {run_id}: {exc}", exc_info=True)
                            print(f"  ERROR: {run_id}: {exc}", flush=True)

                        done += 1

                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()

    print(f"\n  Ablation complete. Results: {OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    run_ablation()
