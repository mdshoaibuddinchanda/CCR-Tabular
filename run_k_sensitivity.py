"""
run_k_sensitivity.py — CCR K (History Window) Sensitivity Analysis
===================================================================
Runs CCR with K in {3, 5, 10} on ALL 6 datasets.
Conditions: clean + asym@20% + asym@30%.
5 folds x 3 seeds = 15 runs per (dataset, K, condition).
Total: 3 K x 6 datasets x 3 conditions x 15 runs = 810 runs.

Output: outputs/metrics/results_k_sensitivity.csv
Usage:  python run_k_sensitivity.py
"""

import gc
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.data.load_data import load_dataset
from src.loss.ccr_loss import CCRLoss
from src.utils.config import (
    BETA, DATASETS, N_FOLDS, OUTPUTS_METRICS, SEEDS, TAU,
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
K_VALUES = [3, 5, 10]
K_DATASETS = list(DATASETS.keys())  # all 6
NOISE_CONDITIONS = [
    ("none", 0.0),
    ("asym", 0.2),
    ("asym", 0.3),
]
OUTPUT_CSV = OUTPUTS_METRICS / "results_k_sensitivity.csv"


def make_run_id(dataset, K, noise_type, noise_rate, seed, fold):
    rate_str = f"{int(noise_rate * 100):02d}"
    return f"k{K}_{dataset}_{noise_type}_{rate_str}_seed{seed}_fold{fold}"


def run_k_sensitivity():
    device = get_device()
    total = len(K_VALUES) * len(K_DATASETS) * len(NOISE_CONDITIONS) * N_FOLDS * len(SEEDS)
    done = 0

    print("=" * 65, flush=True)
    print("  CCR K Sensitivity Analysis", flush=True)
    print(f"  K values:   {K_VALUES}", flush=True)
    print(f"  Datasets:   {K_DATASETS}", flush=True)
    print(f"  Total runs: {total}", flush=True)
    print(f"  Device:     {device}", flush=True)
    print("=" * 65, flush=True)

    for dataset_name in K_DATASETS:
        print(f"\nLoading {dataset_name}...", flush=True)
        df_data = load_dataset(dataset_name)
        feature_cols = [c for c in df_data.columns if c != "target"]
        X = df_data[feature_cols]
        y = df_data["target"].values

        for noise_type, noise_rate in NOISE_CONDITIONS:
            for seed in SEEDS:
                fold_splits = make_fold_splits(X, y, seed, N_FOLDS)

                for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
                    fold = fold_idx + 1

                    X_tr, X_val, X_te, y_tr, y_val, y_te = prepare_fold(
                        X, y, train_idx, test_idx, seed, noise_type, noise_rate
                    )

                    n_majority = int(np.sum(y_tr == 0))
                    n_minority = int(np.sum(y_tr == 1))

                    for K in K_VALUES:
                        run_id = make_run_id(dataset_name, K, noise_type, noise_rate, seed, fold)

                        if is_done(run_id, OUTPUT_CSV):
                            done += 1
                            continue

                        print(
                            f"  [{done + 1}/{total}] {dataset_name} | K={K} | "
                            f"{noise_type}@{noise_rate:.0%} | seed={seed} fold={fold}",
                            flush=True,
                        )

                        try:
                            criterion = CCRLoss(
                                n_samples=len(y_tr),
                                n_classes=2,
                                class_counts=[n_majority, n_minority],
                                tau=TAU,
                                beta=BETA,
                                K=K,
                                device=device,
                            )

                            model, train_time, _ = train_ccr_fold(
                                dataset_name, X_tr, y_tr, X_val, y_val, seed, criterion
                            )
                            metrics = evaluate_model(model, X_te, y_te)

                            append_row({
                                "run_id":          run_id,
                                "dataset":         dataset_name,
                                "K":               K,
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

    print(f"\n  K sensitivity complete. Results: {OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    run_k_sensitivity()
