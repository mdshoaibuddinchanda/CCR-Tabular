"""
run_beta_sensitivity.py — CCR Beta Sensitivity Analysis
========================================================
Run CCR with beta ∈ {0.3, 0.5, 0.8} on ALL 6 datasets.
Conditions: asym@20% + asym@30% only.
5 folds × 3 seeds = 15 runs per (dataset, beta, condition).
Total: 3 beta × 6 datasets × 2 conditions × 15 runs = 540 runs.

Output: outputs/metrics/results_beta_sensitivity.csv
Columns: run_id, dataset, beta, noise_type, noise_rate, fold, seed,
         macro_f1, minority_recall, auc_roc, auc_pr, train_time_s

Usage:
    python run_beta_sensitivity.py
"""

import gc
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score, f1_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from src.data.load_data import load_dataset
from src.data.noise_injection import inject_asymmetric_noise
from src.data.preprocess import preprocess_split
from src.loss.ccr_loss import CCRLoss
from src.models.mlp import TabularDataset, get_mlp_for_dataset
from src.utils.config import (
    BATCH_SIZE, EARLY_STOP_PATIENCE, K, LEARNING_RATE,
    MAX_EPOCHS, OUTPUTS_METRICS, SEEDS, TAU, VAL_SIZE, WEIGHT_DECAY,
)
from src.utils.logger import setup_logging
from src.utils.reproducibility import fix_all_seeds, get_device

setup_logging(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
BETA_VALUES = [0.3, 0.5, 0.8]
DATASETS = ["adult", "bank", "magic", "phoneme", "credit_g", "spambase"]
NOISE_CONDITIONS = [
    ("asym", 0.2),
    ("asym", 0.3),
]
N_FOLDS = 5
OUTPUT_CSV = OUTPUTS_METRICS / "results_beta_sensitivity.csv"


def make_run_id(dataset, beta, noise_type, noise_rate, seed, fold):
    beta_str = str(beta).replace(".", "_")
    rate_str = f"{int(noise_rate * 100):02d}"
    return f"beta_{beta_str}_{dataset}_{noise_type}_{rate_str}_seed{seed}_fold{fold}"


def is_done(run_id):
    if not OUTPUT_CSV.exists():
        return False
    try:
        df = pd.read_csv(OUTPUT_CSV)
        return run_id in df["run_id"].values
    except Exception:
        return False


def append_row(row: dict):
    df_new = pd.DataFrame([row])
    if OUTPUT_CSV.exists():
        try:
            df_existing = pd.read_csv(OUTPUT_CSV)
            if row["run_id"] in df_existing["run_id"].values:
                return
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception:
            df_combined = df_new
    else:
        df_combined = df_new
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(OUTPUT_CSV, index=False)


def inject_noise(X_tr, y_tr, noise_type, noise_rate, seed):
    if noise_type == "none" or noise_rate == 0.0:
        return y_tr
    if noise_type == "asym":
        y_noisy, _ = inject_asymmetric_noise(y_tr, noise_rate, seed)
        return y_noisy
    return y_tr


def train_and_evaluate(dataset_name, beta, X_tr, y_tr, X_val, y_val, X_te, y_te, seed):
    """Train CCR with given beta, return metrics."""
    fix_all_seeds(seed)
    device = get_device()

    model = get_mlp_for_dataset(dataset_name, X_tr.shape[1]).to(device)

    n_majority = int(np.sum(y_tr == 0))
    n_minority = int(np.sum(y_tr == 1))

    criterion = CCRLoss(
        n_samples=len(y_tr),
        n_classes=2,
        class_counts=[n_majority, n_minority],
        tau=TAU,
        beta=beta,
        K=K,
        device=device,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    loader = DataLoader(
        TabularDataset(X_tr, y_tr),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
    )

    best_f1 = -1.0
    patience = 0
    best_state = None
    t0 = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        model.train()
        for X_b, y_b, idx_b in loader:
            X_b, y_b, idx_b = X_b.to(device), y_b.to(device), idx_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b, idx_b, epoch)
            if torch.isnan(loss):
                break
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                probs = F.softmax(logits.detach(), dim=1)
                criterion.update_history(probs, idx_b, epoch)

        # Validate
        model.eval()
        with torch.no_grad():
            X_v = torch.FloatTensor(X_val).to(device)
            preds = model(X_v).argmax(dim=1).cpu().numpy()
        val_f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)

    train_time = time.perf_counter() - t0

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_te).to(device)
        logits_te = model(X_t)
        probs_te = F.softmax(logits_te, dim=1).cpu().numpy()
    y_pred = probs_te.argmax(axis=1)
    y_prob = probs_te[:, 1]

    metrics = {
        "macro_f1":        float(f1_score(y_te, y_pred, average="macro", zero_division=0)),
        "minority_recall": float(recall_score(y_te, y_pred, pos_label=1, zero_division=0)),
        "auc_roc":         float(roc_auc_score(y_te, y_prob)) if len(np.unique(y_te)) > 1 else float("nan"),
        "auc_pr":          float(average_precision_score(y_te, y_prob)) if len(np.unique(y_te)) > 1 else float("nan"),
        "train_time_s":    round(train_time, 3),
    }
    return metrics


def run_beta_sensitivity():
    device = get_device()
    total_runs = len(BETA_VALUES) * len(DATASETS) * len(NOISE_CONDITIONS) * N_FOLDS * len(SEEDS)
    done = 0

    print("=" * 65, flush=True)
    print(f"  CCR Beta Sensitivity Analysis", flush=True)
    print(f"  Beta values: {BETA_VALUES}", flush=True)
    print(f"  Datasets:    {DATASETS}", flush=True)
    print(f"  Total runs:  {total_runs}", flush=True)
    print(f"  Device:      {device}", flush=True)
    print("=" * 65, flush=True)

    for dataset_name in DATASETS:
        print(f"\nLoading {dataset_name}...", flush=True)
        df_data = load_dataset(dataset_name)
        feature_cols = [c for c in df_data.columns if c != "target"]
        X = df_data[feature_cols]
        y = df_data["target"].values

        for noise_type, noise_rate in NOISE_CONDITIONS:
            for seed in SEEDS:
                skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

                for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                    fold = fold_idx + 1

                    X_tr_df = X.iloc[train_idx].reset_index(drop=True)
                    X_te_df = X.iloc[test_idx].reset_index(drop=True)
                    y_tr_raw = y[train_idx]
                    y_te_raw = y[test_idx]

                    X_tr_df2, X_val_df, y_tr2, y_val = train_test_split(
                        X_tr_df, y_tr_raw,
                        test_size=VAL_SIZE, stratify=y_tr_raw, random_state=seed,
                    )

                    (X_tr_np, X_val_np, X_te_np,
                     y_tr_np, y_val_np, y_te_np, _) = preprocess_split(
                        X_tr_df2, X_val_df, X_te_df,
                        pd.Series(y_tr2), pd.Series(y_val), pd.Series(y_te_raw),
                    )

                    y_tr_noisy = inject_noise(X_tr_np, y_tr_np, noise_type, noise_rate, seed)

                    for beta in BETA_VALUES:
                        run_id = make_run_id(dataset_name, beta, noise_type, noise_rate, seed, fold)

                        if is_done(run_id):
                            done += 1
                            continue

                        print(
                            f"  [{done+1}/{total_runs}] "
                            f"{dataset_name} | beta={beta} | "
                            f"{noise_type}@{noise_rate:.0%} | "
                            f"seed={seed} fold={fold}",
                            flush=True,
                        )

                        try:
                            metrics = train_and_evaluate(
                                dataset_name, beta,
                                X_tr_np, y_tr_noisy,
                                X_val_np, y_val_np,
                                X_te_np, y_te_np,
                                seed,
                            )

                            append_row({
                                "run_id":          run_id,
                                "dataset":         dataset_name,
                                "beta":            beta,
                                "noise_type":      noise_type,
                                "noise_rate":      noise_rate,
                                "fold":            fold,
                                "seed":            seed,
                                "macro_f1":        metrics["macro_f1"],
                                "minority_recall": metrics["minority_recall"],
                                "auc_roc":         metrics["auc_roc"],
                                "auc_pr":          metrics["auc_pr"],
                                "train_time_s":    metrics["train_time_s"],
                            })

                            print(
                                f"    macro_f1={metrics['macro_f1']:.4f}  "
                                f"recall={metrics['minority_recall']:.4f}  "
                                f"time={metrics['train_time_s']:.1f}s",
                                flush=True,
                            )

                        except Exception as exc:
                            logger.error(f"FAILED {run_id}: {exc}", exc_info=True)
                            print(f"  ERROR: {run_id}: {exc}", flush=True)

                        done += 1

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()

    print("\n" + "=" * 65, flush=True)
    print(f"  Beta sensitivity complete. Results: {OUTPUT_CSV}", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    run_beta_sensitivity()
