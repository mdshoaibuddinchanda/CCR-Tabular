"""
run_ablation.py — CCR Ablation Study Runner
============================================
Runs 4 CCR variants on ALL 6 datasets across all 7 noise configs.
Results appended to outputs/metrics/results_ablation.csv

Variants:
    ccr_full      — Full CCR (focal + variance gate + batch norm)
    ccr_no_gate   — No confidence gate (variance applied to all samples)
    ccr_no_var    — No variance term (focal + class weight only)
    ccr_no_norm   — No batch normalization (raw weights used directly)

Usage:
    python run_ablation.py
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from src.data.load_data import load_dataset
from src.data.noise_injection import inject_asymmetric_noise, inject_feature_correlated_noise
from src.data.preprocess import preprocess_split
from src.loss.ccr_loss import get_ccr_loss
from src.models.mlp import TabularDataset, get_mlp_for_dataset
from src.utils.config import (
    BATCH_SIZE, BETA, EARLY_STOP_PATIENCE, K, LEARNING_RATE,
    MAX_EPOCHS, OUTPUTS_METRICS, SEEDS, TAU, VAL_SIZE, WEIGHT_DECAY,
)
from src.utils.logger import setup_logging
from src.utils.reproducibility import fix_all_seeds, get_device

setup_logging(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
ABLATION_DATASETS = ["adult", "bank", "magic", "phoneme", "credit_g", "spambase"]
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
N_FOLDS = 5
OUTPUT_CSV = OUTPUTS_METRICS / "results_ablation.csv"


def make_ablation_run_id(dataset, variant, noise_type, noise_rate, seed, fold):
    rate_str = f"{int(noise_rate * 100):02d}"
    return f"abl_{dataset}_{variant}_{noise_type}_{rate_str}_seed{seed}_fold{fold}"


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
    if noise_type == "feat":
        y_noisy, _ = inject_feature_correlated_noise(X_tr, y_tr, noise_rate, seed)
        return y_noisy
    return y_tr


def train_ablation_fold(
    dataset_name, variant, X_tr, y_tr, X_val, y_val, seed
):
    """Train one CCR ablation variant on one fold."""
    from sklearn.metrics import f1_score, recall_score, accuracy_score

    fix_all_seeds(seed)
    device = get_device()

    model = get_mlp_for_dataset(dataset_name, X_tr.shape[1]).to(device)

    n_majority = int(np.sum(y_tr == 0))
    n_minority = int(np.sum(y_tr == 1))

    criterion = get_ccr_loss(
        variant=VARIANT_MAP[variant],
        n_samples=len(y_tr),
        n_classes=2,
        class_counts=[n_majority, n_minority],
        device=device,
    )

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
    return model, train_time


def evaluate_fold(model, X_test, y_test, device):
    from sklearn.metrics import (
        accuracy_score, average_precision_score,
        f1_score, recall_score, roc_auc_score,
    )
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test).to(device)
        logits = model(X_t)
        probs = F.softmax(logits, dim=1).cpu().numpy()
    y_pred = probs.argmax(axis=1)
    y_prob = probs[:, 1]

    return {
        "accuracy":        float(accuracy_score(y_test, y_pred)),
        "macro_f1":        float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "minority_recall": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "auc_roc":         float(roc_auc_score(y_test, y_prob)),
        "auc_pr":          float(average_precision_score(y_test, y_prob)),
    }


def run_ablation():
    device = get_device()
    total_runs = (
        len(ABLATION_DATASETS) * len(ABLATION_VARIANTS) *
        len(NOISE_CONFIGS) * N_FOLDS * len(SEEDS)
    )
    done = 0

    print("=" * 60)
    print(f"  CCR Ablation Study")
    print(f"  Datasets:  {ABLATION_DATASETS}")
    print(f"  Variants:  {ABLATION_VARIANTS}")
    print(f"  Total runs: {total_runs}  (6 datasets x 4 variants x 7 configs x 5 folds x 3 seeds)")
    print(f"  Device:    {device}")
    print("=" * 60)

    for dataset_name in ABLATION_DATASETS:
        print(f"\nLoading {dataset_name}...")
        df_data = load_dataset(dataset_name)
        feature_cols = [c for c in df_data.columns if c != "target"]
        X = df_data[feature_cols]
        y = df_data["target"].values

        for noise_type, noise_rate in NOISE_CONFIGS:
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

                    for variant in ABLATION_VARIANTS:
                        run_id = make_ablation_run_id(
                            dataset_name, variant, noise_type, noise_rate, seed, fold
                        )

                        if is_done(run_id):
                            done += 1
                            continue

                        print(
                            f"  [{done+1}/{total_runs}] "
                            f"{dataset_name} | {variant} | "
                            f"{noise_type}@{noise_rate:.0%} | "
                            f"seed={seed} fold={fold}",
                            flush=True,
                        )

                        try:
                            model, train_time = train_ablation_fold(
                                dataset_name, variant,
                                X_tr_np, y_tr_noisy,
                                X_val_np, y_val_np,
                                seed,
                            )
                            metrics = evaluate_fold(model, X_te_np, y_te_np, device)

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
                            })

                            print(
                                f"    macro_f1={metrics['macro_f1']:.4f}  "
                                f"recall={metrics['minority_recall']:.4f}  "
                                f"time={train_time:.1f}s",
                                flush=True,
                            )

                        except Exception as exc:
                            logger.error(f"FAILED {run_id}: {exc}", exc_info=True)

                        done += 1

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()

    print("\n" + "=" * 60)
    print(f"  Ablation complete. Results: {OUTPUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    run_ablation()
