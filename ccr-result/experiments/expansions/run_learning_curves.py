"""
run_learning_curves.py - Learning Curves for CCR and Key Baselines
====================================================================
Extracts per-epoch validation Macro F1 for:
    - mlp_ccr       (from existing JSON logs — full training)
    - mlp_standard  (trained fresh for curve capture)
    - mlp_weighted_ce (trained fresh for curve capture)

Condition: asym@30% noise, all 6 datasets, all seeds x folds.

CCR curves come from the JSON logs already written during the main run.
Baseline curves require a short training run because the baseline training
path (BaselineModel.fit) does not call RunLogger.log_epoch().

Output: outputs/metrics/learning_curves.csv
Columns: dataset, model, noise_type, noise_rate, epoch, mean_val_f1, std_val_f1, n_runs
Usage:   python experiments/expansions/run_learning_curves.py
"""

import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.load_data import load_dataset
from src.data.noise_injection import inject_asymmetric_noise
from src.data.preprocess import preprocess_split
from src.loss.ccr_loss import CCRLoss
from src.models.baselines import get_baseline
from src.models.mlp import TabularDataset, get_mlp_for_dataset
from src.utils.config import (
    BATCH_SIZE, BETA, DATASETS, EARLY_STOP_PATIENCE,
    K, LEARNING_RATE, MAX_EPOCHS, N_FOLDS,
    OUTPUTS_LOGS, OUTPUTS_METRICS, SEEDS, TAU, VAL_SIZE, WEIGHT_DECAY,
)
from src.utils.logger import setup_logging
from src.utils.reproducibility import fix_all_seeds, get_device

setup_logging(level=logging.WARNING)
logger = logging.getLogger(__name__)

NOISE_TYPE = "asym"
NOISE_RATE = 0.3
OUTPUT_CSV = OUTPUTS_METRICS / "learning_curves.csv"

# Models to capture curves for
CCR_MODEL    = "mlp_ccr"
CURVE_MODELS = ["mlp_ccr", "mlp_standard", "mlp_weighted_ce"]


# ── CCR: extract from existing JSON logs ─────────────────────────────────────

def load_epoch_f1_from_log(log_path: Path):
    """Load per-epoch val_macro_f1 from a JSON training log."""
    if not log_path.exists():
        return []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [(rec["epoch"], rec["val_macro_f1"]) for rec in data.get("epochs", [])]
    except Exception as exc:
        logger.warning(f"Could not parse {log_path.name}: {exc}")
        return []


def extract_ccr_curves(dataset_name, rate_str):
    """Extract CCR learning curves from existing JSON logs."""
    epoch_f1_map = {}
    for seed in SEEDS:
        for fold in range(1, N_FOLDS + 1):
            log_name = (
                f"{dataset_name}_{CCR_MODEL}_{NOISE_TYPE}_{rate_str}"
                f"_seed{seed}_fold{fold}_train.json"
            )
            epoch_f1_list = load_epoch_f1_from_log(OUTPUTS_LOGS / log_name)
            for epoch, f1 in epoch_f1_list:
                epoch_f1_map.setdefault(epoch, []).append(f1)
    return epoch_f1_map


# ── Baselines: train fresh and capture per-epoch F1 ──────────────────────────

def train_baseline_with_curves(
    model_name, dataset_name, X_tr, y_tr, X_val, y_val, seed
):
    """Train an MLP baseline and return per-epoch validation Macro F1.

    Args:
        model_name: 'mlp_standard' or 'mlp_weighted_ce'.
        dataset_name: Dataset key.
        X_tr: Training features.
        y_tr: Training labels (may be noisy).
        X_val: Validation features (always clean).
        y_val: Validation labels (always clean).
        seed: Random seed.

    Returns:
        List of (epoch, val_macro_f1) tuples.
    """
    fix_all_seeds(seed)
    device = get_device()

    model = get_mlp_for_dataset(dataset_name, X_tr.shape[1]).to(device)

    # Build loss function matching the baseline
    if model_name == "mlp_standard":
        import torch.nn as nn
        loss_fn = nn.CrossEntropyLoss()
    elif model_name == "mlp_weighted_ce":
        import torch.nn as nn
        n_majority = int(np.sum(y_tr == 0))
        n_minority = int(np.sum(y_tr == 1))
        weights = torch.tensor(
            [1.0 / n_majority, 1.0 / n_minority], dtype=torch.float32
        )
        weights = (weights / weights.sum()).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
    else:
        raise ValueError(f"Unsupported baseline for curve capture: {model_name}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loader = DataLoader(
        TabularDataset(X_tr, y_tr),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
    )

    best_f1 = -1.0
    patience = 0
    epoch_curves = []

    for epoch in range(MAX_EPOCHS):
        model.train()
        for X_b, y_b, _ in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            loss = loss_fn(logits, y_b)
            if torch.isnan(loss):
                break
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            X_v = torch.FloatTensor(X_val).to(device)
            preds = model(X_v).argmax(dim=1).cpu().numpy()
        val_f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))
        epoch_curves.append((epoch + 1, val_f1))

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                break

    return epoch_curves


def extract_baseline_curves(model_name, dataset_name, X, y, rate_str):
    """Train baseline across all seeds/folds and collect per-epoch F1."""
    epoch_f1_map = {}

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

            # Inject noise on train only
            y_tr_noisy, _ = inject_asymmetric_noise(y_tr_np, NOISE_RATE, seed)

            print(
                f"    {model_name} | seed={seed} fold={fold}",
                end="", flush=True,
            )
            try:
                curves = train_baseline_with_curves(
                    model_name, dataset_name,
                    X_tr_np, y_tr_noisy, X_val_np, y_val_np, seed,
                )
                for epoch, f1 in curves:
                    epoch_f1_map.setdefault(epoch, []).append(f1)
                print(f" — {len(curves)} epochs", flush=True)
            except Exception as exc:
                print(f" — FAILED: {exc}", flush=True)
                logger.error(f"Baseline curve failed: {exc}", exc_info=True)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    return epoch_f1_map


# ── Main ──────────────────────────────────────────────────────────────────────

def extract_learning_curves():
    rate_str = f"{int(NOISE_RATE * 100):02d}"
    all_datasets = list(DATASETS.keys())
    records = []

    print("=" * 65, flush=True)
    print("  Learning Curves Extraction", flush=True)
    print(f"  Models:    {CURVE_MODELS}", flush=True)
    print(f"  Condition: {NOISE_TYPE}@{NOISE_RATE:.0%}", flush=True)
    print(f"  Datasets:  {all_datasets}", flush=True)
    print("=" * 65, flush=True)

    for dataset_name in all_datasets:
        print(f"\n  Dataset: {dataset_name}", flush=True)

        df_data = load_dataset(dataset_name)
        feature_cols = [c for c in df_data.columns if c != "target"]
        X = df_data[feature_cols]
        y = df_data["target"].values

        for model_name in CURVE_MODELS:
            print(f"\n  Model: {model_name}", flush=True)

            if model_name == CCR_MODEL:
                # Extract from existing logs — no training needed
                epoch_f1_map = extract_ccr_curves(dataset_name, rate_str)
                if not epoch_f1_map:
                    print(f"    No log data found for {dataset_name}/{model_name}", flush=True)
                    continue
                print(f"    Extracted from logs: {len(epoch_f1_map)} epochs", flush=True)
            else:
                # Train fresh to capture per-epoch curves
                epoch_f1_map = extract_baseline_curves(
                    model_name, dataset_name, X, y, rate_str
                )
                if not epoch_f1_map:
                    print(f"    No curve data for {dataset_name}/{model_name}", flush=True)
                    continue

            for epoch in sorted(epoch_f1_map.keys()):
                vals = epoch_f1_map[epoch]
                records.append({
                    "dataset":     dataset_name,
                    "model":       model_name,
                    "noise_type":  NOISE_TYPE,
                    "noise_rate":  NOISE_RATE,
                    "epoch":       epoch,
                    "mean_val_f1": round(float(np.mean(vals)), 6),
                    "std_val_f1":  round(float(np.std(vals)), 6),
                    "n_runs":      len(vals),
                })

    if not records:
        print("\n  No data found.", flush=True)
        return

    df = pd.DataFrame(records)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n  Saved {len(df):,} rows to {OUTPUT_CSV}", flush=True)
    summary = df.groupby(["dataset", "model"])["epoch"].max().unstack(fill_value=0)
    print(f"\n  Max epochs per dataset/model:\n{summary.to_string()}", flush=True)


if __name__ == "__main__":
    extract_learning_curves()
