"""
run_learning_curves.py — Extract Learning Curves from Existing Logs
====================================================================
For ALL 6 datasets, extract per-epoch validation Macro F1 for:
    CCR (mlp_ccr), MLP-CE (mlp_standard), MLP-WCE (mlp_weighted_ce), XGBoost-W (xgboost_weighted)
Under asym@30% noise, seed=42, all 5 folds.

Reads from: outputs/logs/*_asym_30_seed42_fold*_train.json
Averages across folds.

Output: outputs/metrics/learning_curves.csv
Columns: dataset, model, epoch, mean_val_f1, std_val_f1

Usage:
    python run_learning_curves.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import OUTPUTS_LOGS, OUTPUTS_METRICS
from src.utils.logger import setup_logging

setup_logging(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DATASETS = ["adult", "bank", "magic", "phoneme", "credit_g", "spambase"]
MODELS = {
    "mlp_ccr":          "CCR",
    "mlp_standard":     "MLP-CE",
    "mlp_weighted_ce":  "MLP-WCE",
    "xgboost_weighted": "XGBoost-W",
}

# Note: mlp_standard, mlp_weighted_ce, xgboost_weighted use the baseline
# training path (BaselineModel.fit()) which does NOT call RunLogger.log_epoch().
# Their JSON logs have empty "epochs" arrays. Only mlp_ccr has per-epoch data.
# The script will extract what's available and note missing data.
NOISE_TYPE = "asym"
NOISE_RATE = 0.3
SEED = 42
N_FOLDS = 5
OUTPUT_CSV = OUTPUTS_METRICS / "learning_curves.csv"


def load_epoch_f1(log_path: Path):
    """Load per-epoch val_macro_f1 from a JSON log file.

    Returns:
        List of (epoch, val_macro_f1) tuples, or empty list if file missing/invalid.
    """
    if not log_path.exists():
        logger.warning(f"Log file not found: {log_path}")
        return []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        epochs_data = data.get("epochs", [])
        return [(rec["epoch"], rec["val_macro_f1"]) for rec in epochs_data]
    except Exception as exc:
        logger.warning(f"Failed to parse {log_path}: {exc}")
        return []


def extract_learning_curves():
    rate_str = f"{int(NOISE_RATE * 100):02d}"
    all_records = []

    print("=" * 65, flush=True)
    print(f"  Learning Curves Extraction", flush=True)
    print(f"  Condition: {NOISE_TYPE}@{NOISE_RATE:.0%}, seed={SEED}", flush=True)
    print(f"  Models: {list(MODELS.keys())}", flush=True)
    print(f"  Datasets: {DATASETS}", flush=True)
    print("=" * 65, flush=True)

    for dataset_name in DATASETS:
        for model_key, model_label in MODELS.items():
            print(f"\n  {dataset_name} | {model_label} ({model_key})", flush=True)

            # Collect per-epoch F1 across all folds
            # epoch -> list of f1 values
            epoch_f1_map: dict = {}

            for fold in range(1, N_FOLDS + 1):
                log_name = f"{dataset_name}_{model_key}_{NOISE_TYPE}_{rate_str}_seed{SEED}_fold{fold}_train.json"
                log_path = OUTPUTS_LOGS / log_name

                epoch_f1_list = load_epoch_f1(log_path)

                if not epoch_f1_list:
                    print(f"    Fold {fold}: No data found in {log_name}", flush=True)
                    continue

                print(f"    Fold {fold}: {len(epoch_f1_list)} epochs", flush=True)

                for epoch, f1 in epoch_f1_list:
                    if epoch not in epoch_f1_map:
                        epoch_f1_map[epoch] = []
                    epoch_f1_map[epoch].append(f1)

            if not epoch_f1_map:
                print(f"    WARNING: No data found for {dataset_name}/{model_key}", flush=True)
                continue

            # Compute mean and std across folds for each epoch
            for epoch in sorted(epoch_f1_map.keys()):
                f1_vals = epoch_f1_map[epoch]
                all_records.append({
                    "dataset":      dataset_name,
                    "model":        model_key,
                    "model_label":  model_label,
                    "epoch":        epoch,
                    "mean_val_f1":  round(float(np.mean(f1_vals)), 6),
                    "std_val_f1":   round(float(np.std(f1_vals)), 6),
                    "n_folds":      len(f1_vals),
                })

    if not all_records:
        print("\nWARNING: No learning curve data extracted!", flush=True)
        print("NOTE: Only mlp_ccr logs per-epoch data. Baseline models (mlp_standard,", flush=True)
        print("      mlp_weighted_ce, xgboost_weighted) use a different training path", flush=True)
        print("      that does not call RunLogger.log_epoch(), so their JSON logs", flush=True)
        print("      have empty 'epochs' arrays.", flush=True)
        return

    df_out = pd.DataFrame(all_records)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)

    print(f"\n  Saved {len(df_out)} rows to {OUTPUT_CSV}", flush=True)
    print(f"  Datasets × Models: {df_out.groupby(['dataset', 'model']).size().shape[0]} combinations", flush=True)

    # Print summary
    print("\n  Summary (max epochs per dataset/model):", flush=True)
    summary = df_out.groupby(["dataset", "model"])["epoch"].max().unstack(fill_value=0)
    print(summary.to_string(), flush=True)

    print("\n" + "=" * 65, flush=True)
    print(f"  Learning curves extraction complete.", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    extract_learning_curves()
