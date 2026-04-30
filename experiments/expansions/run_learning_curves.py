"""
run_learning_curves.py â€” Extract Learning Curves from Training Logs
====================================================================
Extracts per-epoch validation Macro F1 for CCR under asym@30% noise
from the JSON training logs generated during the main experiment run.

Only mlp_ccr logs per-epoch data (RunLogger.log_epoch is called in the
CCR training loop). Baseline models use a different path that does not
log per-epoch metrics.

Reads:  outputs/logs/<dataset>_mlp_ccr_asym_30_seed*_fold*_train.json
Output: outputs/metrics/learning_curves.csv
Usage:  python run_learning_curves.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import DATASETS, N_FOLDS, OUTPUTS_LOGS, OUTPUTS_METRICS, SEEDS
from src.utils.logger import setup_logging

setup_logging(level=logging.WARNING)
logger = logging.getLogger(__name__)

NOISE_TYPE = "asym"
NOISE_RATE = 0.3
OUTPUT_CSV = OUTPUTS_METRICS / "learning_curves.csv"


def load_epoch_f1(log_path: Path):
    """Load per-epoch val_macro_f1 from a JSON training log.

    Returns:
        List of (epoch, val_macro_f1) tuples, or empty list if unavailable.
    """
    if not log_path.exists():
        return []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [(rec["epoch"], rec["val_macro_f1"]) for rec in data.get("epochs", [])]
    except Exception as exc:
        logger.warning(f"Could not parse {log_path.name}: {exc}")
        return []


def extract_learning_curves():
    rate_str = f"{int(NOISE_RATE * 100):02d}"
    all_datasets = list(DATASETS.keys())
    records = []

    print("=" * 65, flush=True)
    print("  Learning Curves Extraction", flush=True)
    print(f"  Model:     mlp_ccr", flush=True)
    print(f"  Condition: {NOISE_TYPE}@{NOISE_RATE:.0%}", flush=True)
    print(f"  Datasets:  {all_datasets}", flush=True)
    print("=" * 65, flush=True)

    for dataset_name in all_datasets:
        print(f"\n  {dataset_name}:", flush=True)
        epoch_f1_map: dict = {}

        for seed in SEEDS:
            for fold in range(1, N_FOLDS + 1):
                log_name = (
                    f"{dataset_name}_mlp_ccr_{NOISE_TYPE}_{rate_str}"
                    f"_seed{seed}_fold{fold}_train.json"
                )
                epoch_f1_list = load_epoch_f1(OUTPUTS_LOGS / log_name)

                if not epoch_f1_list:
                    print(f"    seed={seed} fold={fold}: no data", flush=True)
                    continue

                print(f"    seed={seed} fold={fold}: {len(epoch_f1_list)} epochs", flush=True)
                for epoch, f1 in epoch_f1_list:
                    epoch_f1_map.setdefault(epoch, []).append(f1)

        if not epoch_f1_map:
            print(f"    WARNING: No log data found for {dataset_name}", flush=True)
            continue

        for epoch in sorted(epoch_f1_map.keys()):
            vals = epoch_f1_map[epoch]
            records.append({
                "dataset":     dataset_name,
                "model":       "mlp_ccr",
                "noise_type":  NOISE_TYPE,
                "noise_rate":  NOISE_RATE,
                "epoch":       epoch,
                "mean_val_f1": round(float(np.mean(vals)), 6),
                "std_val_f1":  round(float(np.std(vals)), 6),
                "n_runs":      len(vals),
            })

    if not records:
        print(
            "\n  No data found. Run the main experiments first, then re-run this script.",
            flush=True,
        )
        return

    df = pd.DataFrame(records)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n  Saved {len(df)} rows to {OUTPUT_CSV}", flush=True)
    summary = df.groupby("dataset")["epoch"].max()
    print(f"  Max epochs per dataset:\n{summary.to_string()}", flush=True)


if __name__ == "__main__":
    extract_learning_curves()
