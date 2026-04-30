"""
diagnose_gate.py â€” CCR Confidence Gate Diagnostic
===================================================
Measures the fraction of samples per batch where p_i > tau fires.
If consistently > 90%, the gate is always on and contributes nothing.
If consistently < 50%, the gate is suppressing most samples.

Runs on adult (large, real noise) and credit_g (small, controlled).
Tests both clean and asym@20% noise conditions.

Usage:
    python diagnose_gate.py

Output:
    outputs/logs/gate_diagnostic.csv
    Printed summary to stdout
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.load_data import load_dataset
from src.data.noise_injection import inject_asymmetric_noise
from src.data.preprocess import preprocess_split
from src.loss.ccr_loss import CCRLoss
from src.models.mlp import TabularDataset, get_mlp_for_dataset
from src.utils.config import (
    BATCH_SIZE, EARLY_STOP_PATIENCE, LEARNING_RATE,
    MAX_EPOCHS, OUTPUTS_LOGS, SEEDS, TAU, VAL_SIZE, WEIGHT_DECAY,
)
from src.utils.logger import setup_logging
from src.utils.reproducibility import fix_all_seeds, get_device

setup_logging(level=logging.WARNING)  # suppress INFO noise
logger = logging.getLogger(__name__)

OUTPUT_CSV = OUTPUTS_LOGS / "gate_diagnostic.csv"

DIAG_DATASETS = ["adult", "bank", "credit_g", "phoneme"]
DIAG_CONDITIONS = [
    ("none", 0.0),
    ("asym", 0.2),
    ("asym", 0.3),
]
DIAG_SEED = 42
DIAG_FOLD = 1
DIAG_EPOCHS = 30  # enough to see convergence pattern


def run_diagnostic(dataset_name, noise_type, noise_rate):
    """Train CCR for DIAG_EPOCHS and log gate activation rate per epoch."""
    fix_all_seeds(DIAG_SEED)
    device = get_device()

    df_data = load_dataset(dataset_name)
    feature_cols = [c for c in df_data.columns if c != "target"]
    X = df_data[feature_cols]
    y = df_data["target"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=DIAG_SEED)
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        if fold_idx + 1 == DIAG_FOLD:
            break

    X_tr_df = X.iloc[train_idx].reset_index(drop=True)
    X_te_df = X.iloc[test_idx].reset_index(drop=True)
    y_tr_raw = y[train_idx]
    y_te_raw = y[test_idx]

    X_tr_df2, X_val_df, y_tr2, y_val = train_test_split(
        X_tr_df, y_tr_raw,
        test_size=VAL_SIZE, stratify=y_tr_raw, random_state=DIAG_SEED,
    )

    (X_tr_np, X_val_np, X_te_np,
     y_tr_np, y_val_np, y_te_np, _) = preprocess_split(
        X_tr_df2, X_val_df, X_te_df,
        pd.Series(y_tr2), pd.Series(y_val), pd.Series(y_te_raw),
    )

    # Inject noise if needed
    if noise_type == "asym" and noise_rate > 0:
        y_tr_np, _ = inject_asymmetric_noise(y_tr_np, noise_rate, DIAG_SEED)

    n_majority = int(np.sum(y_tr_np == 0))
    n_minority = int(np.sum(y_tr_np == 1))

    model = get_mlp_for_dataset(dataset_name, X_tr_np.shape[1]).to(device)
    criterion = CCRLoss(
        n_samples=len(y_tr_np),
        n_classes=2,
        class_counts=[n_majority, n_minority],
        device=device,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    loader = DataLoader(
        TabularDataset(X_tr_np, y_tr_np),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
    )

    epoch_records = []

    for epoch in range(DIAG_EPOCHS):
        model.train()
        gate_fracs = []
        p_i_means  = []
        p_i_below_tau = []

        for X_b, y_b, idx_b in loader:
            X_b, y_b, idx_b = X_b.to(device), y_b.to(device), idx_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)

            # â”€â”€ Compute p_i (prob of true class) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            probs = F.softmax(logits, dim=1)
            p_i = probs[torch.arange(len(y_b), device=device), y_b]

            # â”€â”€ Gate activation fraction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            gate_fired = (p_i > TAU).float()
            gate_frac  = gate_fired.mean().item()
            gate_fracs.append(gate_frac)
            p_i_means.append(p_i.mean().item())
            p_i_below_tau.append((p_i <= TAU).float().mean().item())

            # Normal training step
            loss = criterion(logits, y_b, idx_b, epoch)
            if not torch.isnan(loss):
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                criterion.update_history(F.softmax(logits.detach(), dim=1), idx_b, epoch)

        epoch_records.append({
            "dataset":        dataset_name,
            "noise_type":     noise_type,
            "noise_rate":     noise_rate,
            "epoch":          epoch + 1,
            "gate_frac_mean": round(np.mean(gate_fracs), 4),
            "gate_frac_min":  round(np.min(gate_fracs), 4),
            "gate_frac_max":  round(np.max(gate_fracs), 4),
            "p_i_mean":       round(np.mean(p_i_means), 4),
            "frac_below_tau": round(np.mean(p_i_below_tau), 4),
            "tau":            TAU,
        })

    return epoch_records


def main():
    print("=" * 65)
    print("  CCR Confidence Gate Diagnostic")
    print(f"  tau = {TAU}  (gate fires when p_i > tau)")
    print(f"  Datasets: {DIAG_DATASETS}")
    print("=" * 65)

    all_records = []

    for dataset_name in DIAG_DATASETS:
        for noise_type, noise_rate in DIAG_CONDITIONS:
            label = f"{noise_type}@{noise_rate:.0%}" if noise_type != "none" else "clean"
            print(f"\n  Running: {dataset_name} | {label} ...", flush=True)

            try:
                records = run_diagnostic(dataset_name, noise_type, noise_rate)
                all_records.extend(records)

                # Print epoch 1, 10, 20, 30 summary
                for r in records:
                    if r["epoch"] in (1, 5, 10, 20, 30):
                        print(
                            f"    Epoch {r['epoch']:>2}: "
                            f"gate_fired={r['gate_frac_mean']:.1%}  "
                            f"p_i_mean={r['p_i_mean']:.3f}  "
                            f"below_tau={r['frac_below_tau']:.1%}"
                        )
            except Exception as exc:
                print(f"  FAILED: {exc}")

    # Save
    df_out = pd.DataFrame(all_records)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Full results saved to {OUTPUT_CSV}")

    # â”€â”€ Summary verdict â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print()
    print("=" * 65)
    print("  GATE DIAGNOSTIC VERDICT")
    print("=" * 65)

    # Average gate activation in final 10 epochs per condition
    late_epochs = df_out[df_out["epoch"] >= 20]
    summary = late_epochs.groupby(["dataset", "noise_type", "noise_rate"])[
        ["gate_frac_mean", "p_i_mean", "frac_below_tau"]
    ].mean().round(4)

    print(summary.to_string())
    print()

    overall_gate = late_epochs["gate_frac_mean"].mean()
    print(f"  Overall gate activation (epochs 20-30): {overall_gate:.1%}")

    if overall_gate > 0.90:
        print()
        print("  FINDING: Gate fires on >90% of samples consistently.")
        print("  The indicator I(p_i > tau) is almost always 1.")
        print("  tau=0.3 is too low â€” most samples exceed it after a few epochs.")
        print("  The gate is effectively disabled. Variance term applies to all samples.")
        print()
        print("  RECOMMENDATION:")
        print("  1. Report this honestly in the paper as a calibration finding.")
        print("  2. Suggest tau should be tuned per dataset or set higher (e.g. 0.6-0.7).")
        print("  3. The variance term still contributes â€” just without selective gating.")
    elif overall_gate < 0.50:
        print()
        print("  FINDING: Gate fires on <50% of samples.")
        print("  The gate is actively suppressing noisy/uncertain samples.")
        print("  This is the intended behavior â€” mechanism is working correctly.")
    else:
        print()
        print("  FINDING: Gate fires on 50-90% of samples.")
        print("  Moderate selectivity. Gate is partially active.")
        print("  Check per-dataset breakdown above for variation.")

    print("=" * 65)


if __name__ == "__main__":
    main()
