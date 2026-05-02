"""
diagnose_gate.py - CCR Confidence Gate Diagnostic (Recheck)
============================================================
Tests gate activation at tau=0.7 on Bank and Adult.
Target: 45-65% gate activation (intended operating range).

Auto-escalates to tau=0.75 if still above 85% at tau=0.7.

Usage:
    python scripts/diagnose_gate.py

Output:
    outputs/logs/gate_diagnostic_recheck.csv
    Printed verdict to stdout
"""

import logging
import sys
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
    BATCH_SIZE, LEARNING_RATE, OUTPUTS_LOGS, VAL_SIZE, WEIGHT_DECAY,
)
from src.utils.logger import setup_logging
from src.utils.reproducibility import fix_all_seeds, get_device

setup_logging(level=logging.WARNING)
logger = logging.getLogger(__name__)

OUTPUT_CSV = OUTPUTS_LOGS / "gate_diagnostic_recheck.csv"

# Recheck config — Bank and Adult only, as specified
RECHECK_DATASETS = ["adult", "bank"]
RECHECK_CONDITIONS = [
    ("none", 0.0),
    ("asym", 0.2),
    ("asym", 0.3),
]
RECHECK_SEED = 42
RECHECK_FOLD = 1
RECHECK_EPOCHS = 30

# Target gate activation range
TARGET_LOW  = 0.45
TARGET_HIGH = 0.65
ESCALATE_THRESHOLD = 0.85  # if still above this at tau=0.7, try tau=0.75


def run_diagnostic_at_tau(dataset_name, noise_type, noise_rate, tau):
    """Train CCR for RECHECK_EPOCHS at given tau, return per-epoch gate stats."""
    fix_all_seeds(RECHECK_SEED)
    device = get_device()

    df_data = load_dataset(dataset_name)
    feature_cols = [c for c in df_data.columns if c != "target"]
    X = df_data[feature_cols]
    y = df_data["target"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RECHECK_SEED)
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        if fold_idx + 1 == RECHECK_FOLD:
            break

    X_tr_df = X.iloc[train_idx].reset_index(drop=True)
    X_te_df = X.iloc[test_idx].reset_index(drop=True)
    y_tr_raw = y[train_idx]
    y_te_raw = y[test_idx]

    X_tr_df2, X_val_df, y_tr2, y_val = train_test_split(
        X_tr_df, y_tr_raw,
        test_size=VAL_SIZE, stratify=y_tr_raw, random_state=RECHECK_SEED,
    )

    (X_tr_np, X_val_np, X_te_np,
     y_tr_np, y_val_np, y_te_np, _) = preprocess_split(
        X_tr_df2, X_val_df, X_te_df,
        pd.Series(y_tr2), pd.Series(y_val), pd.Series(y_te_raw),
    )

    if noise_type == "asym" and noise_rate > 0:
        y_tr_np, _ = inject_asymmetric_noise(y_tr_np, noise_rate, RECHECK_SEED)

    n_majority = int(np.sum(y_tr_np == 0))
    n_minority = int(np.sum(y_tr_np == 1))

    model = get_mlp_for_dataset(dataset_name, X_tr_np.shape[1]).to(device)
    criterion = CCRLoss(
        n_samples=len(y_tr_np),
        n_classes=2,
        class_counts=[n_majority, n_minority],
        tau=tau,
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

    for epoch in range(RECHECK_EPOCHS):
        model.train()
        gate_fracs, p_i_means = [], []

        for X_b, y_b, idx_b in loader:
            X_b, y_b, idx_b = X_b.to(device), y_b.to(device), idx_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)

            probs = F.softmax(logits, dim=1)
            p_i = probs[torch.arange(len(y_b), device=device), y_b]
            gate_fracs.append((p_i > tau).float().mean().item())
            p_i_means.append(p_i.mean().item())

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
            "tau":            tau,
            "epoch":          epoch + 1,
            "gate_frac_mean": round(float(np.mean(gate_fracs)), 4),
            "p_i_mean":       round(float(np.mean(p_i_means)), 4),
            "frac_below_tau": round(float(1 - np.mean(gate_fracs)), 4),
        })

    return epoch_records


def main():
    print("=" * 65, flush=True)
    print("  CCR Gate Diagnostic Recheck", flush=True)
    print(f"  Testing tau=0.7 on: {RECHECK_DATASETS}", flush=True)
    print(f"  Target gate activation: {TARGET_LOW:.0%} – {TARGET_HIGH:.0%}", flush=True)
    print("=" * 65, flush=True)

    all_records = []
    taus_to_test = [0.7]

    for tau in taus_to_test:
        print(f"\n  === Testing tau={tau} ===", flush=True)
        tau_records = []

        for dataset_name in RECHECK_DATASETS:
            for noise_type, noise_rate in RECHECK_CONDITIONS:
                label = f"{noise_type}@{noise_rate:.0%}" if noise_type != "none" else "clean"
                print(f"\n  {dataset_name} | {label} | tau={tau}", flush=True)

                try:
                    records = run_diagnostic_at_tau(dataset_name, noise_type, noise_rate, tau)
                    tau_records.extend(records)
                    all_records.extend(records)

                    # Print key epochs
                    for r in records:
                        if r["epoch"] in (1, 5, 10, 20, 30):
                            print(
                                f"    Epoch {r['epoch']:>2}: "
                                f"gate={r['gate_frac_mean']:.1%}  "
                                f"p_i_mean={r['p_i_mean']:.3f}  "
                                f"below_tau={r['frac_below_tau']:.1%}",
                                flush=True,
                            )
                except Exception as exc:
                    print(f"  FAILED: {exc}", flush=True)
                    logger.error(f"Diagnostic failed: {exc}", exc_info=True)

        # Check if tau=0.7 is in target range
        if tau_records:
            late = [r for r in tau_records if r["epoch"] >= 20]
            overall = float(np.mean([r["gate_frac_mean"] for r in late]))
            print(f"\n  tau={tau} overall gate activation (epochs 20-30): {overall:.1%}", flush=True)

            if overall > ESCALATE_THRESHOLD:
                print(f"  STILL ABOVE {ESCALATE_THRESHOLD:.0%} — escalating to tau=0.75", flush=True)
                taus_to_test.append(0.75)
            elif TARGET_LOW <= overall <= TARGET_HIGH:
                print(f"  IN TARGET RANGE ({TARGET_LOW:.0%}–{TARGET_HIGH:.0%}) — tau={tau} is CONFIRMED", flush=True)
            elif overall < TARGET_LOW:
                print(f"  BELOW TARGET — gate too selective at tau={tau}. Consider tau=0.65.", flush=True)
            else:
                print(f"  ABOVE TARGET but below escalation threshold — tau={tau} is acceptable.", flush=True)

    # Save all records
    df_out = pd.DataFrame(all_records)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Results saved to {OUTPUT_CSV}", flush=True)

    # Final verdict
    print()
    print("=" * 65, flush=True)
    print("  FINAL VERDICT", flush=True)
    print("=" * 65, flush=True)

    for tau in df_out["tau"].unique():
        sub = df_out[(df_out["tau"] == tau) & (df_out["epoch"] >= 20)]
        overall = sub["gate_frac_mean"].mean()
        in_range = TARGET_LOW <= overall <= TARGET_HIGH
        status = "CONFIRMED" if in_range else ("TOO HIGH" if overall > TARGET_HIGH else "TOO LOW")
        print(f"  tau={tau}: gate={overall:.1%}  [{status}]", flush=True)

    print()
    print("  Recommended tau for paper:", flush=True)
    best_tau = None
    best_dist = float("inf")
    target_mid = (TARGET_LOW + TARGET_HIGH) / 2
    for tau in df_out["tau"].unique():
        sub = df_out[(df_out["tau"] == tau) & (df_out["epoch"] >= 20)]
        overall = sub["gate_frac_mean"].mean()
        dist = abs(overall - target_mid)
        if dist < best_dist:
            best_dist = dist
            best_tau = tau
    print(f"  tau={best_tau} (closest to {target_mid:.0%} midpoint)", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
