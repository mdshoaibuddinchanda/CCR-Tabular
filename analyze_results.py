"""
Analyze experiment results and check paper claims.
Run after experiments complete: python analyze_results.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

df = pd.read_csv("outputs/metrics/results.csv")
total = len(df)

CONFIGS = [
    ("none", 0.0,  "clean_run"),
    ("asym", 0.1,  "noisy_asym_10"),
    ("asym", 0.2,  "noisy_asym_20"),
    ("asym", 0.3,  "noisy_asym_30"),
    ("feat", 0.1,  "noisy_feat_10"),
    ("feat", 0.2,  "noisy_feat_20"),
    ("feat", 0.3,  "noisy_feat_30"),
]
DATASETS = ["adult", "bank", "credit_g", "magic", "phoneme", "spambase"]
MODELS   = ["mlp_standard", "mlp_focal", "mlp_weighted_ce", "mlp_smote",
            "xgboost_default", "xgboost_weighted", "lightgbm_default", "mlp_ccr"]
BASELINES = [m for m in MODELS if m != "mlp_ccr"]

SEP = "=" * 65

# ── 1. Completion status ──────────────────────────────────────────────────────
print(SEP)
print(f"  COMPLETION STATUS   ({total}/5040 = {100*total/5040:.1f}%)")
print(SEP)
print(f"{'Config':<22} {'Done':>5} {'Need':>5}  Status")
print("-" * 50)
for nt, nr, name in CONFIGS:
    sub = df[(df.noise_type == nt) & (df.noise_rate.round(2) == round(nr, 2))]
    done = len(sub)
    needed = 720
    if done == needed:
        status = "COMPLETE"
    elif done == 0:
        status = "NOT STARTED"
    else:
        status = f"IN PROGRESS {100*done/needed:.0f}%"
    print(f"{name:<22} {done:>5} {needed:>5}  {status}")

# ── 2. Per-dataset grid ───────────────────────────────────────────────────────
print()
print(SEP)
print("  DATASET x CONFIG GRID  (OK=120 runs done, -- = not started)")
print(SEP)
header = f"{'Dataset':<10}"
for _, _, nm in CONFIGS:
    header += f"  {nm[-8:]:>10}"
print(header)
print("-" * 85)
for ds in DATASETS:
    row = f"{ds:<10}"
    for nt, nr, _ in CONFIGS:
        sub = df[(df.dataset == ds) & (df.noise_type == nt) &
                 (df.noise_rate.round(2) == round(nr, 2))]
        done = len(sub)
        mark = "OK" if done == 120 else (f"{done:>3}" if done > 0 else "--")
        row += f"  {mark:>10}"
    print(row)

# ── 3. CCR vs baselines — mean metrics per noise condition ───────────────────
print()
print(SEP)
print("  CCR vs BEST BASELINE — MEAN METRICS BY NOISE CONDITION")
print(SEP)

for nt, nr, name in CONFIGS:
    sub = df[(df.noise_type == nt) & (df.noise_rate.round(2) == round(nr, 2))]
    if len(sub) == 0:
        continue

    ccr = sub[sub.model == "mlp_ccr"].groupby("dataset")[
        ["macro_f1", "minority_recall", "auc_roc"]].mean()
    best_bl = sub[sub.model.isin(BASELINES)].groupby(["dataset", "model"])[
        ["macro_f1", "minority_recall", "auc_roc"]].mean().reset_index()
    best_bl = best_bl.loc[best_bl.groupby("dataset")["macro_f1"].idxmax()]
    best_bl = best_bl.set_index("dataset")

    print(f"\n  [{name}]")
    print(f"  {'Dataset':<10} {'CCR_F1':>8} {'BL_F1':>8} {'F1_delta':>9} "
          f"{'CCR_Rec':>8} {'BL_Rec':>8} {'Rec_delta':>10}")
    print("  " + "-" * 65)

    f1_wins, rec_wins = 0, 0
    for ds in DATASETS:
        if ds not in ccr.index or ds not in best_bl.index:
            continue
        cf1  = ccr.loc[ds, "macro_f1"]
        bf1  = best_bl.loc[ds, "macro_f1"]
        crec = ccr.loc[ds, "minority_recall"]
        brec = best_bl.loc[ds, "minority_recall"]
        d_f1  = cf1 - bf1
        d_rec = crec - brec
        f1_wins  += (d_f1  > 0)
        rec_wins += (d_rec > 0)
        f1_mark  = "+" if d_f1  > 0 else "-"
        rec_mark = "+" if d_rec > 0 else "-"
        print(f"  {ds:<10} {cf1:>8.4f} {bf1:>8.4f} {f1_mark}{abs(d_f1):>8.4f} "
              f"{crec:>8.4f} {brec:>8.4f} {rec_mark}{abs(d_rec):>9.4f}")

    n_ds = len([d for d in DATASETS if d in ccr.index])
    print(f"  CCR wins on Macro F1:        {f1_wins}/{n_ds} datasets")
    print(f"  CCR wins on Minority Recall: {rec_wins}/{n_ds} datasets")

# ── 4. Noise robustness — how much does each model degrade? ──────────────────
print()
print(SEP)
print("  NOISE ROBUSTNESS — MACRO F1 DROP FROM CLEAN (asym noise)")
print("  Smaller drop = more robust. CCR should drop least.")
print(SEP)

clean = df[df.noise_type == "none"].groupby(["model", "dataset"])["macro_f1"].mean()

for nt, nr, name in CONFIGS:
    if nt != "asym" or nr not in [0.1, 0.2, 0.3]:
        continue
    noisy = df[(df.noise_type == nt) & (df.noise_rate.round(2) == round(nr, 2))]
    if len(noisy) == 0:
        continue
    noisy_mean = noisy.groupby(["model", "dataset"])["macro_f1"].mean()

    print(f"\n  [{name}] — mean F1 drop across all datasets")
    drops = {}
    for m in MODELS:
        ds_drops = []
        for ds in DATASETS:
            try:
                c = clean.loc[(m, ds)]
                n = noisy_mean.loc[(m, ds)]
                ds_drops.append(c - n)
            except KeyError:
                pass
        if ds_drops:
            drops[m] = np.mean(ds_drops)

    for m, d in sorted(drops.items(), key=lambda x: x[1]):
        marker = " <-- CCR" if m == "mlp_ccr" else ""
        print(f"  {m:<22}  drop = {d:+.4f}{marker}")

# ── 5. Paper claim verdict ────────────────────────────────────────────────────
print()
print(SEP)
print("  PAPER CLAIM VERDICT")
print(SEP)

# Claim 1: CCR beats all MLP baselines on minority recall (clean)
clean_df = df[df.noise_type == "none"]
ccr_rec = clean_df[clean_df.model == "mlp_ccr"]["minority_recall"].mean()
mlp_baselines = ["mlp_standard", "mlp_focal", "mlp_weighted_ce"]
bl_rec = clean_df[clean_df.model.isin(mlp_baselines)]["minority_recall"].mean()
claim1 = ccr_rec > bl_rec
print(f"\n  Claim 1: CCR > MLP baselines on minority recall (clean)")
print(f"    CCR recall = {ccr_rec:.4f}  |  MLP baselines avg = {bl_rec:.4f}")
print(f"    Result: {'SUPPORTED' if claim1 else 'NOT SUPPORTED'}")

# Claim 2: CCR is most noise-robust (smallest F1 drop under asym noise)
asym20 = df[(df.noise_type == "asym") & (df.noise_rate.round(2) == 0.2)]
if len(asym20) > 0:
    drops2 = {}
    for m in MODELS:
        c_vals = clean_df[clean_df.model == m]["macro_f1"].values
        n_vals = asym20[asym20.model == m]["macro_f1"].values
        if len(c_vals) > 0 and len(n_vals) > 0:
            drops2[m] = np.mean(c_vals) - np.mean(n_vals)
    if drops2:
        most_robust = min(drops2, key=drops2.get)
        claim2 = most_robust == "mlp_ccr"
        print(f"\n  Claim 2: CCR is most noise-robust (smallest F1 drop @ asym 20%)")
        for m, d in sorted(drops2.items(), key=lambda x: x[1]):
            marker = " <-- CCR" if m == "mlp_ccr" else ""
            print(f"    {m:<22}  drop = {d:+.4f}{marker}")
        print(f"    Most robust: {most_robust}")
        print(f"    Result: {'SUPPORTED' if claim2 else 'NOT SUPPORTED — see above'}")

# Claim 3: CCR beats XGBoost on minority recall under noise
if len(asym20) > 0:
    ccr_noisy_rec  = asym20[asym20.model == "mlp_ccr"]["minority_recall"].mean()
    xgb_noisy_rec  = asym20[asym20.model == "xgboost_weighted"]["minority_recall"].mean()
    claim3 = ccr_noisy_rec > xgb_noisy_rec
    print(f"\n  Claim 3: CCR > XGBoost (weighted) on minority recall under asym 20%")
    print(f"    CCR recall = {ccr_noisy_rec:.4f}  |  XGBoost weighted = {xgb_noisy_rec:.4f}")
    print(f"    Result: {'SUPPORTED' if claim3 else 'NOT SUPPORTED'}")

print()
print(SEP)
print("  Run complete. Check outputs/metrics/ for full CSVs.")
print(SEP)
