import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

from src.utils.config import OUTPUTS_PLOTS, OUTPUTS_METRICS

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.5)

def _save(fig, name):
    OUTPUTS_PLOTS.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUTS_PLOTS / name, dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    results_path = OUTPUTS_METRICS / "results.csv"
    if not results_path.exists():
        print("results.csv not found!")
        return

    df = pd.read_csv(results_path)
    
    # Filter for tau runs (models that have _tau suffix)
    tau_df = df[df["model"].str.contains("_tau")].copy()
    if len(tau_df) == 0:
        print("No tau results found yet.")
        return

    # Extract the tau value from the model string
    tau_df["tau"] = tau_df["model"].apply(lambda x: float(x.split("_tau")[-1]))
    
    # Sort for plotting
    tau_df = tau_df.sort_values("tau")

    # Metrics to plot
    metrics = {
        "macro_f1": "Macro-F1",
        "minority_recall": "Minority Recall",
        "auc_roc": "AUC-ROC",
        "brier_score": "Brier Score",
        "train_time_s": "Train Time (s)"
    }

    print("Generating Tau Analysis Figures...")
    
    for metric, label in metrics.items():
        if metric not in tau_df.columns:
            continue
            
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Averaged across all datasets (grouped by noise type)
        sns.lineplot(data=tau_df, x="tau", y=metric, hue="noise_type", marker="o", ax=axes[0])
        axes[0].set_title(f"Overall {label} vs Tau")
        axes[0].set_xlabel("Tau")
        axes[0].set_ylabel(label)
        axes[0].set_xticks([0.3, 0.5, 0.7, 0.9])
        
        # Plot 2: Per-dataset (averaged across noise types)
        sns.lineplot(data=tau_df, x="tau", y=metric, hue="dataset", marker="s", ax=axes[1])
        axes[1].set_title(f"Per-Dataset {label} vs Tau")
        axes[1].set_xlabel("Tau")
        axes[1].set_ylabel(label)
        axes[1].set_xticks([0.3, 0.5, 0.7, 0.9])
        
        _save(fig, f"tau_sweep_{metric}.png")
        
    # Generate Summary Table
    print("\n" + "="*50)
    print("Tau Summary Table (Averaged across all folds)")
    print("="*50)
    
    # We also want average rank.
    # To compute rank, group by dataset, fold, seed, noise_type, noise_rate.
    def compute_ranks(group):
        # Higher F1 is better (lower rank)
        return group["macro_f1"].rank(ascending=False)
        
    tau_df["rank"] = tau_df.groupby(["dataset", "fold", "seed", "noise_type", "noise_rate"]).apply(
        lambda g: g["macro_f1"].rank(ascending=False)
    ).reset_index(level=[0,1,2,3,4], drop=True)
    
    summary = tau_df.groupby("tau").agg(
        macro_f1=("macro_f1", "mean"),
        minority_recall=("minority_recall", "mean"),
        brier_score=("brier_score", "mean"),
        train_time_s=("train_time_s", "mean"),
        avg_rank=("rank", "mean")
    ).round(4)
    
    print(summary.to_string())
    
    # Run Friedman test on macro_f1
    print("\n" + "="*50)
    print("Statistical Tests (Macro-F1)")
    print("="*50)
    
    # We need matching samples for Friedman test.
    # Group by the unique configuration: dataset, fold, seed, noise_type, noise_rate
    pivot = tau_df.pivot_table(
        index=["dataset", "fold", "seed", "noise_type", "noise_rate"],
        columns="tau",
        values="macro_f1"
    ).dropna()
    
    if len(pivot) > 0:
        stat, p_value = friedmanchisquare(*[pivot[c].values for c in pivot.columns])
        print(f"Friedman Test (N={len(pivot)}): stat={stat:.4f}, p={p_value:.4e}")
        
        if p_value < 0.05:
            print("\nWilcoxon Signed-Rank Pairwise Tests (Holm Corrected):")
            taus = list(pivot.columns)
            pairs = []
            raw_pvals = []
            
            for i in range(len(taus)):
                for j in range(i+1, len(taus)):
                    t1, t2 = taus[i], taus[j]
                    try:
                        _, p = wilcoxon(pivot[t1], pivot[t2])
                    except ValueError:
                        p = 1.0
                    pairs.append((t1, t2))
                    raw_pvals.append(p)
                    
            rej, corr_pvals, _, _ = multipletests(raw_pvals, method="holm")
            
            for (t1, t2), p_raw, p_corr, sig in zip(pairs, raw_pvals, corr_pvals, rej):
                diff = pivot[t1].mean() - pivot[t2].mean()
                better = t1 if diff > 0 else t2
                print(f"Tau {t1} vs {t2}: p_raw={p_raw:.4f} -> p_corr={p_corr:.4f} | Sig={sig} | Better={better} (diff={abs(diff):.4f})")
    else:
        print("Not enough paired data for statistical tests.")

if __name__ == "__main__":
    main()
