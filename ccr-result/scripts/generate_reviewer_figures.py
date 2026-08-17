import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config import OUTPUTS_PLOTS, OUTPUTS_METRICS
from src.utils.statistics import _BASELINES, run_friedman_test, generate_win_tie_loss, run_all_wilcoxon_tests

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.5)

MODELS = ["mlp_ccr"] + _BASELINES

def _save(fig, name):
    OUTPUTS_PLOTS.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUTS_PLOTS / name, dpi=300, bbox_inches="tight")
    plt.close(fig)

def fig1_noise_robustness(df: pd.DataFrame):
    """Figure 1: Noise robustness curves (x=noise_rate, y=Macro-F1)."""
    if len(df) == 0: return
    
    mask = df["noise_type"].isin(["asym", "sym"])
    df_plot = df[mask].copy()
    
    clean = df[df["noise_type"] == "none"].copy()
    clean_asym = clean.copy()
    clean_asym["noise_type"] = "asym"
    clean_sym = clean.copy()
    clean_sym["noise_type"] = "sym"
    
    df_plot = pd.concat([df_plot, clean_asym, clean_sym])
    
    agg = df_plot.groupby(["noise_type", "noise_rate", "model"])["macro_f1"].mean().reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, n_type in enumerate(["asym", "sym"]):
        ax = axes[i]
        d = agg[agg["noise_type"] == n_type]
        sns.lineplot(data=d, x="noise_rate", y="macro_f1", hue="model", marker="o", ax=ax)
        ax.set_title(f"{n_type.capitalize()} Noise")
        ax.set_xlabel("Noise Rate")
        ax.set_ylabel("Macro F1")
        if i == 0:
            ax.legend(bbox_to_anchor=(0.5, -0.2), loc="upper center", ncol=4)
        else:
            if ax.get_legend():
                ax.get_legend().remove()
            
    _save(fig, "fig1_noise_robustness.png")

def fig2_cd_diagram(df: pd.DataFrame):
    """Figure 2: Critical Difference Diagram Proxy (Average Ranks)."""
    if len(df) == 0: return
    
    mask = df["noise_type"] == "asym"
    d = df[mask]
    
    pivoted = d.pivot_table(index=["dataset", "noise_rate", "seed", "fold"], columns="model", values="macro_f1")
    pivoted = pivoted.dropna()
    
    if len(pivoted) == 0: return
    
    model_ranks = pivoted.rank(axis=1, ascending=False)
    mean_ranks = model_ranks.mean().sort_values()
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=mean_ranks.values, y=mean_ranks.index, ax=ax, palette="viridis")
    ax.set_title("Average Ranks across all Datasets (Asymmetric Noise)")
    ax.set_xlabel("Average Rank (Lower is Better)")
    _save(fig, "fig2_average_ranks.png")

def fig3_boxplots(df: pd.DataFrame):
    """Figure 3: Boxplots (CCR vs competitors)."""
    if len(df) == 0: return
    
    mask = df["noise_type"] == "asym"
    d = df[mask]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=d, x="model", y="macro_f1", hue="noise_rate", ax=ax)
    plt.xticks(rotation=45, ha="right")
    ax.set_title("Macro F1 Distributions under Asymmetric Noise")
    _save(fig, "fig3_boxplots.png")

def fig4_calibration(df: pd.DataFrame):
    """Figure 4: Calibration Plot (Brier Score)."""
    if "brier_score" not in df.columns:
        return
        
    mask = df["noise_type"] == "asym"
    d = df[mask]
    
    agg = d.groupby(["noise_rate", "model"])["brier_score"].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=agg, x="noise_rate", y="brier_score", hue="model", ax=ax)
    ax.set_title("Brier Score Calibration Error (Lower is Better)")
    _save(fig, "fig4_calibration_brier.png")

def fig5_runtime(df: pd.DataFrame):
    """Figure 5: Runtime Decomposition."""
    cols = ["preprocess_time_s", "train_time_s", "predict_time_s"]
    for c in cols:
        if c not in df.columns:
            return
            
    df["total_time_s"] = df[cols].sum(axis=1)
    
    agg = df.groupby("model")[cols + ["total_time_s"]].mean()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    agg[cols].plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
    plt.xticks(rotation=45, ha="right")
    ax.set_ylabel("Seconds per Fold")
    ax.set_title("Average Runtime Decomposition")
    
    for i, v in enumerate(agg["total_time_s"]):
        ax.text(i, v + 0.1, f"{v:.1f}s", ha='center')
        
    _save(fig, "fig5_runtime.png")

def fig6_ablation(df: pd.DataFrame):
    """Figure 6: Ablation Progression (CE -> CE+Focal -> CCR)"""
    ablation_models = ["mlp_standard", "mlp_focal", "mlp_ccr"]
    d = df[df["model"].isin(ablation_models)]
    if len(d) == 0: return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=d[d["noise_type"]=="asym"], x="noise_rate", y="macro_f1", hue="model", ax=ax)
    ax.set_title("Ablation Study: Architecture Progression")
    _save(fig, "fig6_ablation.png")

def main():
    results_path = OUTPUTS_METRICS / "results.csv"
    if not results_path.exists():
        print("No results.csv found.")
        return
        
    df = pd.read_csv(results_path)
    
    print("Generating Figure 1: Noise Robustness...")
    fig1_noise_robustness(df)
    
    print("Generating Figure 2: Average Ranks...")
    fig2_cd_diagram(df)
    
    print("Generating Figure 3: Boxplots...")
    fig3_boxplots(df)
    
    print("Generating Figure 4: Calibration...")
    fig4_calibration(df)
    
    print("Generating Figure 5: Runtime...")
    fig5_runtime(df)
    
    print("Generating Figure 6: Ablation...")
    fig6_ablation(df)
    
    print("Generating Statistical Tables...")
    try:
        run_friedman_test(noise_type="asym", noise_rate=0.3)
        w_res = run_all_wilcoxon_tests()
        if "asym_0.3" in w_res:
            wtl = generate_win_tie_loss(w_res["asym_0.3"])
            wtl.to_csv(OUTPUTS_METRICS / "win_tie_loss_asym30.csv", index=False)
            print("Win/Tie/Loss table saved.")
    except Exception as e:
        print(f"Stats failed: {e}")

if __name__ == "__main__":
    main()
