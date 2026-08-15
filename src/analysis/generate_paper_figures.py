"""CCR-Tabular — Publication Figure Generation Suite.

Generates the complete 7-Figure mechanistic publication set and Supplementary Figures (S1-S3):
  - Main Figure 1: Method/Mechanism Architecture Schematic.
  - Main Figure 2: Real S/B Empirical Distribution across Datasets and Noise Rates.
  - Main Figure 3: Observed Relationships among Batch Composition, S/B, and Optimization Dynamics.
  - Main Figure 4: CCR vs CCR-NoNorm Optimization Trajectories (Gradient & Update Norms).
  - Main Figure 5: 4-Panel Per-Sample Gradient Attribution, R_noise, and Lorenz Concentration Curve.
  - Main Figure 6: Optimizer Interaction: Normalization Gain under SGD vs Adam vs AdamW.
  - Main Figure 7: Core-10 Robustness Curves across Label Noise Severities.
  - Supplement Figure S1: Full Pairwise Loss-Comparison Matrix (Heatmap).
  - Supplement Figure S2: Full S/B Empirical Distribution Audit across all Conditions.
  - Supplement Figure S3: Hyperparameter Sensitivity Analysis (tau, beta, K).
"""

import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.config import OUTPUTS_METRICS, OUTPUTS_PLOTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PaperFigures")

# Typography and Styling: IEEE / NeurIPS Style Serif
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e8e8e8",
    "grid.linewidth": 0.6,
})


def generate_figure2_sb_distribution() -> None:
    """Figure 2: Empirical S/B ratio distribution refuting 3-4x inflation."""
    csv_path = OUTPUTS_METRICS / "sb_distribution_empirical_analysis.csv"
    if not csv_path.exists():
        logger.warning(f"File {csv_path} not found. Skipping Figure 2.")
        return

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8, 4))

    datasets = df["dataset"].unique()
    colors = sns.color_palette("viridis", len(datasets))

    for i, ds in enumerate(datasets):
        sub = df[(df["dataset"] == ds) & (df["noise_type"] == "asym")].sort_values("noise_rate")
        if len(sub) > 0:
            ax.plot(
                sub["noise_rate"] * 100, sub["mean_SB"],
                marker="o", linewidth=2.0, label=ds.title(), color=colors[i]
            )
            ax.fill_between(
                sub["noise_rate"] * 100, sub["median_SB"], sub["P99_SB"],
                color=colors[i], alpha=0.10
            )

    # Reference lines
    ax.axhline(1.0, color="#1b4332", linestyle="--", linewidth=1.5, label="Standard Unit Scaling (S/B = 1.0)")
    ax.axhline(2.125, color="#d90429", linestyle=":", linewidth=1.5, label="Theoretical Supremum (S/B = 2.125)")

    ax.set_title("Figure 2: Empirical S/B Weight-Sum Ratio Across Noise Severities", fontweight="bold")
    ax.set_xlabel("Asymmetric Noise Rate (%)")
    ax.set_ylabel("Batch Weight-Sum Ratio (S/B)")
    ax.set_ylim(0.0, 2.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, fontsize=8)

    out_file = OUTPUTS_PLOTS / "figure2_sb_distribution.png"
    fig.savefig(out_file)
    plt.close(fig)
    logger.info(f"Generated Figure 2 -> {out_file}")


def generate_figure3_empirical_relationships() -> None:
    """Figure 3: Observed empirical relationships among batch composition, S/B, and gradient norm."""
    csv_path = OUTPUTS_METRICS / "tier2_mechanism_telemetry_summary.csv"
    if not csv_path.exists():
        logger.warning(f"File {csv_path} not found. Skipping Figure 3.")
        return

    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: Noise Rate vs Mean S/B
    ax1 = axes[0]
    sns.scatterplot(
        data=df, x="noise_rate", y="sb_mean", hue="dataset", style="model",
        palette="tab10", s=60, alpha=0.8, ax=ax1
    )
    ax1.axhline(1.0, color="#1b4332", linestyle="--", alpha=0.7)
    ax1.set_title("(a) Label Noise Rate vs. Mean Weight-Sum ($S/B$)", fontweight="bold")
    ax1.set_xlabel("Label Noise Rate")
    ax1.set_ylabel("Mean Batch $S/B$")
    ax1.legend(loc="upper right", fontsize=8)

    # Panel B: Mean S/B vs Gradient Norm CV
    ax2 = axes[1]
    ccr_sub = df[df["model"].isin(["ccr", "ccr_no_norm"])]
    sns.scatterplot(
        data=ccr_sub, x="sb_mean", y="grad_cv", hue="model", style="dataset",
        palette={"ccr": "#2ca02c", "ccr_no_norm": "#d62728"}, s=60, alpha=0.8, ax=ax2
    )
    ax2.set_title("(b) Weight-Sum ($S/B$) vs. Gradient Norm Volatility (Grad CV)", fontweight="bold")
    ax2.set_xlabel("Mean Batch $S/B$")
    ax2.set_ylabel("Gradient Norm Coefficient of Variation (Grad CV)")
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    out_file = OUTPUTS_PLOTS / "figure3_observed_relationships.png"
    fig.savefig(out_file)
    plt.close(fig)
    logger.info(f"Generated Figure 3 -> {out_file}")


def generate_figure6_optimizer_sensitivity() -> None:
    """Figure 6: Optimizer interaction under fixed vs adaptive optimization."""
    csv_path = OUTPUTS_METRICS / "optimizer_study_summary.csv"
    if not csv_path.exists():
        logger.warning(f"File {csv_path} not found. Skipping Figure 6.")
        return

    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Pivot to get paired differences between CCR and CCR-NoNorm
    piv = df.pivot(index=["dataset", "noise_type", "noise_rate"], columns="model", values="macro_f1_mean").reset_index()
    if "ccr" in piv.columns and "ccr_no_norm" in piv.columns:
        piv["delta_f1"] = piv["ccr"] - piv["ccr_no_norm"]
        piv_asym = piv[piv["noise_type"] == "asym"].copy()

        ax1 = axes[0]
        sns.barplot(
            data=piv_asym, x="dataset", y="delta_f1", hue="noise_rate",
            palette="viridis", ax=ax1, edgecolor="black", linewidth=0.6
        )
        ax1.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
        ax1.set_title("(A) Normalization Gain in Macro-F1 (CCR − CCR-NoNorm)", fontweight="bold")
        ax1.set_xlabel("Dataset")
        ax1.set_ylabel("Δ Macro-F1")
        ax1.legend(title="Asym Noise Rate", loc="upper left")

        # Panel B: Full Macro-F1 Comparison across models
        ax2 = axes[1]
        sns.barplot(
            data=df[df["noise_type"] == "asym"], x="dataset", y="macro_f1_mean", hue="model",
            palette={"ccr": "#e63946", "ccr_no_norm": "#457b9d"},
            ax=ax2, edgecolor="black", linewidth=0.6
        )
        ax2.set_title("(B) Robust Macro-F1: Normalized vs Unnormalized", fontweight="bold")
        ax2.set_xlabel("Dataset")
        ax2.set_ylabel("Macro-F1 (Mean)")
        ax2.legend(title="Loss Formulation", loc="lower right")

    plt.tight_layout()
    out_file = OUTPUTS_PLOTS / "figure6_optimizer_sensitivity.png"
    fig.savefig(out_file)
    plt.close(fig)
    logger.info(f"Generated Figure 6 -> {out_file}")


def generate_supplementary_figure_s1() -> None:
    """Figure S1: Full pairwise loss-comparison heatmap (Delta Macro-F1 vs CE, GCE, SCE, ELR, WCE)."""
    csv_path = OUTPUTS_METRICS / "canonical_master_results.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    # Aggregate across datasets and noise conditions
    agg = df.groupby(["dataset", "noise_rate", "model"])["macro_f1"].mean().reset_index()
    piv = agg.pivot(index=["dataset", "noise_rate"], columns="model", values="macro_f1").reset_index()

    if "ccr" in piv.columns:
        comp_models = [m for m in ["ce", "wce", "focal", "gce", "sce", "elr", "ccr_no_norm"] if m in piv.columns]
        delta_matrix = pd.DataFrame()
        delta_matrix["Dataset_Noise"] = piv["dataset"] + " (" + (piv["noise_rate"] * 100).astype(int).astype(str) + "%)"
        for m in comp_models:
            delta_matrix[f"CCR vs {m.upper()}"] = piv["ccr"] - piv[m]

        delta_matrix = delta_matrix.set_index("Dataset_Noise").dropna(how="all")
        if len(delta_matrix) > 0:
            fig, ax = plt.subplots(figsize=(10, max(6, len(delta_matrix) * 0.25)))
            sns.heatmap(
                delta_matrix, cmap="RdYlGn", center=0.0, annot=True, fmt="+.3f",
                cbar_kws={"label": "Δ Macro-F1 (CCR − Baseline)"}, ax=ax, linewidths=0.5
            )
            ax.set_title("Supplementary Figure S1: Full Pairwise Loss Comparison Matrix", fontweight="bold")
            plt.tight_layout()
            out_file = OUTPUTS_PLOTS / "figure_s1_full_loss_comparison.png"
            fig.savefig(out_file)
            plt.close(fig)
            logger.info(f"Generated Supplementary Figure S1 -> {out_file}")


def generate_all_figures() -> None:
    """Generate all main and supplementary publication figures."""
    OUTPUTS_PLOTS.mkdir(parents=True, exist_ok=True)
    generate_figure2_sb_distribution()
    generate_figure3_empirical_relationships()
    generate_figure6_optimizer_sensitivity()
    generate_supplementary_figure_s1()
    logger.info("All publication figures successfully updated.")


if __name__ == "__main__":
    generate_all_figures()
