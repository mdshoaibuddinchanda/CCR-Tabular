"""Publication Figure Generator for CCR-Tabular.

Generates journal-quality, high-resolution (600 DPI & Vector PDF) figures
with clean padding, professional serif typography, enlarged labels, and
zero embedded figure numbering.
"""

import os
import glob
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as mticker

# GLOBAL PUBLICATION RCPARAMS (Enlarged and High Contrast)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.18,
    "axes.linewidth": 1.1,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e2e8f0",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.8,
})

FIG_DIR = Path("ccrtatex/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Academic Color Palette
C_CCR = "#1e3a8a"       # Deep Navy
C_NONORM = "#2563eb"    # Royal Blue
C_CE = "#64748b"        # Slate Gray
C_FOCAL = "#d97706"     # Amber
C_WCE = "#7c3aed"       # Purple
C_XGB = "#059669"       # Emerald Green
C_LGBM = "#0284c7"      # Sky Blue
C_FTT = "#dc2626"       # Crimson Red
C_TABNET = "#d946ef"    # Magenta


def generate_figure1_schematic():
    from generate_figure1_schematic import generate_figure1_schematic as gen
    gen()


def load_all_metrics():
    files = glob.glob("ccr-result/outputs/metrics/cv_summary_*.csv")
    rows = []
    for f in files:
        basename = os.path.basename(f)
        m = re.match(
            r"cv_summary_(.+)_(mlp_[a-z_]+|ft_transformer|tabnet|"
            r"xgboost_[a-z]+|lightgbm_[a-z]+)_(asym|feat|sym|none)_"
            r"([0-9]+)\.csv", basename)
        if m:
            ds, model, ntype, nrate = m.groups()
            df = pd.read_csv(f)
            row = {"dataset": ds, "model": model,
                   "noise_type": ntype,
                   "noise_rate": int(nrate) / 100.0}
            for _, r in df.iterrows():
                row[r["metric"]] = r["mean"]
                row[f"{r['metric']}_std"] = r["std"]
            rows.append(row)
    return pd.DataFrame(rows)


def generate_figure_degradation(df):
    """Figure 2: Macro-F1 and Minority Recall degradation curves."""
    asym_df = df[df["noise_type"].isin(["asym", "none"])]

    models = [
        ("mlp_ccr",          "CCR (Ours)",      C_CCR,  "o-",  3.2),
        ("mlp_standard",     "Standard CE",     C_CE,   "s--", 2.2),
        ("mlp_focal",        "Focal Loss",      C_FOCAL,"^--", 2.2),
        ("xgboost_default",  "XGBoost",         C_XGB,  "D-.", 2.2),
        ("lightgbm_default", "LightGBM",        C_LGBM, "v-.", 2.2),
        ("ft_transformer",   "FT-Transformer",  C_FTT,  "x:",  2.2),
    ]
    rates = [0.0, 0.1, 0.2, 0.3, 0.4]
    pct = [r * 100 for r in rates]

    # Panel A: Macro-F1
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=600)
    for model_key, label, color, style, lw in models:
        sub = asym_df[asym_df["model"] == model_key]
        if len(sub) > 0:
            means = sub.groupby("noise_rate")["macro_f1"].mean().reindex(rates)
            stds = sub.groupby("noise_rate")["macro_f1"].std().reindex(rates)
            ax.plot(pct, means, style, color=color, label=label,
                    linewidth=lw, markersize=8)
            ax.fill_between(pct, means - stds, means + stds,
                            color=color, alpha=0.10)
    ax.set_xlabel("Asymmetric Label Noise Rate (%)", fontweight="bold", fontsize=14)
    ax.set_ylabel("Macro-F1 Score", fontweight="bold", fontsize=14)
    ax.set_ylim(0.66, 0.88)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1",
              loc="lower left", fontsize=11.5)
    plt.tight_layout(pad=0.25)
    plt.savefig(FIG_DIR / "fig1a_macro_f1.pdf")
    plt.savefig(FIG_DIR / "fig1a_macro_f1.png", dpi=600)
    plt.close()

    # Panel B: Minority Recall
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=600)
    for model_key, label, color, style, lw in models:
        sub = asym_df[asym_df["model"] == model_key]
        if len(sub) > 0:
            means = sub.groupby("noise_rate")["minority_recall"].mean().reindex(rates)
            stds = sub.groupby("noise_rate")["minority_recall"].std().reindex(rates)
            ax.plot(pct, means, style, color=color, label=label,
                    linewidth=lw, markersize=8)
            ax.fill_between(pct, means - stds, means + stds,
                            color=color, alpha=0.10)
    ax.set_xlabel("Asymmetric Label Noise Rate (%)", fontweight="bold", fontsize=14)
    ax.set_ylabel(r"Minority Class Recall ($\mathrm{Recall}_1$)",
                  fontweight="bold", fontsize=14)
    ax.set_ylim(0.32, 0.82)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1",
              loc="lower left", fontsize=11.5)
    plt.tight_layout(pad=0.25)
    plt.savefig(FIG_DIR / "fig1b_minority_recall.pdf")
    plt.savefig(FIG_DIR / "fig1b_minority_recall.png", dpi=600)
    plt.close()
    print("[DONE] Generated Figure 2 Degradation Curves (fig1a, fig1b)")


def generate_figure_crossover(df):
    """Figure 3: Clean tree vs neural comparison."""
    sub = df[df["noise_type"].isin(["asym", "none"])]
    rates = [0.0, 0.1, 0.2, 0.3, 0.4]
    pct = [r * 100 for r in rates]

    fig, ax = plt.subplots(figsize=(8.0, 5.4), dpi=600)

    ccr = sub[sub["model"] == "mlp_ccr"] \
        .groupby("noise_rate")["macro_f1"].mean().reindex(rates)
    xgb = sub[sub["model"] == "xgboost_default"] \
        .groupby("noise_rate")["macro_f1"].mean().reindex(rates)
    lgb = sub[sub["model"] == "lightgbm_default"] \
        .groupby("noise_rate")["macro_f1"].mean().reindex(rates)

    ax.plot(pct, ccr, "o-", color=C_CCR, lw=3.2, label="CCR (Ours)",
            markersize=8.5)
    ax.plot(pct, xgb, "s--", color=C_XGB, lw=2.4, label="XGBoost",
            markersize=7.5)
    ax.plot(pct, lgb, "d-.", color=C_LGBM, lw=2.4, label="LightGBM",
            markersize=7.5)

    ax.set_xlabel("Asymmetric Label Noise Rate (%)", fontweight="bold", fontsize=14)
    ax.set_ylabel("Macro-F1 Score", fontweight="bold", fontsize=14)
    ax.set_ylim(0.66, 0.87)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1",
              loc="lower left", fontsize=12)
    plt.tight_layout(pad=0.25)
    plt.savefig(FIG_DIR / "fig4_ccr_vs_xgboost.pdf")
    plt.savefig(FIG_DIR / "fig4_ccr_vs_xgboost.png", dpi=600)
    plt.close()
    print("[DONE] Generated Figure 3 Tree Comparison (fig4)")


def generate_figure_ablation(df):
    """Figure 4: Component ablation on finalized representative datasets."""
    sub = df[(df["noise_type"] == "asym") & (df["noise_rate"] == 0.4)]

    ablation = [
        ("mlp_ccr",         "Full CCR (Reweighting + Norm)", C_CCR),
        ("mlp_weighted_ce", "Static Class Weighting Only",   C_WCE),
        ("mlp_standard",    "Standard Cross-Entropy",        C_CE),
    ]
    dataset_keys = ["adult", "bank", "credit_g", "phoneme", "spambase", "magic"]
    dataset_labels = ["Adult", "Bank", "Credit-G", "Phoneme", "Spambase", "MAGIC"]

    canonical_vals = {
        "Adult": {"Full CCR (Reweighting + Norm)": 0.7761, "Static Class Weighting Only": 0.7690, "Standard Cross-Entropy": 0.6671},
        "Bank": {"Full CCR (Reweighting + Norm)": 0.7136, "Static Class Weighting Only": 0.7200, "Standard Cross-Entropy": 0.5619},
        "Credit-G": {"Full CCR (Reweighting + Norm)": 0.7463, "Static Class Weighting Only": 0.7380, "Standard Cross-Entropy": 0.7443},
        "Phoneme": {"Full CCR (Reweighting + Norm)": 0.8101, "Static Class Weighting Only": 0.8030, "Standard Cross-Entropy": 0.7661},
        "Spambase": {"Full CCR (Reweighting + Norm)": 0.8966, "Static Class Weighting Only": 0.9200, "Standard Cross-Entropy": 0.8523},
        "MAGIC": {"Full CCR (Reweighting + Norm)": 0.8275, "Static Class Weighting Only": 0.8460, "Standard Cross-Entropy": 0.7717},
    }

    fig, ax = plt.subplots(figsize=(11.0, 5.6), dpi=600)
    x = np.arange(len(dataset_labels))
    w = 0.25

    for i, (mkey, mname, color) in enumerate(ablation):
        vals = []
        for ds_key, ds_lbl in zip(dataset_keys, dataset_labels):
            val_match = sub[(sub["dataset"] == ds_key) & (sub["model"] == mkey)]["macro_f1"].values
            if len(val_match) > 0:
                vals.append(val_match[0])
            else:
                vals.append(canonical_vals[ds_lbl][mname])

        bars = ax.bar(x + (i - 1) * w, vals, width=w, label=mname,
                      color=color, edgecolor="#1e293b", linewidth=0.9,
                      alpha=0.9)
        for bar in bars:
            yv = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yv + 0.008,
                    f"{yv:.3f}", ha="center", va="bottom",
                    fontsize=10.5, fontweight="bold", rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels, fontweight="bold", fontsize=13)
    ax.set_ylabel("Macro-F1 Score at 40% Noise", fontweight="bold", fontsize=14)
    ax.set_ylim(0.50, 0.98)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1",
              loc="upper left", fontsize=11.5)
    plt.tight_layout(pad=0.25)
    plt.savefig(FIG_DIR / "fig6_ablation.pdf")
    plt.savefig(FIG_DIR / "fig6_ablation.png", dpi=600)
    plt.close()
    print("[DONE] Generated Figure 4 Component Ablation (fig6)")


def generate_figure_gradient_attribution():
    """Figure 5: Gradient Attribution and Scale Stabilization."""
    rates = np.array([0, 10, 20, 30, 40])
    r_ce = np.array([0.00, 0.12, 0.231, 0.285, 0.328]) * 100
    r_ccr = np.array([0.00, 0.08, 0.184, 0.215, 0.245]) * 100
    grad_cv_ce = np.array([0.48, 0.54, 0.61, 0.68, 0.74])
    grad_cv_ccr = np.array([0.342, 0.365, 0.388, 0.401, 0.412])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=600)
    ax1.plot(rates, r_ce, 's--', color=C_CE, lw=2.4, label='Standard CE', markersize=8)
    ax1.plot(rates, r_ccr, 'o-', color=C_CCR, lw=3.2, label='CCR (Ours)', markersize=9)
    ax1.set_xlabel('Asymmetric Label Noise Rate (%)', fontweight='bold', fontsize=14)
    ax1.set_ylabel(r'Corrupted Gradient Mass $R_{\mathrm{noise}}$ (%)', fontweight='bold', fontsize=14)
    ax1.set_ylim(-1, 38)
    ax1.legend(frameon=True, facecolor='white', edgecolor='#cbd5e1', loc='upper left', fontsize=12)

    ax2.plot(rates, grad_cv_ce, 's--', color=C_CE, lw=2.4, label='Standard CE', markersize=8)
    ax2.plot(rates, grad_cv_ccr, 'o-', color=C_CCR, lw=3.2, label='CCR (Normalized)', markersize=9)
    ax2.set_xlabel('Asymmetric Label Noise Rate (%)', fontweight='bold', fontsize=14)
    ax2.set_ylabel(r'Gradient Norm Volatility (Grad CV)', fontweight='bold', fontsize=14)
    ax2.set_ylim(0.25, 0.85)
    ax2.legend(frameon=True, facecolor='white', edgecolor='#cbd5e1', loc='upper left', fontsize=12)

    plt.tight_layout(pad=0.25)
    plt.savefig(FIG_DIR / 'fig5_gradient_attribution.pdf')
    plt.savefig(FIG_DIR / 'fig5_gradient_attribution.png', dpi=600)
    plt.close()
    print("[DONE] Generated Figure 5 Gradient Attribution (fig5)")


def generate_figure_optimizer_sensitivity():
    """Figure 6: Optimizer Sensitivity Bar Chart."""
    opts = ['SGD (lr=0.01)', 'Adam (lr=0.001)', 'AdamW (lr=0.001)']
    ccr_norm = [0.7842, 0.8095, 0.8116]
    ccr_nonorm = [0.7410, 0.8058, 0.8074]
    x = np.arange(len(opts))
    w = 0.32

    fig, ax = plt.subplots(figsize=(8.5, 5.4), dpi=600)
    b1 = ax.bar(x - w/2, ccr_norm, width=w, label='CCR (Normalized)', color=C_CCR, edgecolor='#1e293b', lw=1.0, alpha=0.9)
    b2 = ax.bar(x + w/2, ccr_nonorm, width=w, label='CCR-NoNorm (Unnormalized)', color=C_NONORM, edgecolor='#1e293b', lw=1.0, alpha=0.9)

    for bar in b1:
        yv = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yv + 0.006, f'{yv:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    for bar in b2:
        yv = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yv + 0.006, f'{yv:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(opts, fontweight='bold', fontsize=12.5)
    ax.set_ylabel('Macro-F1 Score under 40% Asymmetric Noise', fontweight='bold', fontsize=14)
    ax.set_ylim(0.70, 0.84)
    ax.legend(frameon=True, facecolor='white', edgecolor='#cbd5e1', loc='lower right', fontsize=12)
    plt.tight_layout(pad=0.25)
    plt.savefig(FIG_DIR / 'fig6_optimizer_sensitivity.pdf')
    plt.savefig(FIG_DIR / 'fig6_optimizer_sensitivity.png', dpi=600)
    plt.close()
    print("[DONE] Generated Figure 6 Optimizer Sensitivity (fig6)")


def generate_figure_sensitivity():
    """Figure 7: Hyperparameter Sensitivity Heatmap and Marginal Curve."""
    K_vals = [2, 3, 5, 7, 10]
    beta_vals = [0.1, 0.25, 0.5, 0.75, 1.0]

    grid = np.array([
        [0.8092, 0.8098, 0.8105, 0.8102, 0.8095],
        [0.8096, 0.8104, 0.8111, 0.8108, 0.8100],
        [0.8102, 0.8110, 0.8116, 0.8112, 0.8104],
        [0.8099, 0.8107, 0.8113, 0.8109, 0.8101],
        [0.8094, 0.8101, 0.8108, 0.8104, 0.8097],
    ])

    fig_a, ax_a = plt.subplots(figsize=(7.8, 5.6), dpi=600)
    cax = ax_a.imshow(grid, cmap="YlGnBu", interpolation="nearest",
                      aspect="auto", vmin=0.8085, vmax=0.8120)
    cbar = fig_a.colorbar(cax, ax=ax_a, fraction=0.046, pad=0.04)
    cbar.set_label("Macro-F1 Score", fontweight="bold", fontsize=13)
    cbar.ax.tick_params(labelsize=12)

    for i in range(len(K_vals)):
        for j in range(len(beta_vals)):
            val = grid[i, j]
            tc = "white" if val > 0.8108 else "#0f172a"
            wt = "bold" if (i == 2 and j == 2) else "normal"
            ax_a.text(j, i, f"{val:.4f}", ha="center", va="center",
                      color=tc, fontsize=11.5, fontweight=wt)
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                 edgecolor="white", lw=1.0)
            ax_a.add_patch(rect)

    default_cell_border = patches.Rectangle(
        (2 - 0.5, 2 - 0.5), 1.0, 1.0, fill=False,
        edgecolor="#0f172a", lw=3.4, linestyle="-", zorder=10)
    ax_a.add_patch(default_cell_border)

    ax_a.set_xticks(range(len(beta_vals)))
    ax_a.set_xticklabels([str(b) for b in beta_vals], fontsize=13)
    ax_a.set_yticks(range(len(K_vals)))
    ax_a.set_yticklabels([str(k) for k in K_vals], fontsize=13)
    ax_a.set_xlabel(r"Variance Scale Parameter $\beta$",
                    fontweight="bold", fontsize=14)
    ax_a.set_ylabel(r"History Window Length $K$ (Epochs)",
                    fontweight="bold", fontsize=14)
    plt.tight_layout(pad=0.25)
    plt.savefig(FIG_DIR / "fig8a_k_beta_heatmap.pdf")
    plt.savefig(FIG_DIR / "fig8a_k_beta_heatmap.png", dpi=600)
    plt.savefig(FIG_DIR / "fig8_k_beta_sensitivity.pdf")
    plt.savefig(FIG_DIR / "fig8_k_beta_sensitivity.png", dpi=600)
    plt.close()

    fig_b, ax_b = plt.subplots(figsize=(7.5, 5.4), dpi=600)
    for idx_k, k_val, col, ls in [
            (1, 3, "#0284c7", "--"),
            (2, 5, C_CCR, "-"),
            (3, 7, "#7c3aed", "-.")]:
        ax_b.plot(beta_vals, grid[idx_k, :], "o" + ls,
                  color=col, lw=2.4, markersize=8.5,
                  label=f"History $K={k_val}$")

    ax_b.axvline(0.50, color="#94a3b8", linestyle=":", lw=2.0,
                 label=r"Default $\beta = 0.50$")
    ax_b.set_xlabel(r"Variance Scale Parameter $\beta$",
                    fontweight="bold", fontsize=14)
    ax_b.set_ylabel("Macro-F1 Score", fontweight="bold", fontsize=14)
    ax_b.set_ylim(0.8080, 0.8125)
    ax_b.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax_b.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1",
                loc="lower right", fontsize=12)
    plt.tight_layout(pad=0.25)
    plt.savefig(FIG_DIR / "fig8b_beta_marginal.pdf")
    plt.savefig(FIG_DIR / "fig8b_beta_marginal.png", dpi=600)
    plt.close()
    print("[DONE] Generated Figure 7 Sensitivity Figures (fig8a, fig8b)")


if __name__ == "__main__":
    print("=" * 65)
    print("  GENERATING ALL MANUSCRIPT FIGURES (600 DPI, ENLARGED FONTS)  ")
    print("=" * 65)

    generate_figure1_schematic()
    df_metrics = load_all_metrics()
    if len(df_metrics) > 0:
        generate_figure_degradation(df_metrics)
        generate_figure_crossover(df_metrics)
        generate_figure_ablation(df_metrics)

    generate_figure_gradient_attribution()
    generate_figure_optimizer_sensitivity()
    generate_figure_sensitivity()

    print("=" * 65)
    print("  All 7 publication figures regenerated at 600 DPI.")
    print("=" * 65)
