"""
paper_figures.py — CCR-Tabular Publication Figure Generator
============================================================
Generates all tables and figures needed for the IEEE TNNLS paper.

Usage:
    python paper_figures.py

Outputs (outputs/plots/):
    fig1_main_results.png / .pdf        — Main results table as heatmap
    fig2_noise_degradation_asym.png/.pdf — F1 degradation under asymmetric noise
    fig3_noise_degradation_feat.png/.pdf — F1 degradation under feature-correlated noise
    fig4_minority_recall_bar.png/.pdf   — Minority recall comparison (clean + noisy)
    fig5_improvement_heatmap.png/.pdf   — CCR delta over best baseline
    fig6_training_time.png/.pdf         — Training time comparison
    table1_clean_results.csv            — LaTeX-ready main results table
    table2_noise_results.csv            — LaTeX-ready noise results table
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).parent
PLOTS  = ROOT / "outputs" / "plots"
METRICS = ROOT / "outputs" / "metrics"
PLOTS.mkdir(parents=True, exist_ok=True)

# ── Global style — IEEE-compatible ────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "figure.dpi":         150,
    "savefig.dpi":        600,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth":     0.8,
    "grid.linewidth":     0.5,
    "lines.linewidth":    1.5,
    "lines.markersize":   5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

DPI_SAVE = 600

# ── Model display names & colors ──────────────────────────────────────────────
MODEL_ORDER = [
    "mlp_standard", "mlp_focal", "mlp_weighted_ce", "mlp_smote",
    "xgboost_default", "xgboost_weighted", "lightgbm_default", "mlp_ccr",
]
MODEL_LABELS = {
    "mlp_standard":    "MLP-CE",
    "mlp_focal":       "MLP-Focal",
    "mlp_weighted_ce": "MLP-WCE",
    "mlp_smote":       "MLP-SMOTE",
    "xgboost_default": "XGBoost",
    "xgboost_weighted":"XGBoost-W",
    "lightgbm_default":"LightGBM",
    "mlp_ccr":         "CCR (Ours)",
}
MODEL_COLORS = {
    "mlp_standard":    "#adb5bd",
    "mlp_focal":       "#4895ef",
    "mlp_weighted_ce": "#4cc9f0",
    "mlp_smote":       "#f77f00",
    "xgboost_default": "#7b2d8b",
    "xgboost_weighted":"#b5179e",
    "lightgbm_default":"#06d6a0",
    "mlp_ccr":         "#d62828",
}
MODEL_MARKERS = {
    "mlp_standard":    "o",
    "mlp_focal":       "s",
    "mlp_weighted_ce": "^",
    "mlp_smote":       "D",
    "xgboost_default": "v",
    "xgboost_weighted":"P",
    "lightgbm_default":"X",
    "mlp_ccr":         "*",
}

DATASET_LABELS = {
    "adult":    "Adult",
    "bank":     "Bank",
    "credit_g": "Credit-G",
    "magic":    "MAGIC",
    "phoneme":  "Phoneme",
    "spambase": "Spambase",
}
DATASETS = ["adult", "bank", "credit_g", "magic", "phoneme", "spambase"]

NOISE_RATES = [0.0, 0.1, 0.2, 0.3]


def save(fig, name):
    """Save figure as both PNG and PDF at 600 DPI."""
    for ext in ("png", "pdf"):
        path = PLOTS / f"{name}.{ext}"
        fig.savefig(path, dpi=DPI_SAVE, format=ext)
    print(f"  Saved {name}.png / .pdf")
    plt.close(fig)


def load_data():
    p = METRICS / "results.csv"
    if not p.exists():
        raise FileNotFoundError(f"results.csv not found at {p}. Run experiments first.")
    df = pd.read_csv(p)
    print(f"Loaded {len(df)} rows from results.csv")
    return df


def agg(df, noise_type, noise_rate, metric):
    """Return mean ± std per (dataset, model) for a given condition."""
    mask = (df["noise_type"] == noise_type) & (df["noise_rate"].round(3) == round(noise_rate, 3))
    g = df[mask].groupby(["dataset", "model"])[metric]
    return g.mean().rename("mean").reset_index().merge(
        g.std().rename("std").reset_index(), on=["dataset", "model"]
    )


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Main results bar chart: Macro F1 and Minority Recall (clean data)
# Layout: 2 rows × 6 cols  (top=F1, bottom=Recall), one column per dataset
# Size: 7.0 × 4.5 inches  (fits a two-column IEEE page)
# ══════════════════════════════════════════════════════════════════════════════
def fig1_main_results(df):
    print("Generating Figure 1 — Main results (clean data)...")

    metrics = [("macro_f1", "Macro F1"), ("minority_recall", "Minority Recall")]
    n_ds = len(DATASETS)
    n_models = len(MODEL_ORDER)
    x = np.arange(n_models)
    bar_w = 0.65

    fig, axes = plt.subplots(
        2, n_ds,
        figsize=(7.0, 4.5),
        sharey="row",
        gridspec_kw={"hspace": 0.45, "wspace": 0.08},
    )

    for row, (metric, ylabel) in enumerate(metrics):
        data = agg(df, "none", 0.0, metric)
        for col, ds in enumerate(DATASETS):
            ax = axes[row, col]
            sub = data[data["dataset"] == ds].set_index("model")

            means = [sub.loc[m, "mean"] if m in sub.index else 0.0 for m in MODEL_ORDER]
            stds  = [sub.loc[m, "std"]  if m in sub.index else 0.0 for m in MODEL_ORDER]
            colors = [MODEL_COLORS[m] for m in MODEL_ORDER]

            bars = ax.bar(x, means, bar_w, yerr=stds, capsize=2,
                          color=colors, edgecolor="white", linewidth=0.4,
                          error_kw={"linewidth": 0.8, "ecolor": "#555"})

            # Bold border on CCR bar
            ccr_idx = MODEL_ORDER.index("mlp_ccr")
            bars[ccr_idx].set_edgecolor("#000")
            bars[ccr_idx].set_linewidth(1.5)

            ax.set_xticks([])
            ax.set_xlim(-0.5, n_models - 0.5)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(4, prune="both"))
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(axis="y", linewidth=0.4, alpha=0.6)

            if row == 0:
                ax.set_title(DATASET_LABELS[ds], fontsize=9, fontweight="bold", pad=3)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=8)

    # Shared legend below the figure
    patches = [mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_LABELS[m])
               for m in MODEL_ORDER]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               fontsize=7.5, frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.04), columnspacing=0.8, handlelength=1.2)

    fig.suptitle("Macro F1 and Minority Recall — Clean Labels",
                 fontsize=10, fontweight="bold", y=1.01)
    save(fig, "fig1_main_results")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Noise degradation curves: Asymmetric noise
# Layout: 2 rows × 3 cols  (top=F1, bottom=Recall), 3 datasets per row
# Size: 7.0 × 4.5 inches
# ══════════════════════════════════════════════════════════════════════════════
def fig_noise_degradation(df, noise_type, label, fname):
    print(f"Generating noise degradation figure — {label}...")

    metrics = [("macro_f1", "Macro F1"), ("minority_recall", "Minority Recall")]
    rates = [0.0, 0.1, 0.2, 0.3]
    # Show only the most informative models to keep the plot readable
    focus = ["mlp_ccr", "mlp_weighted_ce", "mlp_focal",
             "xgboost_weighted", "lightgbm_default"]

    fig, axes = plt.subplots(
        2, 3,
        figsize=(7.0, 4.5),
        sharey="row",
        gridspec_kw={"hspace": 0.45, "wspace": 0.12},
    )

    for row, (metric, ylabel) in enumerate(metrics):
        for col, ds in enumerate(DATASETS[:3] if row == 0 else DATASETS[3:]):
            # row 0 → first 3 datasets, row 1 → last 3 datasets
            pass

    # Redo with correct dataset assignment
    ds_rows = [DATASETS[:3], DATASETS[3:]]

    for row, (metric, ylabel) in enumerate(metrics):
        for col, ds in enumerate(DATASETS):
            ax_row = row
            ax_col = col if col < 3 else col - 3
            # We need a 2×6 layout for this to work cleanly
            pass

    plt.close(fig)

    # Cleaner approach: 2 rows × 3 cols, top row = first 3 datasets, bottom = last 3
    fig, axes = plt.subplots(
        2, 3,
        figsize=(7.0, 4.5),
        gridspec_kw={"hspace": 0.50, "wspace": 0.25},
    )

    for panel_row, ds_group in enumerate([DATASETS[:3], DATASETS[3:]]):
        for panel_col, ds in enumerate(ds_group):
            ax = axes[panel_row, panel_col]

            for m in focus:
                means, errs, xs = [], [], []
                for rate in rates:
                    nt = "none" if rate == 0.0 else noise_type
                    sub = df[(df["dataset"] == ds) &
                             (df["model"] == m) &
                             (df["noise_type"] == nt) &
                             (df["noise_rate"].round(3) == round(rate, 3))]["macro_f1"]
                    if len(sub) > 0:
                        means.append(sub.mean())
                        errs.append(sub.std())
                        xs.append(rate)

                if means:
                    lw = 2.2 if m == "mlp_ccr" else 1.2
                    ls = "-" if m == "mlp_ccr" else "--"
                    zorder = 5 if m == "mlp_ccr" else 2
                    ax.plot(xs, means,
                            color=MODEL_COLORS[m],
                            marker=MODEL_MARKERS[m],
                            markersize=4 if m == "mlp_ccr" else 3,
                            linewidth=lw, linestyle=ls,
                            zorder=zorder,
                            label=MODEL_LABELS[m])
                    ax.fill_between(xs,
                                    [v - e for v, e in zip(means, errs)],
                                    [v + e for v, e in zip(means, errs)],
                                    alpha=0.10, color=MODEL_COLORS[m], zorder=1)

            ax.set_title(DATASET_LABELS[ds], fontsize=9, fontweight="bold", pad=3)
            ax.set_xlabel("Noise Rate", fontsize=8)
            if panel_col == 0:
                ax.set_ylabel("Macro F1", fontsize=8)
            ax.set_xlim(-0.02, 0.32)
            ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            ax.xaxis.set_major_locator(mticker.FixedLocator([0.0, 0.1, 0.2, 0.3]))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(4, prune="both"))
            ax.tick_params(labelsize=7)
            ax.grid(linewidth=0.4, alpha=0.5)

    # Legend
    handles = [mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_LABELS[m])
               for m in focus]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=7.5, frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.04), columnspacing=0.6, handlelength=1.2)

    fig.suptitle(f"Macro F1 Degradation — {label} Noise",
                 fontsize=10, fontweight="bold", y=1.01)
    save(fig, fname)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — CCR vs Best Baseline: Minority Recall across noise levels
# Layout: 1 row × 6 cols, grouped bars at each noise rate
# Size: 7.0 × 3.0 inches
# ══════════════════════════════════════════════════════════════════════════════
def fig3_recall_comparison(df):
    print("Generating Figure 3 — Minority Recall comparison...")

    noise_configs = [
        ("none", 0.0, "Clean"),
        ("asym", 0.1, "Asym 10%"),
        ("asym", 0.2, "Asym 20%"),
        ("asym", 0.3, "Asym 30%"),
    ]
    focus_models = ["mlp_ccr", "mlp_weighted_ce", "mlp_smote",
                    "xgboost_weighted", "lightgbm_default"]
    n_configs = len(noise_configs)
    n_models  = len(focus_models)
    group_w   = 0.75
    bar_w     = group_w / n_models

    fig, axes = plt.subplots(
        1, len(DATASETS),
        figsize=(7.0, 3.0),
        sharey=True,
        gridspec_kw={"wspace": 0.06},
    )

    for col, ds in enumerate(DATASETS):
        ax = axes[col]
        for mi, m in enumerate(focus_models):
            xs, ys, es = [], [], []
            for ci, (nt, nr, _) in enumerate(noise_configs):
                sub = df[(df["dataset"] == ds) &
                         (df["model"] == m) &
                         (df["noise_type"] == nt) &
                         (df["noise_rate"].round(3) == round(nr, 3))]["minority_recall"]
                if len(sub) > 0:
                    xs.append(ci + (mi - n_models / 2 + 0.5) * bar_w)
                    ys.append(sub.mean())
                    es.append(sub.std())

            if xs:
                bars = ax.bar(xs, ys, bar_w * 0.9,
                              color=MODEL_COLORS[m],
                              edgecolor="white", linewidth=0.3)
                if m == "mlp_ccr":
                    for b in bars:
                        b.set_edgecolor("#000")
                        b.set_linewidth(1.2)

        ax.set_title(DATASET_LABELS[ds], fontsize=8, fontweight="bold", pad=3)
        ax.set_xticks(range(n_configs))
        ax.set_xticklabels([c[2] for c in noise_configs], fontsize=6.5, rotation=30, ha="right")
        ax.yaxis.set_major_locator(mticker.MaxNLocator(4, prune="both"))
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", linewidth=0.4, alpha=0.6)
        if col == 0:
            ax.set_ylabel("Minority Recall", fontsize=8)

    patches = [mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_LABELS[m])
               for m in focus_models]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               fontsize=7.5, frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.08), columnspacing=0.6, handlelength=1.2)

    fig.suptitle("Minority-Class Recall — Clean and Asymmetric Noise Conditions",
                 fontsize=10, fontweight="bold", y=1.02)
    save(fig, "fig3_minority_recall")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Improvement heatmap: CCR delta over best baseline
# Layout: 3 subplots (clean, asym@20%, feat@20%), each a 6×3 heatmap
# Size: 7.0 × 3.5 inches
# ══════════════════════════════════════════════════════════════════════════════
def fig4_improvement_heatmap(df):
    print("Generating Figure 4 — Improvement heatmap...")

    conditions = [
        ("none", 0.0, "Clean"),
        ("asym", 0.2, "Asym 20%"),
        ("feat", 0.2, "Feat 20%"),
    ]
    metrics_show = ["macro_f1", "minority_recall", "auc_roc"]
    metric_labels = ["Macro F1", "Min. Recall", "AUC-ROC"]
    baselines = [m for m in MODEL_ORDER if m != "mlp_ccr"]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(7.0, 3.5),
        gridspec_kw={"wspace": 0.35},
    )

    for ax, (nt, nr, title) in zip(axes, conditions):
        sub = df[(df["noise_type"] == nt) & (df["noise_rate"].round(3) == round(nr, 3))]
        if len(sub) == 0:
            ax.set_visible(False)
            continue

        ccr_means = sub[sub["model"] == "mlp_ccr"].groupby("dataset")[metrics_show].mean()
        bl_means  = sub[sub["model"].isin(baselines)].groupby(["dataset", "model"])[metrics_show].mean()
        best_bl   = bl_means.groupby(level=0).max()

        delta = (ccr_means - best_bl).reindex(DATASETS)
        delta.index = [DATASET_LABELS[d] for d in DATASETS]
        delta.columns = metric_labels

        vmax = max(abs(delta.values.max()), abs(delta.values.min()), 0.01)
        sns.heatmap(
            delta,
            ax=ax,
            annot=True,
            fmt=".3f",
            annot_kws={"size": 7.5},
            center=0,
            vmin=-vmax, vmax=vmax,
            cmap="RdYlGn",
            linewidths=0.4,
            linecolor="#ccc",
            cbar_kws={"shrink": 0.75, "label": "Δ (CCR − Best Baseline)"},
        )
        ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=7.5)
        ax.collections[0].colorbar.ax.tick_params(labelsize=7)
        ax.collections[0].colorbar.set_label("Δ (CCR − Best Baseline)", size=7)

    fig.suptitle("CCR Improvement over Best Baseline per Metric",
                 fontsize=10, fontweight="bold", y=1.02)
    save(fig, "fig4_improvement_heatmap")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Training time: CCR vs baselines (clean condition)
# Layout: 1 row × 6 cols
# Size: 7.0 × 2.8 inches
# ══════════════════════════════════════════════════════════════════════════════
def fig5_training_time(df):
    print("Generating Figure 5 — Training time...")

    if "train_time_s" not in df.columns or df["train_time_s"].isna().all():
        print("  Skipped — train_time_s not available in results.csv")
        return

    clean = df[(df["noise_type"] == "none") & df["train_time_s"].notna()]
    time_data = clean.groupby(["dataset", "model"])["train_time_s"].mean().reset_index()

    n_models = len(MODEL_ORDER)
    x = np.arange(n_models)
    bar_w = 0.65

    fig, axes = plt.subplots(
        1, len(DATASETS),
        figsize=(7.0, 2.8),
        sharey=False,
        gridspec_kw={"wspace": 0.12},
    )

    for col, ds in enumerate(DATASETS):
        ax = axes[col]
        sub = time_data[time_data["dataset"] == ds].set_index("model")
        times  = [sub.loc[m, "train_time_s"] if m in sub.index else 0.0 for m in MODEL_ORDER]
        colors = [MODEL_COLORS[m] for m in MODEL_ORDER]

        bars = ax.bar(x, times, bar_w, color=colors,
                      edgecolor="white", linewidth=0.4)
        ccr_idx = MODEL_ORDER.index("mlp_ccr")
        bars[ccr_idx].set_edgecolor("#000")
        bars[ccr_idx].set_linewidth(1.5)

        ax.set_title(DATASET_LABELS[ds], fontsize=8, fontweight="bold", pad=3)
        ax.set_xticks([])
        ax.yaxis.set_major_locator(mticker.MaxNLocator(4, prune="both"))
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", linewidth=0.4, alpha=0.6)
        if col == 0:
            ax.set_ylabel("Time (s)", fontsize=8)

    patches = [mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_LABELS[m])
               for m in MODEL_ORDER]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               fontsize=7.5, frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.08), columnspacing=0.6, handlelength=1.2)

    fig.suptitle("Mean Training Time per Run (Clean Condition)",
                 fontsize=10, fontweight="bold", y=1.02)
    save(fig, "fig5_training_time")


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 1 — Main results table (clean data): mean ± std for all models
# Saved as CSV (easy to paste into LaTeX)
# ══════════════════════════════════════════════════════════════════════════════
def table1_clean_results(df):
    print("Generating Table 1 — Clean results...")

    clean = df[df["noise_type"] == "none"]
    rows = []
    for m in MODEL_ORDER:
        sub = clean[clean["model"] == m]
        row = {"Model": MODEL_LABELS[m]}
        for ds in DATASETS:
            ds_sub = sub[sub["dataset"] == ds]
            if len(ds_sub) > 0:
                f1  = ds_sub["macro_f1"].mean()
                rec = ds_sub["minority_recall"].mean()
                row[f"{DATASET_LABELS[ds]}_F1"]  = f"{f1:.3f}"
                row[f"{DATASET_LABELS[ds]}_Rec"] = f"{rec:.3f}"
            else:
                row[f"{DATASET_LABELS[ds]}_F1"]  = "—"
                row[f"{DATASET_LABELS[ds]}_Rec"] = "—"
        rows.append(row)

    table = pd.DataFrame(rows)
    out = METRICS / "table1_clean_results.csv"
    table.to_csv(out, index=False)
    print(f"  Saved table1_clean_results.csv")
    print(table.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 2 — Noise robustness: mean Macro F1 across all datasets per noise level
# ══════════════════════════════════════════════════════════════════════════════
def table2_noise_robustness(df):
    print("Generating Table 2 — Noise robustness...")

    configs = [
        ("none", 0.0, "Clean"),
        ("asym", 0.1, "Asym 10%"),
        ("asym", 0.2, "Asym 20%"),
        ("asym", 0.3, "Asym 30%"),
        ("feat", 0.1, "Feat 10%"),
        ("feat", 0.2, "Feat 20%"),
        ("feat", 0.3, "Feat 30%"),
    ]

    rows = []
    for m in MODEL_ORDER:
        row = {"Model": MODEL_LABELS[m]}
        for nt, nr, label in configs:
            sub = df[(df["model"] == m) &
                     (df["noise_type"] == nt) &
                     (df["noise_rate"].round(3) == round(nr, 3))]["macro_f1"]
            row[label] = f"{sub.mean():.3f} ± {sub.std():.3f}" if len(sub) > 0 else "—"
        rows.append(row)

    table = pd.DataFrame(rows)
    out = METRICS / "table2_noise_robustness.csv"
    table.to_csv(out, index=False)
    print(f"  Saved table2_noise_robustness.csv")
    print(table.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  CCR-Tabular — Paper Figure Generator")
    print("=" * 60)

    df = load_data()

    print("\nGenerating figures...")
    fig1_main_results(df)
    fig_noise_degradation(df, "asym", "Asymmetric", "fig2_noise_degradation_asym")
    fig_noise_degradation(df, "feat", "Feature-Correlated", "fig2_noise_degradation_feat")
    fig3_recall_comparison(df)
    fig4_improvement_heatmap(df)
    fig5_training_time(df)

    print("\nGenerating tables...")
    table1_clean_results(df)
    table2_noise_robustness(df)

    print("\n" + "=" * 60)
    print(f"  All outputs saved to: {PLOTS}")
    print(f"  Tables saved to:      {METRICS}")
    print("=" * 60)

