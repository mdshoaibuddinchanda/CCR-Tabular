"""
paper_figures.py - CCR-Tabular Publication Figure Generator
Run: python paper_figures.py
Outputs: outputs/plots/  and  outputs/metrics/
"""

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

ROOT    = Path(__file__).parent
PLOTS   = ROOT / "outputs" / "plots"
METRICS = ROOT / "outputs" / "metrics"
PLOTS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style - IEEE serif, clean, generous whitespace
# ---------------------------------------------------------------------------
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
    "savefig.pad_inches": 0.08,
    "axes.linewidth":     0.6,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "#e8e8e8",
    "grid.linewidth":     0.5,
    "lines.linewidth":    1.6,
    "lines.markersize":   5,
    "xtick.major.size":   3,
    "ytick.major.size":   3,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.major.pad":    3,
    "ytick.major.pad":    3,
})

DPI = 600

MODEL_ORDER = [
    "mlp_standard", "mlp_focal", "mlp_weighted_ce", "mlp_smote",
    "xgboost_default", "xgboost_weighted", "lightgbm_default", "mlp_ccr",
]
LABELS = {
    "mlp_standard":    "MLP-CE",
    "mlp_focal":       "MLP-Focal",
    "mlp_weighted_ce": "MLP-WCE",
    "mlp_smote":       "MLP-SMOTE",
    "xgboost_default": "XGBoost",
    "xgboost_weighted":"XGBoost-W",
    "lightgbm_default":"LightGBM",
    "mlp_ccr":         "CCR (Ours)",
}
COLORS = {
    "mlp_standard":    "#adb5bd",
    "mlp_focal":       "#4895ef",
    "mlp_weighted_ce": "#43aa8b",
    "mlp_smote":       "#f4a261",
    "xgboost_default": "#9b72cf",
    "xgboost_weighted":"#c77dff",
    "lightgbm_default":"#48cae4",
    "mlp_ccr":         "#e63946",
}
MARKERS = {
    "mlp_standard":    "o",
    "mlp_focal":       "s",
    "mlp_weighted_ce": "^",
    "mlp_smote":       "D",
    "xgboost_default": "v",
    "xgboost_weighted":"P",
    "lightgbm_default":"X",
    "mlp_ccr":         "*",
}

DS_ORDER  = ["adult", "bank", "credit_g", "magic", "phoneme", "spambase"]
DS_LABELS = {
    "adult":    "Adult",
    "bank":     "Bank",
    "credit_g": "Credit-G",
    "magic":    "MAGIC",
    "phoneme":  "Phoneme",
    "spambase": "Spambase",
}


def load_data():
    p = METRICS / "results.csv"
    if not p.exists():
        raise FileNotFoundError(f"results.csv not found at {p}")
    df = pd.read_csv(p)
    print(f"Loaded {len(df):,} rows.")
    return df


def get(df, noise_type, noise_rate, metric, model=None, dataset=None):
    mask = (
        (df["noise_type"] == noise_type) &
        (df["noise_rate"].round(3) == round(noise_rate, 3))
    )
    if model:
        mask &= df["model"] == model
    if dataset:
        mask &= df["dataset"] == dataset
    return df[mask][metric]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS / f"{name}.{ext}", dpi=DPI, format=ext)
    plt.close(fig)
    print(f"  -> {name}.png / .pdf")


def patches(models):
    return [mpatches.Patch(color=COLORS[m], label=LABELS[m]) for m in models]


# ---------------------------------------------------------------------------
# FIG 1  Clean-data grouped bar chart
# Layout: 1 row x 6 datasets, two metrics side by side per dataset
# Each subplot has exactly 8 bars with NO x-axis labels (legend only)
# Figure is TALL enough so bars never crowd
# ---------------------------------------------------------------------------
def fig1_clean_results(df):
    print("Fig 1 - Clean results...")

    for metric, ylabel, fname in [
        ("macro_f1",        "Macro F1",        "fig1a_macro_f1"),
        ("minority_recall", "Minority Recall", "fig1b_minority_recall"),
    ]:
        # One row, 6 columns - each column is one dataset
        # Tall enough: 3.8 in height gives bars room to breathe
        fig, axes = plt.subplots(
            1, 6,
            figsize=(10.0, 3.8),
            gridspec_kw={"wspace": 0.18},
        )

        n_m = len(MODEL_ORDER)
        x   = np.arange(n_m)
        bw  = 0.62

        for col, ds in enumerate(DS_ORDER):
            ax = axes[col]
            sub = df[(df["noise_type"] == "none") & (df["dataset"] == ds)]

            means  = [sub[sub["model"] == m][metric].mean() for m in MODEL_ORDER]
            colors = [COLORS[m] for m in MODEL_ORDER]

            bars = ax.bar(x, means, bw, color=colors, edgecolor="none", zorder=3)

            # CCR gets a thin black outline so it stands out
            ccr_i = MODEL_ORDER.index("mlp_ccr")
            bars[ccr_i].set_edgecolor("#111")
            bars[ccr_i].set_linewidth(1.0)

            ax.set_title(DS_LABELS[ds], fontsize=9, fontweight="bold", pad=4)
            ax.set_xticks([])
            ax.set_xlim(-0.6, n_m - 0.4)
            ax.yaxis.set_major_locator(mticker.MaxNLocator(4, prune="both"))
            ax.tick_params(axis="y", labelsize=7.5, length=3, width=0.6, pad=2)
            ax.spines["left"].set_linewidth(0.6)
            ax.spines["bottom"].set_linewidth(0.6)
            ax.grid(axis="y", linewidth=0.4, alpha=0.9, zorder=0)
            ax.grid(axis="x", visible=False)

            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9, labelpad=4)

        # Legend below - 4 items per row = 2 rows, plenty of space
        fig.legend(
            handles=patches(MODEL_ORDER),
            loc="lower center",
            ncol=4,
            fontsize=8,
            frameon=False,
            bbox_to_anchor=(0.5, -0.12),
            columnspacing=1.0,
            handlelength=1.2,
            handletextpad=0.5,
        )

        fig.suptitle(
            f"{ylabel} on Clean Data  (Mean, 5-fold x 3-seed CV)",
            fontsize=10, fontweight="bold", y=1.02,
        )
        save(fig, fname)


# ---------------------------------------------------------------------------
# FIG 2  Noise degradation line plots
# Layout: 2 rows x 3 cols, one subplot per dataset
# Only 5 models shown - keeps lines readable
# ---------------------------------------------------------------------------
def fig2_noise_degradation(df, noise_type, label, fname):
    print(f"Fig 2 - Noise degradation ({label})...")

    focus = ["mlp_ccr", "mlp_weighted_ce", "mlp_focal",
             "xgboost_weighted", "lightgbm_default"]
    rates = [0.0, 0.1, 0.2, 0.3]

    fig, axes = plt.subplots(
        2, 3,
        figsize=(7.5, 5.0),
        gridspec_kw={"hspace": 0.55, "wspace": 0.32},
    )

    for row, ds_group in enumerate([DS_ORDER[:3], DS_ORDER[3:]]):
        for col, ds in enumerate(ds_group):
            ax = axes[row, col]

            for m in focus:
                xs, ys, es = [], [], []
                for r in rates:
                    nt = "none" if r == 0.0 else noise_type
                    s = get(df, nt, r, "macro_f1", model=m, dataset=ds)
                    if len(s):
                        xs.append(r)
                        ys.append(s.mean())
                        es.append(s.std())
                if not xs:
                    continue

                lw = 2.0 if m == "mlp_ccr" else 1.0
                ls = "-"  if m == "mlp_ccr" else "--"
                zo = 5    if m == "mlp_ccr" else 2
                ms = 6    if m == "mlp_ccr" else 4
                ax.plot(xs, ys, color=COLORS[m], marker=MARKERS[m],
                        linewidth=lw, linestyle=ls, markersize=ms, zorder=zo)
                ax.fill_between(
                    xs,
                    [v - e for v, e in zip(ys, es)],
                    [v + e for v, e in zip(ys, es)],
                    alpha=0.07, color=COLORS[m], zorder=1,
                )

            ax.set_title(DS_LABELS[ds], fontsize=9, fontweight="bold", pad=4)
            ax.set_xlabel("Noise Rate", fontsize=8, labelpad=3)
            ax.set_xlim(-0.02, 0.32)
            ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            ax.xaxis.set_major_locator(mticker.FixedLocator([0.0, 0.1, 0.2, 0.3]))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(4, prune="both"))
            ax.tick_params(labelsize=7.5, length=3, width=0.6, pad=2)
            ax.spines["left"].set_linewidth(0.6)
            ax.spines["bottom"].set_linewidth(0.6)
            if col == 0:
                ax.set_ylabel("Macro F1", fontsize=9, labelpad=4)

    fig.legend(
        handles=patches(focus),
        loc="lower center", ncol=5,
        fontsize=8, frameon=False,
        bbox_to_anchor=(0.5, -0.06),
        columnspacing=0.8, handlelength=1.2, handletextpad=0.5,
    )
    fig.suptitle(
        f"Macro F1 Degradation - {label} Noise",
        fontsize=10, fontweight="bold", y=1.01,
    )
    save(fig, fname)


# ---------------------------------------------------------------------------
# FIG 3  Minority Recall degradation (asymmetric noise)
# Same layout as Fig 2
# ---------------------------------------------------------------------------
def fig3_minority_recall(df):
    print("Fig 3 - Minority Recall degradation...")

    focus = ["mlp_ccr", "mlp_weighted_ce", "mlp_smote",
             "xgboost_weighted", "lightgbm_default"]
    rates = [0.0, 0.1, 0.2, 0.3]

    fig, axes = plt.subplots(
        2, 3,
        figsize=(7.5, 5.0),
        gridspec_kw={"hspace": 0.55, "wspace": 0.32},
    )

    for row, ds_group in enumerate([DS_ORDER[:3], DS_ORDER[3:]]):
        for col, ds in enumerate(ds_group):
            ax = axes[row, col]

            for m in focus:
                xs, ys = [], []
                for r in rates:
                    nt = "none" if r == 0.0 else "asym"
                    s = get(df, nt, r, "minority_recall", model=m, dataset=ds)
                    if len(s):
                        xs.append(r)
                        ys.append(s.mean())
                if not xs:
                    continue

                lw = 2.0 if m == "mlp_ccr" else 1.0
                ls = "-"  if m == "mlp_ccr" else "--"
                zo = 5    if m == "mlp_ccr" else 2
                ms = 6    if m == "mlp_ccr" else 4
                ax.plot(xs, ys, color=COLORS[m], marker=MARKERS[m],
                        linewidth=lw, linestyle=ls, markersize=ms, zorder=zo)

            ax.set_title(DS_LABELS[ds], fontsize=9, fontweight="bold", pad=4)
            ax.set_xlabel("Noise Rate", fontsize=8, labelpad=3)
            ax.set_xlim(-0.02, 0.32)
            ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            ax.xaxis.set_major_locator(mticker.FixedLocator([0.0, 0.1, 0.2, 0.3]))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(4, prune="both"))
            ax.tick_params(labelsize=7.5, length=3, width=0.6, pad=2)
            ax.spines["left"].set_linewidth(0.6)
            ax.spines["bottom"].set_linewidth(0.6)
            if col == 0:
                ax.set_ylabel("Minority Recall", fontsize=9, labelpad=4)

    fig.legend(
        handles=patches(focus),
        loc="lower center", ncol=5,
        fontsize=8, frameon=False,
        bbox_to_anchor=(0.5, -0.06),
        columnspacing=0.8, handlelength=1.2, handletextpad=0.5,
    )
    fig.suptitle(
        "Minority-Class Recall - Asymmetric Noise",
        fontsize=10, fontweight="bold", y=1.01,
    )
    save(fig, "fig3_minority_recall")


# ---------------------------------------------------------------------------
# FIG 4  Heatmap: CCR vs XGBoost-W across noise conditions
# Shows delta (CCR - XGBoost-W) for Macro F1 and Minority Recall
# Conditions: clean, asym@20%, asym@30%, feat@20%, feat@30%
# Green = CCR wins, Red = XGBoost-W wins
# This framing shows CCR's noise-robustness advantage clearly
# ---------------------------------------------------------------------------
def fig4_heatmap(df):
    print("Fig 4 - CCR vs XGBoost-W heatmap...")

    conditions = [
        ("none", 0.0, "Clean"),
        ("asym", 0.1, "Asym 10%"),
        ("asym", 0.2, "Asym 20%"),
        ("asym", 0.3, "Asym 30%"),
        ("feat", 0.1, "Feat 10%"),
        ("feat", 0.2, "Feat 20%"),
        ("feat", 0.3, "Feat 30%"),
    ]
    metrics_show  = ["macro_f1", "minority_recall", "auc_roc"]
    metric_labels = ["Macro F1", "Min. Recall", "AUC-ROC"]

    # Build delta matrix: rows = conditions, cols = metrics, one panel per dataset
    # Layout: 2 rows x 3 cols (one subplot per dataset)
    fig, axes = plt.subplots(
        2, 3,
        figsize=(7.5, 5.0),
        gridspec_kw={"hspace": 0.55, "wspace": 0.55},
    )

    for panel_row, ds_group in enumerate([DS_ORDER[:3], DS_ORDER[3:]]):
        for panel_col, ds in enumerate(ds_group):
            ax = axes[panel_row, panel_col]

            rows = []
            for nt, nr, cond_label in conditions:
                sub = df[
                    (df["noise_type"] == nt) &
                    (df["noise_rate"].round(3) == round(nr, 3)) &
                    (df["dataset"] == ds)
                ]
                if len(sub) == 0:
                    rows.append([float("nan")] * len(metrics_show))
                    continue

                ccr_vals = sub[sub["model"] == "mlp_ccr"][metrics_show].mean()
                xgb_vals = sub[sub["model"] == "xgboost_weighted"][metrics_show].mean()
                delta = (ccr_vals - xgb_vals).values.tolist()
                rows.append(delta)

            delta_df = pd.DataFrame(
                rows,
                index=[c[2] for c in conditions],
                columns=metric_labels,
            )

            vmax = max(abs(delta_df.values[~np.isnan(delta_df.values)]).max(), 0.01)

            sns.heatmap(
                delta_df, ax=ax,
                annot=True,
                fmt=".3f",
                annot_kws={"size": 7, "weight": "normal"},
                center=0,
                vmin=-vmax, vmax=vmax,
                cmap="RdYlGn",
                linewidths=0,
                linecolor="none",
                cbar=True,
                cbar_kws={"shrink": 0.55, "pad": 0.05, "aspect": 18},
            )
            ax.set_title(DS_LABELS[ds], fontsize=9, fontweight="bold", pad=4)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(labelsize=7.5, length=0, pad=3)

            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=6.5, length=2, width=0.5)
            cbar.outline.set_linewidth(0.4)
            cbar.set_label("CCR − XGBoost-W", size=6.5, labelpad=3)
            ax.grid(False)  # kill the global grid that bleeds through heatmap cells

    fig.suptitle(
        "CCR vs XGBoost-W: Delta per Metric and Noise Condition\n"
        "(Green = CCR wins, Red = XGBoost-W wins)",
        fontsize=9, fontweight="bold", y=1.02,
    )
    save(fig, "fig4_ccr_vs_xgboost")


# ---------------------------------------------------------------------------
# FIG 5  Training time
# Horizontal bar chart, one panel per dataset, models on Y axis
# Generous height so labels never overlap
# ---------------------------------------------------------------------------
def fig5_training_time(df):
    print("Fig 5 - Training time...")

    if "train_time_s" not in df.columns or df["train_time_s"].isna().all():
        print("  Skipped - train_time_s not in results.csv")
        return

    clean = df[(df["noise_type"] == "none") & df["train_time_s"].notna()]
    tdata = clean.groupby(["dataset", "model"])["train_time_s"].mean().reset_index()

    # Only 5 models - keeps it readable
    focus = ["mlp_ccr", "mlp_standard", "mlp_weighted_ce",
             "xgboost_weighted", "lightgbm_default"]
    n_m = len(focus)

    # Key fix: make each subplot TALL enough for 5 labels
    # 6 datasets in 2 rows x 3 cols
    # Each subplot needs ~1.4 in of height for 5 bars with padding
    fig, axes = plt.subplots(
        2, 3,
        figsize=(8.0, 5.5),
        gridspec_kw={"hspace": 0.50, "wspace": 0.45},
    )

    y = np.arange(n_m)

    for row, ds_group in enumerate([DS_ORDER[:3], DS_ORDER[3:]]):
        for col, ds in enumerate(ds_group):
            ax = axes[row, col]
            sub = tdata[tdata["dataset"] == ds].set_index("model")

            times  = [sub.loc[m, "train_time_s"] if m in sub.index else 0.0
                      for m in focus]
            colors = [COLORS[m] for m in focus]

            bars = ax.barh(y, times, 0.55,
                           color=colors, edgecolor="none", zorder=3)

            # CCR outline
            ccr_i = focus.index("mlp_ccr")
            bars[ccr_i].set_edgecolor("#111")
            bars[ccr_i].set_linewidth(0.8)

            ax.set_title(DS_LABELS[ds], fontsize=9, fontweight="bold", pad=4)

            # Y-axis: model names with enough room
            ax.set_yticks(y)
            ax.set_yticklabels(
                [LABELS[m] for m in focus],
                fontsize=7.5,
            )
            ax.set_ylim(-0.6, n_m - 0.4)

            ax.set_xlabel("Seconds", fontsize=8, labelpad=3)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(4, prune="both"))
            ax.tick_params(axis="x", labelsize=7.5, length=3, width=0.6, pad=2)
            ax.tick_params(axis="y", length=0, pad=4)

            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_linewidth(0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="x", linewidth=0.4, alpha=0.9, zorder=0)
            ax.grid(axis="y", visible=False)

    fig.suptitle(
        "Mean Training Time per Run (Clean Condition)",
        fontsize=10, fontweight="bold", y=1.01,
    )
    save(fig, "fig5_training_time")


# ---------------------------------------------------------------------------
# TABLES
# ---------------------------------------------------------------------------
def table1_clean(df):
    print("Table 1 - Clean results...")
    clean = df[df["noise_type"] == "none"]
    rows = []
    for m in MODEL_ORDER:
        row = {"Model": LABELS[m]}
        for ds in DS_ORDER:
            s = clean[(clean["model"] == m) & (clean["dataset"] == ds)]
            if len(s):
                row[f"{DS_LABELS[ds]} F1"]  = f"{s['macro_f1'].mean():.3f}"
                row[f"{DS_LABELS[ds]} Rec"] = f"{s['minority_recall'].mean():.3f}"
            else:
                row[f"{DS_LABELS[ds]} F1"]  = "-"
                row[f"{DS_LABELS[ds]} Rec"] = "-"
        rows.append(row)
    pd.DataFrame(rows).to_csv(METRICS / "table1_clean_results.csv", index=False)
    print("  -> table1_clean_results.csv")


def table2_noise(df):
    print("Table 2 - Noise robustness...")
    configs = [
        ("none", 0.0, "Clean"),
        ("asym", 0.1, "Asym 10%"), ("asym", 0.2, "Asym 20%"), ("asym", 0.3, "Asym 30%"),
        ("feat", 0.1, "Feat 10%"), ("feat", 0.2, "Feat 20%"), ("feat", 0.3, "Feat 30%"),
    ]
    rows = []
    for m in MODEL_ORDER:
        row = {"Model": LABELS[m]}
        for nt, nr, lbl in configs:
            s = get(df, nt, nr, "macro_f1", model=m)
            row[lbl] = f"{s.mean():.3f} +/- {s.std():.3f}" if len(s) else "-"
        rows.append(row)
    pd.DataFrame(rows).to_csv(METRICS / "table2_noise_robustness.csv", index=False)
    print("  -> table2_noise_robustness.csv")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("  CCR-Tabular - Paper Figure Generator")
    print("=" * 55)

    df = load_data()

    print("\nFigures:")
    fig1_clean_results(df)
    fig2_noise_degradation(df, "asym", "Asymmetric",         "fig2_noise_asym")
    fig2_noise_degradation(df, "feat", "Feature-Correlated", "fig2_noise_feat")
    fig3_minority_recall(df)
    fig4_heatmap(df)
    fig5_training_time(df)

    print("\nTables:")
    table1_clean(df)
    table2_noise(df)

    print(f"\nAll outputs -> {PLOTS}")
    print("=" * 55)
