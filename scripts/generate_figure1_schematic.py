"""Publication-Quality Figure 1: CCR Framework Schematic.

Generates a high-contrast, large-typography, professional research figure
for the CCR pipeline. Designed to be readable at 100% zoom in a two-column
journal PDF and when printed in grayscale.

Key design decisions:
  - NO step numbering in box headers (caption provides figure number)
  - Large 14x8.5 inch canvas at 600 DPI
  - Bold serif headers (15pt), crisp body text (12pt)
  - Thick borders (3pt) with strong academic color palette
  - Visual emphasis callouts for the two mechanisms
  - Clean arrows with large labels
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

FIG_DIR = Path("ccrtatex/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
})


def generate_figure1_schematic():
    fig, ax = plt.subplots(figsize=(14, 8.5), dpi=600)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    # ------------------------------------------------------------------
    # Helper: draw a titled box with body lines
    # ------------------------------------------------------------------
    def draw_box(x, y, w, h, title, lines, header_color, body_color,
                 border_color, header_text_color="#0f172a"):
        # Subtle drop shadow
        shadow = patches.FancyBboxPatch(
            (x + 0.05, y - 0.05), w, h,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            facecolor="#94a3b8", edgecolor="none", alpha=0.25, zorder=1)
        ax.add_patch(shadow)

        # Main box
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            facecolor=body_color, edgecolor=border_color,
            linewidth=3.0, zorder=2)
        ax.add_patch(box)

        # Header band
        hh = 0.72
        header = patches.FancyBboxPatch(
            (x, y + h - hh), w, hh,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            facecolor=header_color, edgecolor=border_color,
            linewidth=2.0, zorder=3)
        ax.add_patch(header)

        # Header text
        ax.text(x + w / 2, y + h - hh / 2, title,
                ha="center", va="center", fontsize=15,
                fontweight="bold", color=header_text_color, zorder=4)

        # Body lines
        n = len(lines)
        avail = h - hh - 0.30
        spacing = avail / max(n, 1)
        for idx, (txt, is_bold, text_color) in enumerate(lines):
            yp = y + h - hh - 0.30 - idx * spacing - spacing * 0.35
            weight = "bold" if is_bold else "normal"
            ax.text(x + w / 2, yp, txt,
                    ha="center", va="center", fontsize=12,
                    fontweight=weight, color=text_color, zorder=4)

    # ------------------------------------------------------------------
    # Helper: draw a thick arrow with optional label
    # ------------------------------------------------------------------
    def draw_arrow(x1, y1, x2, y2, label=None, label_offset=(0, 0.28)):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color="#0f172a",
                            lw=3.0, mutation_scale=22),
            zorder=5)
        if label:
            ax.text(
                (x1 + x2) / 2 + label_offset[0],
                (y1 + y2) / 2 + label_offset[1],
                label, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color="#1e293b",
                bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                          edgecolor="#475569", lw=1.5),
                zorder=6)

    # ------------------------------------------------------------------
    # Helper: mechanism emphasis label
    # ------------------------------------------------------------------
    def draw_mechanism_label(x, y, text, bg_color, text_color, border_color):
        ax.text(x, y, text,
                ha="center", va="center", fontsize=11, fontweight="bold",
                fontstyle="italic", color=text_color,
                bbox=dict(boxstyle="round,pad=0.35", facecolor=bg_color,
                          edgecolor=border_color, lw=1.8, alpha=0.9),
                zorder=7)

    # ==================================================================
    # ROW 1: DATA → PREPROCESSING → NEURAL CLASSIFIER
    # ==================================================================
    row1_y = 4.8
    row1_h = 3.0
    box_gap = 0.25

    # Box A: Tabular Dataset
    bA_x, bA_w = 0.35, 3.5
    draw_box(bA_x, row1_y, bA_w, row1_h, "Tabular Dataset", [
        (r"Features $x_i \in \mathbb{R}^D$", True, "#0f172a"),
        (r"Class Labels $y_i^* \in \{0, 1, \ldots, C{-}1\}$", False, "#1e293b"),
        (r"Imbalance Ratio $\mathrm{IR} \leq 17.5\!:\!1$", False, "#1e293b"),
        ("Stratified 5-Fold CV Split", False, "#475569"),
    ], "#cbd5e1", "#f1f5f9", "#334155")

    # Arrow A→B
    draw_arrow(bA_x + bA_w + 0.05, row1_y + row1_h / 2,
               bA_x + bA_w + 0.05 + 1.05, row1_y + row1_h / 2,
               "Fold Split")

    # Box B: Preprocessing & Noise Injection
    bB_x, bB_w = 4.95, 3.8
    draw_box(bB_x, row1_y, bB_w, row1_h, "Fold-Local Preprocessing", [
        ("Median Imputation + Standard Scaling", False, "#1e293b"),
        ("Ordinal Encoding (Categoricals)", False, "#1e293b"),
        (r"$\mathbf{Asymmetric\ Label\ Noise}$", True, "#991b1b"),
        (r"$\mathbf{Injected\ into\ y_{train}\ only}$", True, "#991b1b"),
    ], "#fecaca", "#fef2f2", "#b91c1c", "#7f1d1d")

    # Arrow B→C
    draw_arrow(bB_x + bB_w + 0.05, row1_y + row1_h / 2,
               bB_x + bB_w + 0.05 + 1.05, row1_y + row1_h / 2,
               "Mini-Batch")

    # Box C: Neural Classifier
    bC_x, bC_w = 9.85, 3.8
    draw_box(bC_x, row1_y, bC_w, row1_h, "Neural Classifier", [
        ("TabularMLP / TabularResNet", False, "#1e293b"),
        ("FT-Transformer", False, "#1e293b"),
        (r"Logits:  $z_i = f_\theta(x_i)$", True, "#1e3a8a"),
        (r"Probabilities:  $p_i = \sigma(z_i)$", True, "#1e3a8a"),
    ], "#bfdbfe", "#eff6ff", "#1d4ed8", "#1e3a8a")

    # ==================================================================
    # CONNECTOR: ROW 1 → ROW 2
    # ==================================================================
    conn_x = bC_x + bC_w / 2
    draw_arrow(conn_x, row1_y - 0.05,
               conn_x, row1_y - 0.05 - 0.95,
               r"$p(y_i | x_i)$", label_offset=(0.85, 0))

    # ==================================================================
    # ROW 2: CCR DYNAMIC WEIGHTING → BATCH NORM & GRADIENT STEP
    # ==================================================================
    row2_y = 0.45
    row2_h = 3.3

    # Box D: CCR Dynamic Weighting (RIGHT side — receives probabilities)
    bD_x, bD_w = 7.3, 6.35
    draw_box(bD_x, row2_y, bD_w, row2_h,
             "Confidence-Calibrated Reweighting (CCR)", [
        (r"$w_i = (1 - p_{i,y_i})\ +\ \beta \cdot \mathrm{Var}_K(p_i)"
         r"\,\mathbf{1}(p_{i,y_i} > \tau)\ +\ \gamma_{y_i}$",
         True, "#14532d"),
        (r"$(1 - p_{i,y_i})$: Confidence-Inverse Penalty", False, "#1e293b"),
        (r"$\mathrm{Var}_K(p_i)$: Historical Prediction Variance",
         False, "#1e293b"),
        (r"$\gamma_{y_i}$: Normalized Inverse Class Weight",
         False, "#1e293b"),
        (r"$\mathbf{Autograd\ Detached:\ \nabla_\theta w_i \equiv 0}$",
         True, "#14532d"),
    ], "#bbf7d0", "#f0fdf4", "#15803d", "#14532d")

    # Mechanism label: Learning-Signal Redistribution
    draw_mechanism_label(
        bD_x + bD_w / 2, row2_y - 0.35,
        "Mechanism A:  Learning-Signal Redistribution",
        "#dcfce7", "#14532d", "#16a34a")

    # Arrow D→E
    draw_arrow(bD_x - 0.05, row2_y + row2_h / 2,
               bD_x - 0.05 - 1.05, row2_y + row2_h / 2,
               r"Raw $w_i$")

    # Box E: Batch Normalization & Gradient Step (LEFT side)
    bE_x, bE_w = 0.35, 5.8
    draw_box(bE_x, row2_y, bE_w, row2_h,
             "Batch Normalization & Gradient Step", [
        (r"$\hat{w}_i = B \cdot \frac{w_i}{\sum_{j=1}^B w_j + \epsilon}"
         r" \quad\Rightarrow\quad"
         r" \frac{1}{B}\sum_{i=1}^B \hat{w}_i \approx 1.0$",
         True, "#92400e"),
        (r"$\mathcal{L}_{\mathrm{CCR}}"
         r" = -\frac{1}{B}\sum_{i=1}^B \hat{w}_i \log p_{i,y_i}$",
         True, "#0f172a"),
        (r"$\nabla_{z_i}\mathcal{L}_{\mathrm{CCR}}"
         r" = \frac{1}{B}\hat{w}_i(p_i - \mathbf{1}_{y_i})$",
         True, "#0f172a"),
        (r"$\mathbf{Scale\ Invariance\ |\ Bounded\ S/B \leq 2.125}$",
         True, "#92400e"),
    ], "#fde68a", "#fffbeb", "#b45309", "#78350f")

    # Mechanism label: Gradient-Scale Stabilization
    draw_mechanism_label(
        bE_x + bE_w / 2, row2_y - 0.35,
        "Mechanism B:  Global Gradient-Scale Stabilization",
        "#fef3c7", "#78350f", "#d97706")

    # ==================================================================
    # SAVE
    # ==================================================================
    plt.tight_layout(pad=0.2)
    plt.savefig(FIG_DIR / "fig1_schematic.png", dpi=600, bbox_inches="tight")
    plt.savefig(FIG_DIR / "fig1_schematic.pdf", bbox_inches="tight")
    plt.close()
    print("[DONE] Generated Publication-Quality Figure 1 Schematic "
          "(PDF + 600 DPI PNG)")


if __name__ == "__main__":
    generate_figure1_schematic()
