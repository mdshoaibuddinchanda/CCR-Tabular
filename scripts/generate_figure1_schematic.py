"""High-Contrast, Large-Typography Figure 1 Schematic for CCR-Tabular.

Generates a bold, crisp 2-tier vector schematic where all text, math formulas,
and borders are prominently visible, thick, and readable even when typeset
in single-column or full textwidth in LaTeX.
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

def generate_figure1_schematic_bold():
    # 2-Tier Modular Grid: 10.5 inches wide by 6.2 inches high
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=300)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    def draw_box(x, y, w, h, title, lines, header_color, body_color, border_color):
        # Drop shadow for clean 2D depth
        shadow = patches.FancyBboxPatch((x+0.04, y-0.04), w, h, boxstyle="round,pad=0.04,rounding_size=0.10",
                                        facecolor="#94a3b8", edgecolor="none", alpha=0.3, zorder=1)
        ax.add_patch(shadow)

        # Main box with thick dark border
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.10",
                                     facecolor=body_color, edgecolor=border_color, linewidth=2.0, zorder=2)
        ax.add_patch(box)
        
        # Header bar
        header_h = 0.62
        header = patches.FancyBboxPatch((x, y + h - header_h), w, header_h,
                                        boxstyle="round,pad=0.04,rounding_size=0.10",
                                        facecolor=header_color, edgecolor=border_color, linewidth=1.5, zorder=3)
        ax.add_patch(header)
        
        # Header text (Bold, High-Contrast)
        ax.text(x + w/2.0, y + h - header_h/2.0, title, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color="#0f172a", zorder=4)
        
        # Body lines (Large, Crisp, High-Contrast)
        n_lines = len(lines)
        avail_h = h - header_h - 0.20
        line_spacing = avail_h / max(n_lines, 1)
        for idx, (txt, is_bold, text_color) in enumerate(lines):
            y_pos = y + h - header_h - 0.20 - idx * line_spacing - line_spacing*0.3
            weight = "bold" if is_bold else "normal"
            ax.text(x + w/2.0, y_pos, txt, ha="center", va="center",
                    fontsize=10.0, fontweight=weight, color=text_color, zorder=4)

    def draw_arrow(x1, y1, x2, y2, label=None, label_offset=(0, 0.22)):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#0f172a", lw=2.4, mutation_scale=18), zorder=5)
        if label:
            ax.text((x1+x2)/2.0 + label_offset[0], (y1+y2)/2.0 + label_offset[1], label,
                    ha="center", va="center", fontsize=9.5, fontweight="bold", color="#1e293b",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#64748b", lw=1.2), zorder=6)

    # ==================== ROW 1: DATA & FORWARD PASS ====================
    # Box 1: Tabular Dataset
    draw_box(0.3, 3.6, 2.7, 2.3, "1. Tabular Dataset", [
        (r"Features $x_i \in \mathbb{R}^D$", True, "#0f172a"),
        (r"Class Labels $y_i^* \in \{0, 1\}$", False, "#1e293b"),
        (r"Imbalance Ratio $\mathrm{IR} \leq 17.5:1$", False, "#1e293b"),
        ("Stratified 5-Fold Split", False, "#475569")
    ], "#e2e8f0", "#f8fafc", "#334155")

    draw_arrow(3.05, 4.75, 3.75, 4.75, "CV Fold")

    # Box 2: Preprocessing & Noise Injection
    draw_box(3.8, 3.6, 2.9, 2.3, "2. Preprocessing", [
        ("Robust Quantile Scaling", False, "#1e293b"),
        ("Target Encoding (Fold-Local)", False, "#1e293b"),
        (r"$\mathbf{Asymmetric\ Label\ Noise}$", True, "#b91c1c"),
        (r"$\mathbf{Injected\ into\ y_{train}\ only}$", True, "#b91c1c")
    ], "#fee2e2", "#fff5f5", "#dc2626")

    draw_arrow(6.75, 4.75, 7.45, 4.75, "Mini-Batch")

    # Box 3: Tabular Neural Classifier
    draw_box(7.5, 3.6, 2.7, 2.3, "3. Neural Classifier", [
        ("TabularMLP / ResNet", False, "#1e293b"),
        ("FT-Transformer", False, "#1e293b"),
        (r"Logits: $z_i = f_\theta(x_i)$", True, "#1d4ed8"),
        (r"Probabilities: $p_i = \sigma(z_i)$", True, "#1d4ed8")
    ], "#dbeafe", "#eff6ff", "#2563eb")


    # ==================== CONNECTOR: ROW 1 TO ROW 2 ====================
    draw_arrow(8.85, 3.55, 8.85, 2.95, r"$p(y_i|x_i)$", label_offset=(0.65, 0))


    # ==================== ROW 2: CCR FORMULATION & OPTIMIZATION ====================
    # Box 4: CCR Dynamic Weighting (Autograd-Detached)
    draw_box(5.5, 0.35, 4.7, 2.5, "4. Dynamic Weighting (CCR)", [
        (r"$\mathbf{w_i = (1 - p_{i,y_i})\ +\ \beta \cdot \mathrm{Var}_K(p_i)\mathbf{1}(p_i > \tau)\ +\ \gamma_{y_i}}$", True, "#15803d"),
        (r"$(1 - p_{i, y_i}): \text{Confidence-Inverse Penalty}$", False, "#1e293b"),
        (r"$\mathrm{Var}_K(p_i): \text{Historical Prediction Variance}$", False, "#1e293b"),
        (r"$\gamma_{y_i}: \text{Normalized Inverse Class Weight}$", False, "#1e293b"),
        (r"$\mathbf{Autograd\ Detached:\ \nabla_\theta w_i \equiv 0}$", True, "#15803d")
    ], "#dcfce7", "#f0fdf4", "#16a34a")

    draw_arrow(5.45, 1.60, 4.85, 1.60, r"Raw $w_i$")

    # Box 5: Invariant Batch Normalization & Gradient Step
    draw_box(0.3, 0.35, 4.5, 2.5, "5. Batch Norm & Gradient Step", [
        (r"$\mathbf{\hat{w}_i = B \cdot \frac{w_i}{\sum_{j=1}^B w_j + \epsilon} \quad\Rightarrow\quad \frac{1}{B}\sum_{i=1}^B \hat{w}_i \equiv 1.0}$", True, "#b45309"),
        (r"$\mathcal{L}_{\mathrm{CCR}} = -\frac{1}{B}\sum_{i=1}^B \hat{w}_i \log p_{i, y_i}$", True, "#0f172a"),
        (r"$\nabla_{z_i} \mathcal{L}_{\mathrm{CCR}} = \frac{1}{B}\hat{w}_i(p_i - \mathbf{1}_{y_i})$", True, "#0f172a"),
        (r"$\mathbf{Exact\ Scale\ Invariance\ \mid\ Bounded\ S/B \leq 2.125}$", True, "#b45309")
    ], "#fef3c7", "#fffbeb", "#d97706")

    plt.tight_layout(pad=0.15)
    plt.savefig(FIG_DIR / "fig1_schematic.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIG_DIR / "fig1_schematic.pdf", bbox_inches="tight")
    plt.close()
    print("Successfully generated High-Contrast, Large-Typography Figure 1 (PDF and 300 DPI PNG)!")

if __name__ == "__main__":
    generate_figure1_schematic_bold()
