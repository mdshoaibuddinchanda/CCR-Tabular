"""Publication-Quality Figure 1: CCR Framework Schematic.

Generates a large-scale, high-contrast, large-typography research figure
for the CCR pipeline at 600 DPI vector PDF & PNG.
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
    "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
    "mathtext.fontset": "dejavuserif",
})


def generate_figure1_schematic():
    fig, ax = plt.subplots(figsize=(15.5, 9.2), dpi=600)
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 9.2)
    ax.axis("off")

    def draw_box(x, y, w, h, title, lines, header_color, body_color,
                 border_color, header_text_color="#0f172a"):
        shadow = patches.FancyBboxPatch(
            (x + 0.06, y - 0.06), w, h,
            boxstyle="round,pad=0.06,rounding_size=0.14",
            facecolor="#94a3b8", edgecolor="none", alpha=0.28, zorder=1)
        ax.add_patch(shadow)

        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.06,rounding_size=0.14",
            facecolor=body_color, edgecolor=border_color,
            linewidth=3.2, zorder=2)
        ax.add_patch(box)

        hh = 0.82
        header = patches.FancyBboxPatch(
            (x, y + h - hh), w, hh,
            boxstyle="round,pad=0.06,rounding_size=0.14",
            facecolor=header_color, edgecolor=border_color,
            linewidth=2.2, zorder=3)
        ax.add_patch(header)

        ax.text(x + w / 2, y + h - hh / 2, title,
                ha="center", va="center", fontsize=16,
                fontweight="bold", color=header_text_color, zorder=4)

        n = len(lines)
        avail = h - hh - 0.35
        spacing = avail / max(n, 1)
        for idx, (txt, is_bold, text_color) in enumerate(lines):
            yp = y + h - hh - 0.35 - idx * spacing - spacing * 0.35
            weight = "bold" if is_bold else "normal"
            ax.text(x + w / 2, yp, txt,
                    ha="center", va="center", fontsize=13,
                    fontweight=weight, color=text_color, zorder=4)

    def draw_arrow(x1, y1, x2, y2, label=None, label_offset=(0, 0.30)):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color="#0f172a",
                            lw=3.2, mutation_scale=24),
            zorder=5)
        if label:
            ax.text(
                (x1 + x2) / 2 + label_offset[0],
                (y1 + y2) / 2 + label_offset[1],
                label, ha="center", va="center",
                fontsize=12, fontweight="bold", color="#1e293b",
                bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                          edgecolor="#475569", lw=1.6),
                zorder=6)

    # Row 1: Data -> Preprocessing -> Model
    draw_box(
        0.6, 5.2, 3.8, 3.4,
        "1. Raw Tabular Input",
        [
            ("Imbalanced & Noisy Data D", True, "#0f172a"),
            ("• Heterogeneous attributes (cont/cat)", False, "#334155"),
            ("• Class Imbalance IR = N₀/N₁ ≥ 1.0", False, "#334155"),
            ("• Asymmetric noise transition T_jk", False, "#334155"),
            ("• Stratified 5-Fold outer CV split", False, "#334155"),
        ],
        header_color="#e2e8f0", body_color="#f8fafc", border_color="#475569")

    draw_arrow(4.4, 6.9, 5.8, 6.9, "Fold-Local")

    draw_box(
        5.8, 5.2, 4.0, 3.4,
        "2. Leakage-Free Preprocessing",
        [
            ("Training Split D_train Only", True, "#0f172a"),
            ("• Median imputation (continuous)", False, "#334155"),
            ("• Constant col filter: Var(X)>0", False, "#334155"),
            ("• Standard scaling (z-score)", False, "#334155"),
            ("• Ordinal encoding (categorical)", False, "#334155"),
            ("• Synthetic noise: train-only", True, "#dc2626"),
        ],
        header_color="#fed7aa", body_color="#fffbeb", border_color="#ea580c")

    draw_arrow(9.8, 6.9, 11.2, 6.9, "x_i, y_i")

    draw_box(
        11.2, 5.2, 3.7, 3.4,
        "3. Deep Neural Backbone",
        [
            ("f_θ(x_i) → z_i (Logits)", True, "#0f172a"),
            ("• TabularMLP (256-128-64)", False, "#334155"),
            ("• TabularResNet / FT-Trans.", False, "#334155"),
            ("• Softmax probabilities p_i", False, "#334155"),
            ("• Cross-Entropy loss ℓ_CE", False, "#334155"),
        ],
        header_color="#bfdbfe", body_color="#eff6ff", border_color="#2563eb")

    # Downward arrow from Box 3 to Box 4
    draw_arrow(13.05, 5.2, 13.05, 4.2)

    # Row 2: Optimization <- Batch Norm <- CCR Weighting
    draw_box(
        10.4, 0.6, 4.5, 3.4,
        "4. Dynamic CCR Reweighting",
        [
            ("Mechanism A: Learning-Signal Redistribution", True, "#1e3a8a"),
            ("w_i = (1 - p_{i,y_i}) + β·Var_K(p_i)·1(p>τ) + γ_{y_i}", True, "#1e3a8a"),
            ("• (1 - p_{i,y_i}): Confidence-inverse penalty", False, "#334155"),
            ("• β·Var_K(p_i): Historical stability gate", False, "#334155"),
            ("• γ_{y_i}: Normalized inverse class prior", False, "#334155"),
            (r"• Explicitly detached: $\nabla_\theta w_i \equiv 0$", True, "#059669"),
        ],
        header_color="#bbf7d0", body_color="#f0fdf4", border_color="#16a34a")

    draw_arrow(10.4, 2.3, 8.8, 2.3, "Raw {w_i}")

    draw_box(
        4.8, 0.6, 4.0, 3.4,
        "5. Invariant Batch Normalization",
        [
            ("Mechanism B: Gradient-Scale Stabilization", True, "#7c3aed"),
            ("ŵ_i = B · w_i / (∑ w_j + 10⁻⁸)", True, "#7c3aed"),
            ("• Enforces (1/B) ∑ ŵ_i ≈ 1.0 (Unit mean)", False, "#334155"),
            ("• Theoretical supremum: S/B ≤ 2.125", False, "#334155"),
            ("• Empirical telemetry: S/B ≤ 1.022", False, "#334155"),
            ("• Stabilizes gradient update step", True, "#7c3aed"),
        ],
        header_color="#ddd6fe", body_color="#faf5ff", border_color="#7c3aed")

    draw_arrow(4.8, 2.3, 3.4, 2.3, "ŵ_i · ℓ_CE")

    draw_box(
        0.6, 0.6, 2.8, 3.4,
        "6. Final Evaluation",
        [
            ("Held-Out Test D_test", True, "#0f172a"),
            ("• 100% Clean test data", True, "#dc2626"),
            ("• Macro-F1 metric", False, "#334155"),
            ("• Minority Recall", False, "#334155"),
            ("• 5-Fold mean ± std", False, "#334155"),
        ],
        header_color="#fecdd3", body_color="#fff1f2", border_color="#e11d48")

    # Mechanism Callout Banners
    mech_a_box = patches.FancyBboxPatch(
        (10.3, 4.15), 4.7, 0.65,
        boxstyle="round,pad=0.06,rounding_size=0.10",
        facecolor="#1e3a8a", edgecolor="#0f172a", lw=2.0, zorder=8)
    ax.add_patch(mech_a_box)
    ax.text(12.65, 4.47, "MECHANISM A: Directional Reweighting",
            ha="center", va="center", fontsize=12.5, fontweight="bold",
            color="white", zorder=9)

    mech_b_box = patches.FancyBboxPatch(
        (4.7, 4.15), 4.2, 0.65,
        boxstyle="round,pad=0.06,rounding_size=0.10",
        facecolor="#7c3aed", edgecolor="#0f172a", lw=2.0, zorder=8)
    ax.add_patch(mech_b_box)
    ax.text(6.8, 4.47, "MECHANISM B: Scale Invariance",
            ha="center", va="center", fontsize=12.5, fontweight="bold",
            color="white", zorder=9)

    plt.tight_layout(pad=0.20)
    plt.savefig(FIG_DIR / "fig1_schematic.pdf")
    plt.savefig(FIG_DIR / "fig1_schematic.png", dpi=600)
    plt.close()
    print("[DONE] Generated High-Quality Figure 1 Schematic (fig1)")


if __name__ == "__main__":
    generate_figure1_schematic()
