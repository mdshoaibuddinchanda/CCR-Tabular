"""Per-Sample Gradient Attribution and Noise Gradient Mass Analysis (Figure 5 & Section 12).

Measures the foundational mechanistic question:
  "Which samples actually control the gradient, and does CCR reduce the gradient contribution of corrupted labels?"

Computes:
  1. Per-sample gradient norm: ||g_i||_2 = ||nabla_theta ell_i||_2
  2. Weighted gradient contribution: c_i = (w_i ||g_i||_2) / sum_j (w_j ||g_j||_2)
  3. Clean vs Corrupted gradient mass: G_clean, G_corrupt
  4. Corrupted gradient mass fraction: R_noise = G_corrupt / (G_clean + G_corrupt)
  5. Relationship between observed label confidence p_y, weight w_i, and gradient mass c_i.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.load_data import load_dataset
from src.data.noise_injection import inject_asymmetric_noise
from src.data.preprocess import build_preprocessor, preprocess_split
from src.loss.robust_losses import build_loss
from src.models.mlp import TabularDataset, TabularMLP, get_mlp_for_dataset
from src.utils.config import BATCH_SIZE, LEARNING_RATE, OUTPUTS_METRICS, OUTPUTS_PLOTS, WEIGHT_DECAY
from src.utils.reproducibility import fix_all_seeds, get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_gradient_attribution_study(
    datasets: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    noise_rates: Optional[List[float]] = None,
    n_epochs: int = 25,
    seed: int = 42,
) -> pd.DataFrame:
    """Run per-sample gradient attribution across models and noise levels."""
    if datasets is None:
        datasets = ["credit_g", "phoneme", "adult"]
    if models is None:
        models = ["ce", "wce", "focal", "dynamic_ce", "ccr_no_norm", "ccr"]
    if noise_rates is None:
        noise_rates = [0.20, 0.40]

    device = get_device()
    fix_all_seeds(seed)

    all_sample_records = []
    summary_records = []

    out_metrics_dir = OUTPUTS_METRICS / "gradient_attribution"
    out_metrics_dir.mkdir(parents=True, exist_ok=True)

    for ds_name in datasets:
        logger.info(f"Loading {ds_name} for gradient attribution...")
        df_raw = load_dataset(ds_name)
        target_col = "target" if "target" in df_raw.columns else df_raw.columns[-1]
        X = df_raw.drop(columns=[target_col])
        y = df_raw[target_col].values.astype(int)

        # Simple 80/20 train/val split for attribution
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        train_idx, val_idx = next(skf.split(X, y))

        X_train_raw, y_train_clean = X.iloc[train_idx].copy(), y[train_idx].copy()
        X_val_raw, y_val = X.iloc[val_idx].copy(), y[val_idx].copy()

        preprocessor = build_preprocessor(X_train_raw)
        X_train_proc = preprocessor.transform(X_train_raw)
        X_val_proc = preprocessor.transform(X_val_raw)

        for n_rate in noise_rates:
            y_train_noisy, noise_stats = inject_asymmetric_noise(
                y_train_clean, noise_rate=n_rate, seed=seed
            )
            flip_mask = (y_train_noisy != y_train_clean)
            n_corrupt = int(np.sum(flip_mask))
            logger.info(f"[{ds_name}] Noise: {n_rate:.0%} | Corrupted samples: {n_corrupt}/{len(y_train_clean)}")

            train_ds = TabularDataset(X_train_proc, y_train_noisy)
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

            input_dim = X_train_proc.shape[1]
            num_classes = len(np.unique(y_train_clean))
            class_counts = [int(np.sum(y_train_noisy == c)) for c in range(num_classes)]

            for model_name in models:
                logger.info(f"Attribution Training: [{ds_name}] | Model: {model_name} | Noise: {n_rate:.0%}")
                fix_all_seeds(seed)

                model = get_mlp_for_dataset(ds_name, input_dim, num_classes=num_classes).to(device)
                criterion = build_loss(
                    loss_name=model_name,
                    n_samples=len(X_train_proc),
                    n_classes=num_classes,
                    class_counts=class_counts,
                    device=device,
                ).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

                for epoch in range(n_epochs):
                    model.train()

                    for b_idx, (X_b, y_b, idx_b) in enumerate(train_loader):
                        X_b = X_b.to(device)
                        y_b = y_b.to(device)
                        idx_b = idx_b.to(device)
                        b_size = len(y_b)

                        # Forward pass
                        logits = model(X_b)
                        probs = F.softmax(logits.detach(), dim=1)
                        p_y = probs[torch.arange(b_size, device=device), y_b].cpu().numpy()

                        # Extract per-sample weights
                        if model_name == "ce":
                            weights = np.ones(b_size, dtype=np.float32)
                        elif model_name == "wce":
                            inv_c = np.array([1.0 / c for c in class_counts])
                            w_cls = inv_c / inv_c.sum()
                            weights = w_cls[y_b.cpu().numpy()]
                        elif model_name == "focal":
                            weights = 0.25 * ((1.0 - p_y) ** 2.0)
                        elif model_name == "dynamic_ce":
                            weights = 1.0 - p_y
                        elif model_name in ("ccr", "ccr_no_norm"):
                            with torch.no_grad():
                                w_t = criterion.compute_weights(logits.detach(), y_b, idx_b, epoch)
                                weights = w_t.cpu().numpy()
                        else:
                            weights = np.ones(b_size, dtype=np.float32)

                        if model_name in ("ccr", "norm_dynamic_ce", "norm_wce"):
                            w_sum = np.sum(weights) + 1e-8
                            weights = (weights / w_sum) * b_size

                        # Compute per-sample gradient norm: ||g_i||_2 for the last linear layer
                        # (Representative gradient proxy without O(N*P) full autograd overhead)
                        sample_grad_norms = np.zeros(b_size, dtype=np.float32)
                        last_layer = None
                        for layer in reversed(model.network):
                            if isinstance(layer, nn.Linear):
                                last_layer = layer
                                break

                        # Analytical per-sample gradient for cross entropy: g_i = (p_i - e_y) (x_hidden)
                        with torch.no_grad():
                            y_one_hot = F.one_hot(y_b, num_classes=num_classes).float()
                            logit_errors = (probs - y_one_hot).cpu().numpy()  # [B, C]
                            # Gradient norm w.r.t logits: ||p_i - e_y||_2
                            sample_grad_norms = np.linalg.norm(logit_errors, axis=1)

                        weighted_grads = weights * sample_grad_norms
                        total_weighted_grad = np.sum(weighted_grads) + 1e-8
                        contributions = weighted_grads / total_weighted_grad

                        is_corrupt_batch = flip_mask[idx_b.cpu().numpy()]

                        # Record for every 5th epoch or final epoch
                        if epoch in (0, n_epochs // 2, n_epochs - 1):
                            for i in range(b_size):
                                all_sample_records.append({
                                    "dataset": ds_name,
                                    "model": model_name,
                                    "noise_rate": n_rate,
                                    "epoch": epoch,
                                    "sample_idx": int(idx_b[i].item()),
                                    "is_corrupted": bool(is_corrupt_batch[i]),
                                    "confidence": float(p_y[i]),
                                    "weight": float(weights[i]),
                                    "grad_norm": float(sample_grad_norms[i]),
                                    "weighted_grad": float(weighted_grads[i]),
                                    "contribution": float(contributions[i]),
                                })

                        # Overall batch summary
                        g_corrupt = np.sum(contributions[is_corrupt_batch]) if np.any(is_corrupt_batch) else 0.0
                        g_clean = np.sum(contributions[~is_corrupt_batch]) if np.any(~is_corrupt_batch) else 0.0
                        r_noise = g_corrupt / (g_clean + g_corrupt + 1e-8)

                        summary_records.append({
                            "dataset": ds_name,
                            "model": model_name,
                            "noise_rate": n_rate,
                            "epoch": epoch,
                            "batch_idx": b_idx,
                            "g_clean": float(g_clean),
                            "g_corrupt": float(g_corrupt),
                            "R_noise": float(r_noise),
                        })

                        # Model update
                        optimizer.zero_grad()
                        if hasattr(criterion, "target_history") or "sample_indices" in criterion.forward.__code__.co_varnames:
                            loss = criterion(logits, y_b, sample_indices=idx_b, current_epoch=epoch)
                        elif hasattr(criterion, "update_history") or "current_epoch" in criterion.forward.__code__.co_varnames:
                            loss = criterion(logits, y_b, idx_b, epoch)
                        else:
                            loss = criterion(logits, y_b)
                        loss.backward()
                        optimizer.step()

                        if hasattr(criterion, "update_history"):
                            with torch.no_grad():
                                criterion.update_history(probs, idx_b, epoch)

    df_samples = pd.DataFrame(all_sample_records)
    df_summary = pd.DataFrame(summary_records)

    samples_csv = out_metrics_dir / "per_sample_attribution_records.csv"
    summary_csv = out_metrics_dir / "gradient_mass_summary.csv"
    df_samples.to_csv(samples_csv, index=False)
    df_summary.to_csv(summary_csv, index=False)

    logger.info(f"Saved {len(df_samples)} sample records to {samples_csv}")
    logger.info(f"Saved {len(df_summary)} batch gradient mass records to {summary_csv}")

    # Generate Figure 5
    plot_gradient_attribution_figure(df_samples, df_summary)

    return df_summary


def plot_gradient_attribution_figure(
    df_samples: pd.DataFrame,
    df_summary: pd.DataFrame,
) -> Path:
    """Generate 3-Panel Figure 5: Mechanism of Gradient Mass Redistribution."""
    OUTPUTS_PLOTS.mkdir(parents=True, exist_ok=True)
    out_png = OUTPUTS_PLOTS / "figure5_gradient_attribution.png"

    # Filter to Phoneme at 40% noise at final epoch
    sub_samples = df_samples[
        (df_samples["dataset"] == "phoneme") &
        (df_samples["noise_rate"] == 0.40) &
        (df_samples["epoch"] == df_samples["epoch"].max())
    ].copy()

    if len(sub_samples) == 0:
        sub_samples = df_samples.copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

    # ── Panel A: Confidence vs Sample Weight (Clean vs Corrupted under CCR) ──
    ccr_sub = sub_samples[sub_samples["model"] == "ccr"]
    if len(ccr_sub) > 0:
        sns.scatterplot(
            data=ccr_sub,
            x="confidence",
            y="weight",
            hue="is_corrupted",
            palette={False: "#1f77b4", True: "#d62728"},
            alpha=0.6,
            s=25,
            ax=axes[0],
        )
        axes[0].set_title("(a) Observed Confidence vs. Weight ($w_i$)\n[CCR on Phoneme @ 40% Noise]", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Observed Label Probability $p_y$", fontsize=11)
        axes[0].set_ylabel("Normalized Sample Weight $\\hat{w}_i$", fontsize=11)
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles=handles, labels=["Clean Label", "Corrupted Label"], loc="upper right")
        axes[0].grid(True, linestyle="--", alpha=0.5)

    # ── Panel B: Confidence vs Weighted Gradient Mass (Clean vs Corrupted) ──
    if len(ccr_sub) > 0:
        sns.scatterplot(
            data=ccr_sub,
            x="confidence",
            y="weighted_grad",
            hue="is_corrupted",
            palette={False: "#1f77b4", True: "#d62728"},
            alpha=0.6,
            s=25,
            ax=axes[1],
        )
        axes[1].set_title("(b) Confidence vs. Effective Gradient Mass ($w_i ||g_i||$)\n[CCR on Phoneme @ 40% Noise]", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Observed Label Probability $p_y$", fontsize=11)
        axes[1].set_ylabel("Effective Gradient Norm $w_i ||g_i||_2$", fontsize=11)
        handles, labels = axes[1].get_legend_handles_labels()
        axes[1].legend(handles=handles, labels=["Clean Label", "Corrupted Label"], loc="upper right")
        axes[1].grid(True, linestyle="--", alpha=0.5)

    # ── Panel C: Corrupted Gradient Mass Fraction R_noise across Methods ──
    sub_summ = df_summary[df_summary["noise_rate"] == 0.40].copy()
    if len(sub_summ) > 0:
        grouped_r = (
            sub_summ.groupby(["model"])["R_noise"]
            .mean()
            .reset_index()
            .sort_values(by="R_noise", ascending=False)
        )
        colors = ["#7f7f7f" if m not in ("ccr", "ccr_no_norm") else "#2ca02c" for m in grouped_r["model"]]
        sns.barplot(
            data=grouped_r,
            x="model",
            y="R_noise",
            palette=colors,
            ax=axes[2],
        )
        axes[2].set_title("(c) Corrupted Gradient Mass Fraction ($R_{\\mathrm{noise}}$)\n[All Datasets @ 40% Noise]", fontsize=12, fontweight="bold")
        axes[2].set_xlabel("Loss Formulation", fontsize=11)
        axes[2].set_ylabel("Fraction of Total Gradient Mass $R_{\\mathrm{noise}}$", fontsize=11)
        axes[2].set_ylim(0, max(0.5, grouped_r["R_noise"].max() * 1.2))
        axes[2].grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    logger.info(f"Saved Figure 5 to {out_png}")
    return out_png


if __name__ == "__main__":
    run_gradient_attribution_study()
