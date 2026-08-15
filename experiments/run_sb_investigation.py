"""Investigation and empirical bounds analysis of the 3-4x S/B inflation claim (Section C).

Reviewer 4 noted:
  "Reviewer 4 explicitly challenged your claim that weight-sum inflation can reach 3-4x,
  saying it appears incompatible with Eq. 5."

This script empirically measures and tabulates:
  - Theoretical upper bound of S/B based on formula: w_i = (1 - p_i) + beta * Var_i * I(p_i > tau) + gamma_y
  - Empirical distribution across batches, epochs, noise models, and datasets:
    mean(S/B), median(S/B), P90(S/B), P95(S/B), P99(S/B), and max(S/B).
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.load_data import load_dataset
from src.data.noise_injection import generate_noise
from src.data.preprocess import preprocess_split
from src.loss.ccr_loss import CCRLoss
from src.models.mlp import TabularDataset, get_mlp_for_dataset
from src.utils.config import (
    BATCH_SIZE,
    BETA,
    DATASETS,
    K,
    OUTPUTS_METRICS,
    OUTPUTS_PLOTS,
    TAU,
)
from src.utils.reproducibility import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_theoretical_upper_bound(
    beta: float = BETA,
    gamma_max: float = 1.0,
) -> Dict[str, float]:
    """Compute mathematical supremum of individual raw weights and batch S/B."""
    max_focal = 1.0
    max_variance = beta * 0.25
    supremum_normalized_gamma = max_focal + max_variance + gamma_max

    return {
        "max_focal_term": max_focal,
        "max_variance_term": max_variance,
        "supremum_normalized_gamma": supremum_normalized_gamma,
        "explanation": (
            f"Under normalized class weights (sum gamma = 1), max raw weight is "
            f"{supremum_normalized_gamma:.3f}. If unnormalized inverse frequencies (IR) "
            f"are used, S/B scales up to 1 + IR."
        ),
    }


def run_sb_empirical_measurement(
    dataset_names: Optional[List[str]] = None,
    noise_rates: Optional[List[float]] = None,
    n_epochs: int = 15,
) -> pd.DataFrame:
    """Measure empirical S/B distribution during training across datasets and noise types."""
    if dataset_names is None:
        dataset_names = ["credit_g", "spambase", "phoneme", "magic", "adult", "bank"]
    if noise_rates is None:
        noise_rates = [0.0, 0.10, 0.20, 0.30, 0.40]

    device = get_device()
    records = []

    for ds_name in dataset_names:
        df = load_dataset(ds_name)
        feature_cols = [c for c in df.columns if c != "target"]
        X = df[feature_cols]
        y = df["target"].values

        n_train = int(len(X) * 0.8)
        X_tr_df, X_val_df = X.iloc[:n_train], X.iloc[n_train:]
        y_tr, y_val = y[:n_train], y[n_train:]

        (X_tr_np, X_val_np, _, y_tr_np, y_val_np, _, _) = preprocess_split(
            X_tr_df, X_val_df, X_val_df,
            pd.Series(y_tr), pd.Series(y_val), pd.Series(y_val),
        )

        for noise_type in ["none", "asym", "feat", "sym"]:
            for rate in noise_rates:
                if noise_type == "none" and rate > 0.0:
                    continue

                y_noisy, _ = generate_noise(
                    X_tr_np, y_tr_np, noise_type=noise_type, noise_rate=rate, seed=42
                )

                model = get_mlp_for_dataset(ds_name, X_tr_np.shape[1]).to(device)
                class_counts = [int(np.sum(y_noisy == c)) for c in range(2)]

                criterion = CCRLoss(
                    n_samples=len(y_noisy),
                    n_classes=2,
                    class_counts=class_counts,
                    tau=TAU,
                    beta=BETA,
                    K=K,
                    device=device,
                ).to(device)

                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                dataset = TabularDataset(X_tr_np, y_noisy)
                loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

                sb_values = []

                for epoch in range(n_epochs):
                    model.train()
                    for X_b, y_b, idx_b in loader:
                        X_b, y_b, idx_b = X_b.to(device), y_b.to(device), idx_b.to(device)
                        optimizer.zero_grad()
                        logits = model(X_b)
                        loss = criterion(logits, y_b, idx_b, epoch)
                        loss.backward()
                        optimizer.step()

                        with torch.no_grad():
                            probs = torch.softmax(logits.detach(), dim=1)
                            criterion.update_history(probs, idx_b, epoch)

                        sb = criterion.last_telemetry.get("S_over_B", 1.0)
                        sb_values.append(sb)

                sb_arr = np.array(sb_values)
                records.append({
                    "dataset": ds_name,
                    "noise_type": noise_type,
                    "noise_rate": rate,
                    "n_batches_measured": len(sb_arr),
                    "mean_SB": round(float(np.mean(sb_arr)), 4),
                    "median_SB": round(float(np.median(sb_arr)), 4),
                    "P90_SB": round(float(np.percentile(sb_arr, 90)), 4),
                    "P95_SB": round(float(np.percentile(sb_arr, 95)), 4),
                    "P99_SB": round(float(np.percentile(sb_arr, 99)), 4),
                    "max_SB": round(float(np.max(sb_arr)), 4),
                    "exceeds_3x": bool(np.max(sb_arr) > 3.0),
                })
                logger.info(
                    f"[{ds_name} | {noise_type}@{rate:.0%}] "
                    f"mean(S/B)={np.mean(sb_arr):.3f}, max(S/B)={np.max(sb_arr):.3f}"
                )

    df_results = pd.DataFrame(records)
    out_csv = OUTPUTS_METRICS / "sb_distribution_empirical_analysis.csv"
    df_results.to_csv(out_csv, index=False)
    logger.info(f"Saved S/B investigation results to {out_csv}")
    return df_results


if __name__ == "__main__":
    bounds = compute_theoretical_upper_bound()
    print("\n--- Theoretical Upper Bound Analysis ---")
    for k, v in bounds.items():
        print(f"{k}: {v}")
    print("\n--- Running Empirical Measurement ---")
    run_sb_empirical_measurement()
