"""Mechanism telemetry analysis and Spearman rank correlation pipeline.

Aggregates batch-level instrumentation data and computes:
  1. Detailed Table per (Dataset x Noise Condition x Model):
     Dataset | Noise_Type | Noise_Rate | Model | S/B mean | P50 | P90 | P95 | P99 | Max | Grad CV | Update CV | Cosine | Delta F1
  2. Spearman Rank Correlations:
     - rho(noise_fraction, S/B)      — Tests whether noise concentration causes S/B inflation
     - rho(minority_fraction, S/B)   — Tests whether minority concentration causes S/B inflation
     - rho(S/B, ||grad_L||_2)        — Tests whether S/B scales raw gradient norms
     - rho(S/B, ||Delta theta||_2)   — Tests whether S/B scales actual parameter updates
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.utils.config import OUTPUTS_METRICS, OUTPUTS_PLOTS, OUTPUTS_TELEMETRY

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def analyze_mechanism_telemetry(
    telemetry_dir: Optional[Path] = None,
    results_csv: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Analyze all telemetry files and compute mechanism summary and correlations."""
    t_dir = telemetry_dir or OUTPUTS_TELEMETRY
    r_csv = results_csv or (OUTPUTS_METRICS / "tier2_mechanism_results.csv")

    telemetry_files = list(t_dir.glob("*_telemetry.csv"))
    if not telemetry_files:
        logger.warning(f"No telemetry CSV files found in {t_dir}.")
        return pd.DataFrame(), {}

    logger.info(f"Aggregating {len(telemetry_files)} telemetry files...")
    all_dfs = []
    for f in telemetry_files:
        try:
            df_t = pd.read_csv(f)
            all_dfs.append(df_t)
        except Exception as e:
            logger.warning(f"Could not read {f}: {e}")

    if not all_dfs:
        return pd.DataFrame(), {}

    full_telemetry = pd.concat(all_dfs, ignore_index=True)

    from src.utils.config import DATASETS, LOSS_NAMES
    from src.loss.robust_losses import LOSS_REGISTRY
    known_datasets = sorted(list(DATASETS.keys()) + ["credit_g", "steel_faults", "heart_disease", "breast_cancer"], key=len, reverse=True)
    known_models = sorted(list(LOSS_REGISTRY.keys()) + list(LOSS_NAMES) + ["ccr_no_norm", "ccr_no_gate", "ccr_no_variance", "norm_wce", "norm_focal", "norm_gce", "norm_sce", "dynamic_ce", "norm_dynamic_ce", "xgboost_default", "xgboost_weighted", "lightgbm_default", "catboost_default", "catboost"], key=len, reverse=True)

    records = []
    for run_id, group in full_telemetry.groupby("run_id"):
        # Match dataset
        dataset = "unknown"
        for ds in known_datasets:
            if run_id.startswith(ds + "_"):
                dataset = ds
                break
        rem = run_id[len(dataset) + 1:] if dataset != "unknown" else run_id

        # Match model
        model = "unknown"
        for m in known_models:
            if rem.startswith(m + "_"):
                model = m
                break
        rem2 = rem[len(model) + 1:] if model != "unknown" else rem

        # Parse noise type and noise rate
        noise_type = "none"
        noise_rate = 0.0
        for nt in ["none", "asym", "sym", "feat", "idn"]:
            if f"_{nt}_" in run_id:
                noise_type = nt
                try:
                    parts = run_id.split(f"_{nt}_")[1].split("_")
                    noise_rate = int(parts[0]) / 100.0 if parts[0].isdigit() else 0.0
                except Exception:
                    pass
                break

        sb_vals = group["S_over_B"].dropna().values
        grad_vals = group["grad_norm_weighted"].dropna().values
        update_vals = group["param_update_norm"].dropna().values
        cosine_vals = group["grad_cosine_sim"].dropna().values

        if len(sb_vals) == 0:
            continue

        grad_cv = float(np.std(grad_vals) / (np.mean(grad_vals) + 1e-8))
        update_cv = float(np.std(update_vals) / (np.mean(update_vals) + 1e-8))

        records.append({
            "run_id": run_id,
            "dataset": dataset,
            "model": model,
            "noise_type": noise_type,
            "noise_rate": noise_rate,
            "n_batches": len(sb_vals),
            "sb_mean": round(float(np.mean(sb_vals)), 4),
            "sb_p50": round(float(np.median(sb_vals)), 4),
            "sb_p90": round(float(np.percentile(sb_vals, 90)), 4),
            "sb_p95": round(float(np.percentile(sb_vals, 95)), 4),
            "sb_p99": round(float(np.percentile(sb_vals, 99)), 4),
            "sb_max": round(float(np.max(sb_vals)), 4),
            "grad_cv": round(grad_cv, 4),
            "update_cv": round(update_cv, 4),
            "mean_cosine_sim": round(float(np.mean(cosine_vals)), 4) if len(cosine_vals) > 0 else float("nan"),
        })

    df_summary = pd.DataFrame(records)

    # Merge Macro F1 if results_csv exists
    if r_csv.exists():
        df_results = pd.read_csv(r_csv)
        if "macro_f1" in df_results.columns and "run_id" in df_results.columns:
            f1_map = df_results.set_index("run_id")["macro_f1"].to_dict()
            df_summary["macro_f1"] = df_summary["run_id"].map(f1_map)

    # Compute condition-level aggregation
    cond_summary = (
        df_summary.groupby(["dataset", "noise_type", "noise_rate", "model"])
        .agg({
            "sb_mean": "mean",
            "sb_p50": "mean",
            "sb_p90": "mean",
            "sb_p95": "mean",
            "sb_p99": "mean",
            "sb_max": "max",
            "grad_cv": "mean",
            "update_cv": "mean",
            "mean_cosine_sim": "mean",
            "macro_f1": "mean",
        })
        .reset_index()
    )

    out_csv = OUTPUTS_METRICS / "tier2_mechanism_telemetry_summary.csv"
    cond_summary.to_csv(out_csv, index=False)
    logger.info(f"Saved mechanism telemetry summary table to {out_csv}")

    # ── 2. Spearman Rank Correlations Across All Batches ─────────────────────
    correlations = {}

    # rho(noise_fraction, S/B)
    if "noise_fraction" in full_telemetry.columns:
        valid = full_telemetry[["noise_fraction", "S_over_B"]].dropna()
        if len(valid) > 10:
            rho, p = spearmanr(valid["noise_fraction"], valid["S_over_B"])
            correlations["rho_noise_frac_vs_SB"] = round(float(rho), 4)
            correlations["p_val_noise_frac_vs_SB"] = float(p)

    # rho(minority_fraction, S/B)
    if "minority_fraction" in full_telemetry.columns:
        valid = full_telemetry[["minority_fraction", "S_over_B"]].dropna()
        if len(valid) > 10:
            rho, p = spearmanr(valid["minority_fraction"], valid["S_over_B"])
            correlations["rho_minority_frac_vs_SB"] = round(float(rho), 4)
            correlations["p_val_minority_frac_vs_SB"] = float(p)

    # rho(S/B, grad_norm)
    if "grad_norm_weighted" in full_telemetry.columns:
        valid = full_telemetry[["S_over_B", "grad_norm_weighted"]].dropna()
        if len(valid) > 10:
            rho, p = spearmanr(valid["S_over_B"], valid["grad_norm_weighted"])
            correlations["rho_SB_vs_grad_norm"] = round(float(rho), 4)
            correlations["p_val_SB_vs_grad_norm"] = float(p)

    # rho(S/B, update_norm)
    if "param_update_norm" in full_telemetry.columns:
        valid = full_telemetry[["S_over_B", "param_update_norm"]].dropna()
        if len(valid) > 10:
            rho, p = spearmanr(valid["S_over_B"], valid["param_update_norm"])
            correlations["rho_SB_vs_update_norm"] = round(float(rho), 4)
            correlations["p_val_SB_vs_update_norm"] = float(p)

    corr_df = pd.DataFrame([correlations])
    corr_csv = OUTPUTS_METRICS / "tier2_spearman_correlations.csv"
    corr_df.to_csv(corr_csv, index=False)
    logger.info(f"Saved Spearman correlation results to {corr_csv}")

    return cond_summary, correlations


if __name__ == "__main__":
    analyze_mechanism_telemetry()
