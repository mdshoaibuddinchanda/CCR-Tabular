"""Statistical significance testing, effect sizes, and FDR correction for CCR-Tabular.

Implements rigorous experimental statistics:
  1. Primary Statistical Unit: Dataset (D).
     - Calculates per-dataset matched difference: Delta_d = Metric(CCR, d) - Metric(baseline, d)
     - Conducts paired hypothesis tests (Wilcoxon Signed-Rank) across independent datasets.
     - Controls for False Discovery Rate via Benjamini-Hochberg (BH-FDR) across all pairwise comparisons.
  2. Within-dataset uncertainty estimation: Mean, standard deviation, 95% Student's t confidence intervals.
  3. Effect sizes:
     - Cohen's d (parametric effect size)
     - Cliff's delta (non-parametric effect size)
     - Absolute and relative percentage delta (Macro F1)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t, wilcoxon

from src.utils.config import OUTPUTS_METRICS

logger = logging.getLogger(__name__)

ALPHA = 0.05
_DEFAULT_METRICS = ["macro_f1", "minority_recall", "auc_roc", "auc_pr", "accuracy", "ece", "brier_score"]


def compute_confidence_interval(
    values: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Compute mean and parametric 95% confidence interval using Student's t distribution."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    if n == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])

    mean = float(np.mean(arr))
    sem = float(stats.sem(arr))
    if sem == 0.0:
        return mean, mean, mean

    h = sem * float(t.ppf((1 + confidence) / 2.0, n - 1))
    return mean, mean - h, mean + h


def compute_cohens_d(x1: np.ndarray, x2: np.ndarray, paired: bool = True) -> float:
    """Compute Cohen's d effect size between two groups."""
    a1 = np.asarray(x1, dtype=np.float64)
    a2 = np.asarray(x2, dtype=np.float64)
    min_len = min(len(a1), len(a2))
    if min_len < 2:
        return 0.0

    a1, a2 = a1[:min_len], a2[:min_len]

    if paired:
        diff = a1 - a2
        std_diff = np.std(diff, ddof=1)
        mean_diff = np.mean(diff)
        if std_diff == 0.0:
            if mean_diff == 0.0:
                return 0.0
            s1, s2 = np.std(a1, ddof=1), np.std(a2, ddof=1)
            s_pooled = np.sqrt((s1**2 + s2**2) / 2.0)
            if s_pooled > 0.0:
                return float(mean_diff / s_pooled)
            return float(np.sign(mean_diff) * 10.0)
        return float(mean_diff / std_diff)
    else:
        n1, n2 = len(a1), len(a2)
        s1, s2 = np.std(a1, ddof=1), np.std(a2, ddof=1)
        s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        if s_pooled == 0.0:
            return 0.0
        return float((np.mean(a1) - np.mean(a2)) / s_pooled)


def compute_cliffs_delta(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute Cliff's delta non-parametric effect size in [-1.0, 1.0]."""
    a1 = np.asarray(x1, dtype=np.float64)
    a2 = np.asarray(x2, dtype=np.float64)
    n1, n2 = len(a1), len(a2)
    if n1 == 0 or n2 == 0:
        return 0.0

    greater = 0
    less = 0
    for val1 in a1:
        greater += np.sum(val1 > a2)
        less += np.sum(val1 < a2)

    return float((greater - less) / (n1 * n2))


def benjamini_hochberg_correction(p_values: List[float], alpha: float = ALPHA) -> Tuple[List[float], List[bool]]:
    """Apply Benjamini-Hochberg False Discovery Rate (BH-FDR) procedure."""
    p_arr = np.asarray(p_values, dtype=np.float64)
    m = len(p_arr)
    if m == 0:
        return [], []

    p_clean = np.where(np.isnan(p_arr), 1.0, p_arr)
    sorted_indices = np.argsort(p_clean)
    sorted_p = p_clean[sorted_indices]

    adjusted_p = np.zeros(m, dtype=np.float64)
    adjusted_p[-1] = sorted_p[-1]
    for i in range(m - 2, -1, -1):
        adjusted_p[i] = min(adjusted_p[i + 1], sorted_p[i] * m / (i + 1))
    adjusted_p = np.clip(adjusted_p, 0.0, 1.0)

    original_adjusted = np.zeros(m, dtype=np.float64)
    original_adjusted[sorted_indices] = adjusted_p
    significant = (original_adjusted < alpha).tolist()

    return original_adjusted.tolist(), significant


def analyze_dataset_level_significance(
    df: pd.DataFrame,
    metric: str = "macro_f1",
    reference_model: str = "ccr",
    alpha: float = ALPHA,
    primary_model: Optional[str] = None,
    baseline_models: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Primary statistical inference using Dataset as the independent observational unit.

    For each noise regime:
      1. Aggregates matched fold/seed results to obtain per-dataset mean performance.
      2. Computes per-dataset difference: Delta_d = Metric(CCR, d) - Metric(baseline, d).
      3. Tests cross-dataset significance using Paired Wilcoxon Signed-Rank Test.
      4. Adjusts p-values across all competing baselines using Benjamini-Hochberg FDR.
    """
    if primary_model is not None:
        reference_model = primary_model

    target_metric = metric
    if metrics is not None and len(metrics) > 0:
        target_metric = metrics[0]

    if df.empty or target_metric not in df.columns:
        return pd.DataFrame()

    # Aggregate to dataset-level mean per condition
    group_cols = ["dataset", "model"]
    if "noise_type" in df.columns and "noise_rate" in df.columns:
        group_cols = ["dataset", "noise_type", "noise_rate", "model"]

    ds_means = (
        df.groupby(group_cols)[target_metric]
        .mean()
        .reset_index()
    )

    records = []
    regimes = ds_means.groupby(["noise_type", "noise_rate"]) if ("noise_type" in ds_means.columns and "noise_rate" in ds_means.columns) else [("none", 0.0, ds_means)]

    for regime_info in regimes:
        if len(regime_info) == 2:
            (n_type, n_rate), group = regime_info
        else:
            n_type, n_rate, group = regime_info

        piv = group.pivot(index="dataset", columns="model", values=target_metric)
        if reference_model not in piv.columns:
            continue

        cand_baselines = baseline_models if baseline_models is not None else [m for m in piv.columns if m != reference_model]
        for base_model in cand_baselines:
            if base_model not in piv.columns or base_model == reference_model:
                continue

            common_ds = piv[[reference_model, base_model]].dropna()
            
            # If multi-dataset:
            if len(common_ds) >= 3:
                ref_series = common_ds[reference_model].values
                base_series = common_ds[base_model].values
                diff = ref_series - base_series

                mean_delta = float(np.mean(diff))
                median_delta = float(np.median(diff))
                cohens_d = compute_cohens_d(ref_series, base_series, paired=True)
                cliffs_d = compute_cliffs_delta(ref_series, base_series)
                _, ci_lower, ci_upper = compute_confidence_interval(diff, confidence=0.95)

                try:
                    if np.all(diff == 0.0):
                        stat, p_val = 0.0, 1.0
                    else:
                        stat, p_val = wilcoxon(ref_series, base_series, alternative="two-sided")
                        stat, p_val = float(stat), float(p_val)
                except Exception:
                    stat, p_val = float("nan"), 1.0

                records.append({
                    "noise_type": n_type,
                    "noise_rate": n_rate,
                    "metric": target_metric,
                    "reference_model": reference_model,
                    "baseline_model": base_model,
                    "n_datasets": len(common_ds),
                    "mean_delta": round(mean_delta, 4),
                    "median_delta": round(median_delta, 4),
                    "abs_delta": round(mean_delta, 4),
                    "ci_95_lower": round(ci_lower, 4),
                    "ci_95_upper": round(ci_upper, 4),
                    "cohens_d": round(cohens_d, 3),
                    "cliffs_delta": round(cliffs_d, 3),
                    "wilcoxon_stat": round(stat, 2) if not np.isnan(stat) else None,
                    "raw_p_value": p_val,
                    "p_fdr_wilcoxon": p_val,
                })
            elif len(df["dataset"].unique()) == 1:
                # Single-dataset fold-level matched analysis
                ds_name = df["dataset"].iloc[0]
                sub_ref = df[df["model"] == reference_model].sort_values(by=["seed", "fold"] if "fold" in df.columns else ["seed"])
                sub_base = df[df["model"] == base_model].sort_values(by=["seed", "fold"] if "fold" in df.columns else ["seed"])
                
                if len(sub_ref) > 0 and len(sub_base) > 0:
                    min_len = min(len(sub_ref), len(sub_base))
                    ref_series = sub_ref[target_metric].iloc[:min_len].values
                    base_series = sub_base[target_metric].iloc[:min_len].values
                    diff = ref_series - base_series

                    mean_delta = float(np.mean(diff))
                    median_delta = float(np.median(diff))
                    cohens_d = compute_cohens_d(ref_series, base_series, paired=True)
                    cliffs_d = compute_cliffs_delta(ref_series, base_series)
                    _, ci_lower, ci_upper = compute_confidence_interval(diff, confidence=0.95)

                    try:
                        if np.all(diff == 0.0):
                            stat, p_val = 0.0, 1.0
                        else:
                            stat, p_val = wilcoxon(ref_series, base_series, alternative="two-sided")
                            stat, p_val = float(stat), float(p_val)
                    except Exception:
                        stat, p_val = float("nan"), 1.0

                    records.append({
                        "dataset": ds_name,
                        "noise_type": n_type,
                        "noise_rate": n_rate,
                        "metric": target_metric,
                        "reference_model": reference_model,
                        "baseline_model": base_model,
                        "n_runs": min_len,
                        "mean_delta": round(mean_delta, 4),
                        "median_delta": round(median_delta, 4),
                        "abs_delta": round(mean_delta, 4),
                        "ci_95_lower": round(ci_lower, 4),
                        "ci_95_upper": round(ci_upper, 4),
                        "cohens_d": round(cohens_d, 3),
                        "cliffs_delta": round(cliffs_d, 3),
                        "wilcoxon_stat": round(stat, 2) if not np.isnan(stat) else None,
                        "raw_p_value": p_val,
                        "p_fdr_wilcoxon": p_val,
                    })

    if not records:
        return pd.DataFrame()

    res_df = pd.DataFrame(records)

    # Multiplicity adjustment across all baseline comparisons per noise regime
    all_adj_p = []
    all_sig = []
    for _, group in res_df.groupby(["noise_type", "noise_rate"]):
        p_vals = group["raw_p_value"].tolist()
        adj_p, sig = benjamini_hochberg_correction(p_vals, alpha=alpha)
        all_adj_p.extend(adj_p)
        all_sig.extend(sig)

    res_df["fdr_p_value"] = [round(p, 5) for p in all_adj_p]
    res_df["significant_fdr"] = all_sig

    return res_df


def compute_within_dataset_summary(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute per-dataset fold/seed descriptive statistics (Mean, Std, 95% CI)."""
    if metrics is None:
        metrics = _DEFAULT_METRICS

    valid_metrics = [m for m in metrics if m in df.columns]
    summary_rows = []

    for (ds, m_name, n_type, n_rate), grp in df.groupby(["dataset", "model", "noise_type", "noise_rate"]):
        row = {
            "dataset": ds,
            "model": m_name,
            "noise_type": n_type,
            "noise_rate": n_rate,
            "n_runs": len(grp),
        }
        for m in valid_metrics:
            vals = grp[m].dropna().values
            mean, ci_low, ci_high = compute_confidence_interval(vals)
            row[f"{m}_mean"] = round(mean, 4)
            row[f"{m}_std"] = round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4)
            row[f"{m}_ci95_low"] = round(ci_low, 4)
            row[f"{m}_ci95_high"] = round(ci_high, 4)
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)
