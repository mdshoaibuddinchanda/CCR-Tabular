"""Statistical significance testing, effect sizes, and FDR correction for CCR-Tabular.

Implements rigorous experimental statistics addressing Reviewer 2 & Reviewer 4 concerns:
  1. No heterogeneous cross-dataset variance pooling.
  2. Per-dataset statistics: Mean, standard deviation, 95% confidence intervals.
  3. Effect sizes:
     - Cohen's d (parametric effect size)
     - Cliff's delta (non-parametric effect size)
     - Absolute and relative percentage delta (Macro F1)
  4. Hypothesis testing:
     - Paired Wilcoxon signed-rank test
     - Paired Student's t-test
  5. Multiplicity control:
     - Benjamini-Hochberg False Discovery Rate (FDR) adjustment across all comparisons.
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

# Default significance level
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
    """Compute Cohen's d effect size between two groups.

    Args:
        x1: Treatment values (e.g. CCR).
        x2: Baseline values.
        paired: If True, computes paired Cohen's d_z = mean(diff) / std(diff).
                If False, computes independent Cohen's d with pooled standard deviation.
    """
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
            # Fallback to pooled standard deviation when diff is identical across pairs
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
    """Apply Benjamini-Hochberg FDR correction to a list of p-values."""
    p_arr = np.asarray(p_values, dtype=np.float64)
    m = len(p_arr)
    if m == 0:
        return [], []

    sorted_indices = np.argsort(p_arr)
    sorted_p = p_arr[sorted_indices]

    adj_p = np.zeros(m, dtype=np.float64)
    running_min = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        q_val = (m / rank) * sorted_p[i]
        running_min = min(running_min, q_val)
        adj_p[i] = min(1.0, running_min)

    original_adj_p = np.zeros(m, dtype=np.float64)
    original_adj_p[sorted_indices] = adj_p
    rejected = original_adj_p < alpha

    return original_adj_p.tolist(), rejected.tolist()


def analyze_dataset_level_significance(
    df: pd.DataFrame,
    primary_model: str = "mlp_ccr",
    baseline_models: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """Perform dataset-level paired statistical analysis with FDR correction."""
    if metrics is None:
        metrics = [m for m in _DEFAULT_METRICS if m in df.columns]

    if baseline_models is None:
        all_models = df["model"].unique().tolist()
        baseline_models = [m for m in all_models if m != primary_model]

    rows = []

    for dataset in df["dataset"].unique():
        df_ds = df[df["dataset"] == dataset]
        df_prim = df_ds[df_ds["model"] == primary_model]

        if len(df_prim) == 0:
            continue

        for baseline in baseline_models:
            df_base = df_ds[df_ds["model"] == baseline]
            if len(df_base) == 0:
                continue

            for metric in metrics:
                if metric not in df_prim.columns or metric not in df_base.columns:
                    continue

                v_prim = df_prim[metric].dropna().values
                v_base = df_base[metric].dropna().values
                min_len = min(len(v_prim), len(v_base))

                if min_len < 3:
                    continue

                v_prim_aligned = v_prim[:min_len]
                v_base_aligned = v_base[:min_len]

                mean_prim, ci_l_prim, ci_u_prim = compute_confidence_interval(v_prim_aligned)
                mean_base, ci_l_base, ci_u_base = compute_confidence_interval(v_base_aligned)

                abs_delta = mean_prim - mean_base
                rel_delta_pct = (abs_delta / (abs(mean_base) + 1e-8)) * 100.0
                cohens_d = compute_cohens_d(v_prim_aligned, v_base_aligned, paired=True)
                cliffs_d = compute_cliffs_delta(v_prim_aligned, v_base_aligned)

                try:
                    _, p_wilcoxon = wilcoxon(v_prim_aligned, v_base_aligned, alternative="two-sided")
                except Exception:
                    p_wilcoxon = 1.0

                try:
                    _, p_ttest = stats.ttest_rel(v_prim_aligned, v_base_aligned)
                except Exception:
                    p_ttest = 1.0

                rows.append({
                    "dataset": dataset,
                    "metric": metric,
                    "primary_model": primary_model,
                    "baseline_model": baseline,
                    "n_runs": min_len,
                    "primary_mean": round(mean_prim, 4),
                    "primary_ci95": f"[{ci_l_prim:.4f}, {ci_u_prim:.4f}]",
                    "baseline_mean": round(mean_base, 4),
                    "baseline_ci95": f"[{ci_l_base:.4f}, {ci_u_base:.4f}]",
                    "abs_delta": round(abs_delta, 4),
                    "rel_delta_pct": round(rel_delta_pct, 2),
                    "cohens_d": round(cohens_d, 3),
                    "cliffs_delta": round(cliffs_d, 3),
                    "p_raw_wilcoxon": float(p_wilcoxon),
                    "p_raw_ttest": float(p_ttest),
                })

    res_df = pd.DataFrame(rows)
    if len(res_df) == 0:
        return res_df

    q_wilcoxon, rej_wilcoxon = benjamini_hochberg_correction(res_df["p_raw_wilcoxon"].tolist(), alpha=alpha)
    q_ttest, rej_ttest = benjamini_hochberg_correction(res_df["p_raw_ttest"].tolist(), alpha=alpha)

    res_df["p_fdr_wilcoxon"] = [round(q, 4) for q in q_wilcoxon]
    res_df["sig_fdr_wilcoxon"] = rej_wilcoxon
    res_df["p_fdr_ttest"] = [round(q, 4) for q in q_ttest]
    res_df["sig_fdr_ttest"] = rej_ttest

    return res_df
