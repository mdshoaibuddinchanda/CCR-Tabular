"""Statistical significance testing for CCR-Tabular results.

Implements Wilcoxon signed-rank tests comparing CCR against all baselines,
as required by IEEE TNNLS reviewers. Reports p-values and effect sizes.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

from src.utils.config import OUTPUTS_METRICS

logger = logging.getLogger(__name__)

# Significance threshold (PRD Section 5.3)
ALPHA = 0.05

# Metrics to test (in order of paper importance)
_TEST_METRICS = ["minority_recall", "macro_f1", "auc_roc", "auc_pr"]

# All baseline model names to compare against CCR
_BASELINES = [
    "mlp_standard",
    "mlp_focal",
    "mlp_weighted_ce",
    "mlp_smote",
    "xgboost_default",
    "xgboost_weighted",
    "lightgbm_default",
]


def run_wilcoxon_tests(
    results_csv: Optional[Path] = None,
    noise_type: str = "none",
    noise_rate: float = 0.0,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Run Wilcoxon signed-rank tests: CCR vs each baseline.

    For each dataset × metric combination, tests whether CCR scores are
    significantly different from each baseline's scores across all folds
    and seeds (15 paired observations per condition).

    Args:
        results_csv: Path to results.csv. Defaults to outputs/metrics/results.csv.
        noise_type: Filter to this noise type ('none', 'asym', 'feat').
        noise_rate: Filter to this noise rate (0.0, 0.1, 0.2, 0.3).
        output_path: Where to save the significance table CSV.
            Defaults to outputs/metrics/wilcoxon_<noise_type>_<rate>.csv.

    Returns:
        DataFrame with columns: dataset, metric, baseline, ccr_mean, baseline_mean,
        delta, p_value, significant, better.

    Raises:
        FileNotFoundError: If results.csv does not exist.
        ValueError: If CCR results are missing for the requested condition.
    """
    if results_csv is None:
        results_csv = OUTPUTS_METRICS / "results.csv"

    if not results_csv.exists():
        raise FileNotFoundError(
            f"results.csv not found at '{results_csv}'. "
            f"Run experiments first before computing significance tests."
        )

    df = pd.read_csv(results_csv)

    # Filter to requested noise condition
    mask = (df["noise_type"] == noise_type) & (df["noise_rate"].round(3) == round(noise_rate, 3))
    df_filtered = df[mask].copy()

    if len(df_filtered) == 0:
        raise ValueError(
            f"No results found for noise_type='{noise_type}', noise_rate={noise_rate}. "
            f"Available conditions: {df[['noise_type','noise_rate']].drop_duplicates().to_dict('records')}"
        )

    ccr_df = df_filtered[df_filtered["model"] == "mlp_ccr"]
    if len(ccr_df) == 0:
        raise ValueError(
            f"No CCR results found for noise_type='{noise_type}', noise_rate={noise_rate}. "
            f"Run CCR experiments first."
        )

    rows = []
    datasets = df_filtered["dataset"].unique()

    for dataset in sorted(datasets):
        ccr_dataset = ccr_df[ccr_df["dataset"] == dataset]

        for baseline in _BASELINES:
            bl_dataset = df_filtered[
                (df_filtered["dataset"] == dataset) &
                (df_filtered["model"] == baseline)
            ]

            if len(bl_dataset) == 0:
                logger.warning(
                    f"No results for baseline '{baseline}' on dataset '{dataset}'. Skipping."
                )
                continue

            for metric in _TEST_METRICS:
                if metric not in ccr_dataset.columns or metric not in bl_dataset.columns:
                    continue

                ccr_scores = ccr_dataset[metric].dropna().values
                bl_scores = bl_dataset[metric].dropna().values

                # Align lengths — pair by position (same fold/seed order)
                min_len = min(len(ccr_scores), len(bl_scores))
                if min_len < 5:
                    logger.warning(
                        f"Too few paired observations ({min_len}) for "
                        f"{dataset}/{baseline}/{metric}. Need ≥5. Skipping."
                    )
                    continue

                ccr_scores = ccr_scores[:min_len]
                bl_scores = bl_scores[:min_len]

                ccr_mean = float(np.mean(ccr_scores))
                bl_mean = float(np.mean(bl_scores))
                delta = ccr_mean - bl_mean

                # Wilcoxon signed-rank test (two-sided)
                try:
                    stat, p_value = wilcoxon(ccr_scores, bl_scores, alternative="two-sided")
                except ValueError as exc:
                    # Happens when all differences are zero
                    logger.warning(
                        f"Wilcoxon test failed for {dataset}/{baseline}/{metric}: {exc}. "
                        f"Setting p=1.0."
                    )
                    p_value = 1.0

                rows.append({
                    "dataset": dataset,
                    "metric": metric,
                    "baseline": baseline,
                    "ccr_mean": round(ccr_mean, 4),
                    "baseline_mean": round(bl_mean, 4),
                    "delta": round(delta, 4),
                    "p_value": round(float(p_value), 4),
                    "significant": bool(p_value < ALPHA),
                    "better": bool(delta > 0 and p_value < ALPHA),
                })

    result_df = pd.DataFrame(rows)

    if len(result_df) == 0:
        logger.warning("No Wilcoxon tests could be computed. Check that results.csv has data.")
        return result_df

    # Apply Holm-Bonferroni correction per metric across all baselines/datasets
    from statsmodels.stats.multitest import multipletests
    
    result_df["p_value_corrected"] = 1.0
    for metric in _TEST_METRICS:
        mask = result_df["metric"] == metric
        if mask.sum() > 0:
            raw_pvals = result_df.loc[mask, "p_value"]
            rejected, corrected_pvals, _, _ = multipletests(raw_pvals, alpha=ALPHA, method="holm")
            result_df.loc[mask, "p_value_corrected"] = corrected_pvals

    result_df["significant"] = result_df["p_value_corrected"] < ALPHA
    result_df["better"] = (result_df["delta"] > 0) & (result_df["p_value_corrected"] < ALPHA)

    # ── Save ──────────────────────────────────────────────────────────────────
    if output_path is None:
        rate_str = f"{int(noise_rate * 100):02d}"
        output_path = OUTPUTS_METRICS / f"wilcoxon_{noise_type}_{rate_str}.csv"

    result_df.to_csv(output_path, index=False)
    logger.info(f"Wilcoxon results saved to {output_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    _print_significance_summary(result_df, noise_type, noise_rate)

    return result_df


def run_friedman_test(
    results_csv: Optional[Path] = None,
    noise_type: str = "none",
    noise_rate: float = 0.0,
    metric: str = "macro_f1",
) -> Dict[str, float]:
    if results_csv is None:
        results_csv = OUTPUTS_METRICS / "results.csv"

    df = pd.read_csv(results_csv)
    mask = (df["noise_type"] == noise_type) & (df["noise_rate"].round(3) == round(noise_rate, 3))
    df_filtered = df[mask]

    models = ["mlp_ccr"] + _BASELINES
    
    # We need to compute the average metric per dataset per model
    dataset_model_means = df_filtered.groupby(["dataset", "model"])[metric].mean().unstack("model")
    
    # Only use models that actually exist in the dataframe
    available_models = [m for m in models if m in dataset_model_means.columns]
    
    # Drop datasets that don't have all available models evaluated
    dataset_model_means = dataset_model_means.dropna(subset=available_models)

    if len(dataset_model_means) < 3 or len(available_models) < 2:
        logger.warning(f"Not enough complete datasets ({len(dataset_model_means)}) or models for Friedman test.")
        return {}

    # Extract score arrays for the test
    model_scores = [dataset_model_means[m].values for m in available_models]
    
    stat, p_value = friedmanchisquare(*model_scores)
    
    logger.info(f"Friedman Test for {metric} @ {noise_type}_{noise_rate}: p={p_value:.4e} (statistic={stat:.4f})")
    
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": bool(p_value < ALPHA)
    }

def generate_win_tie_loss(
    wilcoxon_df: pd.DataFrame
) -> pd.DataFrame:
    """Generate a Win/Tie/Loss table comparing CCR vs baselines from Wilcoxon results."""
    rows = []
    
    for metric in _TEST_METRICS:
        if metric not in wilcoxon_df["metric"].values:
            continue
            
        metric_df = wilcoxon_df[wilcoxon_df["metric"] == metric]
        
        for baseline in metric_df["baseline"].unique():
            bl_df = metric_df[metric_df["baseline"] == baseline]
            
            wins = len(bl_df[(bl_df["delta"] > 0) & (bl_df["p_value_corrected"] < ALPHA)])
            losses = len(bl_df[(bl_df["delta"] < 0) & (bl_df["p_value_corrected"] < ALPHA)])
            ties = len(bl_df) - wins - losses
            
            rows.append({
                "metric": metric,
                "baseline": baseline,
                "win": wins,
                "tie": ties,
                "loss": losses
            })
            
    return pd.DataFrame(rows)


def run_all_wilcoxon_tests() -> Dict[str, pd.DataFrame]:
    """Run Wilcoxon tests for all noise conditions present in results.csv.

    Returns:
        Dict mapping '<noise_type>_<rate>' to significance DataFrame.
    """
    results_csv = OUTPUTS_METRICS / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(
            f"results.csv not found. Run experiments first."
        )

    df = pd.read_csv(results_csv)
    conditions = df[["noise_type", "noise_rate"]].drop_duplicates()

    all_results = {}
    for _, row in conditions.iterrows():
        noise_type = row["noise_type"]
        noise_rate = float(row["noise_rate"])
        key = f"{noise_type}_{int(noise_rate * 100):02d}"

        try:
            result = run_wilcoxon_tests(
                noise_type=noise_type,
                noise_rate=noise_rate,
            )
            all_results[key] = result
        except ValueError as exc:
            logger.warning(f"Skipping condition {key}: {exc}")

    return all_results


def _print_significance_summary(
    result_df: pd.DataFrame,
    noise_type: str,
    noise_rate: float,
) -> None:
    """Print a compact significance summary table.

    Args:
        result_df: Wilcoxon results DataFrame.
        noise_type: Noise type for display.
        noise_rate: Noise rate for display.
    """
    print(f"\n{'='*70}")
    print(f"Wilcoxon Significance: CCR vs Baselines | {noise_type}@{noise_rate:.0%}")
    print(f"{'='*70}")
    print(f"{'Dataset':<12} {'Metric':<18} {'Baseline':<20} {'Delta':>7} {'p':>7} {'Sig':>5}")
    print(f"{'-'*70}")

    for _, row in result_df.iterrows():
        sig_marker = "✓" if row["significant"] else " "
        better_marker = "↑" if row["better"] else ("↓" if row["delta"] < 0 else "=")
        print(
            f"{row['dataset']:<12} {row['metric']:<18} {row['baseline']:<20} "
            f"{row['delta']:>+7.4f} {row['p_value']:>7.4f} {sig_marker}{better_marker}"
        )

    n_sig = result_df["significant"].sum()
    n_better = result_df["better"].sum()
    n_total = len(result_df)
    print(f"{'-'*70}")
    print(f"Significant: {n_sig}/{n_total} | CCR better (p<0.05): {n_better}/{n_total}")
    print(f"{'='*70}\n")
