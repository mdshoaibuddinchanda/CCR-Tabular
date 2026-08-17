"""Master experiment runner for CCR-Tabular.

Reads all YAML configs, runs all experiments, supports graceful resume.
Skips already-completed runs based on results.csv.
"""

import argparse
import gc
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

# Ensure project root is on the path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.cross_validation import run_cross_validation
from src.utils.config import OUTPUTS_LOGS
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)

_STATUS_LOG = OUTPUTS_LOGS / "experiment_status.json"
_CONFIGS_DIR = Path(__file__).parent / "configs"


def load_all_configs(config_dir: Path = _CONFIGS_DIR) -> List[Dict[str, Any]]:
    """Load all YAML experiment configs from the configs directory.

    Args:
        config_dir: Directory containing YAML config files.

    Returns:
        List of config dicts.

    Raises:
        FileNotFoundError: If config_dir does not exist.
    """
    if not config_dir.exists():
        raise FileNotFoundError(
            f"Config directory not found: {config_dir}. "
            f"Expected YAML files at {config_dir}/*.yaml"
        )

    configs = []
    for yaml_path in sorted(config_dir.glob("*.yaml")):
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        config["_config_file"] = str(yaml_path.name)
        configs.append(config)
        logger.info(f"Loaded config: {yaml_path.name}")

    if not configs:
        raise ValueError(
            f"No YAML configs found in {config_dir}. "
            f"Expected files matching *.yaml"
        )

    return configs


def load_single_config(config_path: Path) -> Dict[str, Any]:
    """Load a single YAML config file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Config dict.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["_config_file"] = str(config_path.name)
    return config


def update_status_log(
    dataset: str,
    model: str,
    config: Dict[str, Any],
    status: str,
    error: Optional[str] = None,
) -> None:
    """Update the experiment status log JSON file.

    Args:
        dataset: Dataset name.
        model: Model name.
        config: Experiment config dict.
        status: 'COMPLETED', 'FAILED', or 'SKIPPED'.
        error: Optional error message for failed runs.
    """
    status_records: List[Dict] = []

    if _STATUS_LOG.exists():
        try:
            with open(_STATUS_LOG, "r") as f:
                status_records = json.load(f)
        except Exception:
            status_records = []

    status_records.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "model": model,
        "noise_type": config.get("noise_type", "none"),
        "noise_rate": config.get("noise_rate", 0.0),
        "config_file": config.get("_config_file", ""),
        "status": status,
        "error": error,
    })

    try:
        with open(_STATUS_LOG, "w") as f:
            json.dump(status_records, f, indent=2)
    except OSError as exc:
        logger.warning(f"Could not write status log: {exc}")


def print_summary(results_df: Any) -> None:
    """Print a brief summary of completed results.

    Args:
        results_df: DataFrame with experiment results.
    """
    if results_df is None or len(results_df) == 0:
        print("  No results to summarize.")
        return

    metric_cols = ["macro_f1", "minority_recall", "auc_roc"]
    available = [c for c in metric_cols if c in results_df.columns]

    for col in available:
        vals = results_df[col].dropna()
        if len(vals) > 0:
            print(f"  {col}: {vals.mean():.4f} ± {vals.std():.4f}")


def main() -> None:
    """Run all experiments. Supports graceful interruption and resume."""
    # Setup logging FIRST — before any imports or config loading so nothing is silent
    setup_logging()
    print("CCR-Tabular experiment runner starting...", flush=True)

    parser = argparse.ArgumentParser(description="CCR-Tabular experiment runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a single YAML config file. If omitted, runs all configs.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Override datasets to run (space-separated).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Override models to run (space-separated).",
    )
    args = parser.parse_args()

    # ── Load configs ──────────────────────────────────────────────────────────
    if args.config is not None:
        configs = [load_single_config(args.config)]
    else:
        configs = load_all_configs()

    logger.info(f"Loaded {len(configs)} experiment config(s).")

    # ── Run experiments ───────────────────────────────────────────────────────
    total_runs = 0
    failed_runs = 0

    for config in configs:
        datasets = args.datasets or config.get("datasets", [])
        models = args.models or config.get("models", [])
        noise_type = config.get("noise_type", "none")
        noise_rate = float(config.get("noise_rate", 0.0))
        seeds = config.get("seeds", [42, 123, 2024])
        n_folds = config.get("n_folds", 5)

        logger.info(
            f"\nConfig: {config.get('experiment_name', 'unnamed')} | "
            f"noise={noise_type}@{noise_rate:.0%} | "
            f"{len(datasets)} datasets × {len(models)} models"
        )

        for dataset in datasets:
            for model in models:
                print(f"\n{'='*60}")
                print(
                    f"Running: {dataset} | {model} | "
                    f"{noise_type} {noise_rate:.0%}"
                )
                print(f"{'='*60}")

                try:
                    results = run_cross_validation(
                        dataset_name=dataset,
                        model_name=model,
                        noise_type=noise_type,
                        noise_rate=noise_rate,
                        seeds=seeds,
                        n_folds=n_folds,
                    )
                    print_summary(results)
                    update_status_log(dataset, model, config, status="COMPLETED")
                    total_runs += 1

                except KeyboardInterrupt:
                    logger.warning("Interrupted by user. Exiting.")
                    sys.exit(0)

                except Exception as exc:
                    logger.error(
                        f"FAILED: {dataset}/{model} "
                        f"({noise_type}@{noise_rate:.0%}): {exc}",
                        exc_info=True,
                    )
                    update_status_log(
                        dataset, model, config,
                        status="FAILED", error=str(exc)
                    )
                    failed_runs += 1
                    continue

                finally:
                    # Always clean up memory after each model run
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

    print(f"\n{'='*60}")
    print(f"Experiment run complete.")
    print(f"  Total completed: {total_runs}")
    print(f"  Failed: {failed_runs}")
    print(f"  Status log: {_STATUS_LOG}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
