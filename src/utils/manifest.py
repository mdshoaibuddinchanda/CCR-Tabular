"""Provenance Manifest Generator for CCR-Tabular Master Runs.

Records deterministic hardware, environment, commit SHA, and experimental matrix parameters.
"""

import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import psutil
import torch

from src.utils.config import (
    BATCH_SIZE,
    BETA,
    CORE_10_DATASETS,
    DATASETS,
    K,
    LEARNING_RATE,
    LOSS_NAMES,
    MULTICLASS_DATASETS,
    N_FOLDS,
    OPTIMIZER,
    OUTPUTS_FINAL_MASTER,
    REAL_WORLD_DATASETS,
    SEEDS,
    TAU,
    WEIGHT_DECAY,
)

logger = logging.getLogger(__name__)


def get_current_git_commit() -> str:
    """Retrieve current Git commit hash or fallback string."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("ascii").strip()
        return commit
    except Exception:
        return "UNKNOWN_COMMIT"


def get_current_git_branch() -> str:
    """Retrieve current Git branch name or fallback string."""
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode("ascii").strip()
        return branch
    except Exception:
        return "UNKNOWN_BRANCH"


def generate_experiment_manifest(output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Generate and write a frozen provenance manifest for the final benchmark."""
    target_dir = output_dir or OUTPUTS_FINAL_MASTER
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = target_dir / "manifest.json"

    manifest_data: Dict[str, Any] = {
        "title": "CCR-Tabular Master Benchmark Provenance Manifest",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": get_current_git_commit(),
            "branch": get_current_git_branch(),
        },
        "environment": {
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "cpu_logical_cores": psutil.cpu_count(logical=True),
            "cpu_physical_cores": psutil.cpu_count(logical=False),
            "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
            "os": platform.platform(),
        },
        "experimental_design": {
            "tier1_core10_datasets": CORE_10_DATASETS,
            "tier2_multiclass_datasets": MULTICLASS_DATASETS,
            "tier3_real_world_datasets": REAL_WORLD_DATASETS,
            "canonical_10_losses": LOSS_NAMES,
            "noise_regimes": [
                {"type": "none", "rate": 0.00},
                {"type": "asym", "rate": 0.20},
                {"type": "asym", "rate": 0.40},
                {"type": "sym", "rate": 0.20},
            ],
            "seeds": SEEDS,
            "n_folds": N_FOLDS,
            "tier1_expected_runs": len(CORE_10_DATASETS) * len(LOSS_NAMES) * 4 * len(SEEDS) * N_FOLDS,
        },
        "hyperparameters": {
            "tau": TAU,
            "beta": BETA,
            "K_history": K,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "optimizer": OPTIMIZER,
        },
    }

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info(f"Generated frozen benchmark provenance manifest at {manifest_file}")
    return manifest_data


if __name__ == "__main__":
    generate_experiment_manifest()
