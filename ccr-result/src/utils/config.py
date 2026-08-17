"""Central configuration for CCR-Tabular experiments.

This module is the single source of truth for all hyperparameters and paths.
Nothing is hardcoded anywhere else in the codebase.
"""

from pathlib import Path
from typing import Dict, List

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_NOISY = ROOT / "data" / "noisy"
OUTPUTS_MODELS = ROOT / "outputs" / "models"
OUTPUTS_LOGS = ROOT / "outputs" / "logs"
OUTPUTS_METRICS = ROOT / "outputs" / "metrics"
OUTPUTS_PLOTS = ROOT / "outputs" / "plots"

# Create all directories on import
for _p in [
    DATA_RAW, DATA_PROCESSED, DATA_NOISY,
    OUTPUTS_MODELS, OUTPUTS_LOGS, OUTPUTS_METRICS, OUTPUTS_PLOTS,
]:
    try:
        _p.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(
            f"Failed to create required directory '{_p}': {e}. "
            f"Check filesystem permissions."
        ) from e

# ── CCR Hyperparameters (FIXED — do not tune per dataset) ─────────────────────
TAU: float = 0.3    # Confidence gate threshold
BETA: float = 0.5   # Variance scaling factor
K: int = 5          # Rolling epoch window for variance

# ── Training Hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE: int = 512
MAX_EPOCHS: int = 200
EARLY_STOP_PATIENCE: int = 20
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
DROPOUT: float = 0.3

# ── Validation ────────────────────────────────────────────────────────────────
N_FOLDS: int = 5
SEEDS: List[int] = [42, 123, 2024, 1, 2]
VAL_SIZE: float = 0.15  # fraction of training fold used for early stopping

# ── Noise Levels ──────────────────────────────────────────────────────────────
NOISE_RATES: List[float] = [0.10, 0.20, 0.30]

# ── Dataset Registry ──────────────────────────────────────────────────────────
DATASETS: Dict[str, Dict] = {
    "adult": {"openml_id": 1590, "target": "class", "n_samples": 48842},
    "default_credit": {"openml_id": 42477, "target": "y", "n_samples": 30000},
    "credit_fraud": {"openml_id": 1597, "target": "Class", "n_samples": 284807},
    "bank": {"openml_id": 1461, "target": "Class", "n_samples": 45211},
    "aps_failure": {"openml_id": 41138, "target": "class", "n_samples": 76000},
    "covertype": {"openml_id": 150, "target": "class", "n_samples": 581012},
    "higgs": {"openml_id": 42769, "target": "target", "n_samples": 1000000},
    "electricity": {"openml_id": 151, "target": "class", "n_samples": 45312},
    "kddcup99": {"openml_id": 1113, "target": "label", "n_samples": 494020},
    "spambase": {"openml_id": 44, "target": "class", "n_samples": 4601},
    "ionosphere": {"openml_id": 59, "target": "class", "n_samples": 351},
    "phoneme": {"openml_id": 1489, "target": "Class", "n_samples": 5404},
    "magic": {"openml_id": 1120, "target": "class:", "n_samples": 19020},
    "breast_cancer": {"openml_id": 15, "target": "Class", "n_samples": 699},
    "heart_disease": {"openml_id": 53, "target": "class", "n_samples": 270},
    "diabetes": {"openml_id": 37, "target": "class", "n_samples": 768},
    "mammography": {"openml_id": 310, "target": "class", "n_samples": 11183},
    "sonar": {"openml_id": 40, "target": "Class", "n_samples": 208},
    "mushroom": {"openml_id": 24, "target": "class", "n_samples": 8124},
    "census_kdd": {"openml_id": 4535, "target": "V42", "n_samples": 299285}
}

# ── Model Names ───────────────────────────────────────────────────────────────
MODEL_NAMES: List[str] = [
    "mlp_standard",     # B1: vanilla CE
    "mlp_focal",        # B2: focal loss
    "mlp_weighted_ce",  # B3: class-weighted CE
    "mlp_smote",        # B4: SMOTE + MLP
    "xgboost_default",  # B5
    "xgboost_weighted", # B6
    "lightgbm_default", # B7
    "tabnet",           # B8: TabNet
    "ft_transformer",   # B9: FT-Transformer
    "mlp_ccr",          # OURS
]
