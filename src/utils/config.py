"""Central configuration registry for CCR-Tabular experiments.

Structured according to the rigorous 10 + 2 + 2 experimental design:
  - Tier 1: 10 Core Binary Benchmark datasets spanning 5 orders of variation:
            N from 351 to 48,842, IR from 1.35 to 17.50, and 10 diverse domains.
  - Tier 2 / Multiclass: 2 representative multiclass datasets (Segment, Steel Faults).
  - Tier 3 / Real-World External Validation: 2 real-world clinical datasets (Heart Disease, Breast Cancer).
"""

from pathlib import Path
from typing import Any, Dict, List

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_NOISY = ROOT / "data" / "noisy"
OUTPUTS_MODELS = ROOT / "outputs" / "models"
OUTPUTS_LOGS = ROOT / "outputs" / "logs"
OUTPUTS_METRICS = ROOT / "outputs" / "metrics"
OUTPUTS_PLOTS = ROOT / "outputs" / "plots"
OUTPUTS_TELEMETRY = ROOT / "outputs" / "telemetry"

for _p in [
    DATA_RAW, DATA_PROCESSED, DATA_NOISY,
    OUTPUTS_MODELS, OUTPUTS_LOGS, OUTPUTS_METRICS, OUTPUTS_PLOTS, OUTPUTS_TELEMETRY,
]:
    try:
        _p.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create directory '{_p}': {e}.") from e

# ── CCR Core Hyperparameters (Fixed Defaults) ─────────────────────────────────
TAU: float = 0.3      # Confidence gate threshold
BETA: float = 0.5     # Variance scaling factor
K: int = 5            # Rolling epoch history window

# ── Training Defaults ─────────────────────────────────────────────────────────
BATCH_SIZE: int = 512
MAX_EPOCHS: int = 200
EARLY_STOP_PATIENCE: int = 20
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
DROPOUT: float = 0.3
OPTIMIZER: str = "AdamW"  # Options: 'AdamW', 'Adam', 'SGD'
GRAD_CLIP_NORM: float = 5.0

# ── Cross-Validation ──────────────────────────────────────────────────────────
N_FOLDS: int = 5
SEEDS: List[int] = [42, 123, 2024]
VAL_SIZE: float = 0.15

# ── Noise Configurations ──────────────────────────────────────────────────────
NOISE_TYPES: List[str] = ["none", "asym", "sym", "feat", "idn"]
NOISE_RATES: List[float] = [0.10, 0.20, 0.30, 0.40]

# ── Dataset Registry (10 Core Binary + 2 Multiclass + 2 Real-World External) ──
DATASETS: Dict[str, Dict[str, Any]] = {
    # ── Tier 1: Core 10 Binary Benchmark Datasets ──
    "adult":        {"openml_id": 1590,  "target": "class",  "n_samples": 48842, "type": "binary", "ir": 3.17,  "domain": "Socioeconomic / Census"},
    "bank":         {"openml_id": 1461,  "target": "Class",  "n_samples": 45211, "type": "binary", "ir": 7.55,  "domain": "Marketing / Finance"},
    "magic":        {"openml_id": 1120,  "target": "class",  "n_samples": 19020, "type": "binary", "ir": 1.84,  "domain": "Physics / Gamma Radiation"},
    "phoneme":      {"openml_id": 1489,  "target": "Class",  "n_samples": 5404,  "type": "binary", "ir": 2.41,  "domain": "Acoustics / Signal"},
    "spambase":     {"openml_id": 44,    "target": "class",  "n_samples": 4601,  "type": "binary", "ir": 1.54,  "domain": "Text / Email"},
    "credit_g":     {"openml_id": 31,    "target": "class",  "n_samples": 1000,  "type": "binary", "ir": 2.33,  "domain": "Finance / Credit"},
    "churn":        {"openml_id": 40701, "target": "class",  "n_samples": 5000,  "type": "binary", "ir": 6.07,  "domain": "Customer Behavior"},
    "electricity":  {"openml_id": 151,   "target": "class",  "n_samples": 45312, "type": "binary", "ir": 1.36,  "domain": "Energy / Electricity"},
    "wilt":         {"openml_id": 40983, "target": "class",  "n_samples": 4839,  "type": "binary", "ir": 17.54, "domain": "Remote Sensing / Forestry"},
    "ionosphere":   {"openml_id": 59,    "target": "class",  "n_samples": 351,   "type": "binary", "ir": 1.79,  "domain": "Radar / Aerospace"},

    # ── Tier 2 / Multiclass Transfer (2 Datasets) ──
    "segment":      {"openml_id": 36,    "target": "class",  "n_samples": 2310,  "type": "multiclass", "n_classes": 7, "domain": "Image Vision"},
    "vehicle":      {"openml_id": 54,    "target": "Class",  "n_samples": 846,   "type": "multiclass", "n_classes": 4, "domain": "Silhouette Vision"},

    # ── Tier 3 / Real-World External Validation (2 Datasets) ──
    # Note: Labeled as 'real_world_external' as annotation noise provenance is clinical ambiguity rather than multiple adjudicated label disagreement.
    "heart_disease":{"openml_id": 1498,  "target": "Class",  "n_samples": 462,   "type": "real_world_external", "ir": 1.89, "domain": "Clinical Cardiology"},
    "breast_cancer":{"openml_id": 13,    "target": "class",  "n_samples": 286,   "type": "real_world_external", "ir": 2.36, "domain": "Clinical Pathology"},
}

CORE_10_DATASETS: List[str] = [
    "adult", "bank", "magic", "phoneme", "spambase",
    "credit_g", "churn", "electricity", "wilt", "ionosphere",
]

MULTICLASS_DATASETS: List[str] = ["segment", "vehicle"]
REAL_WORLD_DATASETS: List[str] = ["heart_disease", "breast_cancer"]
NATURAL_NOISE_DATASETS = REAL_WORLD_DATASETS  # Alias for backward compatibility

# ── Loss Matrix (10 Canonical Losses) ─────────────────────────────────────────
LOSS_NAMES: List[str] = [
    # Baseline losses
    "ce",
    "wce",
    "norm_wce",
    "focal",
    "norm_focal",
    "gce",
    "sce",
    "elr",
    # CCR variants
    "ccr_no_norm",
    "ccr",
]

# ── Architecture Types ────────────────────────────────────────────────────────
ARCHITECTURES: List[str] = ["mlp", "resnet", "ft_transformer"]
