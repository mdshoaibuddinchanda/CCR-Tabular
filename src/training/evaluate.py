"""Model evaluation and results persistence for CCR-Tabular.

Loads saved models, runs inference on test sets, computes all metrics,
and appends results to the central results CSV.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.utils.config import OUTPUTS_METRICS
from src.utils.metrics import compute_all_metrics

logger = logging.getLogger(__name__)

_RESULTS_CSV = OUTPUTS_METRICS / "results.csv"

_RESULTS_COLUMNS = [
    "run_id", "dataset", "model", "fold", "seed",
    "noise_type", "noise_rate",
    "accuracy", "macro_f1", "minority_recall", "auc_roc", "auc_pr",
    "train_time_s", "n_epochs",
    "timestamp",
]


def evaluate_model(
    model_or_path: Union[Any, Path],
    X_test: np.ndarray,
    y_test: np.ndarray,
    run_id: str,
    metadata: Dict[str, Any],
    train_time_s: float = 0.0,
    n_epochs: int = -1,
) -> Dict[str, float]:
    """Evaluate a trained model on the test set.

    Args:
        model_or_path: Either a trained model object or path to saved checkpoint.
        X_test: Test features. NEVER noisy.
        y_test: Test labels. NEVER noisy.
        run_id: Unique identifier for this run.
        metadata: Dict with keys: dataset, model, fold, seed, noise_type, noise_rate.
        train_time_s: Total wall-clock training time in seconds.
        n_epochs: Number of epochs trained (after early stopping).

    Returns:
        Dict with all metrics: accuracy, macro_f1, minority_recall, auc_roc, auc_pr.

    Saves:
        Appends one row to outputs/metrics/results.csv.
        Columns: run_id, dataset, model, fold, seed, noise_type, noise_rate,
                 accuracy, macro_f1, minority_recall, auc_roc, auc_pr, timestamp.

    Note:
        AUC metrics use predicted PROBABILITIES, not hard labels.
        results.csv is append-mode with run_id deduplication on load.
    """
    # ── Get predictions ───────────────────────────────────────────────────────
    y_pred, y_prob = _get_predictions(model_or_path, X_test, metadata.get("model", ""))

    # ── Compute metrics ───────────────────────────────────────────────────────
    metrics = compute_all_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        minority_class=1,
    )

    logger.info(
        f"[{run_id}] Test metrics: "
        f"macro_f1={metrics['macro_f1']:.4f}, "
        f"minority_recall={metrics['minority_recall']:.4f}, "
        f"auc_roc={metrics['auc_roc']:.4f}"
    )

    # ── Build results row ─────────────────────────────────────────────────────
    results_row = {
        "run_id": run_id,
        "dataset": metadata.get("dataset", ""),
        "model": metadata.get("model", ""),
        "fold": metadata.get("fold", -1),
        "seed": metadata.get("seed", -1),
        "noise_type": metadata.get("noise_type", "none"),
        "noise_rate": metadata.get("noise_rate", 0.0),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "minority_recall": metrics["minority_recall"],
        "auc_roc": metrics["auc_roc"],
        "auc_pr": metrics["auc_pr"],
        "train_time_s": round(train_time_s, 3),
        "n_epochs": n_epochs,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    append_results(results_row, _RESULTS_CSV)

    return metrics


def append_results(results_row: Dict, results_path: Path) -> None:
    """Append one row to results.csv. Handles file creation and deduplication.

    If results.csv doesn't exist, creates it with headers.
    If run_id already exists in file, SKIPS (idempotent — safe to re-run).
    Never overwrites existing results.

    Args:
        results_row: Dict with all required columns.
        results_path: Path to the results CSV file.
    """
    df_new = pd.DataFrame([results_row])

    if results_path.exists():
        try:
            df_existing = pd.read_csv(results_path)
        except Exception as exc:
            logger.warning(
                f"Could not read existing results.csv: {exc}. "
                f"Creating a new file."
            )
            df_existing = pd.DataFrame(columns=_RESULTS_COLUMNS)

        if results_row["run_id"] in df_existing["run_id"].values:
            logger.info(
                f"[SKIP] run_id '{results_row['run_id']}' already in results.csv. "
                f"Skipping to preserve idempotency."
            )
            return

        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    results_path.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(results_path, index=False)
    logger.info(f"Results appended to {results_path}")


def _get_predictions(
    model_or_path: Union[Any, Path],
    X_test: np.ndarray,
    model_name: str,
) -> tuple:
    """Get hard predictions and probabilities from a model.

    Args:
        model_or_path: Trained model or path to checkpoint.
        X_test: Test features.
        model_name: Model name for routing inference logic.

    Returns:
        Tuple of (y_pred, y_prob) where y_prob is for the minority class.
    """
    from src.training.train import _MLP_MODELS

    # ── Load from path if needed ──────────────────────────────────────────────
    if isinstance(model_or_path, Path):
        model = _load_model_from_path(model_or_path, model_name)
    else:
        model = model_or_path

    # ── PyTorch MLP inference ─────────────────────────────────────────────────
    if isinstance(model, torch.nn.Module):
        from src.utils.reproducibility import get_device
        device = get_device()
        model.eval()
        model = model.to(device)
        with torch.no_grad():
            X_t = torch.FloatTensor(X_test).to(device)
            logits = model(X_t)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        y_pred = probs.argmax(axis=1)
        y_prob = probs[:, 1]
        return y_pred, y_prob

    # ── Sklearn-compatible inference ──────────────────────────────────────────
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
        y_pred = probs.argmax(axis=1)
        y_prob = probs[:, 1]
        return y_pred, y_prob

    # ── Baseline wrapper inference ────────────────────────────────────────────
    if hasattr(model, "predict") and hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
        y_prob = probs[:, 1]
        return y_pred, y_prob

    raise RuntimeError(
        f"Cannot get predictions from model of type {type(model)}. "
        f"Model must have predict() and predict_proba() methods, "
        f"or be a torch.nn.Module."
    )


def _load_model_from_path(path: Path, model_name: str = "") -> Any:  # noqa: ARG001
    """Load a model from a checkpoint file.

    Args:
        path: Path to the saved model file.
        model_name: Unused — reserved for future architecture-aware loading.

    Returns:
        Loaded model.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    import pickle

    if not path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at '{path}'. "
            f"Ensure training completed successfully before evaluation."
        )

    if path.suffix == ".pt":
        # PyTorch state dict — caller must provide model architecture
        logger.warning(
            f"Loading raw state dict from {path}. "
            f"Caller should pass a model object, not a path, for MLP models."
        )
        return torch.load(path, map_location="cpu", weights_only=True)

    with open(path, "rb") as f:
        return pickle.load(f)
