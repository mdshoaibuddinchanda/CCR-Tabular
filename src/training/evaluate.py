"""Model evaluation and results persistence for CCR-Tabular.

Loads saved models, runs inference on test sets, computes all metrics
(including calibration metrics ECE and Brier score), and appends results to CSV
with process-safe locking, explicit run status tracking, and noise rate transparency.
"""

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.utils.config import OUTPUTS_METRICS
from src.utils.metrics import compute_all_metrics

logger = logging.getLogger(__name__)

_RESULTS_CSV = OUTPUTS_METRICS / "results.csv"

_RESULTS_COLUMNS = [
    "run_id", "dataset", "model", "architecture", "optimizer", "fold", "seed",
    "noise_type", "noise_rate", "actual_noise_rate",
    "target_conditional_noise_rate", "actual_conditional_noise_rate",
    "target_overall_noise_rate", "actual_overall_noise_rate",
    "status", "accuracy", "macro_f1", "minority_recall", "auc_roc", "auc_pr",
    "ece", "brier_score",
    "train_time_s", "peak_vram_mb", "n_epochs",
    "timestamp",
]


def evaluate_model(
    model_or_path: Union[Any, Path],
    X_test: np.ndarray,
    y_test: np.ndarray,
    run_id: str,
    metadata: Dict[str, Any],
    train_time_s: float = 0.0,
    peak_vram_mb: float = 0.0,
    n_epochs: int = -1,
    results_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Evaluate trained model on untouched test set and persist results."""
    y_pred, y_prob = _get_predictions(model_or_path, X_test, device=device)

    metrics = compute_all_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        minority_class=1,
    )

    target_csv = results_path or _RESULTS_CSV

    results_row = {
        "run_id": run_id,
        "dataset": metadata.get("dataset", ""),
        "model": metadata.get("model", ""),
        "architecture": metadata.get("architecture", "mlp"),
        "optimizer": metadata.get("optimizer", "AdamW"),
        "fold": metadata.get("fold", -1),
        "seed": metadata.get("seed", -1),
        "noise_type": metadata.get("noise_type", "none"),
        "noise_rate": metadata.get("noise_rate", 0.0),
        "actual_noise_rate": metadata.get("actual_noise_rate", metadata.get("noise_rate", 0.0)),
        "target_conditional_noise_rate": metadata.get("target_conditional_noise_rate", metadata.get("noise_rate", 0.0)),
        "actual_conditional_noise_rate": metadata.get("actual_conditional_noise_rate", metadata.get("actual_noise_rate", 0.0)),
        "target_overall_noise_rate": metadata.get("target_overall_noise_rate", metadata.get("noise_rate", 0.0)),
        "actual_overall_noise_rate": metadata.get("actual_overall_noise_rate", metadata.get("actual_noise_rate", 0.0)),
        "status": metadata.get("status", "SUCCESS"),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "minority_recall": metrics["minority_recall"],
        "auc_roc": metrics["auc_roc"],
        "auc_pr": metrics["auc_pr"],
        "ece": metrics["ece"],
        "brier_score": metrics["brier_score"],
        "train_time_s": round(train_time_s, 3),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "n_epochs": n_epochs,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    append_results(results_row, target_csv)
    return results_row


def append_results(results_row: Dict, results_path: Path) -> None:
    """Append one row to results CSV with process-safe file locking, timeout enforcement, and deduplication."""
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = results_path.with_suffix(".lock")

    df_new = pd.DataFrame([results_row])

    # Spin-lock with stale lock detection (30s) and timeout enforcement
    lock_acquired = False
    max_wait = 20.0
    start_wait = time.time()

    while time.time() - start_wait < max_wait:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            lock_acquired = True
            break
        except FileExistsError:
            # Check for stale lock file older than 30s
            try:
                if lock_path.exists() and (time.time() - lock_path.stat().st_mtime > 30.0):
                    lock_path.unlink(missing_ok=True)
            except Exception:
                pass
            time.sleep(0.05)

    if not lock_acquired:
        raise TimeoutError(f"Could not acquire process-safe write lock on '{lock_path}' after {max_wait}s.")

    try:
        if results_path.exists() and results_path.stat().st_size > 0:
            try:
                df_existing = pd.read_csv(results_path)
                if "run_id" in df_existing.columns and results_row["run_id"] in df_existing["run_id"].values:
                    return
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            except Exception:
                df_combined = df_new
        else:
            df_combined = df_new

        df_combined.to_csv(results_path, index=False)
    finally:
        if lock_acquired and lock_path.exists():
            try:
                lock_path.unlink(missing_ok=True)
            except Exception:
                pass


def _get_predictions(
    model_or_path: Union[Any, Path],
    X_test: np.ndarray,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract hard predictions and probability array."""
    if isinstance(model_or_path, Path):
        import pickle
        if model_or_path.suffix == ".pkl":
            with open(model_or_path, "rb") as f:
                model = pickle.load(f)
        else:
            raise RuntimeError("Direct loading from .pt path requires model class.")
    else:
        model = model_or_path

    if isinstance(model, torch.nn.Module):
        if device is None:
            from src.utils.reproducibility import get_device
            device = get_device()
        model.eval()
        model = model.to(device)
        with torch.no_grad():
            X_t = torch.FloatTensor(X_test).to(device)
            logits = model(X_t)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        return preds, probs

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
        if hasattr(model, "predict"):
            preds = model.predict(X_test)
        else:
            preds = probs.argmax(axis=1)
        return preds, probs

    raise TypeError(f"Unsupported model type: {type(model_or_path)}.")
