"""Main training loop for CCR and all MLP baselines.

Handles single-fold training with early stopping, checkpointing,
and structured logging.
"""

import gc
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.loss.ccr_loss import CCRLoss
from src.models.baselines import get_baseline
from src.models.mlp import TabularDataset, get_mlp_for_dataset
from src.utils.config import (
    BATCH_SIZE,
    BETA,
    EARLY_STOP_PATIENCE,
    K,
    LEARNING_RATE,
    MAX_EPOCHS,
    OUTPUTS_MODELS,
    TAU,
    WEIGHT_DECAY,
)
from src.utils.logger import RunLogger
from src.utils.reproducibility import fix_all_seeds, get_device

logger = logging.getLogger(__name__)

# MLP-based model names that use the PyTorch training loop
_MLP_MODELS = {"mlp_standard", "mlp_focal", "mlp_weighted_ce", "mlp_smote", "mlp_ccr"}
# Sklearn-compatible models
_SKLEARN_MODELS = {"xgboost_default", "xgboost_weighted", "lightgbm_default"}


def make_run_id(
    dataset_name: str,
    model_name: str,
    noise_type: str,
    noise_rate: float,
    seed: int,
    fold: int,
) -> str:
    """Generate a unique run identifier.

    Args:
        dataset_name: Dataset key.
        model_name: Model name.
        noise_type: 'none', 'asym', or 'feat'.
        noise_rate: Noise rate (0.0–0.3).
        seed: Random seed.
        fold: Fold number (1-indexed).

    Returns:
        Run ID string, e.g. 'adult_mlp_ccr_asym_20_seed42_fold1'.
    """
    rate_str = f"{int(noise_rate * 100):02d}"
    return f"{dataset_name}_{model_name}_{noise_type}_{rate_str}_seed{seed}_fold{fold}"


def train_one_fold(
    model_name: str,
    dataset_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    fold: int,
    seed: int,
    noise_type: str = "none",
    noise_rate: float = 0.0,
    run_id: Optional[str] = None,
) -> Tuple[Any, Dict[str, float]]:
    """Train a single model on a single fold.

    Args:
        model_name: One of config.MODEL_NAMES.
        dataset_name: One of config.DATASETS keys.
        X_train: Training features (preprocessed, noise-injected).
        y_train: Training labels.
        X_val: Validation features (never noisy, never scaled on its own).
        y_val: Validation labels.
        fold: Fold number (1-indexed).
        seed: Random seed.
        noise_type: 'none', 'asym', or 'feat'.
        noise_rate: 0.0, 0.1, 0.2, or 0.3.
        run_id: Optional explicit run ID. Auto-generated if None.

    Returns:
        Tuple of (best_model, best_val_metrics_dict).

    Saves:
        Best model checkpoint to outputs/models/<run_id>.pt or .pkl.
        Training log to outputs/logs/<run_id>_train.json.

    Raises:
        ValueError: If model_name is not recognized.

    Note:
        Early stopping monitors val_macro_f1, NOT accuracy.
    """
    fix_all_seeds(seed)

    if run_id is None:
        run_id = make_run_id(dataset_name, model_name, noise_type, noise_rate, seed, fold)

    wall_start = time.perf_counter()

    run_logger = RunLogger(
        run_id=run_id,
        config_dict={
            "tau": TAU, "beta": BETA, "K": K,
            "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
            "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "early_stop_patience": EARLY_STOP_PATIENCE,
        },
        dataset_name=dataset_name,
        model_name=model_name,
        seed=seed,
        fold=fold,
        noise_config={"type": noise_type, "rate": noise_rate},
    )

    # ── Route to appropriate training function ────────────────────────────────
    if model_name == "mlp_ccr":
        model, best_metrics = _train_ccr(
            dataset_name, X_train, y_train, X_val, y_val,
            seed, run_id, run_logger,
        )
    elif model_name in _MLP_MODELS:
        model, best_metrics = _train_mlp_baseline(
            model_name, dataset_name, X_train, y_train, X_val, y_val,
            seed, run_id, run_logger,
        )
    elif model_name in _SKLEARN_MODELS:
        model, best_metrics = _train_sklearn_baseline(
            model_name, dataset_name, X_train, y_train, X_val, y_val,
            seed, run_id, run_logger,
        )
    else:
        raise ValueError(
            f"Unknown model_name '{model_name}'. "
            f"Valid options: {list(_MLP_MODELS | _SKLEARN_MODELS)}."
        )

    # ── Save model ────────────────────────────────────────────────────────────
    suffix = ".pt" if model_name in _MLP_MODELS else ".pkl"
    save_path = OUTPUTS_MODELS / f"{run_id}{suffix}"
    _save_model(model, model_name, save_path)
    run_logger.log(f"Model saved to {save_path}")

    wall_time_s = time.perf_counter() - wall_start
    best_metrics["train_time_s"] = round(wall_time_s, 3)
    run_logger.log(f"Total training time: {wall_time_s:.1f}s")

    run_logger.finalize(
        best_epoch=best_metrics.get("best_epoch", -1),
        best_val_f1=best_metrics.get("macro_f1", 0.0),
    )

    # ── GPU memory cleanup ────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return model, best_metrics


# ── CCR training ──────────────────────────────────────────────────────────────

def _train_ccr(
    dataset_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    run_id: str,
    run_logger: RunLogger,
) -> Tuple[Any, Dict[str, float]]:
    """Train the CCR model with the full CCRLoss training loop.

    Args:
        dataset_name: Dataset key.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        seed: Random seed.
        run_id: Run identifier for checkpointing.
        run_logger: RunLogger instance.

    Returns:
        Tuple of (trained_model, best_val_metrics).
    """
    device = get_device()
    run_logger.log(f"Device: {device}")

    model = get_mlp_for_dataset(dataset_name, X_train.shape[1])
    model = model.to(device)

    # Class counts for CCR weight computation
    n_majority = int(np.sum(y_train == 0))
    n_minority = int(np.sum(y_train == 1))
    class_counts = [n_majority, n_minority]

    criterion = CCRLoss(
        n_samples=len(y_train),
        n_classes=2,
        class_counts=class_counts,
        device=device,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    train_dataset = TabularDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )

    checkpoint_path = OUTPUTS_MODELS / f"{run_id}_ckpt.pt"
    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    best_metrics: Dict[str, float] = {}
    epoch_times: list = []

    for epoch in range(MAX_EPOCHS):
        epoch_start = time.perf_counter()
        # ── Training epoch ────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_batch, y_batch, idx_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            idx_batch = idx_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)

            try:
                loss = criterion(logits, y_batch, idx_batch, epoch)
            except RuntimeError as exc:
                run_logger.log(f"Loss computation error at epoch {epoch + 1}: {exc}")
                raise

            if torch.isnan(loss):
                run_logger.log(f"NaN loss detected at epoch {epoch + 1}. Stopping.")
                break

            loss.backward()
            optimizer.step()

            # Update CCR history AFTER optimizer step
            with torch.no_grad():
                probs = F.softmax(logits.detach(), dim=1)
                criterion.update_history(probs, idx_batch, epoch)

            epoch_loss += loss.item()
            n_batches += 1

        if n_batches == 0:
            break

        train_loss = epoch_loss / n_batches

        # ── Validation ────────────────────────────────────────────────────────
        val_metrics = _validate_mlp(model, X_val, y_val, device)
        val_f1 = val_metrics["macro_f1"]

        # ── Early stopping (monitor macro F1, NOT accuracy) ───────────────────
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0
            best_metrics = {**val_metrics, "best_epoch": best_epoch}
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                run_logger.log(f"Early stopping at epoch {epoch + 1}.")
                break

        lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)
        run_logger.log_epoch(epoch, train_loss, val_metrics, lr=lr)

    # ── Record mean epoch time ────────────────────────────────────────────────
    if epoch_times:
        best_metrics["mean_epoch_time_s"] = round(sum(epoch_times) / len(epoch_times), 4)
        best_metrics["n_epochs"] = len(epoch_times)

    # ── Load best checkpoint ──────────────────────────────────────────────────
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        checkpoint_path.unlink(missing_ok=True)
    else:
        run_logger.log(
            f"WARNING: Checkpoint not found at {checkpoint_path}. "
            f"Returning model from last epoch."
        )

    return model, best_metrics


# ── MLP baseline training ─────────────────────────────────────────────────────

def _train_mlp_baseline(
    model_name: str,
    dataset_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    run_id: str,
    run_logger: RunLogger,
) -> Tuple[Any, Dict[str, float]]:
    """Train an MLP baseline model.

    Args:
        model_name: Baseline model name.
        dataset_name: Dataset key.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        seed: Random seed.
        run_id: Run identifier (used for logging context).
        run_logger: RunLogger instance.

    Returns:
        Tuple of (trained_model, best_val_metrics).
    """
    t0 = time.perf_counter()
    baseline = get_baseline(model_name, dataset_name, X_train.shape[1], seed)
    baseline.fit(X_train, y_train, X_val, y_val)

    # Compute final validation metrics
    y_prob = baseline.predict_proba(X_val)[:, 1]
    y_pred = baseline.predict(X_val)
    val_metrics = _compute_val_metrics(y_val, y_pred, y_prob)
    val_metrics["best_epoch"] = -1   # not tracked for baselines
    val_metrics["n_epochs"] = -1
    val_metrics["train_time_s"] = round(time.perf_counter() - t0, 3)

    run_logger.log(
        f"Baseline training complete. val_macro_f1={val_metrics['macro_f1']:.4f}"
    )
    return baseline, val_metrics


# ── Sklearn baseline training ─────────────────────────────────────────────────

def _train_sklearn_baseline(
    model_name: str,
    dataset_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    run_id: str,
    run_logger: RunLogger,
) -> Tuple[Any, Dict[str, float]]:
    """Train a sklearn-compatible baseline (XGBoost, LightGBM).

    Args:
        model_name: Baseline model name.
        dataset_name: Dataset key.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        seed: Random seed.
        run_id: Run identifier (used for logging context).
        run_logger: RunLogger instance.

    Returns:
        Tuple of (trained_model, best_val_metrics).
    """
    t0 = time.perf_counter()
    baseline = get_baseline(model_name, dataset_name, X_train.shape[1], seed)
    baseline.fit(X_train, y_train, X_val, y_val)

    y_prob = baseline.predict_proba(X_val)[:, 1]
    y_pred = baseline.predict(X_val)
    val_metrics = _compute_val_metrics(y_val, y_pred, y_prob)
    val_metrics["best_epoch"] = -1
    val_metrics["n_epochs"] = -1
    val_metrics["train_time_s"] = round(time.perf_counter() - t0, 3)

    run_logger.log(
        f"Sklearn baseline training complete. val_macro_f1={val_metrics['macro_f1']:.4f}"
    )
    return baseline, val_metrics


# ── Validation helpers ────────────────────────────────────────────────────────

def _validate_mlp(
    model: torch.nn.Module,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
) -> Dict[str, float]:
    """Run inference on validation set and compute metrics.

    Args:
        model: Trained PyTorch model.
        X_val: Validation features.
        y_val: Validation labels.
        device: Torch device.

    Returns:
        Dict with accuracy, macro_f1, minority_recall.
    """
    model.eval()
    with torch.no_grad():
        X_val_t = torch.FloatTensor(X_val).to(device)
        logits = model(X_val_t)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        y_pred = probs.argmax(axis=1)

    return _compute_val_metrics(y_val, y_pred, probs[:, 1])


def _compute_val_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,  # noqa: ARG001 — reserved for future val AUC computation
) -> Dict[str, float]:
    """Compute lightweight validation metrics (no AUC — speed-optimised for early stopping).

    Args:
        y_true: Ground truth labels.
        y_pred: Hard predictions.
        y_prob: Predicted probabilities for minority class (reserved for future use).

    Returns:
        Dict with accuracy, macro_f1, minority_recall.
    """
    from sklearn.metrics import accuracy_score, f1_score, recall_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "minority_recall": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
    }


def _save_model(model: Any, model_name: str, path: Path) -> None:
    """Save a model to disk.

    Args:
        model: Trained model (PyTorch or sklearn-compatible).
        model_name: Model name for routing save logic.
        path: Destination path.
    """
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)

    if model_name in _MLP_MODELS:
        if hasattr(model, "state_dict"):
            torch.save(model.state_dict(), path)
        elif hasattr(model, "model") and hasattr(model.model, "state_dict"):
            torch.save(model.model.state_dict(), path)
        else:
            raise RuntimeError(
                f"Cannot save MLP model of type {type(model)}. "
                f"Expected a PyTorch module with state_dict()."
            )
    else:
        with open(path, "wb") as f:
            pickle.dump(model, f)
