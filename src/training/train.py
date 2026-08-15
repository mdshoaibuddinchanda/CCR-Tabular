"""Main training loop for CCR-Tabular models and baselines.

Handles single-fold training with early stopping, batch instrumentation,
profiling, and structured logging.
"""

import gc
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.loss.robust_losses import build_loss
from src.models.baselines import build_neural_model, get_baseline
from src.models.mlp import TabularDataset
from src.training.instrumentation import BatchInstrumenter
from src.utils.config import (
    BATCH_SIZE,
    BETA,
    EARLY_STOP_PATIENCE,
    GRAD_CLIP_NORM,
    K,
    LEARNING_RATE,
    MAX_EPOCHS,
    OPTIMIZER,
    OUTPUTS_MODELS,
    TAU,
    WEIGHT_DECAY,
)
from src.utils.logger import RunLogger
from src.utils.metrics import compute_all_metrics
from src.utils.reproducibility import fix_all_seeds, get_device

logger = logging.getLogger(__name__)

_SKLEARN_MODELS = {
    "xgboost_default",
    "xgboost_weighted",
    "lightgbm_default",
    "catboost",
    "catboost_default",
}


def make_run_id(
    dataset_name: str,
    model_name: str,
    noise_type: str,
    noise_rate: float,
    seed: int,
    fold: int,
    architecture: str = "mlp",
) -> str:
    """Generate unique run identifier."""
    rate_str = f"{int(noise_rate * 100):02d}"
    return f"{dataset_name}_{model_name}_{architecture}_{noise_type}_{rate_str}_seed{seed}_fold{fold}"


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
    architecture: str = "mlp",
    optimizer_name: str = OPTIMIZER,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    tau: float = TAU,
    beta: float = BETA,
    K_hist: int = K,
    instrument_batch: bool = False,
    run_id: Optional[str] = None,
    clean_y_train: Optional[np.ndarray] = None,
) -> Tuple[Any, Dict[str, float]]:
    """Train a model on a single fold."""
    fix_all_seeds(seed)

    if run_id is None:
        run_id = make_run_id(dataset_name, model_name, noise_type, noise_rate, seed, fold, architecture)

    wall_start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    run_logger = RunLogger(
        run_id=run_id,
        config_dict={
            "tau": tau, "beta": beta, "K": K_hist,
            "architecture": architecture, "optimizer": optimizer_name,
            "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
            "lr": lr, "weight_decay": weight_decay,
            "early_stop_patience": EARLY_STOP_PATIENCE,
        },
        dataset_name=dataset_name,
        model_name=model_name,
        seed=seed,
        fold=fold,
        noise_config={"type": noise_type, "rate": noise_rate},
    )

    if model_name in _SKLEARN_MODELS:
        model, best_metrics = _train_sklearn_baseline(
            model_name=model_name,
            dataset_name=dataset_name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            seed=seed,
            run_id=run_id,
            run_logger=run_logger,
        )
    else:
        model, best_metrics = _train_neural_model(
            loss_name=model_name,
            architecture=architecture,
            dataset_name=dataset_name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            seed=seed,
            run_id=run_id,
            run_logger=run_logger,
            optimizer_name=optimizer_name,
            lr=lr,
            weight_decay=weight_decay,
            tau=tau,
            beta=beta,
            K_hist=K_hist,
            instrument_batch=instrument_batch,
            clean_y_train=clean_y_train,
        )

    # Record profiling
    wall_time_s = time.perf_counter() - wall_start
    best_metrics["train_time_s"] = round(wall_time_s, 3)

    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        best_metrics["peak_vram_mb"] = round(peak_vram_mb, 2)
        torch.cuda.empty_cache()
    else:
        best_metrics["peak_vram_mb"] = 0.0

    gc.collect()
    return model, best_metrics


def _train_neural_model(
    loss_name: str,
    architecture: str,
    dataset_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    run_id: str,
    run_logger: RunLogger,
    optimizer_name: str = "AdamW",
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    tau: float = TAU,
    beta: float = BETA,
    K_hist: int = K,
    instrument_batch: bool = False,
    clean_y_train: Optional[np.ndarray] = None,
) -> Tuple[Any, Dict[str, float]]:
    """Train neural model with configurable architecture, loss, optimizer, and telemetry."""
    device = get_device()
    n_samples, input_dim = X_train.shape
    num_classes = len(np.unique(np.concatenate([y_train, y_val])))

    model = build_neural_model(
        architecture=architecture,
        input_dim=input_dim,
        num_classes=num_classes,
        dataset_name=dataset_name,
    ).to(device)

    class_counts = [int(np.sum(y_train == c)) for c in range(num_classes)]

    criterion = build_loss(
        loss_name=loss_name,
        n_samples=n_samples,
        n_classes=num_classes,
        class_counts=class_counts,
        device=device,
        tau=tau,
        beta=beta,
        K=K_hist,
    ).to(device)

    # Optimizer selection
    if optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_dataset = TabularDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    instrumenter = BatchInstrumenter(run_id=run_id, enabled=instrument_batch)

    checkpoint_path = OUTPUTS_MODELS / f"{run_id}_ckpt.pt"
    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    best_metrics: Dict[str, float] = {}

    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for b_idx, (X_batch, y_batch, idx_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            idx_batch = idx_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)

            # Compute loss
            if hasattr(criterion, "target_history") or "sample_indices" in criterion.forward.__code__.co_varnames:
                loss = criterion(logits, y_batch, sample_indices=idx_batch, current_epoch=epoch)
            elif hasattr(criterion, "update_history") or "current_epoch" in criterion.forward.__code__.co_varnames:
                loss = criterion(logits, y_batch, idx_batch, epoch)
            else:
                loss = criterion(logits, y_batch)

            if torch.isnan(loss):
                break

            loss.backward(retain_graph=instrument_batch)

            # Instrument batch before step (autograd graph alive for unweighted grads)
            prev_params = None
            if instrument_batch:
                clean_batch = None
                if clean_y_train is not None:
                    clean_batch = torch.LongTensor(clean_y_train[idx_batch.cpu().numpy()]).to(device)

                instrumenter.record_pre_step(
                    epoch=epoch,
                    batch_idx=b_idx,
                    loss_val=loss.item(),
                    loss_fn=criterion,
                    model=model,
                    logits=logits,
                    targets=y_batch,
                    sample_indices=idx_batch,
                    clean_targets=clean_batch,
                )
                prev_params = {name: p.data.clone() for name, p in model.named_parameters()}

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            # Record exact parameter step ||theta_{t+1} - theta_t||_2
            if instrument_batch:
                instrumenter.record_post_step(prev_params=prev_params, model=model)

            # Update history tensor after step
            if hasattr(criterion, "update_history"):
                with torch.no_grad():
                    probs = F.softmax(logits.detach(), dim=1)
                    criterion.update_history(probs, idx_batch, epoch)

            epoch_loss += loss.item()
            n_batches += 1

        # Validation
        val_metrics = _validate_neural_model(model, X_val, y_val, device)
        val_f1 = val_metrics["macro_f1"]

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0
            best_metrics = {**val_metrics, "best_epoch": best_epoch}
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                break

    # Load best checkpoint
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    # Save telemetry if instrumented
    if instrument_batch:
        instrumenter.save()

    return model, best_metrics


def _validate_neural_model(
    model: nn.Module,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on validation fold and return all metrics including calibration."""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_val).to(device)
        logits = model(X_t)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

    return compute_all_metrics(y_true=y_val, y_pred=preds, y_prob=probs)


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
    """Train sklearn / GBDT baseline."""
    model = get_baseline(
        model_name=model_name,
        dataset_name=dataset_name,
        input_dim=X_train.shape[1],
        seed=seed,
    )
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    probs = model.predict_proba(X_val)
    preds = model.predict(X_val)
    metrics = compute_all_metrics(y_true=y_val, y_pred=preds, y_prob=probs)
    metrics["best_epoch"] = 0

    save_path = OUTPUTS_MODELS / f"{run_id}.pkl"
    model.save(save_path)
    return model, metrics
