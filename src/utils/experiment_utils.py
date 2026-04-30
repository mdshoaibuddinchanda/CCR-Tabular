"""Shared utilities for expansion experiment scripts.

Provides common functions used across run_ablation.py, run_tau_sensitivity.py,
run_k_sensitivity.py, run_beta_sensitivity.py, and run_noise40.py to avoid
code duplication and ensure consistency.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

from src.data.noise_injection import (
    inject_asymmetric_noise,
    inject_feature_correlated_noise,
)
from src.data.preprocess import preprocess_split
from src.loss.ccr_loss import CCRLoss
from src.models.mlp import TabularDataset, get_mlp_for_dataset
from src.utils.config import (
    BATCH_SIZE,
    EARLY_STOP_PATIENCE,
    LEARNING_RATE,
    MAX_EPOCHS,
    N_FOLDS,
    SEEDS,
    VAL_SIZE,
    WEIGHT_DECAY,
)
from src.utils.reproducibility import fix_all_seeds, get_device

logger = logging.getLogger(__name__)


# ── CSV helpers ───────────────────────────────────────────────────────────────

def is_done(run_id: str, output_csv: Path) -> bool:
    """Check if a run_id already exists in the output CSV.

    Args:
        run_id: Unique run identifier to check.
        output_csv: Path to the output CSV file.

    Returns:
        True if run_id is already recorded.
    """
    if not output_csv.exists():
        return False
    try:
        df = pd.read_csv(output_csv)
        return run_id in df["run_id"].values
    except Exception:
        return False


def append_row(row: Dict[str, Any], output_csv: Path) -> None:
    """Append one row to a CSV file with run_id deduplication.

    Creates the file if it doesn't exist. Skips silently if run_id
    already present (idempotent).

    Args:
        row: Dict with at least a 'run_id' key.
        output_csv: Destination CSV path.
    """
    df_new = pd.DataFrame([row])
    if output_csv.exists():
        try:
            df_existing = pd.read_csv(output_csv)
            if row["run_id"] in df_existing["run_id"].values:
                return
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception:
            df_combined = df_new
    else:
        df_combined = df_new
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(output_csv, index=False)


# ── Noise injection ───────────────────────────────────────────────────────────

def apply_noise(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    noise_type: str,
    noise_rate: float,
    seed: int,
) -> np.ndarray:
    """Apply label noise to training data only.

    Args:
        X_tr: Training features (used for feature-correlated noise).
        y_tr: Training labels.
        noise_type: 'none', 'asym', or 'feat'.
        noise_rate: Fraction of labels to corrupt.
        seed: Random seed.

    Returns:
        Possibly-noisy training labels (same shape as y_tr).
    """
    if noise_type == "none" or noise_rate == 0.0:
        return y_tr
    if noise_type == "asym":
        y_noisy, _ = inject_asymmetric_noise(y_tr, noise_rate, seed)
        return y_noisy
    if noise_type == "feat":
        y_noisy, _ = inject_feature_correlated_noise(X_tr, y_tr, noise_rate, seed)
        return y_noisy
    raise ValueError(f"Unknown noise_type '{noise_type}'. Must be 'none', 'asym', or 'feat'.")


# ── Data splitting ────────────────────────────────────────────────────────────

def make_fold_splits(
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    n_folds: int = N_FOLDS,
) -> List[Tuple]:
    """Generate stratified fold splits.

    Args:
        X: Feature DataFrame.
        y: Label array.
        seed: Random seed for StratifiedKFold.
        n_folds: Number of folds.

    Returns:
        List of (train_idx, test_idx) tuples.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(X, y))


def prepare_fold(
    X: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    noise_type: str = "none",
    noise_rate: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """Prepare one fold: split, preprocess, inject noise.

    Preprocessing is fit on training data ONLY. Noise is applied to
    training labels ONLY. Val and test are always clean.

    Args:
        X: Full feature DataFrame.
        y: Full label array.
        train_idx: Training indices from StratifiedKFold.
        test_idx: Test indices from StratifiedKFold.
        seed: Random seed for val split and noise injection.
        noise_type: 'none', 'asym', or 'feat'.
        noise_rate: Fraction of training labels to corrupt.

    Returns:
        (X_tr, X_val, X_te, y_tr_noisy, y_val, y_te) — all numpy arrays.
        y_tr_noisy: training labels with noise applied.
        y_val, y_te: always clean (noise never applied).
    """
    X_tr_df = X.iloc[train_idx].reset_index(drop=True)
    X_te_df = X.iloc[test_idx].reset_index(drop=True)
    y_tr_raw = y[train_idx]
    y_te_raw = y[test_idx]

    # Val split from training fold
    X_tr_df2, X_val_df, y_tr2, y_val = train_test_split(
        X_tr_df, y_tr_raw,
        test_size=VAL_SIZE,
        stratify=y_tr_raw,
        random_state=seed,
    )

    # Preprocess — fit on train ONLY
    (X_tr_np, X_val_np, X_te_np,
     y_tr_np, y_val_np, y_te_np, _) = preprocess_split(
        X_tr_df2, X_val_df, X_te_df,
        pd.Series(y_tr2), pd.Series(y_val), pd.Series(y_te_raw),
    )

    # Noise injection — train ONLY
    y_tr_noisy = apply_noise(X_tr_np, y_tr_np, noise_type, noise_rate, seed)

    return X_tr_np, X_val_np, X_te_np, y_tr_noisy, y_val_np, y_te_np


# ── Training loop ─────────────────────────────────────────────────────────────

def train_ccr_fold(
    dataset_name: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    criterion: CCRLoss,
) -> Tuple[Any, float, List[float]]:
    """Train a CCR model for one fold with early stopping on macro F1.

    Args:
        dataset_name: Dataset key for architecture selection.
        X_tr: Training features.
        y_tr: Training labels (may be noisy).
        X_val: Validation features (always clean).
        y_val: Validation labels (always clean).
        seed: Random seed (fix_all_seeds called internally).
        criterion: Instantiated CCRLoss (or variant) to use.

    Returns:
        Tuple of (trained_model, train_time_seconds, epoch_gate_fracs).
        epoch_gate_fracs: per-epoch mean gate activation fraction.
    """
    fix_all_seeds(seed)
    device = get_device()

    model = get_mlp_for_dataset(dataset_name, X_tr.shape[1]).to(device)
    criterion = criterion.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    loader = DataLoader(
        TabularDataset(X_tr, y_tr),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
    )

    best_f1 = -1.0
    patience = 0
    best_state: Optional[Dict] = None
    epoch_gate_fracs: List[float] = []
    t0 = time.perf_counter()

    for epoch in range(MAX_EPOCHS):
        model.train()
        batch_gate_fracs: List[float] = []

        for X_b, y_b, idx_b in loader:
            X_b = X_b.to(device)
            y_b = y_b.to(device)
            idx_b = idx_b.to(device)

            optimizer.zero_grad()
            logits = model(X_b)

            # Track gate activation (p_i > tau fraction)
            with torch.no_grad():
                probs_d = F.softmax(logits.detach(), dim=1)
                p_i = probs_d[torch.arange(len(y_b), device=device), y_b]
                batch_gate_fracs.append((p_i > criterion.tau).float().mean().item())

            loss = criterion(logits, y_b, idx_b, epoch)
            if torch.isnan(loss):
                logger.warning(f"NaN loss at epoch {epoch + 1}. Stopping.")
                break
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                probs = F.softmax(logits.detach(), dim=1)
                criterion.update_history(probs, idx_b, epoch)

        if batch_gate_fracs:
            epoch_gate_fracs.append(float(np.mean(batch_gate_fracs)))

        # Validate
        model.eval()
        with torch.no_grad():
            X_v = torch.FloatTensor(X_val).to(device)
            preds = model(X_v).argmax(dim=1).cpu().numpy()
        val_f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, time.perf_counter() - t0, epoch_gate_fracs


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(
    model: torch.nn.Module,
    X_te: np.ndarray,
    y_te: np.ndarray,
) -> Dict[str, float]:
    """Evaluate a trained model on the test set.

    Args:
        model: Trained PyTorch model.
        X_te: Test features (always clean — noise never applied here).
        y_te: Test labels (always clean).

    Returns:
        Dict with accuracy, macro_f1, minority_recall, auc_roc, auc_pr.
    """
    device = get_device()
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        X_t = torch.FloatTensor(X_te).to(device)
        logits = model(X_t)
        probs = F.softmax(logits, dim=1).cpu().numpy()

    y_pred = probs.argmax(axis=1)
    y_prob = probs[:, 1]
    has_both_classes = len(np.unique(y_te)) > 1

    return {
        "accuracy":        float(accuracy_score(y_te, y_pred)),
        "macro_f1":        float(f1_score(y_te, y_pred, average="macro", zero_division=0)),
        "minority_recall": float(recall_score(y_te, y_pred, pos_label=1, zero_division=0)),
        "auc_roc":         float(roc_auc_score(y_te, y_prob)) if has_both_classes else float("nan"),
        "auc_pr":          float(average_precision_score(y_te, y_prob)) if has_both_classes else float("nan"),
    }
