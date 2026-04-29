"""Baseline models for CCR-Tabular experiments.

All baselines implement a unified interface compatible with the training
and evaluation pipeline. MLP baselines use the same architecture as CCR.
"""

import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.mlp import TabularDataset, TabularMLP, get_mlp_for_dataset
from src.utils.config import (
    BATCH_SIZE,
    EARLY_STOP_PATIENCE,
    LEARNING_RATE,
    MAX_EPOCHS,
    WEIGHT_DECAY,
)
from src.utils.reproducibility import get_device

logger = logging.getLogger(__name__)


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaselineModel(ABC):
    """Abstract base class for all CCR-Tabular baseline models.

    All baselines must implement fit, predict, predict_proba, save, and load.
    """

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train the model.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Optional validation features for early stopping.
            y_val: Optional validation labels.
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard class predictions.

        Args:
            X: Feature array, shape [N, F].

        Returns:
            Predicted class indices, shape [N].
        """

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities.

        Args:
            X: Feature array, shape [N, F].

        Returns:
            Probability array, shape [N, 2].
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model to disk.

        Args:
            path: Destination file path.
        """

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaselineModel":
        """Load model from disk.

        Args:
            path: Source file path.

        Returns:
            Loaded model instance.
        """


# ── MLP training helper ───────────────────────────────────────────────────────

def _train_mlp(
    model: TabularMLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    loss_fn: nn.Module,
    seed: int = 42,
) -> TabularMLP:
    """Generic MLP training loop with early stopping on macro F1.

    Args:
        model: TabularMLP instance.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features (optional).
        y_val: Validation labels (optional).
        loss_fn: Loss function module.
        seed: Random seed.

    Returns:
        Trained model with best validation weights loaded.
    """
    from sklearn.metrics import f1_score

    device = get_device()
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    train_dataset = TabularDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )

    best_val_f1 = -1.0
    patience_counter = 0
    best_state: Optional[dict] = None

    for epoch in range(MAX_EPOCHS):
        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        for X_batch, y_batch, _ in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            if torch.isnan(loss):
                logger.warning(f"NaN loss at epoch {epoch + 1}. Stopping training.")
                break
            loss.backward()
            optimizer.step()

        # ── Validation ────────────────────────────────────────────────────────
        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                X_val_t = torch.FloatTensor(X_val).to(device)
                logits_val = model(X_val_t)
                y_pred = logits_val.argmax(dim=1).cpu().numpy()

            val_f1 = float(f1_score(y_val, y_pred, average="macro", zero_division=0))

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    logger.info(f"Early stopping at epoch {epoch + 1}.")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ── B1: Standard cross-entropy MLP ───────────────────────────────────────────

class MLPStandardBaseline(BaselineModel):
    """MLP with standard cross-entropy loss (Baseline B1).

    Args:
        dataset_name: Dataset key for architecture selection.
        input_dim: Number of input features.
        seed: Random seed.
    """

    def __init__(self, dataset_name: str, input_dim: int, seed: int = 42) -> None:
        self.dataset_name = dataset_name
        self.input_dim = input_dim
        self.seed = seed
        self.model: Optional[TabularMLP] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        self.model = get_mlp_for_dataset(self.dataset_name, self.input_dim)
        loss_fn = nn.CrossEntropyLoss()
        self.model = _train_mlp(
            self.model, X_train, y_train, X_val, y_val, loss_fn, self.seed
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        device = get_device()
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(device)
            logits = self.model(X_t)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save: model not fitted.")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "dataset_name": self.dataset_name,
            "input_dim": self.input_dim,
            "seed": self.seed,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "MLPStandardBaseline":
        checkpoint = torch.load(path, map_location="cpu")
        obj = cls(
            dataset_name=checkpoint["dataset_name"],
            input_dim=checkpoint["input_dim"],
            seed=checkpoint["seed"],
        )
        obj.model = get_mlp_for_dataset(checkpoint["dataset_name"], checkpoint["input_dim"])
        obj.model.load_state_dict(checkpoint["state_dict"])
        return obj


# ── B2: Focal Loss MLP ────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss for binary classification.

    focal_loss = -alpha * (1 - p_t)^gamma * log(p_t + eps)

    Args:
        alpha: Weighting factor for the rare class. Default 0.25.
        gamma: Focusing parameter. Default 2.0.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Raw model outputs, shape [B, C].
            targets: Ground truth labels, shape [B].

        Returns:
            Scalar focal loss.
        """
        probs = F.softmax(logits, dim=1)
        batch_size = logits.shape[0]
        # p_t = probability of the true class
        p_t = probs[torch.arange(batch_size, device=logits.device), targets]
        focal_weight = self.alpha * (1.0 - p_t) ** self.gamma
        loss = -focal_weight * torch.log(p_t + 1e-8)
        return loss.mean()


class MLPFocalLossBaseline(BaselineModel):
    """MLP with Focal Loss (Baseline B2).

    Args:
        dataset_name: Dataset key for architecture selection.
        input_dim: Number of input features.
        seed: Random seed.
        alpha: Focal loss alpha. Default 0.25.
        gamma: Focal loss gamma. Default 2.0.
    """

    def __init__(
        self,
        dataset_name: str,
        input_dim: int,
        seed: int = 42,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> None:
        self.dataset_name = dataset_name
        self.input_dim = input_dim
        self.seed = seed
        self.alpha = alpha
        self.gamma = gamma
        self.model: Optional[TabularMLP] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        self.model = get_mlp_for_dataset(self.dataset_name, self.input_dim)
        loss_fn = FocalLoss(alpha=self.alpha, gamma=self.gamma)
        self.model = _train_mlp(
            self.model, X_train, y_train, X_val, y_val, loss_fn, self.seed
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        device = get_device()
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(device)
            logits = self.model(X_t)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save: model not fitted.")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "dataset_name": self.dataset_name,
            "input_dim": self.input_dim,
            "seed": self.seed,
            "alpha": self.alpha,
            "gamma": self.gamma,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "MLPFocalLossBaseline":
        checkpoint = torch.load(path, map_location="cpu")
        obj = cls(
            dataset_name=checkpoint["dataset_name"],
            input_dim=checkpoint["input_dim"],
            seed=checkpoint["seed"],
            alpha=checkpoint.get("alpha", 0.25),
            gamma=checkpoint.get("gamma", 2.0),
        )
        obj.model = get_mlp_for_dataset(checkpoint["dataset_name"], checkpoint["input_dim"])
        obj.model.load_state_dict(checkpoint["state_dict"])
        return obj


# ── B3: Class-Weighted CE MLP ─────────────────────────────────────────────────

class MLPWeightedCEBaseline(BaselineModel):
    """MLP with class-weighted cross-entropy loss (Baseline B3).

    Args:
        dataset_name: Dataset key for architecture selection.
        input_dim: Number of input features.
        seed: Random seed.
    """

    def __init__(self, dataset_name: str, input_dim: int, seed: int = 42) -> None:
        self.dataset_name = dataset_name
        self.input_dim = input_dim
        self.seed = seed
        self.model: Optional[TabularMLP] = None
        self._class_weights: Optional[torch.Tensor] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        n_majority = int(np.sum(y_train == 0))
        n_minority = int(np.sum(y_train == 1))

        if n_majority == 0 or n_minority == 0:
            raise ValueError(
                f"Both classes must be present in y_train. "
                f"Got n_majority={n_majority}, n_minority={n_minority}."
            )

        weights = torch.tensor(
            [1.0 / n_majority, 1.0 / n_minority], dtype=torch.float32
        )
        weights = weights / weights.sum()
        self._class_weights = weights

        device = get_device()
        loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))

        self.model = get_mlp_for_dataset(self.dataset_name, self.input_dim)
        self.model = _train_mlp(
            self.model, X_train, y_train, X_val, y_val, loss_fn, self.seed
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        device = get_device()
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(device)
            logits = self.model(X_t)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save: model not fitted.")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "dataset_name": self.dataset_name,
            "input_dim": self.input_dim,
            "seed": self.seed,
            "class_weights": self._class_weights,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "MLPWeightedCEBaseline":
        checkpoint = torch.load(path, map_location="cpu")
        obj = cls(
            dataset_name=checkpoint["dataset_name"],
            input_dim=checkpoint["input_dim"],
            seed=checkpoint["seed"],
        )
        obj.model = get_mlp_for_dataset(checkpoint["dataset_name"], checkpoint["input_dim"])
        obj.model.load_state_dict(checkpoint["state_dict"])
        obj._class_weights = checkpoint.get("class_weights")
        return obj


# ── B4: SMOTE + MLP ───────────────────────────────────────────────────────────

class MLPSMOTEBaseline(BaselineModel):
    """MLP trained on SMOTE-resampled data (Baseline B4).

    SMOTE is applied INSIDE fit() on training data only.
    Never applied to val or test.

    Args:
        dataset_name: Dataset key for architecture selection.
        input_dim: Number of input features.
        seed: Random seed.
    """

    def __init__(self, dataset_name: str, input_dim: int, seed: int = 42) -> None:
        self.dataset_name = dataset_name
        self.input_dim = input_dim
        self.seed = seed
        self.model: Optional[TabularMLP] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        from imblearn.over_sampling import SMOTE

        n_minority = int(np.sum(y_train == 1))

        if n_minority < 2:
            raise ValueError(
                f"SMOTE requires at least 2 minority samples. Got {n_minority}."
            )

        k_neighbors = min(5, n_minority - 1)
        if k_neighbors < 5:
            logger.warning(
                f"SMOTE: n_minority={n_minority} < 6, "
                f"using k_neighbors={k_neighbors} instead of default 5."
            )

        smote = SMOTE(random_state=self.seed, k_neighbors=k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        logger.info(
            f"SMOTE resampling: {len(y_train)} → {len(y_resampled)} samples "
            f"(minority: {n_minority} → {int(np.sum(y_resampled == 1))})"
        )

        self.model = get_mlp_for_dataset(self.dataset_name, self.input_dim)
        loss_fn = nn.CrossEntropyLoss()
        self.model = _train_mlp(
            self.model, X_resampled, y_resampled, X_val, y_val, loss_fn, self.seed
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        device = get_device()
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(device)
            logits = self.model(X_t)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save: model not fitted.")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "dataset_name": self.dataset_name,
            "input_dim": self.input_dim,
            "seed": self.seed,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "MLPSMOTEBaseline":
        checkpoint = torch.load(path, map_location="cpu")
        obj = cls(
            dataset_name=checkpoint["dataset_name"],
            input_dim=checkpoint["input_dim"],
            seed=checkpoint["seed"],
        )
        obj.model = get_mlp_for_dataset(checkpoint["dataset_name"], checkpoint["input_dim"])
        obj.model.load_state_dict(checkpoint["state_dict"])
        return obj


# ── B5: XGBoost Default ───────────────────────────────────────────────────────

class XGBoostDefaultBaseline(BaselineModel):
    """XGBoost with default settings (Baseline B5).

    Args:
        seed: Random seed.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.model = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        import xgboost as xgb

        n_minority = int(np.sum(y_train == 1))
        if n_minority == 0:
            raise ValueError(
                "XGBoost training requires at least one minority sample. "
                "Got n_minority=0. Check your data split."
            )

        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model = xgb.XGBClassifier(
            eval_metric="logloss",
            random_state=self.seed,
            use_label_encoder=False,
            verbosity=0,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict_proba(X)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "XGBoostDefaultBaseline":
        with open(path, "rb") as f:
            return pickle.load(f)


# ── B6: XGBoost Weighted ──────────────────────────────────────────────────────

class XGBoostWeightedBaseline(BaselineModel):
    """XGBoost with scale_pos_weight for class imbalance (Baseline B6).

    Args:
        seed: Random seed.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.model = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        import xgboost as xgb

        n_majority = int(np.sum(y_train == 0))
        n_minority = int(np.sum(y_train == 1))

        if n_minority == 0:
            raise ValueError(
                "XGBoostWeighted requires at least one minority sample. "
                "Got n_minority=0."
            )

        scale_pos_weight = n_majority / n_minority
        logger.info(f"XGBoostWeighted: scale_pos_weight={scale_pos_weight:.2f}")

        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=self.seed,
            use_label_encoder=False,
            verbosity=0,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict_proba(X)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "XGBoostWeightedBaseline":
        with open(path, "rb") as f:
            return pickle.load(f)


# ── B7: LightGBM Default ──────────────────────────────────────────────────────

class LightGBMDefaultBaseline(BaselineModel):
    """LightGBM with default settings (Baseline B7).

    Args:
        seed: Random seed.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.model = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        import lightgbm as lgb

        callbacks = [lgb.log_evaluation(period=-1)]  # suppress output
        eval_set = [(X_val, y_val)] if X_val is not None else None

        self.model = lgb.LGBMClassifier(
            random_state=self.seed,
            verbose=-1,
        )
        # Convert to DataFrame to avoid "feature names" warning during inference
        import pandas as _pd
        X_train_df = _pd.DataFrame(X_train)
        X_val_df = _pd.DataFrame(X_val) if X_val is not None else None
        eval_set_df = [(X_val_df, y_val)] if X_val_df is not None else None
        self.model.fit(
            X_train_df, y_train,
            eval_set=eval_set_df,
            callbacks=callbacks,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        import pandas as _pd
        return self.model.predict(_pd.DataFrame(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        import pandas as _pd
        return self.model.predict_proba(_pd.DataFrame(X))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "LightGBMDefaultBaseline":
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Factory ───────────────────────────────────────────────────────────────────

def get_baseline(
    model_name: str,
    dataset_name: str,
    input_dim: int,
    seed: int = 42,
) -> BaselineModel:
    """Factory function: return the appropriate baseline model instance.

    Args:
        model_name: One of config.MODEL_NAMES (excluding 'mlp_ccr').
        dataset_name: Dataset key from config.DATASETS.
        input_dim: Number of input features.
        seed: Random seed.

    Returns:
        Instantiated (unfitted) baseline model.

    Raises:
        ValueError: If model_name is not recognized.
    """
    registry = {
        "mlp_standard":     lambda: MLPStandardBaseline(dataset_name, input_dim, seed),
        "mlp_focal":        lambda: MLPFocalLossBaseline(dataset_name, input_dim, seed),
        "mlp_weighted_ce":  lambda: MLPWeightedCEBaseline(dataset_name, input_dim, seed),
        "mlp_smote":        lambda: MLPSMOTEBaseline(dataset_name, input_dim, seed),
        "xgboost_default":  lambda: XGBoostDefaultBaseline(seed),
        "xgboost_weighted": lambda: XGBoostWeightedBaseline(seed),
        "lightgbm_default": lambda: LightGBMDefaultBaseline(seed),
    }

    if model_name not in registry:
        raise ValueError(
            f"Unknown baseline model '{model_name}'. "
            f"Valid options: {list(registry.keys())}. "
            f"For CCR, use the dedicated CCRLoss in src/loss/ccr_loss.py."
        )

    return registry[model_name]()
