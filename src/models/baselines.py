"""Baseline models and neural architecture factory for CCR-Tabular experiments.

Supports:
  1. Neural architectures: TabularMLP, TabularResNet, TabularFTTransformer
  2. Neural baselines with arbitrary loss functions (CE, WCE, Focal, GCE, SCE, ELR, CCR, Norm-losses)
  3. GBDT models: XGBoost (default & weighted), LightGBM (default)
"""

import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.mlp import TabularDataset, TabularMLP, get_mlp_for_dataset
from src.models.resnet import TabularResNet
from src.models.transformer import TabularFTTransformer
from src.utils.config import (
    BATCH_SIZE,
    EARLY_STOP_PATIENCE,
    GRAD_CLIP_NORM,
    LEARNING_RATE,
    MAX_EPOCHS,
    WEIGHT_DECAY,
)
from src.utils.reproducibility import get_device

logger = logging.getLogger(__name__)


def build_neural_model(
    architecture: str,
    input_dim: int,
    num_classes: int = 2,
    dataset_name: Optional[str] = None,
    dropout: float = 0.3,
) -> nn.Module:
    """Factory function for neural architectures.

    Args:
        architecture: 'mlp', 'resnet', or 'ft_transformer' / 'transformer'.
        input_dim: Number of input features.
        num_classes: Number of target classes.
        dataset_name: Dataset name for MLP scaling heuristic.
        dropout: Dropout rate.

    Returns:
        Instantiated nn.Module.
    """
    arch = architecture.lower().strip()
    if arch == "mlp":
        if dataset_name is not None:
            return get_mlp_for_dataset(dataset_name, input_dim, num_classes=num_classes)
        return TabularMLP(input_dim=input_dim, num_classes=num_classes, dropout=dropout)
    elif arch in ("resnet", "tabular_resnet"):
        return TabularResNet(input_dim=input_dim, num_classes=num_classes, dropout=dropout)
    elif arch in ("ft_transformer", "transformer", "fttransformer"):
        return TabularFTTransformer(input_dim=input_dim, num_classes=num_classes, dropout=dropout)
    else:
        raise ValueError(
            f"Unknown architecture '{architecture}'. Valid options: ['mlp', 'resnet', 'ft_transformer']."
        )


class BaselineModel(ABC):
    """Abstract base class for all CCR-Tabular baseline models."""

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Fit model on training fold."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict hard class labels."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model checkpoint."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaselineModel":
        """Load model checkpoint."""


# ── Generic Neural Baseline Wrapper ───────────────────────────────────────────

class NeuralBaseline(BaselineModel):
    """Generic neural baseline supporting any architecture + loss combination."""

    def __init__(
        self,
        architecture: str = "mlp",
        loss_name: str = "ce",
        dataset_name: str = "adult",
        input_dim: int = 14,
        num_classes: int = 2,
        seed: int = 42,
        lr: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        optimizer_name: str = "AdamW",
    ) -> None:
        self.architecture = architecture
        self.loss_name = loss_name
        self.dataset_name = dataset_name
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.seed = seed
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.model: Optional[nn.Module] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        from sklearn.metrics import f1_score
        from src.loss.robust_losses import build_loss

        device = get_device()
        self.model = build_neural_model(
            self.architecture,
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            dataset_name=self.dataset_name,
        ).to(device)

        class_counts = [int(np.sum(y_train == c)) for c in range(self.num_classes)]

        loss_fn = build_loss(
            loss_name=self.loss_name,
            n_samples=len(y_train),
            n_classes=self.num_classes,
            class_counts=class_counts,
            device=device,
        )

        # Optimizer selection
        if self.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        elif self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        train_dataset = TabularDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

        best_val_f1 = -1.0
        patience_counter = 0
        best_state = None

        for epoch in range(MAX_EPOCHS):
            self.model.train()
            for X_b, y_b, idx_b in train_loader:
                X_b, y_b, idx_b = X_b.to(device), y_b.to(device), idx_b.to(device)
                optimizer.zero_grad()
                logits = self.model(X_b)

                # Loss forward
                if hasattr(loss_fn, "target_history") or "sample_indices" in loss_fn.forward.__code__.co_varnames:
                    loss = loss_fn(logits, y_b, sample_indices=idx_b, current_epoch=epoch)
                elif hasattr(loss_fn, "update_history") or "current_epoch" in loss_fn.forward.__code__.co_varnames:
                    loss = loss_fn(logits, y_b, idx_b, epoch)
                else:
                    loss = loss_fn(logits, y_b)

                if torch.isnan(loss):
                    break
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

                if hasattr(loss_fn, "update_history"):
                    with torch.no_grad():
                        probs = F.softmax(logits.detach(), dim=1)
                        loss_fn.update_history(probs, idx_b, epoch, targets=y_b)

            # Validation
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    X_val_t = torch.FloatTensor(X_val).to(device)
                    logits_val = self.model(X_val_t)
                    y_pred = logits_val.argmax(dim=1).cpu().numpy()

                val_f1 = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= EARLY_STOP_PATIENCE:
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        device = get_device()
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(device)
            logits = self.model(X_t)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save unfitted model.")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "architecture": self.architecture,
            "loss_name": self.loss_name,
            "dataset_name": self.dataset_name,
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "seed": self.seed,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "NeuralBaseline":
        checkpoint = torch.load(path, map_location="cpu")
        obj = cls(
            architecture=checkpoint["architecture"],
            loss_name=checkpoint["loss_name"],
            dataset_name=checkpoint["dataset_name"],
            input_dim=checkpoint["input_dim"],
            num_classes=checkpoint.get("num_classes", 2),
            seed=checkpoint["seed"],
        )
        obj.model = build_neural_model(
            obj.architecture, input_dim=obj.input_dim, num_classes=obj.num_classes, dataset_name=obj.dataset_name
        )
        obj.model.load_state_dict(checkpoint["state_dict"])
        return obj


# ── Tree Baselines (XGBoost, LightGBM) ────────────────────────────────────────

class XGBoostBaseline(BaselineModel):
    """XGBoost baseline with optional class reweighting."""

    def __init__(self, weighted: bool = False, seed: int = 42) -> None:
        self.weighted = weighted
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

        kwargs = {
            "random_state": self.seed,
            "verbosity": 0,
            "n_jobs": 1,
        }
        n_classes = len(np.unique(y_train))
        if n_classes == 2 and self.weighted:
            n_0 = int(np.sum(y_train == 0))
            n_1 = int(np.sum(y_train == 1))
            if n_1 > 0:
                kwargs["scale_pos_weight"] = n_0 / n_1

        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None
        self.model = xgb.XGBClassifier(**kwargs)
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "XGBoostBaseline":
        with open(path, "rb") as f:
            return pickle.load(f)


class LightGBMBaseline(BaselineModel):
    """LightGBM baseline with single-thread process isolation."""

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
        import pandas as pd

        callbacks = [lgb.log_evaluation(period=-1)]
        self.model = lgb.LGBMClassifier(random_state=self.seed, n_jobs=1, verbose=-1)
        X_train_df = pd.DataFrame(X_train)
        X_val_df = pd.DataFrame(X_val) if X_val is not None else None
        eval_set_df = [(X_val_df, y_val)] if X_val_df is not None else None
        self.model.fit(X_train_df, y_train, eval_set=eval_set_df, callbacks=callbacks)

    def predict(self, X: np.ndarray) -> np.ndarray:
        import pandas as pd
        return self.model.predict(pd.DataFrame(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import pandas as pd
        return self.model.predict_proba(pd.DataFrame(X))

class CatBoostBaseline(BaselineModel):
    """CatBoost gradient boosting baseline with single-thread process isolation."""

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
        from catboost import CatBoostClassifier
        eval_set = (X_val, y_val) if X_val is not None and y_val is not None else None
        self.model = CatBoostClassifier(
            random_seed=self.seed,
            thread_count=1,
            verbose=False,
            early_stopping_rounds=20 if eval_set is not None else None,
        )
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).astype(int).flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "CatBoostBaseline":
        with open(path, "rb") as f:
            return pickle.load(f)


def get_baseline(
    model_name: str,
    dataset_name: str,
    input_dim: int,
    num_classes: int = 2,
    seed: int = 42,
) -> BaselineModel:
    """Factory returning configured baseline model."""
    name = model_name.lower().strip()

    if name == "xgboost_default":
        return XGBoostBaseline(weighted=False, seed=seed)
    elif name == "xgboost_weighted":
        return XGBoostBaseline(weighted=True, seed=seed)
    elif name == "lightgbm_default":
        return LightGBMBaseline(seed=seed)
    elif name in ("catboost", "catboost_default"):
        return CatBoostBaseline(seed=seed)
    elif name.startswith("mlp_"):
        loss_key = name.replace("mlp_", "")
        if loss_key == "standard":
            loss_key = "ce"
        elif loss_key == "weighted_ce":
            loss_key = "wce"
        return NeuralBaseline(
            architecture="mlp",
            loss_name=loss_key,
            dataset_name=dataset_name,
            input_dim=input_dim,
            num_classes=num_classes,
            seed=seed,
        )
    else:
        return NeuralBaseline(
            architecture="mlp",
            loss_name=name,
            dataset_name=dataset_name,
            input_dim=input_dim,
            num_classes=num_classes,
            seed=seed,
        )
