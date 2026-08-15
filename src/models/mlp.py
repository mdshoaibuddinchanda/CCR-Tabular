"""MLP architecture for CCR-Tabular experiments.

Explicit Architecture Specification (Resolving Reviewer 2 Ambiguity):
  For standard datasets:
    Input [B, D]
    -> Linear(D, 256) -> BatchNorm1d(256) -> ReLU() -> Dropout(p)
    -> Linear(256, 128) -> BatchNorm1d(128) -> ReLU() -> Dropout(p)
    -> Linear(128, num_classes) [Raw Logits]

  For small datasets (< 5000 samples):
    Input [B, D]
    -> Linear(D, 128) -> BatchNorm1d(128) -> ReLU() -> Dropout(p)
    -> Linear(128, 64) -> BatchNorm1d(64) -> ReLU() -> Dropout(p)
    -> Linear(64, num_classes) [Raw Logits]
"""

import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.utils.config import DATASETS, DROPOUT

logger = logging.getLogger(__name__)

_SMALL_DATASET_THRESHOLD = 5000
_SMALL_HIDDEN_DIMS = [128, 64]
_DEFAULT_HIDDEN_DIMS = [256, 128]


class TabularMLP(nn.Module):
    """Configurable MLP for tabular classification."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = _DEFAULT_HIDDEN_DIMS

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}.")

        layers: List[nn.Module] = []
        in_features = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(p=dropout))
            in_features = hidden_dim

        # Final linear classification head (raw logits)
        layers.append(nn.Linear(in_features, num_classes))

        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming uniform initialization for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits [B, num_classes]."""
        return self.network(x)


def get_mlp_for_dataset(
    dataset_name: str,
    input_dim: int,
    num_classes: int = 2,
) -> TabularMLP:
    """Factory function: selects MLP architecture based on dataset size."""
    n_samples = DATASETS.get(dataset_name, {}).get("n_samples", 10000)

    if n_samples < _SMALL_DATASET_THRESHOLD:
        hidden_dims = _SMALL_HIDDEN_DIMS
    else:
        hidden_dims = _DEFAULT_HIDDEN_DIMS

    return TabularMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
    )


class TabularDataset(torch.utils.data.Dataset):
    """Tabular dataset yielding (features, label, global_index)."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.FloatTensor(np.array(X, dtype=np.float32))
        self.y = torch.LongTensor(np.array(y, dtype=np.int64))

        if len(self.X) != len(self.y):
            raise ValueError(f"X ({len(self.X)}) and y ({len(self.y)}) length mismatch.")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], idx
