"""MLP architecture for CCR-Tabular experiments.

Defines the TabularMLP model and a factory function that selects the
appropriate architecture based on dataset size.
"""

import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn

from src.utils.config import DATASETS, DROPOUT

logger = logging.getLogger(__name__)

# Datasets with fewer than this many samples use a smaller architecture
_SMALL_DATASET_THRESHOLD = 5000
_SMALL_HIDDEN_DIMS = [128, 64]
_DEFAULT_HIDDEN_DIMS = [256, 128]


class TabularMLP(nn.Module):
    """Configurable MLP for tabular classification.

    Architecture (default):
        Linear(input_dim) -> BatchNorm1d -> ReLU -> Dropout
        -> Linear(256) -> BatchNorm1d -> ReLU -> Dropout
        -> Linear(128) -> BatchNorm1d -> ReLU
        -> Linear(num_classes)

    Output: Raw logits (NOT softmax). Softmax is applied inside loss functions.

    Args:
        input_dim: Number of input features.
        num_classes: Number of output classes (2 for binary).
        hidden_dims: List of hidden layer sizes. Default [256, 128].
        dropout: Dropout probability. Default 0.3 from config.

    Note:
        For datasets with < 5000 samples (credit_g, spambase), use
        hidden_dims=[128, 64] to reduce overfitting risk.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int] = None,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = _DEFAULT_HIDDEN_DIMS

        if input_dim <= 0:
            raise ValueError(
                f"input_dim must be positive, got {input_dim}."
            )
        if num_classes < 2:
            raise ValueError(
                f"num_classes must be >= 2, got {num_classes}."
            )

        layers: List[nn.Module] = []
        in_features = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            # No dropout after the last hidden layer
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(p=dropout))
            in_features = hidden_dim

        # Output layer — no BN, no activation (raw logits)
        layers.append(nn.Linear(in_features, num_classes))

        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims

        self._init_weights()

        logger.info(
            f"TabularMLP created: input_dim={input_dim}, "
            f"hidden_dims={hidden_dims}, num_classes={num_classes}, "
            f"dropout={dropout}"
        )

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming uniform for ReLU activations."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor, shape [B, input_dim].

        Returns:
            Raw logits, shape [B, num_classes].
        """
        return self.network(x)


def get_mlp_for_dataset(dataset_name: str, input_dim: int) -> TabularMLP:
    """Factory function: select MLP architecture based on dataset size.

    For small datasets (< 5000 samples), uses a reduced architecture
    [128, 64] to reduce overfitting risk. For larger datasets, uses
    the default [256, 128] architecture.

    Args:
        dataset_name: Key from config.DATASETS (e.g., 'adult', 'credit_g').
        input_dim: Number of input features (after preprocessing).

    Returns:
        TabularMLP instance with appropriate architecture.

    Raises:
        ValueError: If dataset_name is not in config.DATASETS.
    """
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Valid options: {list(DATASETS.keys())}."
        )

    n_samples = DATASETS[dataset_name]["n_samples"]

    if n_samples < _SMALL_DATASET_THRESHOLD:
        hidden_dims = _SMALL_HIDDEN_DIMS
        logger.info(
            f"[{dataset_name}] Small dataset ({n_samples} samples): "
            f"using reduced MLP architecture {hidden_dims}."
        )
    else:
        hidden_dims = _DEFAULT_HIDDEN_DIMS
        logger.info(
            f"[{dataset_name}] Standard dataset ({n_samples} samples): "
            f"using default MLP architecture {hidden_dims}."
        )

    return TabularMLP(
        input_dim=input_dim,
        num_classes=2,
        hidden_dims=hidden_dims,
    )


class TabularDataset(torch.utils.data.Dataset):
    """Tabular dataset that yields (features, label, global_index) tuples.

    The global_index is essential for CCR's history tracking.
    Without it, the history tensor cannot be indexed correctly
    when the DataLoader shuffles samples.

    Args:
        X: Feature array, shape [N, F].
        y: Label array, shape [N].
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.FloatTensor(np.array(X, dtype=np.float32))
        self.y = torch.LongTensor(np.array(y, dtype=np.int64))

        if len(self.X) != len(self.y):
            raise ValueError(
                f"X and y must have the same number of samples. "
                f"Got X={len(self.X)}, y={len(self.y)}."
            )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        """Return (features, label, global_index) for sample at idx.

        Args:
            idx: Sample index (IS the global index when dataset is not subsetted).

        Returns:
            Tuple of (X[idx], y[idx], idx).
        """
        return self.X[idx], self.y[idx], idx
