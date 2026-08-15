"""Tabular ResNet architecture for architecture transfer experiments (Section J).

Follows established tabular deep learning protocols (Gorishniy et al., NeurIPS 2021):
  Input -> Input Linear Projection -> Stack of Residual Blocks with Skip Connections -> Classification Head.
"""

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config import DROPOUT

logger = logging.getLogger(__name__)


class ResNetBlock(nn.Module):
    """Single Tabular ResNet residual block with skip connection."""

    def __init__(
        self,
        d_main: int,
        d_hidden: int,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm1d(d_main)
        self.linear1 = nn.Linear(d_main, d_hidden)
        self.dropout1 = nn.Dropout(p=dropout)
        self.norm2 = nn.BatchNorm1d(d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_main)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm1(x)
        out = F.relu(self.linear1(out))
        out = self.dropout1(out)
        out = self.norm2(out)
        out = self.linear2(out)
        out = self.dropout2(out)
        return residual + out


class TabularResNet(nn.Module):
    """Tabular ResNet for tabular classification."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        d_main: int = 128,
        d_hidden: int = 256,
        n_blocks: int = 3,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Input projection
        self.input_layer = nn.Linear(input_dim, d_main)

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResNetBlock(d_main=d_main, d_hidden=d_hidden, dropout=dropout)
            for _ in range(n_blocks)
        ])

        # Final classification head
        self.final_norm = nn.BatchNorm1d(d_main)
        self.head = nn.Linear(d_main, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.input_layer(x))
        for block in self.blocks:
            out = block(out)
        out = self.final_norm(out)
        out = F.relu(out)
        logits = self.head(out)
        return logits
