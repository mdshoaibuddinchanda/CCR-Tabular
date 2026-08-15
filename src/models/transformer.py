"""FT-Transformer (Feature Tokenizer + Transformer) for tabular learning.

Follows Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data", NeurIPS 2021:
  - Feature Tokenizer transforms each continuous feature into an embedding vector.
  - [CLS] token is prepended.
  - Stack of Transformer Encoder layers with Multi-Head Self-Attention.
  - Classification head applied to the [CLS] output token.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config import DROPOUT

logger = logging.getLogger(__name__)


class NumericalFeatureTokenizer(nn.Module):
    """Transforms numerical features x in R^{B x D} into token embeddings in R^{B x D x d_token}."""

    def __init__(self, n_features: int, d_token: int) -> None:
        super().__init__()
        # Weight per feature: [n_features, d_token]
        self.weight = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n_features]
        # output: [B, n_features, d_token]
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class TabularFTTransformer(nn.Module):
    """FT-Transformer architecture for tabular classification."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        d_token: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        d_ffn_factor: float = 4.0,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.d_token = d_token

        # Feature tokenizer
        self.tokenizer = NumericalFeatureTokenizer(n_features=input_dim, d_token=d_token)

        # [CLS] token embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)

        # Transformer encoder layers
        d_ffn = int(d_token * d_ffn_factor)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output classification head
        self.head_norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Tokenize features -> [B, D, d_token]
        tokens = self.tokenizer(x)

        # Prepend [CLS] token -> [B, 1 + D, d_token]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        # Transformer processing
        out = self.transformer(tokens)

        # Take representation of [CLS] token at index 0
        cls_out = out[:, 0, :]
        cls_out = self.head_norm(cls_out)
        cls_out = F.relu(cls_out)
        logits = self.head(cls_out)
        return logits
