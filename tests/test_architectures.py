"""Unit tests for Tabular MLP, Tabular ResNet, and Tabular FT-Transformer."""

import numpy as np
import pytest
import torch

from src.models.mlp import TabularMLP
from src.models.resnet import TabularResNet
from src.models.transformer import TabularFTTransformer


def test_tabular_mlp_binary_and_multiclass():
    """TabularMLP forward and backward passes."""
    B, D = 16, 10

    for num_classes in [2, 5]:
        x = torch.randn(B, D)
        model = TabularMLP(input_dim=D, num_classes=num_classes)
        out = model(x)
        assert out.shape == (B, num_classes)
        loss = out.sum()
        loss.backward()


def test_tabular_resnet_forward_backward():
    """TabularResNet forward and backward passes."""
    B, D = 16, 12

    for num_classes in [2, 4]:
        x = torch.randn(B, D)
        model = TabularResNet(input_dim=D, num_classes=num_classes, d_main=64, d_hidden=128, n_blocks=2)
        out = model(x)
        assert out.shape == (B, num_classes)
        loss = out.sum()
        loss.backward()


def test_tabular_transformer_forward_backward():
    """TabularFTTransformer forward and backward passes."""
    B, D = 8, 6

    for num_classes in [2, 3]:
        x = torch.randn(B, D)
        model = TabularFTTransformer(input_dim=D, num_classes=num_classes, d_token=32, n_layers=2, n_heads=2)
        out = model(x)
        assert out.shape == (B, num_classes)
        loss = out.sum()
        loss.backward()


def test_catboost_baseline_fit_predict():
    """CatBoostBaseline fits on tabular data and produces valid probability predictions."""
    from src.models.baselines import CatBoostBaseline
    N, D = 50, 4
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((N, D))
    y_train = rng.integers(0, 2, N)
    X_val = rng.standard_normal((20, D))
    y_val = rng.integers(0, 2, 20)

    model = CatBoostBaseline(seed=42)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)

    assert len(preds) == len(X_val)
    assert probs.shape == (len(X_val), 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
