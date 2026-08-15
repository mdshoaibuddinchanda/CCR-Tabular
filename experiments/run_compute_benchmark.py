"""Computational cost and memory profiling benchmark (Section S).

Measures and compares:
  - Training time per epoch (ms/epoch)
  - Sample throughput (samples/sec)
  - Peak GPU VRAM (MB)
  - Memory overhead (%) relative to standard Cross Entropy
across:
  - Loss functions: CE, WCE, Focal, GCE, SCE, ELR, CCR-NoNorm, CCR
  - Architectures: TabularMLP, TabularResNet, TabularFTTransformer
"""

import argparse
import gc
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.load_data import load_dataset
from src.data.preprocess import preprocess_split
from src.loss.robust_losses import build_loss
from src.models.baselines import build_neural_model
from src.models.mlp import TabularDataset
from src.utils.config import BATCH_SIZE, OUTPUTS_METRICS
from src.utils.reproducibility import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def benchmark_model_loss_pair(
    dataset_name: str = "adult",
    architecture: str = "mlp",
    loss_name: str = "ccr",
    n_epochs: int = 5,
    n_warmup: int = 2,
    batch_size: int = BATCH_SIZE,
) -> Dict[str, float]:
    """Profile wall-clock epoch duration, throughput, and peak memory for a model-loss pair."""
    device = get_device()
    df = load_dataset(dataset_name)
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols]
    y = df["target"].values

    n_tr = int(len(X) * 0.8)
    X_tr, X_val = X.iloc[:n_tr], X.iloc[n_tr:]
    y_tr, y_val = y[:n_tr], y[n_tr:]
    (X_tr_np, _, _, y_tr_np, _, _, _) = preprocess_split(
        X_tr, X_val, X_val, pd.Series(y_tr), pd.Series(y_val), pd.Series(y_val)
    )

    n_samples, input_dim = X_tr_np.shape
    num_classes = 2
    class_counts = [int(np.sum(y_tr_np == 0)), int(np.sum(y_tr_np == 1))]

    model = build_neural_model(architecture, input_dim=input_dim, num_classes=num_classes).to(device)
    criterion = build_loss(
        loss_name=loss_name,
        n_samples=n_samples,
        n_classes=num_classes,
        class_counts=class_counts,
        device=device,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataset = TabularDataset(X_tr_np, y_tr_np)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model.train()
    for _ in range(n_warmup):
        for X_b, y_b, idx_b in loader:
            X_b, y_b, idx_b = X_b.to(device), y_b.to(device), idx_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            if hasattr(criterion, "update_history") or "current_epoch" in criterion.forward.__code__.co_varnames:
                loss = criterion(logits, y_b, idx_b, 0)
            else:
                loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

    epoch_times = []
    total_samples = 0

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    for epoch in range(n_epochs):
        ep_start = time.perf_counter()
        for X_b, y_b, idx_b in loader:
            X_b, y_b, idx_b = X_b.to(device), y_b.to(device), idx_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            if hasattr(criterion, "update_history") or "current_epoch" in criterion.forward.__code__.co_varnames:
                loss = criterion(logits, y_b, idx_b, epoch)
            else:
                loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            total_samples += len(X_b)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ep_dur = time.perf_counter() - ep_start
        epoch_times.append(ep_dur)

    total_time = time.perf_counter() - t0
    avg_epoch_ms = (np.mean(epoch_times)) * 1000.0
    samples_per_sec = total_samples / total_time

    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        peak_vram_mb = 0.0

    return {
        "dataset": dataset_name,
        "architecture": architecture,
        "loss": loss_name,
        "avg_epoch_ms": round(avg_epoch_ms, 2),
        "samples_per_sec": round(samples_per_sec, 1),
        "peak_vram_mb": round(peak_vram_mb, 2),
    }


def run_full_compute_benchmark() -> pd.DataFrame:
    """Run profiling benchmark across all loss functions and architectures."""
    losses = ["ce", "wce", "focal", "gce", "sce", "elr", "ccr_no_norm", "ccr"]

    results = []

    logger.info("Profiling losses on TabularMLP...")
    for loss in losses:
        res = benchmark_model_loss_pair(dataset_name="adult", architecture="mlp", loss_name=loss)
        results.append(res)
        logger.info(f"MLP + {loss:12s}: {res['avg_epoch_ms']:6.2f} ms/epoch | {res['samples_per_sec']:7.1f} samples/s | {res['peak_vram_mb']:6.1f} MB VRAM")

    logger.info("Profiling ResNet and FT-Transformer...")
    for arch in ["resnet", "ft_transformer"]:
        for loss in ["ce", "ccr"]:
            res = benchmark_model_loss_pair(dataset_name="adult", architecture=arch, loss_name=loss)
            results.append(res)
            logger.info(f"{arch:15s} + {loss:6s}: {res['avg_epoch_ms']:6.2f} ms/epoch | {res['samples_per_sec']:7.1f} samples/s | {res['peak_vram_mb']:6.1f} MB VRAM")

    df = pd.DataFrame(results)

    base_mlp_ce = df[(df["architecture"] == "mlp") & (df["loss"] == "ce")]["avg_epoch_ms"].values[0]
    df["time_overhead_vs_ce_pct"] = (((df["avg_epoch_ms"] - base_mlp_ce) / base_mlp_ce) * 100.0).round(2)

    out_csv = OUTPUTS_METRICS / "computational_overhead_benchmark.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved compute benchmark results to {out_csv}")
    return df


if __name__ == "__main__":
    run_full_compute_benchmark()
