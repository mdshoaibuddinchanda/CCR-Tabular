"""Batch-level instrumentation and telemetry logger for CCR-Tabular.

Records:
  - Batch weight sum: S = sum(w_i), S/B
  - Individual weight dispersion: max(w_i), std(w_i), CV(w_i)
  - Gradient norm of weighted loss: ||nabla_theta L_weighted||_2
  - Gradient norm of unweighted loss: ||nabla_theta L_unweighted||_2
  - Gradient cosine alignment: cos(nabla_theta L_weighted, nabla_theta L_unweighted)
  - Parameter update norm: ||Delta theta_t||_2 = ||theta_{t+1} - theta_t||_2
  - In-batch noise and minority concentration
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config import OUTPUTS_TELEMETRY

logger = logging.getLogger(__name__)


class BatchInstrumenter:
    """Instruments neural network training at the mini-batch level."""

    def __init__(
        self,
        run_id: str,
        enabled: bool = True,
        save_dir: Optional[Path] = None,
    ) -> None:
        self.run_id = run_id
        self.enabled = enabled
        self.save_dir = save_dir or OUTPUTS_TELEMETRY
        self.records: List[Dict[str, Any]] = []
        self._pending_record: Optional[Dict[str, Any]] = None

    def record_pre_step(
        self,
        epoch: int,
        batch_idx: int,
        loss_val: float,
        loss_fn: nn.Module,
        model: nn.Module,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_indices: Optional[torch.Tensor] = None,
        clean_targets: Optional[torch.Tensor] = None,
        minority_class: int = 1,
    ) -> None:
        """Record gradient and weight statistics before optimizer.step()."""
        if not self.enabled:
            return

        batch_size = logits.shape[0]
        if batch_size == 0:
            return

        # 1. Weight Statistics from Loss Module Telemetry
        telemetry = getattr(loss_fn, "last_telemetry", {})
        raw_sum = telemetry.get("raw_sum", float(batch_size))
        s_over_b = telemetry.get("S_over_B", raw_sum / batch_size)
        max_w = telemetry.get("max_weight", 1.0)
        std_w = telemetry.get("std_weight", 0.0)
        cv_w = std_w / (s_over_b + 1e-8)

        # 2. Gradient Norm of Weighted Loss
        grad_norm_weighted = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm_weighted += p.grad.data.norm(2).item() ** 2
        grad_norm_weighted = np.sqrt(grad_norm_weighted)

        # 3. Unweighted Gradient Norm & Cosine Similarity
        unweighted_loss = F.cross_entropy(logits, targets)
        unweighted_grads = torch.autograd.grad(
            unweighted_loss,
            [p for p in model.parameters() if p.requires_grad],
            retain_graph=True,
            allow_unused=True,
        )

        grad_norm_unweighted = 0.0
        dot_product = 0.0
        for g_unw, p in zip(unweighted_grads, [p for p in model.parameters() if p.requires_grad]):
            if g_unw is not None and p.grad is not None:
                g_unw_flat = g_unw.data.flatten()
                g_w_flat = p.grad.data.flatten()
                grad_norm_unweighted += g_unw_flat.norm(2).item() ** 2
                dot_product += torch.dot(g_unw_flat, g_w_flat).item()

        grad_norm_unweighted = np.sqrt(grad_norm_unweighted)
        if grad_norm_weighted > 1e-8 and grad_norm_unweighted > 1e-8:
            cosine_similarity = dot_product / (grad_norm_weighted * grad_norm_unweighted)
        else:
            cosine_similarity = 1.0

        # 4. Batch Composition Metadata
        minority_count = int(torch.sum(targets == minority_class).item())
        minority_fraction = minority_count / batch_size
        noise_fraction = 0.0
        if clean_targets is not None:
            noise_count = int(torch.sum(targets != clean_targets).item())
            noise_fraction = noise_count / batch_size

        self._pending_record = {
            "run_id": self.run_id,
            "epoch": epoch,
            "batch_idx": batch_idx,
            "loss": round(float(loss_val), 5),
            "S_over_B": round(float(s_over_b), 4),
            "raw_weight_sum": round(float(raw_sum), 4),
            "max_weight": round(float(max_w), 4),
            "weight_std": round(float(std_w), 4),
            "weight_CV": round(float(cv_w), 4),
            "grad_norm_weighted": round(float(grad_norm_weighted), 5),
            "grad_norm_unweighted": round(float(grad_norm_unweighted), 5),
            "grad_cosine_sim": round(float(cosine_similarity), 4),
            "minority_fraction": round(float(minority_fraction), 4),
            "noise_fraction": round(float(noise_fraction), 4),
            "param_update_norm": 0.0,
        }

    def record_post_step(
        self,
        prev_params: Optional[Dict[str, torch.Tensor]],
        model: nn.Module,
    ) -> None:
        """Measure exact parameter update norm ||Delta theta||_2 after optimizer.step()."""
        if not self.enabled or self._pending_record is None:
            return

        update_norm = 0.0
        if prev_params is not None:
            for name, param in model.named_parameters():
                if name in prev_params:
                    diff = param.data - prev_params[name].to(param.device)
                    update_norm += diff.norm(2).item() ** 2
            update_norm = np.sqrt(update_norm)

        self._pending_record["param_update_norm"] = round(float(update_norm), 6)
        self.records.append(self._pending_record)
        self._pending_record = None

    def record_batch(
        self,
        epoch: int,
        batch_idx: int,
        loss_val: float,
        loss_fn: nn.Module,
        model: nn.Module,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_indices: Optional[torch.Tensor] = None,
        prev_params: Optional[Dict[str, torch.Tensor]] = None,
        clean_targets: Optional[torch.Tensor] = None,
        minority_class: int = 1,
    ) -> None:
        """Legacy single-call method."""
        self.record_pre_step(
            epoch=epoch,
            batch_idx=batch_idx,
            loss_val=loss_val,
            loss_fn=loss_fn,
            model=model,
            logits=logits,
            targets=targets,
            sample_indices=sample_indices,
            clean_targets=clean_targets,
            minority_class=minority_class,
        )
        self.record_post_step(prev_params=prev_params, model=model)

    def save(self) -> Optional[Path]:
        """Save telemetry records to CSV file."""
        if not self.enabled or not self.records:
            return None

        self.save_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.save_dir / f"{self.run_id}_telemetry.csv"
        df = pd.DataFrame(self.records)
        df.to_csv(out_path, index=False)
        return out_path
