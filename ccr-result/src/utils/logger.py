"""Structured logging for CCR-Tabular training runs.

Every training run writes a JSON log file to outputs/logs/<run_id>_train.json
and prints timestamped progress to stdout.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.utils.config import MAX_EPOCHS, OUTPUTS_LOGS

# Module-level stdlib logger for library use
_stdlib_logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging to stdout with timestamps.

    Args:
        level: Logging level. Default INFO.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


class RunLogger:
    """Per-run structured logger that writes JSON logs and prints to stdout.

    Args:
        run_id: Unique identifier for this training run.
        config_dict: Hyperparameter dict to embed in the log header.
        dataset_name: Name of the dataset being trained on.
        model_name: Name of the model being trained.
        seed: Random seed used for this run.
        fold: Cross-validation fold number.
        noise_config: Dict with keys 'type' and 'rate' describing noise setup.
    """

    def __init__(
        self,
        run_id: str,
        config_dict: Dict[str, Any],
        dataset_name: str,
        model_name: str,
        seed: int,
        fold: int,
        noise_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.run_id = run_id
        self.log_path = OUTPUTS_LOGS / f"{run_id}_train.json"
        self.epoch_records: List[Dict[str, Any]] = []
        self.start_time = datetime.now(timezone.utc).isoformat()

        self.header: Dict[str, Any] = {
            "run_id": run_id,
            "start_time": self.start_time,
            "dataset": dataset_name,
            "model": model_name,
            "seed": seed,
            "fold": fold,
            "noise": noise_config or {"type": "none", "rate": 0.0},
            "config": config_dict,
            "epochs": [],
        }

        _stdlib_logger.info(
            f"RunLogger initialized | run_id={run_id} | "
            f"dataset={dataset_name} | model={model_name} | "
            f"seed={seed} | fold={fold}"
        )

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_metrics: Dict[str, float],
        lr: float = 0.0,
    ) -> None:
        """Log one epoch's metrics to JSON file and print to stdout.

        Args:
            epoch: Current epoch number (0-indexed).
            train_loss: Mean training loss for this epoch.
            val_metrics: Dict containing at least 'macro_f1' and 'minority_recall'.
            lr: Current learning rate.
        """
        record: Dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss": round(float(train_loss), 6),
            "val_macro_f1": round(float(val_metrics.get("macro_f1", 0.0)), 6),
            "val_minority_recall": round(float(val_metrics.get("minority_recall", 0.0)), 6),
            "val_accuracy": round(float(val_metrics.get("accuracy", 0.0)), 6),
            "lr": lr,
        }
        self.epoch_records.append(record)

        # Print to stdout
        print(
            f"[EPOCH {epoch + 1:03d}/{MAX_EPOCHS}] "
            f"loss={record['train_loss']:.4f} | "
            f"val_f1={record['val_macro_f1']:.4f} | "
            f"val_recall={record['val_minority_recall']:.4f}",
            flush=True,
        )

        self._flush()

    def log(self, message: str) -> None:
        """Log a free-form message to stdout and the JSON log.

        Args:
            message: Message string to log.
        """
        _stdlib_logger.info(f"[{self.run_id}] {message}")
        self._flush()

    def finalize(self, best_epoch: int, best_val_f1: float) -> None:
        """Write final summary to the JSON log.

        Args:
            best_epoch: Epoch number with best validation F1.
            best_val_f1: Best validation macro F1 achieved.
        """
        self.header["end_time"] = datetime.now(timezone.utc).isoformat()
        self.header["best_epoch"] = best_epoch
        self.header["best_val_macro_f1"] = round(best_val_f1, 6)
        self._flush()
        _stdlib_logger.info(
            f"[{self.run_id}] Training complete | "
            f"best_epoch={best_epoch} | best_val_f1={best_val_f1:.4f}"
        )

    def _flush(self) -> None:
        """Write current state to JSON log file. Handles write failures gracefully."""
        try:
            payload = dict(self.header)
            payload["epochs"] = self.epoch_records
            with open(self.log_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        except OSError as exc:
            # Log to stdout but do NOT crash training
            _stdlib_logger.warning(
                f"[{self.run_id}] Failed to write log file '{self.log_path}': {exc}. "
                f"Training continues."
            )
