"""Reproducibility utilities — fix all sources of randomness.

Must be called once at the start of every training run before any model
initialization or data loading.
"""

import os
import random

import numpy as np
import torch


def fix_all_seeds(seed: int) -> None:
    """Fix seeds for Python, NumPy, and PyTorch (CPU + GPU).

    Args:
        seed: Integer seed. Use values from config.SEEDS.

    Note:
        Must be called before any model initialization or data loading.
        Also sets PYTHONHASHSEED environment variable.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the best available device: cuda > mps > cpu.

    Returns:
        torch.device object. Never hardcodes 'cuda'.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
