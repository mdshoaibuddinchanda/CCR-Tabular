"""CCR-Tabular — single entry point.

Running this file does everything:
  1. Installs all required packages (CPU or CUDA-accelerated PyTorch).
  2. Downloads all 6 benchmark datasets from OpenML.
  3. Runs all experiments (all datasets × all models × all noise configs).

Usage:
    # Full run — everything
    python main.py

    # Quick single run (one dataset, one model)
    python main.py --dataset credit_g --model mlp_ccr

    # With noise
    python main.py --dataset adult --model mlp_ccr --noise_type asym --noise_rate 0.2

    # Fast smoke test
    python main.py --dataset credit_g --model mlp_ccr --n_folds 2 --seeds 42

    # Skip auto-install (if deps already installed)
    python main.py --no_install

    # Skip dataset pre-download (downloads on demand during training)
    python main.py --no_prefetch
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# ── Project root on path ──────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Dependency installation
# ─────────────────────────────────────────────────────────────────────────────

def _pip(*args: str) -> None:
    """Run a pip install command using the current interpreter.

    Args:
        *args: Arguments to pass after 'pip install'.
    """
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", *args],
        stdout=subprocess.DEVNULL,
    )


def _pkg_installed(name: str) -> bool:
    """Check if a package is importable.

    Args:
        name: Module name to try importing.

    Returns:
        True if the package can be imported.
    """
    import importlib
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def _detect_cuda() -> bool:
    """Detect whether a CUDA-capable GPU is available on this machine.

    Tries nvidia-smi first (fast), then falls back to checking torch if
    already installed.

    Returns:
        True if CUDA is available.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if _pkg_installed("torch"):
        import torch
        return torch.cuda.is_available()

    return False


def install_dependencies(requirements_path: Path) -> None:
    """Install all project dependencies, choosing GPU or CPU PyTorch.

    Reads requirements.txt for non-torch packages. Installs PyTorch
    separately with the correct index URL for CUDA or CPU.

    Args:
        requirements_path: Path to requirements.txt.
    """
    print("=" * 60)
    print("PHASE 1 — Installing dependencies")
    print("=" * 60)

    has_cuda = _detect_cuda()
    torch_installed = _pkg_installed("torch")

    # ── Install / upgrade PyTorch ─────────────────────────────────────────────
    if not torch_installed:
        if has_cuda:
            print("  GPU detected — installing PyTorch with CUDA 11.8 support...")
            _pip(
                "torch==2.1.0", "torchvision==0.16.0",
                "--index-url", "https://download.pytorch.org/whl/cu118",
            )
        else:
            print("  No GPU detected — installing CPU-only PyTorch...")
            _pip(
                "torch==2.1.0", "torchvision==0.16.0",
                "--index-url", "https://download.pytorch.org/whl/cpu",
            )
        print("  PyTorch installed.")
    else:
        import torch
        gpu_str = f"CUDA {torch.version.cuda}" if torch.cuda.is_available() else "CPU"
        print(f"  PyTorch {torch.__version__} already installed ({gpu_str}).")

    # ── Install remaining requirements ────────────────────────────────────────
    # Filter out torch lines — already handled above
    if requirements_path.exists():
        lines = requirements_path.read_text().splitlines()
        non_torch = [
            ln.strip() for ln in lines
            if ln.strip()
            and not ln.startswith("#")
            and not ln.lower().startswith("torch")
        ]
        if non_torch:
            print(f"  Installing {len(non_torch)} packages from requirements.txt...")
            _pip(*non_torch)
            print("  All packages installed.")
    else:
        print(f"  WARNING: {requirements_path} not found. Skipping package install.")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Dataset pre-download
# ─────────────────────────────────────────────────────────────────────────────

def prefetch_all_datasets() -> None:
    """Download and cache all 6 benchmark datasets from OpenML.

    Skips datasets that are already cached in data/raw/.
    """
    from src.data.load_data import load_dataset
    from src.utils.config import DATASETS

    print("=" * 60)
    print("PHASE 2 — Downloading datasets")
    print("=" * 60)

    for name in DATASETS:
        print(f"  [{name}] ", end="", flush=True)
        try:
            load_dataset(name)
            print("ready.")
        except Exception as exc:
            print(f"FAILED: {exc}")
            raise RuntimeError(
                f"Could not download dataset '{name}'. "
                f"Check your internet connection and try again."
            ) from exc

    print()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — Run experiments
# ─────────────────────────────────────────────────────────────────────────────

def run_all_experiments(datasets: list[str] | None, models: list[str] | None) -> None:
    """Run all experiments across all configs.

    Args:
        datasets: Optional list of dataset names to restrict to.
        models: Optional list of model names to restrict to.
    """
    import gc
    import torch
    from experiments.run_experiments import load_all_configs, print_summary, update_status_log
    from src.training.cross_validation import run_cross_validation
    from src.utils.logger import setup_logging

    setup_logging()

    print("=" * 60, flush=True)
    print("PHASE 3 — Running experiments", flush=True)
    print("=" * 60, flush=True)

    configs_dir = _ROOT / "experiments" / "configs"
    configs = load_all_configs(configs_dir)

    total, failed = 0, 0

    for config in configs:
        run_datasets = datasets or config.get("datasets", [])
        run_models = models or config.get("models", [])
        noise_type = config.get("noise_type", "none")
        noise_rate = float(config.get("noise_rate", 0.0))
        seeds = config.get("seeds", [42, 123, 2024])
        n_folds = config.get("n_folds", 5)

        for dataset in run_datasets:
            for model in run_models:
                print(f"\n  {dataset} | {model} | {noise_type}@{int(noise_rate*100)}%")
                try:
                    results = run_cross_validation(
                        dataset_name=dataset,
                        model_name=model,
                        noise_type=noise_type,
                        noise_rate=noise_rate,
                        seeds=seeds,
                        n_folds=n_folds,
                    )
                    print_summary(results)
                    update_status_log(dataset, model, config, status="COMPLETED")
                    total += 1
                except KeyboardInterrupt:
                    print("\nInterrupted. Re-run to resume — completed runs are skipped.")
                    sys.exit(0)
                except Exception as exc:
                    import logging
                    logging.getLogger(__name__).error(
                        f"FAILED: {dataset}/{model}: {exc}", exc_info=True
                    )
                    update_status_log(dataset, model, config, status="FAILED", error=str(exc))
                    failed += 1
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

    print(f"\n{'='*60}")
    print(f"All experiments complete: {total} succeeded, {failed} failed.")
    print(f"Results: outputs/metrics/results.csv")
    print(f"{'='*60}\n")


def run_single(args: argparse.Namespace) -> None:
    """Run a single dataset/model combination.

    Args:
        args: Parsed CLI arguments.
    """
    from src.data.load_data import load_dataset
    from src.training.cross_validation import run_cross_validation
    from src.utils.logger import setup_logging

    setup_logging()

    print(f"\nCCR-Tabular — Single Run")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Model:      {args.model}")
    print(f"  Noise:      {args.noise_type} @ {args.noise_rate:.0%}")
    print(f"  Folds:      {args.n_folds}")
    print(f"  Seeds:      {args.seeds or [42, 123, 2024]}\n")

    load_dataset(args.dataset)

    run_cross_validation(
        dataset_name=args.dataset,
        model_name=args.model,
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
        seeds=args.seeds,
        n_folds=args.n_folds,
    )

    print("\nDone. Results saved to outputs/metrics/results.csv")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "CCR-Tabular — single entry point.\n"
            "With no arguments: installs deps, downloads datasets, runs all experiments.\n"
            "With --dataset/--model: runs a single combination."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Run scope ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Run a single dataset instead of all. "
            "Choices: adult, bank, magic, phoneme, credit_g, spambase."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Run a single model instead of all. "
            "Choices: mlp_ccr, mlp_standard, mlp_focal, mlp_weighted_ce, "
            "mlp_smote, xgboost_default, xgboost_weighted, lightgbm_default."
        ),
    )
    parser.add_argument(
        "--noise_type",
        choices=["none", "asym", "feat"],
        default="none",
        help="Noise type for single-run mode.",
    )
    parser.add_argument(
        "--noise_rate",
        type=float,
        default=0.0,
        help="Noise rate for single-run mode (0.0–0.3).",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of CV folds.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Random seeds. Default: 42 123 2024.",
    )

    # ── Control flags ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--no_install",
        action="store_true",
        help="Skip automatic dependency installation.",
    )
    parser.add_argument(
        "--no_prefetch",
        action="store_true",
        help="Skip pre-downloading all datasets (they download on demand).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── Phase 1: Install dependencies ─────────────────────────────────────────
    if not args.no_install:
        install_dependencies(_ROOT / "requirements.txt")

    # ── Phase 2: Pre-download datasets ────────────────────────────────────────
    if not args.no_prefetch:
        prefetch_all_datasets()

    # ── Phase 3: Run ──────────────────────────────────────────────────────────
    single_run = args.dataset is not None or args.model is not None

    if single_run:
        # Validate choices now that imports are available
        from src.utils.config import DATASETS, MODEL_NAMES
        if args.dataset and args.dataset not in DATASETS:
            print(f"ERROR: Unknown dataset '{args.dataset}'. "
                  f"Choices: {list(DATASETS.keys())}")
            sys.exit(1)
        if args.model and args.model not in MODEL_NAMES:
            print(f"ERROR: Unknown model '{args.model}'. "
                  f"Choices: {MODEL_NAMES}")
            sys.exit(1)
        if args.dataset is None:
            args.dataset = "credit_g"
        if args.model is None:
            args.model = "mlp_ccr"
        run_single(args)
    else:
        run_all_experiments(datasets=None, models=None)
