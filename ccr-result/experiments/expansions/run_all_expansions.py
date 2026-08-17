"""
run_all_expansions.py â€” Master Expansion Experiment Runner
===========================================================
Runs all expansion experiments in sequence after the main experiment completes.
Each step is fully resumable â€” completed run_ids are skipped automatically.

Sequence:
    1. Main experiments     (run via main.py first)
    2. Ablation study       (4 CCR variants x all 6 datasets x 7 configs)
    3. Tau sensitivity      (tau in {0.3, 0.5, 0.6, 0.7, 0.8})
    4. K sensitivity        (K in {3, 5, 10})
    5. Beta sensitivity     (beta in {0.3, 0.5, 0.8})
    6. Noise@40% extension  (all 8 models x all 6 datasets)
    7. Learning curves      (extracted from logs, no training)

Total new runs: ~6,210 (on top of the 5,040 main runs)

Usage:
    python run_all_expansions.py                  # run everything
    python run_all_expansions.py --skip-ablation  # skip ablation
    python run_all_expansions.py --skip-tau       # skip tau sensitivity
    python run_all_expansions.py --skip-k         # skip K sensitivity
    python run_all_expansions.py --skip-beta      # skip beta sensitivity
    python run_all_expansions.py --skip-noise40   # skip noise@40%
    python run_all_expansions.py --only-curves    # only extract learning curves
"""

import argparse
import gc
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logging

setup_logging(level=logging.WARNING)
logger = logging.getLogger(__name__)


def run_step(name: str, func, skip: bool = False) -> bool:
    """Run one expansion step with timing and error handling.

    Args:
        name: Display name for this step.
        func: Callable to execute.
        skip: If True, skip this step entirely.

    Returns:
        True if completed successfully, False if skipped or failed.
    """
    if skip:
        print(f"\n  [SKIP] {name}", flush=True)
        return False

    print(f"\n{'='*65}", flush=True)
    print(f"  STARTING: {name}", flush=True)
    print(f"{'='*65}", flush=True)

    t0 = time.perf_counter()
    try:
        func()
        elapsed = time.perf_counter() - t0
        print(f"\n  DONE: {name} in {elapsed / 60:.1f} min", flush=True)
        return True
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.error(f"FAILED: {name}: {exc}", exc_info=True)
        print(f"\n  FAILED: {name} after {elapsed / 60:.1f} min: {exc}", flush=True)
        print("  Continuing to next step...", flush=True)
        return False
    finally:
        gc.collect()


def print_output_summary():
    """Print row counts for all output CSVs."""
    import pandas as pd
    from src.utils.config import OUTPUTS_METRICS

    files = [
        ("results.csv",                  "Main results (incl. noise@40%)"),
        ("results_ablation.csv",         "Ablation study"),
        ("results_tau_sensitivity.csv",  "Tau sensitivity"),
        ("results_k_sensitivity.csv",    "K sensitivity"),
        ("results_beta_sensitivity.csv", "Beta sensitivity"),
        ("learning_curves.csv",          "Learning curves"),
    ]

    print("\n  Output files:", flush=True)
    for fname, label in files:
        fpath = OUTPUTS_METRICS / fname
        if fpath.exists():
            try:
                rows = len(pd.read_csv(fpath))
                print(f"    {label:<35} {rows:>6} rows  ({fname})", flush=True)
            except Exception:
                print(f"    {label:<35} exists  ({fname})", flush=True)
        else:
            print(f"    {label:<35} NOT FOUND", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run all CCR expansion experiments in sequence."
    )
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-tau",      action="store_true")
    parser.add_argument("--skip-k",        action="store_true")
    parser.add_argument("--skip-beta",     action="store_true")
    parser.add_argument("--skip-noise40",  action="store_true")
    parser.add_argument("--only-curves",   action="store_true",
                        help="Only extract learning curves (no training)")
    args = parser.parse_args()

    if args.only_curves:
        args.skip_ablation = True
        args.skip_tau      = True
        args.skip_k        = True
        args.skip_beta     = True
        args.skip_noise40  = True

    print("\n" + "=" * 65, flush=True)
    print("  CCR-Tabular Expansion Experiments", flush=True)
    print("=" * 65, flush=True)

    wall_start = time.perf_counter()

    # Step 1: Ablation
    from experiments.expansions.run_ablation import run_ablation
    run_step("Ablation study (2,520 runs)", run_ablation, skip=args.skip_ablation)

    # Step 2: Tau sensitivity
    from experiments.expansions.run_tau_sensitivity import run_tau_sensitivity
    run_step("Tau sensitivity (1,350 runs)", run_tau_sensitivity, skip=args.skip_tau)

    # Step 3: K sensitivity
    from experiments.expansions.run_k_sensitivity import run_k_sensitivity
    run_step("K sensitivity (810 runs)", run_k_sensitivity, skip=args.skip_k)

    # Step 4: Beta sensitivity
    from experiments.expansions.run_beta_sensitivity import run_beta_sensitivity
    run_step("Beta sensitivity (810 runs)", run_beta_sensitivity, skip=args.skip_beta)

    # Step 5: Noise@40%
    from experiments.expansions.run_noise40 import run_noise40
    run_step("Noise@40% extension (720 runs)", run_noise40, skip=args.skip_noise40)

    # Step 6: Learning curves (always run â€” fast log extraction, no training)
    from experiments.expansions.run_learning_curves import extract_learning_curves
    run_step("Learning curves extraction", extract_learning_curves, skip=False)

    total_elapsed = time.perf_counter() - wall_start
    print(f"\n{'='*65}", flush=True)
    print(f"  ALL STEPS COMPLETE  ({total_elapsed / 3600:.2f} hours total)", flush=True)
    print_output_summary()
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
