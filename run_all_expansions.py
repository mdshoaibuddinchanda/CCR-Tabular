"""
run_all_expansions.py — Master Expansion Experiment Runner
===========================================================
Runs ALL expansion experiments in sequence:
    1. Tau sensitivity (1,350 runs)
    2. K sensitivity (540 runs)
    3. Beta sensitivity (540 runs)
    4. Noise@40% extension (720 runs, appended to results.csv)
    5. Learning curves extraction (from existing logs, no training)

Each sub-experiment is resumable (skips completed run_ids).
Progress is printed after each sub-experiment completes.

Usage:
    python run_all_expansions.py
    python run_all_expansions.py --skip-tau      # skip tau sensitivity
    python run_all_expansions.py --skip-k        # skip K sensitivity
    python run_all_expansions.py --skip-beta     # skip beta sensitivity
    python run_all_expansions.py --skip-noise40  # skip noise@40%
    python run_all_expansions.py --only-curves   # only extract learning curves
"""

import argparse
import gc
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import setup_logging

setup_logging(level=logging.WARNING)
logger = logging.getLogger(__name__)


def run_step(name: str, func, skip: bool = False):
    """Run a single expansion step with timing and error handling."""
    if skip:
        print(f"\n{'='*65}", flush=True)
        print(f"  SKIPPING: {name}", flush=True)
        print(f"{'='*65}", flush=True)
        return

    print(f"\n{'='*65}", flush=True)
    print(f"  STARTING: {name}", flush=True)
    print(f"{'='*65}", flush=True)

    t0 = time.perf_counter()
    try:
        func()
        elapsed = time.perf_counter() - t0
        print(f"\n  ✓ COMPLETED: {name} in {elapsed/60:.1f} min", flush=True)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.error(f"FAILED: {name}: {exc}", exc_info=True)
        print(f"\n  ✗ FAILED: {name} after {elapsed/60:.1f} min: {exc}", flush=True)
        print(f"  Continuing with next step...", flush=True)

    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Run all CCR expansion experiments")
    parser.add_argument("--skip-tau",     action="store_true", help="Skip tau sensitivity")
    parser.add_argument("--skip-k",       action="store_true", help="Skip K sensitivity")
    parser.add_argument("--skip-beta",    action="store_true", help="Skip beta sensitivity")
    parser.add_argument("--skip-noise40", action="store_true", help="Skip noise@40% extension")
    parser.add_argument("--only-curves",  action="store_true", help="Only extract learning curves")
    args = parser.parse_args()

    if args.only_curves:
        args.skip_tau = True
        args.skip_k = True
        args.skip_beta = True
        args.skip_noise40 = True

    print("\n" + "=" * 65, flush=True)
    print("  CCR-Tabular Expansion Experiments — Master Runner", flush=True)
    print("=" * 65, flush=True)
    print(f"  Steps to run:", flush=True)
    print(f"    1. Tau sensitivity:   {'SKIP' if args.skip_tau else 'RUN'}", flush=True)
    print(f"    2. K sensitivity:     {'SKIP' if args.skip_k else 'RUN'}", flush=True)
    print(f"    3. Beta sensitivity:  {'SKIP' if args.skip_beta else 'RUN'}", flush=True)
    print(f"    4. Noise@40%:         {'SKIP' if args.skip_noise40 else 'RUN'}", flush=True)
    print(f"    5. Learning curves:   RUN (log extraction, no training)", flush=True)
    print("=" * 65, flush=True)

    wall_start = time.perf_counter()

    # ── Step 1: Tau sensitivity ───────────────────────────────────────────────
    if not args.skip_tau:
        from run_tau_sensitivity import run_tau_sensitivity
        run_step("Tau Sensitivity (1,350 runs)", run_tau_sensitivity, skip=False)
    else:
        run_step("Tau Sensitivity", None, skip=True)

    # ── Step 2: K sensitivity ─────────────────────────────────────────────────
    if not args.skip_k:
        from run_k_sensitivity import run_k_sensitivity
        run_step("K Sensitivity (540 runs)", run_k_sensitivity, skip=False)
    else:
        run_step("K Sensitivity", None, skip=True)

    # ── Step 3: Beta sensitivity ──────────────────────────────────────────────
    if not args.skip_beta:
        from run_beta_sensitivity import run_beta_sensitivity
        run_step("Beta Sensitivity (540 runs)", run_beta_sensitivity, skip=False)
    else:
        run_step("Beta Sensitivity", None, skip=True)

    # ── Step 4: Noise@40% ─────────────────────────────────────────────────────
    if not args.skip_noise40:
        from run_noise40 import run_noise40
        run_step("Noise@40% Extension (720 runs)", run_noise40, skip=False)
    else:
        run_step("Noise@40% Extension", None, skip=True)

    # ── Step 5: Learning curves (always run — fast log extraction) ────────────
    from run_learning_curves import extract_learning_curves
    run_step("Learning Curves Extraction (from logs)", extract_learning_curves, skip=False)

    # ── Final summary ─────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - wall_start

    print("\n" + "=" * 65, flush=True)
    print(f"  ALL EXPANSION EXPERIMENTS COMPLETE", flush=True)
    print(f"  Total wall time: {total_elapsed/3600:.2f} hours", flush=True)
    print("=" * 65, flush=True)

    # Print output file summary
    from src.utils.config import OUTPUTS_METRICS
    import pandas as pd

    output_files = [
        ("results_tau_sensitivity.csv", "Tau sensitivity"),
        ("results_k_sensitivity.csv",   "K sensitivity"),
        ("results_beta_sensitivity.csv", "Beta sensitivity"),
        ("results.csv",                  "Main results (incl. noise@40%)"),
        ("learning_curves.csv",          "Learning curves"),
    ]

    print("\n  Output files:", flush=True)
    for fname, label in output_files:
        fpath = OUTPUTS_METRICS / fname
        if fpath.exists():
            try:
                df = pd.read_csv(fpath)
                print(f"    {label}: {len(df)} rows → {fpath.name}", flush=True)
            except Exception:
                print(f"    {label}: exists → {fpath.name}", flush=True)
        else:
            print(f"    {label}: NOT FOUND → {fpath.name}", flush=True)

    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
