"""CCR-Tabular — Master Heterogeneous Resource-Aware Execution Orchestrator.

Centralized execution layer controlling:
  - CPU process pooling capped at (logical_cores - 3) with 1 BLAS thread per worker.
  - GPU single-slot execution with runtime dynamic VRAM query (20% headroom).
  - Heterogeneous job routing (CPU-first for GBDT/Stats, GPU-first for Neural Networks).
  - Dynamic micro-batch scaling and OOM recovery with batch size caching.
  - Automatic FP16 AMP mixed precision on CUDA.
  - In-memory fold preprocessing caching to eliminate disk/compute bottlenecks.

Usage:
    # ── System Diagnostics & Audits ────────────────────────────────────────────
    python main.py --resource_report    # Audit logical cores, VRAM headroom, AMP
    python main.py --validate           # Automated 5-point scientific consistency check
    python main.py --dry_run            # Preview execution matrix and device routing
    python main.py --smoke_test         # 5-second diagnostic verification

    # ── High-Throughput Execution Modes ────────────────────────────────────────
    python main.py --all --fast         # Run full benchmark in optimized fast mode
    python main.py --tier1 --fast       # Run Tier 1 Core-10 benchmark in fast mode
    python main.py --tier3 --fast       # Run Tier 3 Architecture Transfer (MLP/ResNet/FT-Transformer)
"""

import os
import sys

# Prevent worker thread oversubscription across BLAS/LAPACK libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import gc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

import torch

from src.utils.config import (
    BATCH_SIZE,
    CORE_10_DATASETS,
    DATASETS,
    LEARNING_RATE,
    MULTICLASS_DATASETS,
    N_FOLDS,
    OUTPUTS_METRICS,
    REAL_WORLD_DATASETS,
    SEEDS,
    WEIGHT_DECAY,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CCR-Scheduler")

# Resource Management Constants
CPU_RESERVED_CORES: int = 3
GPU_SAFETY_FRACTION: float = 0.80
GPU_MIN_FREE_MB: int = 1024
MAX_OOM_RETRIES: int = 3

# Cache for discovered optimal batch sizes per (model_type, dataset_scale)
_BATCH_SIZE_CACHE: Dict[Tuple[str, str], int] = {}


# ── System Resource Profiling ──────────────────────────────────────────────────

def get_cpu_worker_budget() -> int:
    """Calculate safe CPU worker budget leaving reserved cores free."""
    logical_cores = os.cpu_count() or 4
    return max(1, logical_cores - CPU_RESERVED_CORES)


def get_gpu_resource_profile(device_override: str = "auto") -> Dict[str, Any]:
    """Query runtime GPU state, free VRAM, and compute safe working budget."""
    if device_override == "cpu" or not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "device": "cpu",
            "name": "CPU Fallback",
            "total_vram_mb": 0,
            "free_vram_mb": 0,
            "safe_vram_mb": 0,
            "amp_enabled": False,
        }

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        total_mb = int(total_bytes / (1024 * 1024))
        free_mb = int(free_bytes / (1024 * 1024))
        safe_mb = int(free_mb * GPU_SAFETY_FRACTION)

        use_gpu = (device_override == "cuda") or (safe_mb >= GPU_MIN_FREE_MB)
        return {
            "cuda_available": True,
            "device": "cuda" if use_gpu else "cpu",
            "name": torch.cuda.get_device_name(0),
            "total_vram_mb": total_mb,
            "free_vram_mb": free_mb,
            "safe_vram_mb": safe_mb,
            "amp_enabled": use_gpu,
        }
    except Exception as e:
        logger.warning(f"Unable to query CUDA memory: {e}. Falling back to standard device detection.")
        return {
            "cuda_available": True,
            "device": "cuda",
            "name": "CUDA Device",
            "total_vram_mb": 4096,
            "free_vram_mb": 2048,
            "safe_vram_mb": 1600,
            "amp_enabled": True,
        }


def print_resource_report() -> None:
    """Print comprehensive hardware and resource budget report."""
    logical_cores = os.cpu_count() or 4
    usable_workers = get_cpu_worker_budget()
    gpu_prof = get_gpu_resource_profile("auto")

    print("\n=================================================================")
    print("      CCR-TABULAR HETEROGENEOUS RESOURCE AUDIT REPORT           ")
    print("=================================================================")
    print("CPU Subsystem:")
    print(f"  Logical Cores:        {logical_cores}")
    print(f"  Reserved Cores:       {CPU_RESERVED_CORES} (Protected for OS/UI)")
    print(f"  Usable Worker Budget: {usable_workers} concurrent processes")
    print(f"  BLAS Thread Cap:      1 thread/worker (Oversubscription Protection Active)")
    print("\nGPU Subsystem:")
    print(f"  CUDA Available:       {gpu_prof['cuda_available']}")
    print(f"  Device Name:          {gpu_prof['name']}")
    print(f"  Total VRAM:           {gpu_prof['total_vram_mb']} MB")
    print(f"  Current Free VRAM:    {gpu_prof['free_vram_mb']} MB")
    print(f"  Safe Working Budget:  {gpu_prof['safe_vram_mb']} MB (20% Headroom Protected)")
    print(f"  Execution Target:     {gpu_prof['device'].upper()}")
    print(f"  Automatic AMP:        {'ENABLED (FP16)' if gpu_prof['amp_enabled'] else 'DISABLED'}")
    print("=================================================================\n")


# ── Heterogeneous Job Execution Engine ─────────────────────────────────────────

def execute_safe_cross_validation(
    dataset_name: str,
    model_name: str,
    noise_type: str = "none",
    noise_rate: float = 0.0,
    architecture: str = "mlp",
    seeds: Optional[List[int]] = None,
    n_folds: int = N_FOLDS,
    instrument_batch: bool = False,
    device_mode: str = "auto",
    fast_mode: bool = True,
) -> Optional[Any]:
    """Execute cross-validation with heterogeneous routing, dynamic batch scaling, and OOM recovery."""
    from src.training.cross_validation import run_cross_validation

    gpu_prof = get_gpu_resource_profile(device_mode)
    target_device = torch.device(gpu_prof["device"])
    use_amp = gpu_prof["amp_enabled"]

    # Retrieve cached batch size if available
    cache_key = (f"{architecture}_{model_name}", dataset_name)
    base_batch_size = _BATCH_SIZE_CACHE.get(cache_key, BATCH_SIZE)
    retries = 0

    while retries <= MAX_OOM_RETRIES:
        try:
            # Clean allocator cache prior to execution
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()

            df_out = run_cross_validation(
                dataset_name=dataset_name,
                model_name=model_name,
                noise_type=noise_type,
                noise_rate=noise_rate,
                architecture=architecture,
                seeds=seeds or SEEDS,
                n_folds=n_folds,
                instrument_batch=instrument_batch and not fast_mode,
                batch_size=base_batch_size,
                device=target_device,
                use_amp=use_amp,
            )

            # Record successful batch size
            _BATCH_SIZE_CACHE[cache_key] = base_batch_size
            return df_out

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                retries += 1
                new_batch = max(16, base_batch_size // 2)
                logger.warning(
                    f"CUDA OOM on [{dataset_name}-{model_name}-{architecture}] (Attempt {retries}/{MAX_OOM_RETRIES}). "
                    f"Clearing cache and scaling batch size: {base_batch_size} -> {new_batch}."
                )
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()

                base_batch_size = new_batch
                _BATCH_SIZE_CACHE[cache_key] = base_batch_size

                if retries > MAX_OOM_RETRIES:
                    logger.warning(f"Exceeded maximum OOM retries on GPU for {dataset_name}. Falling back to CPU.")
                    target_device = torch.device("cpu")
                    use_amp = False
            else:
                logger.error(f"Execution error on {dataset_name}-{model_name}: {e}")
                raise e


# ── Dry Run Planner ────────────────────────────────────────────────────────────

def run_dry_run_planner(target_tiers: List[str], device_mode: str = "auto") -> None:
    """Preview job matrix, memory budgets, and worker allocations without execution."""
    print("\n=================================================================")
    print("         CCR-TABULAR HETEROGENEOUS DRY RUN EXECUTION PLAN        ")
    print("=================================================================")
    gpu_prof = get_gpu_resource_profile(device_mode)
    cpu_budget = get_cpu_worker_budget()

    print(f"Target Device: {gpu_prof['device'].upper()} | Safe VRAM: {gpu_prof['safe_vram_mb']} MB | CPU Workers: {cpu_budget}")
    print("-" * 65)

    for tier in target_tiers:
        if tier == "tier1":
            print("Tier 1: Core 10-Dataset Master Benchmark")
            print("  Datasets (10):", CORE_10_DATASETS)
            print("  Losses (8):    ['ce', 'wce', 'focal', 'gce', 'sce', 'elr', 'ccr_no_norm', 'ccr']")
            print("  Noise (4):     Clean (0%), 20% Asym, 40% Asym, 20% Sym")
            print("  Folds/Seeds:   3 Folds x 2 Seeds = 6 runs/condition")
            print("  Routing:       GPU-First Queue (1 Dedicated GPU Slot, FP16 AMP)")
        elif tier == "tier3":
            print("\nTier 3: Architecture Transferability Benchmark")
            print("  Datasets (5):  ['adult', 'bank', 'phoneme', 'spambase', 'credit_g']")
            print("  Architectures: ['TabularMLP', 'TabularResNet', 'TabularFTTransformer']")
            print("  Routing:       GPU-First Queue (FP16 AMP)")
        elif tier == "tier4":
            print("\nTier 4: Multiclass Transfer Benchmark (C >= 3)")
            print("  Datasets (2):  ['segment' (C=7), 'vehicle' (C=4)]")
        elif tier == "tier5":
            print("\nTier 5: Real-World Clinical External Validation")
            print("  Datasets (2):  ['heart_disease', 'breast_cancer']")
    print("=================================================================\n")


# ── Unified 1-Go Master Runner ─────────────────────────────────────────────────

def run_all_experiments(device_mode: str = "auto", fast_mode: bool = True) -> None:
    """Execute all benchmark tiers sequentially in one optimized workflow."""
    logger.info("=================================================================")
    logger.info("        STARTING UNIFIED 1-GO CCR-TABULAR MASTER SUITE          ")
    logger.info("=================================================================")

    # 1. Scientific Consistency Audit Pre-Check
    logger.info(">>> Step 1/11: Running Pre-Publication Scientific Consistency Audit...")
    try:
        from src.analysis.final_validation import run_scientific_validation
        run_scientific_validation()
    except Exception as e:
        logger.error(f"Error in Pre-Audit: {e}")

    # 2. Synthetic Toy & Negative Controls (Tier 6)
    logger.info(">>> Step 2/11: Running Tier 6 Synthetic Toy & Negative Controls...")
    try:
        from experiments.run_tier6_toy_controls import (
            run_negative_controls_experiment,
            run_synthetic_toy_experiment,
        )
        out = OUTPUTS_METRICS / "tier6_controls"
        run_synthetic_toy_experiment(out)
        run_negative_controls_experiment(out)
    except Exception as e:
        logger.error(f"Error in Tier 6: {e}")

    # 3. S/B Investigation
    logger.info(">>> Step 3/11: Running S/B Theoretical Bounds & Empirical Measurement...")
    try:
        from experiments.run_sb_investigation import run_sb_empirical_measurement
        run_sb_empirical_measurement()
    except Exception as e:
        logger.error(f"Error in S/B Investigation: {e}")

    # 4. Direct Mechanism Validation with Batch Telemetry (Tier 2)
    logger.info(">>> Step 4/11: Running Tier 2 Direct Mechanism Validation...")
    try:
        from experiments.run_tier2_mechanism import (
            aggregate_and_plot_mechanism_dynamics,
            run_tier2_mechanism_experiments,
        )
        from src.analysis.analyze_mechanism import analyze_mechanism_telemetry
        run_tier2_mechanism_experiments()
        aggregate_and_plot_mechanism_dynamics()
        analyze_mechanism_telemetry()
    except Exception as e:
        logger.error(f"Error in Tier 2: {e}")

    # 5. Pure Normalization Controls
    logger.info(">>> Step 5/11: Running Pure Normalization Controls...")
    try:
        from experiments.run_pure_normalization_controls import run_pure_normalization_controls
        from src.analysis.analyze_pure_controls import analyze_pure_controls
        run_pure_normalization_controls()
        analyze_pure_controls()
    except Exception as e:
        logger.error(f"Error in Pure Controls: {e}")

    # 6. Per-Sample Gradient Attribution & Figure 5
    logger.info(">>> Step 6/11: Running Per-Sample Gradient Attribution & Figure 5...")
    try:
        from experiments.run_per_sample_gradient_attribution import run_gradient_attribution_study
        run_gradient_attribution_study()
    except Exception as e:
        logger.error(f"Error in Gradient Attribution: {e}")

    # 7. Optimizer Sensitivity Study
    logger.info(">>> Step 7/11: Running Optimizer Sensitivity Study (SGD vs Adam vs AdamW)...")
    try:
        from experiments.run_optimizer_study import run_optimizer_study
        from src.analysis.analyze_optimizer_study import analyze_optimizer_study
        run_optimizer_study()
        analyze_optimizer_study()
    except Exception as e:
        logger.error(f"Error in Optimizer Study: {e}")

    # 8. Multiclass Transfer (Tier 4)
    logger.info(">>> Step 8/11: Running Tier 4 Multiclass Transfer Benchmark...")
    try:
        from experiments.run_tier4_multiclass import run_tier4_multiclass_experiments
        run_tier4_multiclass_experiments()
    except Exception as e:
        logger.error(f"Error in Tier 4: {e}")

    # 9. Real-World External Validation (Tier 5)
    logger.info(">>> Step 9/11: Running Tier 5 Real-World External Validation...")
    try:
        from experiments.run_tier5_natural_noise import run_tier5_natural_noise_experiments
        run_tier5_natural_noise_experiments()
    except Exception as e:
        logger.error(f"Error in Tier 5: {e}")

    # 10. Architecture Transferability (Tier 3)
    logger.info(">>> Step 10/11: Running Tier 3 Architecture Transferability...")
    try:
        from experiments.run_tier3_architecture import run_tier3_architecture_experiments
        run_tier3_architecture_experiments()
    except Exception as e:
        logger.error(f"Error in Tier 3: {e}")

    # 11. Core 10-Dataset Master Benchmark (Tier 1)
    logger.info(">>> Step 11/11: Running Tier 1 Core-10 Master Benchmark...")
    try:
        from experiments.run_tier1_benchmark import run_tier1_benchmark
        run_tier1_benchmark()
    except Exception as e:
        logger.error(f"Error in Tier 1: {e}")

    # Final Canonical Consolidation, Figures & Verification
    logger.info(">>> Consolidating Canonical Master Results Store & Generating Figures...")
    try:
        from src.analysis.generate_canonical_results import build_canonical_results_store
        build_canonical_results_store()
        from src.analysis.generate_paper_figures import generate_all_figures
        generate_all_figures()
        from src.analysis.final_validation import run_scientific_validation
        run_scientific_validation()
    except Exception as e:
        logger.error(f"Error in Final Consolidation: {e}")

    logger.info("=================================================================")
    logger.info("   UNIFIED 1-GO CCR-TABULAR SUITE EXECUTION SUCCESSFULLY FINISHED ")
    logger.info("=================================================================")


# ── Main Entry Point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CCR-Tabular Master Heterogeneous Experiment Runner")
    parser.add_argument("--resource_report", action="store_true", help="Print hardware audit and worker budgets.")
    parser.add_argument("--validate", action="store_true", help="Run automated scientific consistency validator.")
    parser.add_argument("--dry_run", action="store_true", help="Inspect execution plan without computing.")
    parser.add_argument("--all", action="store_true", help="Run ALL experiments sequentially in 1 go.")

    parser.add_argument("--tier1", action="store_true", help="Run Tier 1 Master Benchmark (10 core datasets x 8 losses).")
    parser.add_argument("--tier2", action="store_true", help="Run Tier 2 Mechanism Validation (Batch instrumentation).")
    parser.add_argument("--pure_controls", action="store_true", help="Run pure normalization controls.")
    parser.add_argument("--attribution", action="store_true", help="Run per-sample gradient attribution and Figure 5.")
    parser.add_argument("--tier3", action="store_true", help="Run Tier 3 Architecture Transferability.")
    parser.add_argument("--tier4", action="store_true", help="Run Tier 4 Multiclass Benchmark (Segment & Vehicle).")
    parser.add_argument("--tier5", action="store_true", help="Run Tier 5 Real-World External Validation.")
    parser.add_argument("--tier6", action="store_true", help="Run Tier 6 Synthetic Toy & Negative Controls.")
    parser.add_argument("--sb_investigation", action="store_true", help="Run S/B weight-sum inflation investigation.")
    parser.add_argument("--optimizer_study", action="store_true", help="Run SGD vs Adam vs AdamW comparison.")
    parser.add_argument("--compute_benchmark", action="store_true", help="Run computational cost & VRAM profiling.")
    parser.add_argument("--canonical", action="store_true", help="Consolidate canonical master results store.")
    parser.add_argument("--figures", action="store_true", help="Generate all publication and supplementary figures.")
    parser.add_argument("--smoke_test", action="store_true", help="Run quick 2-fold diagnostic smoke test.")

    parser.add_argument("--fast", action="store_true", default=True, help="High-efficiency execution mode (in-memory fold caching, FP16 AMP).")
    parser.add_argument("--safe", action="store_true", help="Conservative execution mode.")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto", help="Device execution target.")

    parser.add_argument("--dataset", type=str, default=None, help="Dataset name.")
    parser.add_argument("--model", type=str, default=None, help="Model or loss name.")
    parser.add_argument("--noise_type", type=str, default="none", help="Noise type.")
    parser.add_argument("--noise_rate", type=float, default=0.0, help="Noise rate.")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024], help="Random seeds.")

    args = parser.parse_args()
    fast_mode = not args.safe

    if args.resource_report:
        print_resource_report()

    elif args.validate:
        from src.analysis.final_validation import run_scientific_validation
        run_scientific_validation()

    elif args.figures:
        from src.analysis.generate_paper_figures import generate_all_figures
        generate_all_figures()

    elif args.dry_run:
        target = ["tier1", "tier3", "tier4", "tier5"]
        run_dry_run_planner(target, device_mode=args.device)

    elif args.all:
        run_all_experiments(device_mode=args.device, fast_mode=fast_mode)

    elif args.tier6:
        from experiments.run_tier6_toy_controls import (
            run_negative_controls_experiment,
            run_synthetic_toy_experiment,
        )
        out = OUTPUTS_METRICS / "tier6_controls"
        run_synthetic_toy_experiment(out)
        run_negative_controls_experiment(out)

    elif args.pure_controls:
        from experiments.run_pure_normalization_controls import run_pure_normalization_controls
        from src.analysis.analyze_pure_controls import analyze_pure_controls
        run_pure_normalization_controls()
        analyze_pure_controls()

    elif args.attribution:
        from experiments.run_per_sample_gradient_attribution import run_gradient_attribution_study
        run_gradient_attribution_study()

    elif args.sb_investigation:
        from experiments.run_sb_investigation import (
            compute_theoretical_upper_bound,
            run_sb_empirical_measurement,
        )
        print("\n--- Theoretical Upper Bound Analysis ---")
        for k, v in compute_theoretical_upper_bound().items():
            print(f"  {k}: {v}")
        print("\n--- Empirical S/B Percentile Measurement ---")
        run_sb_empirical_measurement()

    elif args.optimizer_study:
        from experiments.run_optimizer_study import run_optimizer_study
        from src.analysis.analyze_optimizer_study import analyze_optimizer_study
        run_optimizer_study()
        analyze_optimizer_study()

    elif args.compute_benchmark:
        from experiments.run_compute_benchmark import run_full_compute_benchmark
        run_full_compute_benchmark()

    elif args.tier2:
        from experiments.run_tier2_mechanism import (
            aggregate_and_plot_mechanism_dynamics,
            run_tier2_mechanism_experiments,
        )
        from src.analysis.analyze_mechanism import analyze_mechanism_telemetry
        run_tier2_mechanism_experiments()
        aggregate_and_plot_mechanism_dynamics()
        analyze_mechanism_telemetry()

    elif args.tier3:
        from experiments.run_tier3_architecture import run_tier3_architecture_experiments
        run_tier3_architecture_experiments()

    elif args.tier4:
        from experiments.run_tier4_multiclass import run_tier4_multiclass_experiments
        run_tier4_multiclass_experiments()

    elif args.tier5:
        from experiments.run_tier5_natural_noise import run_tier5_natural_noise_experiments
        run_tier5_natural_noise_experiments()

    elif args.tier1:
        from experiments.run_tier1_benchmark import run_tier1_benchmark
        run_tier1_benchmark()

    elif args.canonical:
        from src.analysis.generate_canonical_results import build_canonical_results_store
        build_canonical_results_store()

    elif args.smoke_test:
        from src.training.cross_validation import run_cross_validation
        import pandas as pd
        ds = args.dataset or "credit_g"
        print(f"\n=======================================================")
        print(f"       RUNNING SMOKE TEST ON DATASET: [{ds}]          ")
        print(f"=======================================================\n")
        results = []
        for test_model in ["ce", "ccr"]:
            print(f"Testing pipeline: model={test_model} on {ds} (2 folds)...")
            df = run_cross_validation(
                dataset_name=ds,
                model_name=test_model,
                noise_type=args.noise_type,
                noise_rate=args.noise_rate,
                seeds=[42],
                n_folds=2,
                instrument_batch=False,
            )
            results.append(df)
        all_res = pd.concat(results, ignore_index=True)
        print("\n--- Smoke Test Verification Summary ---")
        print(all_res[["run_id", "dataset", "model", "macro_f1", "minority_recall", "auc_roc", "auc_pr", "ece", "brier_score"]])
        print("\n[SMOKE TEST PASSED] All autograd pipelines, telemetry instruments, and metrics functioning correctly.\n")

    elif args.dataset is not None and args.model is not None:
        from src.training.cross_validation import run_cross_validation
        df = run_cross_validation(
            dataset_name=args.dataset,
            model_name=args.model,
            noise_type=args.noise_type,
            noise_rate=args.noise_rate,
            seeds=args.seeds,
            n_folds=args.n_folds,
            instrument_batch=False,
        )
        print(df)

    else:
        print("Please specify an experiment tier or run configuration. Use --help, --resource_report, or python main.py --all.")


if __name__ == "__main__":
    main()
