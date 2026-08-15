"""CCR-Tabular — Master Execution Entry Point.

Usage:
    # 1. Direct Mechanism Validation (Tier 2 - Section B)
    python main.py --tier2

    # 2. Pure Normalization Controls (Uniform, Static, Plain Dynamic, CCR)
    python main.py --pure_controls

    # 3. S/B Distribution Empirical Investigation (Section C)
    python main.py --sb_investigation

    # 4. Optimizer Comparison (SGD vs Adam vs AdamW - Section D)
    python main.py --optimizer_study

    # 5. Computational Overhead & VRAM Benchmark (Section S)
    python main.py --compute_benchmark

    # 6. Synthetic Toy Verification & Negative Controls (Tier 6 - Sections T & U)
    python main.py --tier6

    # 7. Architecture Transferability (Tier 3 - Section J)
    python main.py --tier3

    # 8. Multiclass Benchmark (Tier 4 - Section R)
    python main.py --tier4

    # 9. Real-World External Validation (Tier 5 - Section G)
    python main.py --tier5

    # 10. Full 10-Dataset Master Benchmark (Tier 1 - Section H & I)
    python main.py --tier1

    # 11. Fast Smoke Test (Single dataset, 2 folds, 1 seed)
    python main.py --smoke_test
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="CCR-Tabular Master Experiment Runner")
    parser.add_argument("--tier1", action="store_true", help="Run Tier 1 Master Benchmark (10 core datasets x 10 losses).")
    parser.add_argument("--tier2", action="store_true", help="Run Tier 2 Mechanism Validation (Batch instrumentation).")
    parser.add_argument("--pure_controls", action="store_true", help="Run pure normalization controls (uniform, static, dynamic, CCR).")
    parser.add_argument("--tier3", action="store_true", help="Run Tier 3 Architecture Transferability (MLP vs ResNet vs FT-Transformer).")
    parser.add_argument("--tier4", action="store_true", help="Run Tier 4 Multiclass Benchmark.")
    parser.add_argument("--tier5", action="store_true", help="Run Tier 5 Real-World External Validation.")
    parser.add_argument("--tier6", action="store_true", help="Run Tier 6 Synthetic Toy & Negative Controls.")
    parser.add_argument("--sb_investigation", action="store_true", help="Run S/B weight-sum inflation investigation.")
    parser.add_argument("--optimizer_study", action="store_true", help="Run SGD vs Adam vs AdamW optimizer comparison.")
    parser.add_argument("--compute_benchmark", action="store_true", help="Run computational cost & VRAM profiling.")
    parser.add_argument("--smoke_test", action="store_true", help="Run quick 2-fold smoke test.")

    parser.add_argument("--dataset", type=str, default=None, help="Dataset name.")
    parser.add_argument("--model", type=str, default=None, help="Model or loss name.")
    parser.add_argument("--noise_type", type=str, default="none", help="Noise type: none, asym, sym, feat, idn.")
    parser.add_argument("--noise_rate", type=float, default=0.0, help="Noise rate.")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024], help="Random seeds.")

    args = parser.parse_args()

    if args.tier6:
        from experiments.run_tier6_toy_controls import (
            run_negative_controls_experiment,
            run_synthetic_toy_experiment,
        )
        from src.utils.config import OUTPUTS_METRICS
        out = OUTPUTS_METRICS / "tier6_controls"
        run_synthetic_toy_experiment(out)
        run_negative_controls_experiment(out)

    elif args.pure_controls:
        from experiments.run_pure_normalization_controls import (
            run_pure_normalization_controls,
        )
        run_pure_normalization_controls()

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
        run_optimizer_study()

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

    elif args.smoke_test:
        from src.training.cross_validation import run_cross_validation
        ds = args.dataset or "credit_g"
        model = args.model or "ccr"
        print(f"Running Smoke Test on {ds} with model={model}...")
        df = run_cross_validation(
            dataset_name=ds,
            model_name=model,
            noise_type=args.noise_type,
            noise_rate=args.noise_rate,
            seeds=[42],
            n_folds=2,
            instrument_batch=True,
        )
        print("\nSmoke Test Results:")
        print(df[["run_id", "macro_f1", "minority_recall", "auc_roc", "auc_pr", "ece", "brier_score"]])

    elif args.dataset is not None and args.model is not None:
        from src.training.cross_validation import run_cross_validation
        df = run_cross_validation(
            dataset_name=args.dataset,
            model_name=args.model,
            noise_type=args.noise_type,
            noise_rate=args.noise_rate,
            seeds=args.seeds,
            n_folds=args.n_folds,
            instrument_batch=True,
        )
        print(df)

    else:
        print("Please specify an experiment tier or run configuration. Use --help for options.")


if __name__ == "__main__":
    main()
