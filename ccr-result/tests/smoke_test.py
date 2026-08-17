import os
import sys
import yaml
import traceback
from pathlib import Path
from src.utils.config import DATASETS
from src.training.cross_validation import run_cross_validation

def main():
    print("=" * 60)
    print("SMOKE TEST: 20 Datasets, 1 config, 1 seed, 1 fold, 2 models")
    print("Models: xgboost_default, ft_transformer")
    print("=" * 60)

    config_path = "experiments/configs/clean_run.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    models = ["xgboost_default", "ft_transformer"]
    seeds = [42]
    n_folds = 2
    
    success_count = 0
    fail_count = 0
    failures = []

    for dataset in DATASETS:
        for model in models:
            print(f"\n[SMOKE] Testing {dataset} | {model}...")
            try:
                run_cross_validation(
                    dataset_name=dataset,
                    model_name=model,
                    noise_type="none",
                    noise_rate=0.0,
                    seeds=seeds,
                    n_folds=n_folds
                )
                print(f"[SMOKE] ---> SUCCESS")
                success_count += 1
            except Exception as e:
                print(f"[SMOKE] ---> FAILED: {e}")
                traceback.print_exc()
                failures.append(f"{dataset} | {model}")
                fail_count += 1

    print("\n" + "=" * 60)
    print(f"SMOKE TEST COMPLETE: {success_count} succeeded, {fail_count} failed")
    if fail_count > 0:
        print("Failures:")
        for f in failures:
            print(f"  - {f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
