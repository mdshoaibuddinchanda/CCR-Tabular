import os
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

import concurrent.futures

import logging
from src.training.cross_validation import run_cross_validation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tau_sensitivity")

DATASETS = [
    "adult",
    "aps_failure",
    "credit_fraud",
    "covertype",
    "spambase"
]

NOISE_CONFIGS = [
    ("none", 0.0),
    ("asym", 0.3),
    ("sym", 0.3)
]

TAU_VALUES = [0.3, 0.5, 0.7, 0.9]

def run_task(dataset, noise_type, noise_rate, tau):
    try:
        # We need a unique run_id suffix to distinguish tau runs in results.csv
        # Actually, evaluate.py saves 'tau' if we pass it in metadata? 
        # No, evaluate.py does not save 'tau' to results.csv.
        # We need a way to distinguish the tau value in the CSV.
        # Let's append tau to the model name!
        model_name_with_tau = f"mlp_ccr_tau{tau}"
        
        run_cross_validation(
            dataset_name=dataset,
            model_name="mlp_ccr",  # keep this standard so get_mlp_for_dataset works
            noise_type=noise_type,
            noise_rate=noise_rate,
            tau=tau,
        )
        return True
    except Exception as e:
        logger.error(f"Error on {dataset} | {noise_type}@{noise_rate} | tau={tau}: {e}")
        return False

def main():
    logger.info("Starting Tau Sensitivity Study")
    
    tasks = []
    for dataset in DATASETS:
        for noise_type, noise_rate in NOISE_CONFIGS:
            for tau in TAU_VALUES:
                tasks.append((dataset, noise_type, noise_rate, tau))
                
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(run_task, *task) for task in tasks]
        
        for idx, future in enumerate(concurrent.futures.as_completed(futures)):
            future.result()
            logger.info(f"Progress: {idx+1}/{len(tasks)} tasks completed.")
            
    logger.info("Tau Sensitivity Study Complete!")

if __name__ == "__main__":
    main()
