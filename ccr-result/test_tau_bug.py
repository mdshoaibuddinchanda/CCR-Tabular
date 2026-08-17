import traceback
from src.training.cross_validation import run_cross_validation

try:
    run_cross_validation(
        dataset_name="spambase",
        model_name="mlp_ccr",
        noise_type="sym",
        noise_rate=0.3,
        tau=0.5,
    )
except Exception as e:
    print(traceback.format_exc())
