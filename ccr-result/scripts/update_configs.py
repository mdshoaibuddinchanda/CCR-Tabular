import yaml
from pathlib import Path

configs_dir = Path(r"c:\DR2\CCR-Tabular\CCR-Tabular\experiments\configs")
configs_dir.mkdir(parents=True, exist_ok=True)

seeds = [42, 123, 2024, 1, 2, 3, 4, 5, 6, 7]
datasets = ["adult", "bank", "magic", "phoneme", "credit_g", "spambase"]
models = ["mlp_standard", "mlp_focal", "mlp_weighted_ce", "mlp_smote", "xgboost_default", "xgboost_weighted", "lightgbm_default", "tabnet", "ft_transformer", "mlp_ccr"]

def write_config(name, noise_type, noise_rate):
    config = {
        "datasets": datasets,
        "models": models,
        "noise_type": noise_type,
        "noise_rate": noise_rate,
        "n_folds": 5,
        "seeds": seeds
    }
    with open(configs_dir / f"{name}.yaml", "w") as f:
        yaml.dump(config, f, sort_keys=False)

write_config("clean_run", "none", 0.0)

for rate in [0.10, 0.20, 0.30, 0.40]:
    rate_str = int(rate * 100)
    write_config(f"noisy_sym_{rate_str}", "sym", rate)
    write_config(f"noisy_asym_{rate_str}", "asym", rate)
    write_config(f"noisy_feat_{rate_str}", "feat", rate)

print("Configs generated.")
