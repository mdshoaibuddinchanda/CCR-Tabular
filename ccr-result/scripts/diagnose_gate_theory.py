"""
diagnose_gate_theory.py
Produces the exact two tables requested by the reviewer to prove the full hypothesis.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.load_data import load_dataset
from src.data.noise_injection import inject_asymmetric_noise
from src.data.preprocess import preprocess_split
from src.models.mlp import get_mlp_for_dataset
from src.utils.reproducibility import fix_all_seeds, get_device

def main():
    dataset_name = "adult"
    noise_rate = 0.30
    seed = 42
    epochs = 30
    K = 5
    batch_size = 512

    fix_all_seeds(seed)
    device = get_device()

    df_data = load_dataset(dataset_name)
    feature_cols = [c for c in df_data.columns if c != "target"]
    X = df_data[feature_cols]
    y = df_data["target"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, test_idx = next(skf.split(X, y))

    X_tr_df = X.iloc[train_idx].reset_index(drop=True)
    y_tr_raw = y[train_idx]
    
    (X_tr_np, _, _, y_tr_clean, _, _, _) = preprocess_split(
        X_tr_df, X_tr_df, X_tr_df, pd.Series(y_tr_raw), pd.Series(y_tr_raw), pd.Series(y_tr_raw)
    )

    y_tr_noisy, _ = inject_asymmetric_noise(y_tr_clean, noise_rate, seed)
    is_noisy = (y_tr_clean != y_tr_noisy)

    model = get_mlp_for_dataset(dataset_name, X_tr_np.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    X_tr_t = torch.tensor(X_tr_np, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr_noisy, dtype=torch.long).to(device)
    
    history = torch.zeros((len(X_tr_t), K), dtype=torch.float32, device=device)
    epoch_variances = np.zeros((len(X_tr_t), epochs))
    epoch_confidences = np.zeros((len(X_tr_t), epochs))

    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(len(X_tr_t))
        for i in range(0, len(X_tr_t), batch_size):
            indices = permutation[i:i+batch_size]
            batch_X, batch_y = X_tr_t[indices], y_tr_t[indices]
            
            optimizer.zero_grad()
            logits = model(batch_X)
            probs = F.softmax(logits, dim=1)
            p_true = probs[torch.arange(len(batch_y)), batch_y]
            
            history[indices, epoch % K] = p_true.detach()
            n_t = min(epoch + 1, K)
            
            if n_t > 1:
                var = torch.var(history[indices, :n_t], dim=1, unbiased=True)
            else:
                var = torch.zeros_like(p_true)
            
            epoch_variances[indices.cpu().numpy(), epoch] = var.cpu().numpy()
            epoch_confidences[indices.cpu().numpy(), epoch] = p_true.detach().cpu().numpy()
            
            loss = F.cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()

    # Get metrics over last 10 epochs
    mean_variance_per_sample = np.mean(epoch_variances[:, -10:], axis=1)
    mean_confidence_per_sample = np.mean(epoch_confidences[:, -10:], axis=1)
    
    clean_conf = np.mean(mean_confidence_per_sample[~is_noisy])
    noisy_conf = np.mean(mean_confidence_per_sample[is_noisy])
    clean_var = np.mean(mean_variance_per_sample[~is_noisy])
    noisy_var = np.mean(mean_variance_per_sample[is_noisy])
    
    print("\n| Sample Type | Mean Confidence | Mean Variance |")
    print("| ----------- | --------------: | ------------: |")
    print(f"| Clean       |         {clean_conf:.5f} |       {clean_var:.5f} |")
    print(f"| Noisy       |         {noisy_conf:.5f} |       {noisy_var:.5f} |")

    print("\n| tau | Gate Activation |")
    print("| --- | --------------: |")
    for tau in [0.3, 0.5, 0.7, 0.9]:
        gate_active = np.mean(mean_confidence_per_sample > tau) * 100
        print(f"| {tau:.1f} |          {gate_active:.1f}% |")

if __name__ == "__main__":
    main()
