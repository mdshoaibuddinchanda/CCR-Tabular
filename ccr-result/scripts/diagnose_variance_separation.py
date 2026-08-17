"""
diagnose_variance_separation.py

This script explicitly tests the core hypothesis: 
Does the 'variance of confidence' actually separate clean samples from noisy samples?

It runs on Adult with 30% asymmetric noise, tracks the variance of every sample 
during training, and compares the average variance of cleanly labeled samples 
versus corrupted (flipped) samples.
"""

import logging
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

    print(f"Loading {dataset_name}...")
    df_data = load_dataset(dataset_name)
    feature_cols = [c for c in df_data.columns if c != "target"]
    X = df_data[feature_cols]
    y = df_data["target"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, test_idx = next(skf.split(X, y))

    X_tr_df = X.iloc[train_idx].reset_index(drop=True)
    y_tr_raw = y[train_idx]
    
    # Preprocess
    (X_tr_np, _, _, y_tr_clean, _, _, _) = preprocess_split(
        X_tr_df, X_tr_df, X_tr_df, pd.Series(y_tr_raw), pd.Series(y_tr_raw), pd.Series(y_tr_raw)
    )

    # Inject noise and track WHICH indices were flipped
    y_tr_noisy, _ = inject_asymmetric_noise(y_tr_clean, noise_rate, seed)
    
    is_noisy = (y_tr_clean != y_tr_noisy)
    num_noisy = np.sum(is_noisy)
    print(f"Injected 30% Asymmetric Noise. Corrupted {num_noisy} minority samples.")

    model = get_mlp_for_dataset(dataset_name, X_tr_np.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    X_tr_t = torch.tensor(X_tr_np, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr_noisy, dtype=torch.long).to(device)
    
    history = torch.zeros((len(X_tr_t), K), dtype=torch.float32, device=device)
    
    # To track the final variance assigned to each sample across the last 10 epochs
    epoch_variances = np.zeros((len(X_tr_t), epochs))

    print(f"Training for {epochs} epochs to track variance...")
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
            
            # Save variance for analysis
            epoch_variances[indices.cpu().numpy(), epoch] = var.cpu().numpy()
            
            # Dummy loss just to train the network
            loss = F.cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()

    print("\n=== VARIANCE ANALYSIS RESULTS ===")
    
    # Calculate average variance assigned to each sample during the last 10 epochs
    mean_variance_per_sample = np.mean(epoch_variances[:, -10:], axis=1)
    
    clean_variance = np.mean(mean_variance_per_sample[~is_noisy])
    noisy_variance = np.mean(mean_variance_per_sample[is_noisy])
    
    print(f"Average variance of CLEAN samples:   {clean_variance:.5f}")
    print(f"Average variance of NOISY samples:   {noisy_variance:.5f}")
    
    if noisy_variance > clean_variance:
        ratio = noisy_variance / clean_variance
        print(f"\nVerdict: The variance gate works! Noisy samples have {ratio:.2f}x higher variance than clean samples.")
    else:
        print("\nVerdict: The gate is doomed. Variance does not separate noisy samples from clean samples.")

if __name__ == "__main__":
    main()
