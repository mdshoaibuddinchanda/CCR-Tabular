"""
diagnose_confidence_distribution.py
Generates the exact confidence distribution table requested by the reviewer
to check the overlap of clean vs noisy samples.
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
            
            epoch_confidences[indices.cpu().numpy(), epoch] = p_true.detach().cpu().numpy()
            
            loss = F.cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()

    # Get mean confidence over last 10 epochs
    mean_conf = np.mean(epoch_confidences[:, -10:], axis=1)
    
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    print("\n| Confidence Range | Total Samples | Clean Count | Noisy Count | % Noisy in Bin |")
    print("| ---------------- | ------------: | ----------: | ----------: | -------------: |")
    
    for i in range(len(bins)-1):
        low = bins[i]
        high = bins[i+1]
        
        # inclusive of upper bound for the last bin
        if i == len(bins) - 2:
            mask = (mean_conf >= low) & (mean_conf <= high)
        else:
            mask = (mean_conf >= low) & (mean_conf < high)
            
        total_in_bin = np.sum(mask)
        if total_in_bin == 0:
            print(f"| {low:.1f}-{high:.1f}          |             0 |           0 |           0 |           0.0% |")
            continue
            
        noisy_in_bin = np.sum(mask & is_noisy)
        clean_in_bin = np.sum(mask & ~is_noisy)
        
        pct_noisy = (noisy_in_bin / total_in_bin) * 100
        
        print(f"| {low:.1f}-{high:.1f}          | {total_in_bin:13d} | {clean_in_bin:11d} | {noisy_in_bin:11d} |          {pct_noisy:5.1f}% |")

    # Also show the percentage of the ENTIRE noisy population that falls into each bin
    total_noisy = np.sum(is_noisy)
    total_clean = np.sum(~is_noisy)
    
    print("\n| Confidence Range | % of all Clean Samples | % of all Noisy Samples |")
    print("| ---------------- | ---------------------: | ---------------------: |")
    for i in range(len(bins)-1):
        low = bins[i]
        high = bins[i+1]
        if i == len(bins) - 2:
            mask = (mean_conf >= low) & (mean_conf <= high)
        else:
            mask = (mean_conf >= low) & (mean_conf < high)
            
        noisy_in_bin = np.sum(mask & is_noisy)
        clean_in_bin = np.sum(mask & ~is_noisy)
        
        pct_of_clean = (clean_in_bin / total_clean) * 100
        pct_of_noisy = (noisy_in_bin / total_noisy) * 100
        
        print(f"| {low:.1f}-{high:.1f}          |                  {pct_of_clean:5.1f}% |                  {pct_of_noisy:5.1f}% |")

if __name__ == "__main__":
    main()
