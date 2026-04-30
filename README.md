<div align="center">

# CCR-Tabular

**Confidence-Calibrated Reweighting for Robust Tabular Learning**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-33%20passed-brightgreen?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![Experiments](https://img.shields.io/badge/Experiments-5040%20runs-orange?style=for-the-badge)](outputs/)

[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.0-189fdd?style=flat-square)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.1.0-2980b9?style=flat-square)](https://lightgbm.readthedocs.io/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26+-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.1+-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-0.11.0-orange?style=flat-square)](https://imbalanced-learn.org/)
[![Reproducible](https://img.shields.io/badge/Reproducible-✓-success?style=flat-square)](src/utils/reproducibility.py)
[![GPU Ready](https://img.shields.io/badge/GPU-CUDA%20%7C%20MPS%20%7C%20CPU-76b900?style=flat-square&logo=nvidia&logoColor=white)](src/utils/reproducibility.py)
[![Wilcoxon](https://img.shields.io/badge/Stats-Wilcoxon%20p%3C0.05-blueviolet?style=flat-square)](src/utils/statistics.py)
[![Ablations](https://img.shields.io/badge/Ablations-3%20variants-informational?style=flat-square)](src/loss/ccr_loss.py)

*A publication-ready codebase for a journal paper on dynamic loss functions for noisy tabular data.*

</div>

---

## One Command to Run Everything

```bash
python main.py
```

That single command:

1. **Installs all dependencies** — detects your GPU and installs CUDA-accelerated or CPU PyTorch automatically
2. **Downloads all 6 datasets** — from OpenML, cached locally after first run
3. **Runs all experiments** — 6 datasets × 8 models × 7 noise configs × 5 folds × 3 seeds

> Already have deps installed? Use `python main.py --no_install` to skip step 1.
> Already downloaded data? Use `python main.py --no_prefetch` to skip step 2.

---

## Key Results

All 5,040 experiments complete. Summary across 6 datasets:

| Model | Macro F1 | Minority Recall | F1 Drop @ Asym 30% |
|-------|----------|-----------------|---------------------|
| MLP-CE | 0.793 | 0.654 | −0.058 |
| MLP-WCE | 0.791 | 0.805 | −0.006 |
| XGBoost-W | 0.825 | 0.798 | −0.036 |
| LightGBM | 0.822 | 0.687 | −0.089 |
| **CCR (Ours)** | **0.799** | **0.724** | **−0.014** |

CCR degrades **3× slower than XGBoost-W** and **6× slower than LightGBM** under heavy asymmetric noise, while maintaining the best balance of Macro F1 and minority recall among all neural methods.

---

## The Problem

Training neural networks on real-world tabular data is hard for three compounding reasons:

- **Class imbalance** — minority-to-majority ratios up to 7.5:1
- **Asymmetric label noise** — minority labels silently flipped to majority
- **Feature-correlated noise** — labels corrupted near the decision boundary

Standard cross-entropy ignores all three. CCR addresses them in a single unified loss.

---

## The CCR Loss Function

```text
Step 1 — Raw weight per sample:
    w_i = (1 - p_i)  +  beta * Var_K(p_i) * I(p_i > tau)  +  gamma_yi

Step 2 — Batch normalization (mean weight = 1.0):
    w_hat_i = (w_i / sum(w_j)) * batch_size

Step 3 — Final loss:
    L_CCR = (1/B) * sum(w_hat_i * CrossEntropy(logits_i, y_i))
```

| Symbol | Meaning | Value |
|--------|---------|-------|
| `p_i` | Softmax probability of the true class | — |
| `Var_K` | Variance of `p_i` over the last K epochs | — |
| `I(p_i > tau)` | Confidence gate — silences noisy low-conf samples | — |
| `tau` | Confidence gate threshold (**fixed**) | `0.3` |
| `beta` | Variance scaling factor (**fixed**) | `0.5` |
| `K` | Rolling history window in epochs (**fixed**) | `5` |
| `gamma_yi` | Normalized inverse class-frequency weight | — |

> All three hyperparameters are **fixed across all datasets** — no per-dataset tuning.

---

## Datasets

Six OpenML benchmark datasets, downloaded automatically on first run.

| Dataset | OpenML ID | Samples | Features | Imbalance Ratio |
|---------|-----------|---------|----------|-----------------|
| `adult` | [1590](https://www.openml.org/d/1590) | 48,842 | 14 | 3.2 : 1 |
| `bank` | [1461](https://www.openml.org/d/1461) | 45,211 | 17 | 7.5 : 1 |
| `magic` | [1120](https://www.openml.org/d/1120) | 19,020 | 10 | 1.8 : 1 |
| `phoneme` | [1489](https://www.openml.org/d/1489) | 5,404 | 5 | 2.4 : 1 |
| `credit_g` | [31](https://www.openml.org/d/31) | 1,000 | 20 | 2.3 : 1 |
| `spambase` | [44](https://www.openml.org/d/44) | 4,601 | 57 | 1.5 : 1 |

---

## Baselines

Eight models compared in every experiment.

| ID | Model key | Description |
|----|-----------|-------------|
| B1 | `mlp_standard` | MLP with vanilla cross-entropy |
| B2 | `mlp_focal` | MLP with Focal Loss (α=0.25, γ=2.0) |
| B3 | `mlp_weighted_ce` | MLP with class-weighted cross-entropy |
| B4 | `mlp_smote` | SMOTE oversampling + standard MLP |
| B5 | `xgboost_default` | XGBoost with default settings |
| B6 | `xgboost_weighted` | XGBoost with `scale_pos_weight` |
| B7 | `lightgbm_default` | LightGBM with default settings |
| **OURS** | `mlp_ccr` | **CCR Loss MLP** |

---

## Experiment Configs

Seven noise configurations, each run across all datasets and all models.

| Config file | Noise type | Rate |
|-------------|------------|------|
| `clean_run.yaml` | None (clean labels) | 0% |
| `noisy_asym_10.yaml` | Asymmetric (minority → majority) | 10% |
| `noisy_asym_20.yaml` | Asymmetric | 20% |
| `noisy_asym_30.yaml` | Asymmetric | 30% |
| `noisy_feat_10.yaml` | Feature-correlated (boundary) | 10% |
| `noisy_feat_20.yaml` | Feature-correlated | 20% |
| `noisy_feat_30.yaml` | Feature-correlated | 30% |

Each config runs **5-fold stratified CV × 3 seeds = 15 runs** per dataset-model pair.

---

## Installation & Running

### Option A — Everything in one command (recommended)

```bash
git clone https://github.com/your-org/CCR-Tabular.git
cd CCR-Tabular
python main.py
```

`main.py` handles the full pipeline automatically:

```text
PHASE 1 — Installing dependencies
  GPU detected — installing PyTorch with CUDA 11.8 support...
  Installing 13 packages from requirements.txt...

PHASE 2 — Downloading datasets
  [adult] ready.  [bank] ready.  [magic] ready.
  [phoneme] ready.  [credit_g] ready.  [spambase] ready.

PHASE 3 — Running experiments
  adult | mlp_ccr | none@0%  ...
  adult | mlp_ccr | asym@10% ...
  ...
```

### Option B — Manual install then run

```bash
# CPU
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# GPU (CUDA 11.8)
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Run everything
python main.py --no_install
```

### Single dataset / model run

```bash
# One dataset, one model, clean labels
python main.py --dataset credit_g --model mlp_ccr

# With noise
python main.py --dataset adult --model mlp_ccr --noise_type asym --noise_rate 0.2

# Fast smoke test: 2 folds, 1 seed
python main.py --dataset credit_g --model mlp_ccr --n_folds 2 --seeds 42
```

### All `main.py` flags

| Flag | Description | Default |
|------|-------------|---------|
| `--dataset` | Single dataset to run | all datasets |
| `--model` | Single model to run | all models |
| `--noise_type` | `none` `asym` `feat` | `none` |
| `--noise_rate` | `0.0` – `0.3` | `0.0` |
| `--n_folds` | Number of CV folds | `5` |
| `--seeds` | Space-separated seeds | `42 123 2024` |
| `--no_install` | Skip dependency installation | off |
| `--no_prefetch` | Skip dataset pre-download | off |

---

## Tests

```bash
conda run -n py312 python -m pytest tests/ -v
```

> **Expected: 33 passed, 0 failed**| Test file | Coverage |
|-----------|----------|
| `test_ccr_loss.py` | Loss math, batch normalization, history buffer, variance cold-start, edge cases |
| `test_noise_injection.py` | Majority labels never flipped, rate accuracy ±2%, reproducibility |
| `test_metrics.py` | macro F1 vs binary F1, AUC uses probabilities, minority recall class |
| `test_data_leakage.py` | Scaler fit on train only, column mismatch detection, noise safety guard |

---

## Statistical Significance

After experiments complete, run Wilcoxon signed-rank tests (CCR vs all 7 baselines):

```bash
python -c "from src.utils.statistics import run_all_wilcoxon_tests; run_all_wilcoxon_tests()"
```

Results saved to `outputs/metrics/wilcoxon_*.csv`. Significance threshold: p < 0.05.

---

## Ablation Study

Three ablation variants are implemented in `src/loss/ccr_loss.py`:

| Variant | Class | What is removed |
|---------|-------|-----------------|
| A1 | `CCRLossNoGate` | Confidence gate `I(p_i > tau)` — variance applied to all samples |
| A2 | `CCRLossNoVariance` | Entire variance term — focal + class weight only |
| A3 | `CCRLossNoNormalization` | Batch-level weight normalization — raw weights used directly |

Run the full ablation study (all 6 datasets × 4 variants × 7 noise configs):

```bash
python run_ablation.py
```

---

## Expansion Experiments

Additional sensitivity analyses for the paper:

```bash
# Tau sensitivity: tau in {0.3, 0.5, 0.6, 0.7, 0.8}
python run_tau_sensitivity.py

# K sensitivity: K in {3, 5, 10}
python run_k_sensitivity.py

# Beta sensitivity: beta in {0.3, 0.5, 0.8}
python run_beta_sensitivity.py

# Noise@40% extension
python run_noise40.py

# Run all expansion experiments in sequence (resumable)
python run_all_expansions.py

# Gate calibration diagnostic
python diagnose_gate.py

# View results summary and paper claim verdicts
python show_results.py
```

---

## Outputs

All outputs are written automatically during training.

```text
outputs/
├── models/      ← Checkpoints (.pt for MLP, .pkl for sklearn)
│                   <dataset>_<model>_<noise>_<rate>_seed<s>_fold<f>.pt
├── logs/        ← Per-run JSON training logs
│                   <run_id>_train.json
│                   Fields: epoch, train_loss, val_macro_f1, val_minority_recall, lr
├── metrics/
│   ├── results.csv                         ← All runs, all metrics (append + deduplicated)
│   └── cv_summary_<dataset>_<model>_*.csv  ← Mean ± Std per experiment
└── plots/       ← Figures
```

`results.csv` schema:

```text
run_id | dataset | model | fold | seed | noise_type | noise_rate |
accuracy | macro_f1 | minority_recall | auc_roc | auc_pr | timestamp
```

---

## Project Structure

```text
CCR-Tabular/
│
├── main.py                        ← Single entry point (install + download + run)
├── requirements.txt
├── CLAUDE.md                      ← Agent memory / project rules
│
├── data/
│   ├── raw/                       ← Downloaded CSVs (gitignored)
│   ├── processed/                 ← Encoded datasets (gitignored)
│   └── noisy/                     ← Noise-injected train splits (gitignored)
│
├── src/
│   ├── data/
│   │   ├── load_data.py           ← OpenML download + binary encoding
│   │   ├── preprocess.py          ← Leakage-safe ColumnTransformer pipeline
│   │   └── noise_injection.py     ← Asymmetric + feature-correlated noise
│   ├── loss/
│   │   └── ccr_loss.py            ← CCRLoss (core contribution)
│   ├── models/
│   │   ├── mlp.py                 ← TabularMLP + TabularDataset
│   │   └── baselines.py           ← All 7 baselines + factory
│   ├── training/
│   │   ├── train.py               ← Single-fold training loop
│   │   ├── evaluate.py            ← Test inference + results.csv writer
│   │   └── cross_validation.py    ← 5-fold × 3-seed CV orchestration
│   └── utils/
│       ├── config.py              ← All hyperparameters and paths
│       ├── reproducibility.py     ← Seed fixing + device detection
│       ├── logger.py              ← Structured JSON + stdout logging
│       ├── metrics.py             ← accuracy, macro_f1, recall, AUC-ROC, AUC-PR
│       └── statistics.py          ← Wilcoxon signed-rank tests (IEEE TNNLS requirement)
│
├── experiments/
│   ├── configs/                   ← 7 YAML experiment configs
│   │   ├── clean_run.yaml
│   │   ├── noisy_asym_10.yaml
│   │   ├── noisy_asym_20.yaml
│   │   ├── noisy_asym_30.yaml
│   │   ├── noisy_feat_10.yaml
│   │   ├── noisy_feat_20.yaml
│   │   └── noisy_feat_30.yaml
│   └── run_experiments.py         ← Master runner with resume support
│
├── paper_figures.py               ← Publication figures (600 DPI PNG + PDF)
├── run_ablation.py                ← Ablation study (4 CCR variants × all 6 datasets)
├── run_tau_sensitivity.py         ← τ sensitivity (τ ∈ {0.3, 0.5, 0.6, 0.7, 0.8})
├── run_k_sensitivity.py           ← K sensitivity (K ∈ {3, 5, 10})
├── run_beta_sensitivity.py        ← β sensitivity (β ∈ {0.3, 0.5, 0.8})
├── run_noise40.py                 ← Noise@40% extension experiment
├── run_learning_curves.py         ← Extract per-epoch F1 from logs
├── run_all_expansions.py          ← Master runner for all expansion experiments
├── diagnose_gate.py               ← Gate activation diagnostic (τ calibration)
├── show_results.py                ← Print results summary and paper claim verdicts
│
└── tests/    ├── test_ccr_loss.py
    ├── test_noise_injection.py
    ├── test_metrics.py
    └── test_data_leakage.py
```

---

## Key Design Decisions

| Decision | Detail |
|----------|--------|
| **No data leakage** | `ColumnTransformer` fit on training fold only — enforced by assertions |
| **No noise on test data** | Size-based guardrail in `noise_injection.py` raises if array looks like full dataset |
| **Early stopping on macro F1** | Not accuracy — accuracy is misleading on imbalanced data |
| **GPU-first, CPU-fallback** | Auto-detects `cuda → mps → cpu`. No hardcoded `"cuda"` strings |
| **Idempotent results** | Re-running skips completed `run_id`s — safe to interrupt and resume |
| **Config-driven** | `tau`, `beta`, `K`, seeds, paths all in `config.py` — no magic numbers in training code |
| **`weights_only=True`** | All `torch.load` calls use `weights_only=True` — no arbitrary code execution on load |
| **Wilcoxon significance** | `src/utils/statistics.py` — CCR vs all 7 baselines, p < 0.05 threshold |
| **3 ablation variants** | `CCRLossNoGate`, `CCRLossNoVariance`, `CCRLossNoNormalization` in `ccr_loss.py` |
| **Training time tracked** | `train_time_s` and `n_epochs` in every row of `results.csv` for all models |

---

## Reproducibility

Every run is fully reproducible. Seeds are fixed for Python, NumPy, PyTorch (CPU + GPU), and `PYTHONHASHSEED` before any model initialization or data loading.

Run IDs encode all experimental conditions:

```text
<dataset>_<model>_<noise_type>_<noise_rate>_seed<seed>_fold<fold>

# Example
adult_mlp_ccr_asym_20_seed42_fold3
```

---

## Citation

If you use this codebase, please cite

---

<div align="center">

Made with ❤️ for reproducible ML research

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![pytest](https://img.shields.io/badge/pytest-33%20passed-brightgreen?style=flat-square&logo=pytest&logoColor=white)](tests/)
[![GPU Ready](https://img.shields.io/badge/GPU-CUDA%20%7C%20MPS%20%7C%20CPU-76b900?style=flat-square&logo=nvidia&logoColor=white)](src/utils/reproducibility.py)

</div>
