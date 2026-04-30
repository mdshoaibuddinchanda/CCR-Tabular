# CCR-Tabular — Project Progress

> Internal tracking document. Not pushed to GitHub.

## Project Identity

| Field | Value |
|-------|-------|
| Full title | Dynamic Sample Reweighting via Confidence-Gated Variance for Robust Tabular Learning |
| Target venue | IEEE Transactions on Neural Networks and Learning Systems (TNNLS) |
| Hardware | NVIDIA RTX 3050 (4 GB VRAM) |
| Framework | PyTorch 2.6 + scikit-learn 1.3 |
| Python | 3.12 |
| Repository | https://github.com/mdshoaibuddinchanda/CCR-Tabular- |

---

## Final Project Structure

```
CCR-Tabular/
├── main.py                              ← SINGLE ENTRY POINT (runs everything)
├── requirements.txt
├── src/                                 ← All library code
│   ├── data/load_data.py
│   ├── data/preprocess.py
│   ├── data/noise_injection.py
│   ├── loss/ccr_loss.py                 ← CCRLoss + 3 ablation variants
│   ├── models/mlp.py
│   ├── models/baselines.py
│   ├── training/train.py
│   ├── training/evaluate.py
│   ├── training/cross_validation.py
│   └── utils/config.py, logger.py, metrics.py,
│         reproducibility.py, statistics.py, experiment_utils.py
├── experiments/
│   ├── configs/                         ← 7 YAML noise configs
│   ├── run_experiments.py               ← Main experiment runner (Phase 3)
│   └── expansions/                      ← All expansion scripts (Phase 4)
│       ├── run_all_expansions.py        ← Master expansion runner
│       ├── run_ablation.py
│       ├── run_tau_sensitivity.py
│       ├── run_k_sensitivity.py
│       ├── run_beta_sensitivity.py
│       ├── run_noise40.py
│       └── run_learning_curves.py
├── scripts/                             ← Analysis utilities
│   ├── paper_figures.py
│   └── diagnose_gate.py
└── tests/                               ← 49 tests, all passing
    ├── test_ccr_loss.py
    ├── test_noise_injection.py
    ├── test_metrics.py
    ├── test_data_leakage.py
    └── test_experiment_utils.py
```

---

## Implementation Status — ALL COMPLETE

| Component | Status |
|-----------|--------|
| CCR Loss (focal + variance gate + batch norm) | ✅ |
| 3 CCR ablation variants (no gate, no variance, no norm) | ✅ |
| 7 baselines (B1–B7) | ✅ |
| 6 datasets via OpenML (with sklearn fallback) | ✅ |
| 5-fold × 3-seed CV | ✅ |
| 7 noise configs (clean + asym 10/20/30 + feat 10/20/30) | ✅ |
| All 5 metrics (macro F1, recall, AUC-ROC, AUC-PR, accuracy) | ✅ |
| No data leakage (asserted architecturally) | ✅ |
| No noise on test data (size guardrail) | ✅ |
| Fixed hyperparams τ=0.3, β=0.5, K=5 | ✅ |
| Wilcoxon signed-rank tests | ✅ |
| Training time tracking | ✅ |
| Resume support (idempotent run_ids) | ✅ |
| GPU-first device detection | ✅ |
| 49 unit tests passing | ✅ |
| Publication figures (600 DPI PNG + PDF) | ✅ |
| Single entry point (main.py) | ✅ |
| Shared experiment utilities (experiment_utils.py) | ✅ |
| τ sensitivity (1,350 runs) | ✅ Script ready |
| K sensitivity (810 runs) | ✅ Script ready |
| β sensitivity (810 runs) | ✅ Script ready |
| Noise@40% extension (720 runs) | ✅ Script ready |
| Ablation study (2,520 runs) | ✅ Script ready |
| Learning curves extraction | ✅ Script ready |
| Gate diagnostic | ✅ Done — τ=0.3 fires 97.3% of time |

---

## How to Run

```bash
# FULL RUN — everything (recommended)
python main.py

# Skip install (deps already installed)
python main.py --no_install

# Skip install + download (data already cached)
python main.py --no_install --no_prefetch

# Single quick test
python main.py --dataset credit_g --model mlp_ccr --n_folds 2 --seeds 42

# Expansion experiments only (after main.py completes)
python experiments/expansions/run_all_expansions.py

# Generate paper figures (after all experiments complete)
python scripts/paper_figures.py

# Run tests
python -m pytest tests/ -v
```

---

## Experiment Scope

| Phase | Script | Runs | Output |
|-------|--------|------|--------|
| Phase 3 (main) | `experiments/run_experiments.py` | 5,040 | `results.csv` |
| Phase 4 — Ablation | `experiments/expansions/run_ablation.py` | 2,520 | `results_ablation.csv` |
| Phase 4 — τ sensitivity | `experiments/expansions/run_tau_sensitivity.py` | 1,350 | `results_tau_sensitivity.csv` |
| Phase 4 — K sensitivity | `experiments/expansions/run_k_sensitivity.py` | 810 | `results_k_sensitivity.csv` |
| Phase 4 — β sensitivity | `experiments/expansions/run_beta_sensitivity.py` | 810 | `results_beta_sensitivity.csv` |
| Phase 4 — Noise@40% | `experiments/expansions/run_noise40.py` | 720 | appended to `results.csv` |
| Phase 4 — Curves | `experiments/expansions/run_learning_curves.py` | 0 (log parse) | `learning_curves.csv` |
| **Total** | | **~11,250** | |

---

## Key Results (from previous run)

### Clean Data (mean across 6 datasets)

| Model | Macro F1 | Minority Recall |
|-------|----------|-----------------|
| MLP-CE | 0.793 | 0.654 |
| MLP-WCE | 0.791 | 0.805 |
| XGBoost-W | 0.825 | 0.798 |
| LightGBM | 0.822 | 0.687 |
| **CCR (Ours)** | **0.799** | **0.724** |

### Noise Robustness (mean F1 drop from clean)

| Model | Asym@10% | Asym@20% | Asym@30% |
|-------|----------|----------|----------|
| MLP-CE | +0.0098 | +0.0283 | +0.0578 |
| XGBoost-W | +0.0103 | +0.0234 | +0.0357 |
| LightGBM | +0.0204 | +0.0498 | +0.0889 |
| **CCR** | **+0.0014** | **+0.0070** | **+0.0144** |

CCR degrades **3× slower than XGBoost-W**, **6× slower than LightGBM**.

### Gate Diagnostic Finding

τ=0.3 fires on 97.3% of samples — gate is effectively disabled. τ=0.7 gives ~57% activation (intended range). τ sensitivity experiment will confirm optimal value.

---

## Audit Log

| Change | Reason |
|--------|--------|
| Initial build | PRD implementation |
| Multiple bug fixes (torch.load, datetime, device bug) | Correctness |
| Added train_time_s, n_epochs to results.csv | PRD requirement |
| Added 3 CCR ablation variants | PRD requirement |
| Added Wilcoxon tests | PRD requirement |
| Fixed OpenML download with sklearn fallback | Compatibility |
| Fixed LightGBM feature names warning | Code cleanliness |
| Added experiment_utils.py shared module | DRY principle |
| Moved all run_*.py to experiments/expansions/ | Clean structure |
| Moved paper_figures.py, diagnose_gate.py to scripts/ | Clean structure |
| Added Phase 4 to main.py | Single entry point |
| 49/49 tests passing | Verification |
| All imports verified from new locations | Final audit |
