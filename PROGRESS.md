# CCR-Tabular — Project Progress

> Single source of truth for what is built, what is running, and what remains.
> Updated after every significant change.

## Project Identity

| Field | Value |
|-------|-------|
| Full title | Dynamic Sample Reweighting via Confidence-Gated Variance for Robust Tabular Learning |
| Target venue | IEEE Transactions on Neural Networks and Learning Systems (TNNLS) |
| Hardware | NVIDIA RTX 3050 (4 GB VRAM) |
| Framework | PyTorch 2.1 + scikit-learn 1.3 |
| Python | 3.12 |
| Repository | CCR-Tabular/ |

---

## Implementation Status

### Core Components

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| CCR Loss (full) | `src/loss/ccr_loss.py` | ✅ Complete | Focal + variance gate + batch norm |
| CCR Ablation A1 (no gate) | `src/loss/ccr_loss.py` | ✅ Complete | `CCRLossNoGate` |
| CCR Ablation A2 (no variance) | `src/loss/ccr_loss.py` | ✅ Complete | `CCRLossNoVariance` |
| CCR Ablation A3 (no norm) | `src/loss/ccr_loss.py` | ✅ Complete | `CCRLossNoNormalization` |
| MLP architecture | `src/models/mlp.py` | ✅ Complete | Auto-selects arch by dataset size |
| All 7 baselines | `src/models/baselines.py` | ✅ Complete | B1–B7 with unified interface |
| Data loading (OpenML) | `src/data/load_data.py` | ✅ Complete | Retry logic, local cache |
| Preprocessing (no leakage) | `src/data/preprocess.py` | ✅ Complete | Fit on train only, asserted |
| Asymmetric noise injection | `src/data/noise_injection.py` | ✅ Complete | Majority labels never flipped |
| Feature-correlated noise | `src/data/noise_injection.py` | ✅ Complete | Confidence-threshold based |
| Training loop (CCR) | `src/training/train.py` | ✅ Complete | Early stop on macro F1 |
| Training loop (baselines) | `src/training/train.py` | ✅ Complete | Unified routing |
| Training time tracking | `src/training/train.py` | ✅ Complete | `train_time_s`, `n_epochs` in all runs |
| Evaluation + metrics | `src/training/evaluate.py` | ✅ Complete | All 5 metrics + timing |
| 5-fold × 3-seed CV | `src/training/cross_validation.py` | ✅ Complete | StratifiedKFold, resume support |
| Wilcoxon significance tests | `src/utils/statistics.py` | ✅ Complete | All baselines × all datasets |
| Config (single source) | `src/utils/config.py` | ✅ Complete | τ=0.3, β=0.5, K=5 fixed |
| Reproducibility | `src/utils/reproducibility.py` | ✅ Complete | All seeds fixed |
| Structured logging | `src/utils/logger.py` | ✅ Complete | JSON + stdout, no deprecated APIs |
| All metrics | `src/utils/metrics.py` | ✅ Complete | macro F1, recall, AUC-ROC, AUC-PR |

### Experiment Infrastructure

| Component | File | Status |
|-----------|------|--------|
| 7 YAML configs | `experiments/configs/*.yaml` | ✅ Complete |
| Master runner (resume) | `experiments/run_experiments.py` | ✅ Complete |
| Single entry point | `main.py` | ✅ Complete — auto-installs, downloads, runs |

### Tests

| Test file | Tests | Status |
|-----------|-------|--------|
| `tests/test_ccr_loss.py` | 9 tests | ✅ 9/9 passing |
| `tests/test_noise_injection.py` | 12 tests | ✅ 12/12 passing |
| `tests/test_metrics.py` | 6 tests | ✅ 6/6 passing |
| `tests/test_data_leakage.py` | 6 tests | ✅ 6/6 passing |
| **Total** | **33 tests** | **✅ 33/33 passing** |

### Outputs and Notebooks

| Component | Status | Notes |
|-----------|--------|-------|
| Results CSV | ✅ Auto-generated | `outputs/metrics/results.csv` |
| Per-run JSON logs | ✅ Auto-generated | `outputs/logs/<run_id>_train.json` |
| CV summary CSVs | ✅ Auto-generated | `outputs/metrics/cv_summary_*.csv` |
| Wilcoxon tables | ✅ Auto-generated | `outputs/metrics/wilcoxon_*.csv` |
| Results visualization | ✅ Complete | `notebooks/02_results_viz.ipynb` |
| Paper table export | ✅ In notebook | `outputs/metrics/paper_table_clean.csv` |

---

## Experiment Progress

### Datasets Downloaded

| Dataset | Samples | Status |
|---------|---------|--------|
| adult | 48,842 | ✅ Cached in `data/raw/adult.csv` |
| bank | 45,211 | ⏳ Not yet downloaded |
| magic | 19,020 | ⏳ Not yet downloaded |
| phoneme | 5,404 | ⏳ Not yet downloaded |
| credit_g | 1,000 | ✅ Cached in `data/raw/credit_g.csv` |
| spambase | 4,601 | ⏳ Not yet downloaded |

### Experiment Runs Completed

> Check `outputs/metrics/results.csv` for live count.

| Config | Status |
|--------|--------|
| `clean_run.yaml` | 🔄 In progress (adult: mlp_standard ✅, mlp_focal ✅, mlp_weighted_ce partial) |
| `noisy_asym_10.yaml` | ⏳ Not started |
| `noisy_asym_20.yaml` | ⏳ Not started |
| `noisy_asym_30.yaml` | ⏳ Not started |
| `noisy_feat_10.yaml` | ⏳ Not started |
| `noisy_feat_20.yaml` | ⏳ Not started |
| `noisy_feat_30.yaml` | ⏳ Not started |

Total runs needed: **5,040** (6 datasets × 8 models × 7 configs × 5 folds × 3 seeds)

---

## PRD Compliance Checklist

| PRD Requirement | Status | Notes |
|-----------------|--------|-------|
| 6 real-world datasets | ✅ | adult, bank, magic, phoneme, credit_g, spambase |
| 7 baselines | ✅ | B1–B7 all implemented |
| CCR loss (3 components) | ✅ | Focal + variance gate + batch norm |
| Fixed hyperparams (τ, β, K) | ✅ | Never tuned per dataset |
| 5-fold × 3-seed CV | ✅ | 15 runs per condition |
| Macro F1 as primary metric | ✅ | Early stopping on macro F1 |
| Minority recall reported | ✅ | pos_label=1 |
| AUC-ROC + AUC-PR | ✅ | Uses probabilities, not hard labels |
| No data leakage | ✅ | Asserted architecturally |
| No noise on test data | ✅ | Size guardrail enforced |
| Wilcoxon signed-rank test | ✅ | `src/utils/statistics.py` |
| Ablation study (3 variants) | ✅ | No gate / no variance / no norm |
| Training time reporting | ✅ | `train_time_s` in results.csv for all models |
| Results visualization | ✅ | `notebooks/02_results_viz.ipynb` |
| Reproducibility (seeds) | ✅ | Python + NumPy + PyTorch + PYTHONHASHSEED |
| Pinned requirements | ✅ | `requirements.txt` |
| Resume support | ✅ | run_id deduplication |
| GPU-first device | ✅ | cuda → mps → cpu |
| Paper docs (LaTeX) | ❌ | Intentionally excluded per project decision |

---

## Known Issues and Risks

| Issue | Severity | Status |
|-------|----------|--------|
| `bank` dataset (7.5:1 IR) is the hardest test — CCR must show clear gains here | HIGH | ⏳ Awaiting results |
| Ablation variants not yet wired into experiment YAML configs | MEDIUM | Manual run needed |
| Wilcoxon tests require ≥5 paired observations — some conditions may have fewer if runs are partial | LOW | Auto-handled with warning |
| `outputs/plots/` empty until notebook is run | LOW | Expected — run notebook after experiments |

---

## How to Run

```bash
# Everything (install + download + all experiments)
python main.py

# Resume interrupted run
python main.py --no_install --no_prefetch

# Single quick test
python main.py --dataset credit_g --model mlp_ccr --n_folds 2 --seeds 42

# Run significance tests after experiments complete
python -c "from src.utils.statistics import run_all_wilcoxon_tests; run_all_wilcoxon_tests()"

# Generate all paper figures
jupyter notebook notebooks/02_results_viz.ipynb
```

---

## Audit Log

| Date | Change | Reason |
|------|--------|--------|
| Initial build | Full codebase scaffolded | PRD implementation |
| Audit 1 | Removed unused imports (`tempfile`, `DROPOUT`, `List`, `append_results`) | Code cleanliness |
| Audit 1 | Fixed `torch.load` → `weights_only=True` | Security / PyTorch 2.x deprecation |
| Audit 1 | Fixed `_compute_variance` device bug | Correctness on multi-device |
| Audit 1 | Fixed `datetime.utcnow()` → `datetime.now(timezone.utc)` | Python 3.12 deprecation |
| Audit 1 | Moved `train_test_split` import out of inner loop | Performance |
| Audit 2 | Added `train_time_s` and `n_epochs` to results.csv | PRD requirement |
| Audit 2 | Added 3 CCR ablation variants to `ccr_loss.py` | PRD requirement |
| Audit 2 | Added `src/utils/statistics.py` (Wilcoxon tests) | PRD requirement |
| Audit 2 | Added `notebooks/02_results_viz.ipynb` | PRD requirement |
| Audit 2 | Removed dead `history_filled` buffer from CCRLoss | Dead code removal |
| Audit 2 | Fixed `TabularDataset` string annotation → real `np.ndarray` import | Type correctness |
| Audit 3 (mega) | Removed `Path` import from `logger.py` | Unused import |
| Audit 3 (mega) | Removed `List` import from `statistics.py` | Unused import |
| Audit 3 (mega) | Fixed `datetime.utcnow()` in `logger.py` (two remaining instances) | Python 3.12 deprecation |
| Audit 3 (mega) | Fixed hardcoded `200` in epoch print → uses `MAX_EPOCHS` from config | Config-driven |
| Audit 3 (mega) | Fixed corrupted line in `_train_ccr` (two statements merged on one line) | Syntax correctness |
| Audit 3 (mega) | Added `train_time_s` and `n_epochs` to MLP and sklearn baseline returns | Consistency |
| Audit 3 (mega) | Documented `y_prob` and `model_name` unused params with `noqa` comments | Clarity |
| Audit 3 (mega) | All diagnostics clean — 0 warnings across all 6 audited files | Code quality |
| Audit 3 (mega) | 33/33 tests passing after all changes | Verification |
