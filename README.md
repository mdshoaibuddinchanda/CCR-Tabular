<div align="center">

# CCR-Tabular

**Confidence-Calibrated Reweighting with Invariant Batch Normalization for Robust Tabular Learning**

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20(CUDA%20%7C%20CPU)-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-87%20passed%20(100%25)-brightgreen?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![Audit Gate](https://img.shields.io/badge/Dataset%20Audit-14%2F14%20PASSED-success?style=for-the-badge)](outputs/metrics/dataset_audit_report.csv)

[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-189fdd?style=flat-square)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.1+-2980b9?style=flat-square)](https://lightgbm.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2+-yellow?style=flat-square)](https://catboost.ai/)
[![Statistical Inference](https://img.shields.io/badge/Stats-Paired%20Wilcoxon%20%2B%20BH--FDR-blueviolet?style=flat-square)](src/utils/statistics.py)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Pre--registered%20Seeds%20(42%2C%20123%2C%202024)-informational?style=flat-square)](src/utils/reproducibility.py)
[![Resource Controller](https://img.shields.io/badge/Heterogeneous-Auto%20AMP%20%2B%20OOM%20Recovery-blue?style=flat-square)](main.py)

*An empirical and theoretical framework investigating dynamic sample reweighting, per-batch gradient scale invariance, and noise robustness on tabular deep neural architectures.*

</div>

---

## 🚀 1-Click Master Execution (Transport & Run on Any Machine)

The entire pipeline is **100% self-contained and automated**. When transferring to a new or higher-spec workstation:

```powershell
# 1. Clone the repository
git clone https://github.com/mdshoaibuddinchanda/CCR-Tabular.git
cd CCR-Tabular

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run 1-Click Master Benchmark (Press Enter and Leave it!)
python main.py
```

### What `python main.py` Does Automatically:
1. **Automatic Dataset Provisioning**: Checks [`data/raw/`](data/raw) for all 14 datasets. If any are missing, it automatically downloads them from OpenML, standardizes the class labels, and caches them locally as CSV.
2. **Dynamic Hardware Profiling**: Automatically detects available CUDA VRAM, sets safe memory budgets (with 20% protected headroom), enables automatic mixed precision (`FP16 AMP`), and budgets CPU worker processes (`logical_cores - 3`).
3. **Frozen Manifest Generation**: Writes `outputs/final_master/manifest.json` recording Git commit SHA, environment specs, and experimental tokens.
4. **Executes the 6,000-Run Core-10 Benchmark**: Concurrently schedules the GPU neural worker and CPU pool across all $10\text{ Datasets} \times 10\text{ Losses} \times 4\text{ Noise Regimes} \times 3\text{ Seeds} \times 5\text{ Folds}$.
5. **Idempotent Resume**: If paused or restarted, it resumes from the exact fold without repeating finished runs.
6. **Automatic Statistical Significance**: Computes paired effect sizes and Benjamini-Hochberg FDR-adjusted $p$-values across datasets upon completion.

---

## 🔬 Core Scientific Finding & Contribution

> **The Settled Contribution Statement**:  
> *Dynamic confidence-aware reweighting is the principal source of predictive robustness under label noise, while per-batch normalization removes batch-dependent global gradient-scale variation and can improve optimization stability, particularly under fixed-step optimization.*

### Three Decisive Empirical Findings:
1. **Dynamic Reweighting Drives Predictive Gain**: Dynamic confidence-aware down-weighting $(1 - p_{i, y_i}) + \beta \text{Var}_K(p_i)\mathbf{1}(p_{i, y_i} > \tau) + \gamma_{y_i}$ actively depresses corrupted-label gradient mass ($R_{\text{noise}}$ down by 15%–25%), generating a decisive Macro-F1 advantage under severe asymmetric label corruption.
2. **Normalization Stabilizes Gradient Norm Scale**: Per-batch normalization ($\frac{1}{B}\sum \hat{w}_i \equiv 1.0$) eliminates batch-composition-dependent optimization step distortion, reducing gradient norm volatility ($\text{Grad CV}$) by **10% to 35%**.
3. **Refutation of $3\text{--}4\times$ Inflation**: Comprehensive batch telemetry across real-world runs proves that real tabular training exhibits batch deflation ($S/B \in [0.327, 1.022]$ with $P_{99} \le 0.96$ and theoretical supremum $S/B \le 2.125$).

---

## 📐 Mathematical Formulation

### 1. Autograd-Detached Dynamic Sample Weighting
For mini-batch $\mathcal{B} = \{(x_i, y_i)\}_{i=1}^B$ with predicted probability vectors $p_i = \text{softmax}(z_i) \in \Delta^{C-1}$:

$$w_i = \text{detach}\Big( (1 - p_{i, y_i}) + \beta \cdot \text{Var}_K(p_i) \cdot \mathbf{1}(p_{i, y_i} > \tau) + \gamma_{y_i} \Big)$$

where:
* **$(1 - p_{i, y_i})$**: Dynamic confidence-inverse weight (down-weights noisy/overconfident mislabeled samples).
* **$\beta \cdot \text{Var}_K(p_i) \cdot \mathbf{1}(p_{i, y_i} > \tau)$**: Historical variance gate over $K=5$ epochs active above $\tau=0.30$ (identifies fluctuating boundary samples).
* **$\gamma_{y_i} = \frac{1/N_{y_i}}{\sum_c 1/N_c}$**: Normalized static inverse-class weight (counteracts class imbalance).

### 2. Invariant Batch Normalization
$$\hat{w}_i = \frac{w_i}{\sum_{j=1}^B w_j + \epsilon} \cdot B \implies \frac{1}{B}\sum_{i=1}^B \hat{w}_i \equiv 1.0$$

### 3. Exact Analytical Loss & Gradient
$$\mathcal{L}_{\text{CCR}} = \frac{1}{B}\sum_{i=1}^B \hat{w}_i \, \ell_{\text{CE}}(z_i, y_i) \implies \frac{\partial \mathcal{L}_{\text{CCR}}}{\partial z_{ik}} = \frac{1}{B} \hat{w}_i \Big( p_{ik} - \mathbf{1}(y_i = k) \Big)$$

### 4. Theoretical Supremum Bound ($S/B$)
$$\sup w_i \le \max(1 - p_y) + \beta \cdot \max(\text{Var}_K) + \max(\gamma_y) = 1.0 + (0.50 \times 0.25) + 1.0 = \mathbf{2.125}$$
$$\sup \left( \frac{1}{B}\sum_{i=1}^B w_i \right) \le \mathbf{2.125}$$

---

## 📊 Complete 14-Dataset Hierarchy (10 + 2 + 2 Design)

All datasets undergo strict fold-local preprocessing with zero leakage and exact metadata verification:

| Category | Dataset | OpenML ID | Samples ($N$) | Features ($D$) | Classes ($C$) | Imbalance Ratio (IR) | Domain / Task Description |
|---|---|---|---|---|---|---|---|
| **Tier 1: Core 10 Benchmark**<br>*(Primary 6,000-run comparison)* | `Adult` | 1590 | 48,842 | 14 | 2 | 3.17:1 | Socioeconomic Census Income |
| | `Bank` | 1461 | 45,211 | 16 | 2 | 7.55:1 | Marketing / Finance Term Deposit |
| | `Electricity` | 151 | 45,312 | 8 | 2 | 1.36:1 | Energy Demand Market Pricing |
| | `MAGIC` | 1120 | 19,020 | 10 | 2 | 1.84:1 | Gamma Ray Astro-Physics |
| | `Churn` | 40701 | 5,000 | 20 | 2 | 6.07:1 | Customer Subscription Churn |
| | `Phoneme` | 1489 | 5,404 | 5 | 2 | 2.41:1 | Acoustics / Speech Signal |
| | `Spambase` | 44 | 4,601 | 57 | 2 | 1.54:1 | Email Text Classification |
| | `WILT` | 40983 | 4,839 | 5 | 2 | **17.54:1** | Forestry Remote Sensing |
| | `Credit-G` | 31 | 1,000 | 20 | 2 | 2.33:1 | Financial Credit Lending |
| | `Ionosphere` | 59 | 351 | 34 | 2 | 1.79:1 | Low-$N$ Aerospace Radar |
| **Tier 4: Multiclass Transfer**<br>*($C \ge 3$ multiclass scaling)* | `Segment` | 36 | 2,310 | 19 | **7** | 1.00:1 | Image Vision Segmentation |
| | `Vehicle` | 54 | 846 | 18 | **4** | 1.10:1 | Silhouette Vision Geometry |
| **Tier 5: Clinical External Validation**<br>*(Real-world diagnostic ambiguity)* | `Heart Disease` | 1498 | 462 | 9 | 2 | 1.89:1 | Clinical Cardiology Diagnosis |
| | `Breast Cancer` | 13 | 286 | 9 | 2 | 2.36:1 | Clinical Pathology Biopsy |

---

## 🧱 Supported Models & Canonical 10-Loss Registry

### 1. Neural Architectures (PyTorch)
* **`TabularMLP`**: Feed-forward linear layers `[256, 128, 64]`, Batch Normalization, ReLU, Dropout ($p=0.30$).
* **`TabularResNet`**: 4 Pre-activation Residual Blocks (dim 128), Layer Normalization, ReLU, Dropout ($p=0.10$).
* **`TabularFTTransformer`**: Feature Tokenizer ($d_{\text{embed}}=64$), 3 Transformer Encoder Layers, 4 Heads, FFN Multiplier 4/3, Dropout ($p=0.10$).

### 2. Canonical 10-Loss Matrix
* **Standard Baselines**: Cross-Entropy (`ce`), Weighted CE (`wce`), Normalized WCE (`norm_wce`).
* **Noise-Robust Losses**: Focal Loss (`focal`), Normalized Focal (`norm_focal`), Generalized CE (`gce`), Symmetric CE (`sce`), Early-Learning Regularization (`elr`).
* **CCR Formulations**: CCR Ablation without Normalization (`ccr_no_norm`), Full CCR (`ccr`).

### 3. Tree-based GBDT Baselines
* **`XGBoost`**, **`LightGBM`**, **`CatBoost`** (Single-threaded worker isolation for deterministic reproducibility).

---

## ⚡ Centralized Heterogeneous Resource Scheduler

[`main.py`](main.py) provides a single unified resource-aware orchestrator:

```text
                    main.py
                       │
             Heterogeneous Scheduler
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
     CPU Queue                   GPU Queue
   (Process Pool)             (1 Dedicated Slot)
   • Tree Baselines           • TabularMLP
   • Preprocessing Caching    • TabularResNet
   • Statistics & Figures     • TabularFTTransformer
   (Max N_cpu - 3 Workers)    • Live VRAM Refresh
   • BLAS 1-Thread Isolation  • Automatic FP16 AMP
   • RAM Backpressure (>=4GB) • Model-Aware Safe Thresholds
```

### CLI Command Reference:

```powershell
# ── Default 1-Click Master Benchmark (Recommended) ─────────────────────────
python main.py                      # Automatically downloads datasets and runs Core-10 benchmark

# ── System Diagnostics & Audits ────────────────────────────────────────────
python main.py --resource_report    # Audit CPU cores, GPU safe VRAM budget, and AMP status
python main.py --validate           # Automated 5-point scientific consistency check
python main.py --dry_run            # Preview execution matrix & device assignments
python main.py --smoke_test         # Quick 2-fold diagnostic verification
python main.py --smoke_test_transformer # 1-Fold FT-Transformer GPU AMP smoke test

# ── Targeted Experiment Tiers ──────────────────────────────────────────────
python main.py --tier1              # Tier 1 Core-10 Benchmark (6,000 runs)
python main.py --tier2              # Tier 2 Batch Instrumentation & Telemetry
python main.py --attribution        # Per-sample gradient attribution & Lorenz curve
python main.py --tier3              # Tier 3 Architecture Transfer (MLP / ResNet / Transformer)
python main.py --tier4              # Tier 4 Multiclass Benchmark (Segment & Vehicle)
python main.py --tier5              # Tier 5 Clinical External Validation
python main.py --figures            # Generate all Publication Figures (1–7) & Supplementals (S1–S3)
```

---

## 📈 Publication Figures Hierarchy

All figures are programmatically generated and stored in `outputs/plots/`:

* **Figure 1**: Method Architecture & Normalization Mechanism Schematic.
* **Figure 2**: Empirical $S/B$ Weight-Sum Distribution vs. Theoretical Bounds ([`figure2_sb_distribution.png`](outputs/plots/figure2_sb_distribution.png)).
* **Figure 3**: Observed Relationships among Batch Composition, $S/B$, and Gradient Volatility ([`figure3_observed_relationships.png`](outputs/plots/figure3_observed_relationships.png)).
* **Figure 4**: CCR vs CCR-NoNorm Optimization Trajectories ([`mechanism_dynamics_training.png`](outputs/plots/mechanism_dynamics_training.png)).
* **Figure 5**: 4-Panel Gradient Attribution & Lorenz Concentration Curve ([`figure5_gradient_attribution.png`](outputs/plots/figure5_gradient_attribution.png)).
* **Figure 6**: Optimizer Interaction under SGD vs Adam vs AdamW ([`figure6_optimizer_sensitivity.png`](outputs/plots/figure6_optimizer_sensitivity.png)).
* **Figure 7**: Core-10 Robustness Curves across Label Noise Severities.
* **Figure S1**: Full Pairwise Loss-Comparison Matrix Heatmap ([`figure_s1_full_loss_comparison.png`](outputs/plots/figure_s1_full_loss_comparison.png)).
* **Figure S2**: Full $S/B$ Distribution Grid across All Datasets and Noise Regimes.
* **Figure S3**: Hyperparameter Sensitivity Curves ($\tau$, $\beta$, $K$).

---

## 🧪 Statistical Inference Framework

Primary statistical inference treats the **Dataset ($d$) as the independent observational unit**:
1. Computes matched dataset differences: $\Delta_d = \text{Macro-F1}_{\text{CCR}, d} - \text{Macro-F1}_{\text{baseline}, d}$.
2. Conducts cross-dataset **Paired Wilcoxon Signed-Rank Tests**.
3. Controls for multiplicity using **Benjamini-Hochberg False Discovery Rate (BH-FDR)** at $\alpha = 0.05$.
4. Reports Mean $\Delta_d$, Median $\Delta_d$, 95% Confidence Intervals, and Paired Cohen's $d$.

---

## ⚙️ Verification & Unit Tests

```powershell
# Run the complete certified test suite (87 unit & integration tests)
pytest tests/ -v

# Run automated scientific consistency validation
python main.py --validate
```

---

<div align="center">

**CCR-Tabular: Rigorous, Transparent, and Reproducible Tabular Learning.**  
Licensed under the [MIT License](LICENSE).

</div>
