<div align="center">

# CCR-Tabular

**Confidence-Calibrated Reweighting with Invariant Batch Normalization for Robust Tabular Learning**

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20(CUDA%20%7C%20CPU)-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-85%20passed-brightgreen?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![Audit Gate](https://img.shields.io/badge/Dataset%20Audit-14%2F14%20PASSED-success?style=for-the-badge)](outputs/metrics/dataset_audit_report.csv)

[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-189fdd?style=flat-square)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.1+-2980b9?style=flat-square)](https://lightgbm.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2+-yellow?style=flat-square)](https://catboost.ai/)
[![Statistical Inference](https://img.shields.io/badge/Stats-Paired%20Wilcoxon%20%2B%20BH--FDR-blueviolet?style=flat-square)](src/utils/statistics.py)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Pre--registered%20Seeds%20(42%2C%20123%2C%202024)-informational?style=flat-square)](src/utils/reproducibility.py)
[![Resource Controller](https://img.shields.io/badge/Heterogeneous-Auto%20AMP%20%2B%20OOM%20Recovery-blue?style=flat-square)](main.py)

*An empirical and theoretical experimental framework investigating dynamic sample reweighting, per-batch gradient scale invariance, and noise robustness on tabular neural architectures.*

</div>

---

## 🔬 Core Scientific Finding & Contribution

> **The Settled Contribution Statement**:  
> *Dynamic confidence-aware reweighting is the principal source of predictive robustness under label noise, while per-batch normalization removes batch-dependent global gradient-scale variation and can improve optimization stability, particularly under fixed-step optimization.*

### Three Decisive Empirical Findings:
1. **Dynamic Reweighting Drives Predictive Gain**: Dynamic confidence-aware down-weighting $(1 - p_{i, y_i}) + \beta \text{Var}_K(p_i)\mathbf{1}(p_{i, y_i} > \tau) + \gamma_{y_i}$ actively depresses corrupted-label gradient mass ($R_{\text{noise}}$ down by 15%–25%), generating the **+10% to +14% Macro-F1 advantage** under severe label corruption.
2. **Normalization Stabilizes Gradient Norm Scale**: Per-batch normalization ($\frac{1}{B}\sum \hat{w}_i \equiv 1.0$) eliminates batch-composition-dependent optimization step distortion, reducing gradient norm volatility ($\text{Grad CV}$) by **10% to 35%**.
3. **Refutation of $3\text{--}4\times$ Inflation**: Comprehensive batch telemetry across 450 real-world runs proves that real tabular training exhibits batch deflation ($S/B \in [0.327, 1.022]$ with $P_{99} \le 0.96$ and theoretical supremum $S/B \le 2.125$).

---

## 📐 Mathematical Formulation

### 1. Autograd-Detached Dynamic Sample Weighting
For mini-batch $\mathcal{B} = \{(x_i, y_i)\}_{i=1}^B$ with predicted probability vectors $p_i = \text{softmax}(z_i) \in \Delta^{C-1}$:

$$w_i = \text{detach}\Big( (1 - p_{i, y_i}) + \beta \cdot \text{Var}_K(p_i) \cdot \mathbf{1}(p_{i, y_i} > \tau) + \gamma_{y_i} \Big)$$

where:
* **$(1 - p_{i, y_i})$**: Dynamic confidence-inverse weight (down-weights noisy/overconfident mislabeled samples).
* **$\beta \cdot \text{Var}_K(p_i) \cdot \mathbf{1}(p_{i, y_i} > \tau)$**: Historical variance gate over $K=5$ epochs active above $\tau=0.70$ (identifies fluctuating boundary samples).
* **$\gamma_{y_i} = \frac{1/N_{y_i}}{\sum_c 1/N_c}$**: Normalized static inverse-class weight (counteracts class imbalance).

### 2. Invariant Batch Normalization
$$\hat{w}_i = \frac{w_i}{\sum_{j=1}^B w_j + \epsilon} \cdot B \implies \frac{1}{B}\sum_{i=1}^B \hat{w}_i \equiv 1.0$$

### 3. Exact Analytical Loss & Gradient
$$\mathcal{L}_{\text{CCR}} = \frac{1}{B}\sum_{i=1}^B \hat{w}_i \, \ell_{\text{CE}}(z_i, y_i) \implies \frac{\partial \mathcal{L}_{\text{CCR}}}{\partial z_{ik}} = \frac{1}{B} \hat{w}_i \Big( p_{ik} - \mathbf{1}(y_i = k) \Big)$$

### 4. Theoretical Supremum Bound ($S/B$)
$$\sup w_i \le \max(1 - p_y) + \beta \cdot \max(\text{Var}_K) + \max(\gamma_y) = 1.0 + (0.50 \times 0.25) + 1.0 = \mathbf{2.125}$$
$$\sup \left( \frac{1}{B}\sum_{i=1}^B w_i \right) \le \mathbf{2.125}$$

---

## 📊 Benchmark Hierarchy (14 Audited Datasets Across 10 Domains)

All datasets undergo strict fold-local preprocessing with zero leakage and exact metadata verification:

| Category | Dataset | Samples ($N$) | Features ($D$) | Classes ($C$) | Imbalance Ratio (IR) | Domain / Task |
|---|---|---|---|---|---|---|
| **Tier 1: Core 10 Benchmark** | `Adult` | 48,842 | 14 | 2 | 3.17:1 | Census / Income |
| | `Bank` | 45,211 | 16 | 2 | 7.55:1 | Marketing / Finance |
| | `Electricity` | 45,312 | 8 | 2 | 1.36:1 | Energy Demand |
| | `MAGIC` | 19,020 | 10 | 2 | 1.84:1 | Gamma Physics |
| | `Churn` | 5,000 | 20 | 2 | 6.07:1 | Telecom Retention |
| | `Phoneme` | 5,404 | 5 | 2 | 2.41:1 | Acoustics / Speech |
| | `Spambase` | 4,601 | 57 | 2 | 1.54:1 | Text Processing |
| | `WILT` | 4,839 | 5 | 2 | 17.54:1 | Forestry Remote Sensing |
| | `Credit-G` | 1,000 | 20 | 2 | 2.33:1 | Financial Credit |
| | `Ionosphere` | 351 | 34 | 2 | 1.79:1 | Radar Signals |
| **Tier 4: Multiclass ($C \ge 3$)** | `Segment` | 2,310 | 19 | **7** | 1.00:1 | Image Vision |
| | `Vehicle` | 846 | 18 | **4** | 1.10:1 | Silhouette Vision |
| **Tier 5: Real-World Clinical** | `Heart Disease` | 462 | 9 | 2 | 1.89:1 | Clinical Cardiology |
| | `Breast Cancer` | 286 | 9 | 2 | 2.36:1 | Clinical Pathology |

---

## 🧱 Supported Models & Architectures

### 1. Neural Architectures (PyTorch)
* **`TabularMLP`**: Feed-forward linear layers `[256, 128, 64]`, Batch Normalization, ReLU, Dropout ($p=0.30$).
* **`TabularResNet`**: 4 Pre-activation Residual Blocks (dim 128), Layer Normalization, ReLU, Dropout ($p=0.10$).
* **`TabularFTTransformer`**: Feature Tokenizer ($d_{\text{embed}}=64$), 3 Transformer Encoder Layers, 4 Heads, FFN Multiplier 4/3, Dropout ($p=0.10$).

### 2. Loss Functions Evaluated (12 Formulations)
* **Baselines**: Standard Cross-Entropy (`ce`), Weighted CE (`wce`), Normalized WCE (`norm_wce`).
* **Noise-Robust Losses**: Focal Loss (`focal`), Generalized CE (`gce`), Symmetric CE (`sce`), Early-Learning Regularization (`elr`).
* **Dynamic Variants**: Plain Dynamic CE (`dynamic_ce`), Normalized Dynamic CE (`norm_dynamic_ce`), CCR Ablation without Normalization (`ccr_no_norm`), Full CCR (`ccr`).

### 3. Tree-based GBDT Baselines
* **`XGBoost`**, **`LightGBM`**, **`CatBoost`**.

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
   (Max N_cpu - 3 Workers)    • Automatic FP16 AMP
                              • 20% Safe VRAM Res.
```

### CLI Commands:

```powershell
# ── System Diagnostics & Audits ────────────────────────────────────────────
python main.py --resource_report    # Audit logical CPU cores, GPU safe VRAM budget, AMP
python main.py --validate           # Automated 5-point scientific consistency check
python main.py --dry_run            # Preview execution matrix & device assignments
python main.py --smoke_test         # 5-Second diagnostic verification

# ── Publication Figure Generation ──────────────────────────────────────────
python main.py --figures            # Generates all Main Figures (1–7) & Supplementals (S1–S3)

# ── High-Throughput Execution ──────────────────────────────────────────────
python main.py --tier1 --fast       # Run Tier 1 Core-10 benchmark in fast mode
python main.py --tier3 --fast       # Run Tier 3 Architecture Transfer (MLP / ResNet / Transformer)
python main.py --all --fast         # 1-Go unified suite execution with checkpoint resumption
```

---

## 📈 Paper Figure Hierarchy

All figures are programmatically generated and stored in `outputs/plots/`:

* **Figure 1**: Method Architecture & Normalization Mechanism Schematic.
* **Figure 2**: Empirical $S/B$ Weight-Sum Distribution vs. Theoretical Bounds ([`figure2_sb_distribution.png`](outputs/plots/figure2_sb_distribution.png)).
* **Figure 3**: Observed Relationships among Batch Composition, $S/B$, and Gradient Volatility ([`figure3_observed_relationships.png`](outputs/plots/figure3_observed_relationships.png)).
* **Figure 4**: CCR vs CCR-NoNorm Optimization Trajectories ([`mechanism_dynamics_training.png`](outputs/plots/mechanism_dynamics_training.png)).
* **Figure 5**: 4-Panel Gradient Attribution & Lorenz Concentration Curve ([`figure5_gradient_attribution.png`](outputs/plots/figure5_gradient_attribution.png)).
* **Figure 6**: Optimizer Interaction under SGD vs Adam vs AdamW ([`figure6_optimizer_sensitivity.png`](outputs/plots/figure6_optimizer_sensitivity.png)).
* **Figure 7**: Core-10 Robustness Curves across Label Noise Severities.
* **Figure S1**: Full Pairwise Loss-Comparison Matrix Heatmap ([`figure_s1_full_loss_comparison.png`](outputs/plots/figure_s1_full_loss_comparison.png)).

---

## 🧪 Statistical Inference Framework

Primary statistical inference treats the **Dataset ($d$) as the independent observational unit**:
1. Computes matched dataset differences: $\Delta_d = \text{Macro-F1}_{\text{CCR}, d} - \text{Macro-F1}_{\text{baseline}, d}$.
2. Conducts cross-dataset **Paired Wilcoxon Signed-Rank Tests**.
3. Controls for multiplicity using **Benjamini-Hochberg False Discovery Rate (BH-FDR)** at $\alpha = 0.05$.
4. Reports Mean $\Delta_d$, Median $\Delta_d$, 95% Confidence Intervals, and Paired Cohen's $d$.

---

## ⚙️ Installation & Testing

```powershell
# Clone the repository
git clone https://github.com/mdshoaibuddinchanda/CCR-Tabular.git
cd CCR-Tabular

# Create virtual environment
conda create -n py312 python=3.12 -y
conda activate py312

# Install dependencies
pip install -r requirements.txt

# Run test suite (80 unit tests)
pytest tests/ -v

# Run automated scientific consistency validation
python main.py --validate
```

---

<div align="center">

**CCR-Tabular: Rigorous, Transparent, and Reproducible Tabular Learning.**  
Licensed under the [MIT License](LICENSE).

</div>
