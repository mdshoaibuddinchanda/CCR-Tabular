<div align="center">

# CCR-Tabular

**Confidence-Calibrated Reweighting with Invariant Batch Normalization for Robust Tabular Learning**

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20(CUDA%20%7C%20CPU)-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-87%20passed%20(100%25)-brightgreen?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![Dataset Audit](https://img.shields.io/badge/Dataset%20Audit-14%2F14%20PASSED-success?style=for-the-badge)](outputs/tables/table1_dataset_taxonomy.csv)

[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-189fdd?style=flat-square)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.1+-2980b9?style=flat-square)](https://lightgbm.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2+-yellow?style=flat-square)](https://catboost.ai/)
[![Statistical Inference](https://img.shields.io/badge/Stats-Paired%20Wilcoxon%20%2B%20BH--FDR-blueviolet?style=flat-square)](src/utils/statistics.py)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Pre--registered%20Seeds%20(42%2C%20123%2C%202024)-informational?style=flat-square)](src/utils/reproducibility.py)
[![Resource Controller](https://img.shields.io/badge/Heterogeneous-Auto%20AMP%20%2B%20OOM%20Recovery-blue?style=flat-square)](main.py)

*An empirical and theoretical framework investigating dynamic sample reweighting, per-batch gradient scale invariance, and noise robustness on tabular deep neural architectures.*

</div>

---

## 📌 Overview

Tabular datasets in high-stakes domains (finance, healthcare, fraud detection) routinely exhibit **concurrent class imbalance** and **asymmetric label noise**. Under asymmetric corruption (where minority samples are mislabeled as majority instances), standard loss functions treat noisy instances as "hard examples" and paradoxically amplify corrupted gradients, collapsing minority-class recall.

**Confidence-Calibrated Reweighting (CCR)** is an autograd-detached loss function and optimization framework that separates two foundational mechanisms:
1. **Mechanism A (Directional Reweighting)**: Dynamically depresses the learning signal of corrupted and overconfident instances while preserving minority-class representation.
2. **Mechanism B (Gradient Scale Invariance)**: Per-batch weight normalization enforces a unit batch-mean weight ($\frac{1}{B}\sum \hat{w}_i \approx 1.0$), eliminating batch-composition-dependent optimization step distortion and reducing gradient norm volatility by **10% to 35%**.

---

## 🔬 Key Scientific & Theoretical Contributions

* **Analytical Supremum Bound**: We prove that for all probability vectors and variance states, the batch weight sum ratio satisfies:
  $$\sup\left(\frac{1}{B}\sum_{i=1}^B w_i\right) \le 1.0 + (0.50 \times 0.25) + 1.0 = \mathbf{2.125}$$
* **Empirical Weight Deflation Telemetry**: Across 450 full training trajectories, real tabular batches predominantly exhibit *weight deflation* ($S/B \in [0.327, 1.022]$ with $P_{99} \le 0.985$).
* **Superior Robustness under Severe Noise**: Under 40% asymmetric label noise, CCR achieves **0.6690 minority recall** versus **0.5478 for Cross-Entropy** (+12.12 pp / 22.1% relative gain) and **~0.39–0.42 for tree ensembles** (LightGBM 0.3897, CatBoost 0.4080, XGBoost 0.4156), while gaining **+4.86 pp Macro-F1** over CE with zero clean-data penalty (+0.15 pp).
* **Mechanistic Gradient Attribution**: Directly demonstrates that CCR attenuates corrupted-sample gradient mass ($R_{\text{noise}}$ down by 15%–25%) and stabilizes gradient norm variance across training iterations.

---

## 📐 Mathematical Formulation

### 1. Autograd-Detached Dynamic Sample Weighting
For mini-batch $\mathcal{B} = \{(x_i, y_i)\}_{i=1}^B$ with predicted probability vectors $p_i = \text{softmax}(z_i) \in \Delta^{C-1}$:

$$w_i = \text{detach}\Big( (1 - p_{i, y_i}) + \beta \cdot \text{Var}_K(p_i) \cdot \mathbf{1}(p_{i, y_i} > \tau) + \gamma_{y_i} \Big)$$

where:
* **$(1 - p_{i, y_i})$**: Dynamic confidence-inverse penalty (down-weights noisy/overconfident mislabeled samples).
* **$\beta \cdot \text{Var}_K(p_i) \cdot \mathbf{1}(p_{i, y_i} > \tau)$**: Historical variance gate over $K=5$ epochs active above $\tau=0.30$ (identifies fluctuating boundary samples).
* **$\gamma_{y_i} = \frac{1/N_{y_i}}{\sum_c 1/N_c}$**: Normalized static inverse-class weight (counteracts class imbalance).

### 2. Invariant Batch Normalization
$$\hat{w}_i = \frac{w_i}{\sum_{j=1}^B w_j + \epsilon} \cdot B \implies \frac{1}{B}\sum_{i=1}^B \hat{w}_i = \frac{S}{S + \epsilon} \xrightarrow{\epsilon \to 0} 1.0$$

### 3. Exact Analytical Loss & Gradient
$$\mathcal{L}_{\text{CCR}} = \frac{1}{B}\sum_{i=1}^B \hat{w}_i \, \ell_{\text{CE}}(z_i, y_i) \implies \frac{\partial \mathcal{L}_{\text{CCR}}}{\partial z_{ik}} = \frac{1}{B} \hat{w}_i \Big( p_{ik} - \mathbf{1}(y_i = k) \Big)$$

---

## 📊 Benchmark Dataset Taxonomy ($10 + 2 + 2$ Hierarchy)

All datasets undergo strict fold-local preprocessing with zero data leakage:

| Benchmark Tier | Dataset Name | Domain / Application | Total Samples ($N$) | Features ($D$) | Imbalance Ratio (IR) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Core-10 Binary** | `Adult Census` | Census / Demographics | 48,842 | 14 | 3.17 : 1 |
| | `Bank Marketing` | Banking / Telemarketing | 45,211 | 16 | 7.55 : 1 |
| | `Electricity` | Energy Market Pricing | 45,312 | 8 | 1.35 : 1 |
| | `MAGIC Gamma` | High-Energy Physics | 19,020 | 10 | 1.84 : 1 |
| | `Customer Churn` | Telecom / Churn Analysis | 7,043 | 19 | 2.77 : 1 |
| | `Phoneme` | Acoustics / Speech Signal | 5,404 | 5 | 2.41 : 1 |
| | `Spambase` | Text / Spam Filtering | 4,601 | 57 | 1.54 : 1 |
| | `WILT Forest` | Remote Sensing / Forestry | 4,839 | 5 | 17.54 : 1 |
| | `Credit-G` | Finance / Credit Scoring | 1,000 | 20 | 2.33 : 1 |
| | `Ionosphere` | Radar / Atmospheric Physics | 351 | 34 | 1.79 : 1 |
| **Multiclass ($C \ge 3$)** | `Image Segment` | Land-Cover Pixel Classification ($C=7$) | 2,310 | 19 | 1.00 : 1 |
| | `Vehicle` | Vehicle Silhouette Recognition ($C=4$) | 846 | 18 | 1.08 : 1 |
| **Clinical External** | `Heart Disease` | Clinical Cardiology ($C=2$) | 462 | 9 | 1.89 : 1 |
| | `Breast Cancer` | Clinical Oncology ($C=2$) | 286 | 9 | 2.36 : 1 |

---

## ⚡ Quick Start & Execution

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/mdshoaibuddinchanda/CCR-Tabular.git
cd CCR-Tabular

# Create virtual environment
conda create -n ccr python=3.11 -y
conda activate ccr

# Install dependencies
pip install -r requirements.txt
```

### 2. Reproduce All Paper Artifacts (1-Click)

```bash
# Generate all 7 publication figures (600 DPI vector PDF + PNG)
python main.py --figures

# Generate all 8 canonical manuscript tables (CSV + LaTeX)
python main.py --tables

# Run automated 5-point scientific consistency validator
python main.py --validate
```

### 3. Run Experimental Tiers

```bash
# 5-second end-to-end diagnostic verification
python main.py --smoke_test

# Run Tier 1 Core-10 benchmark across all 10 losses and noise regimes
python main.py --tier1 --fast

# Run Tier 2 Mechanism telemetry (S/B logging & gradient volatility)
python main.py --tier2

# Run Tier 3 Architecture Transferability (TabularMLP, ResNet, FT-Transformer)
python main.py --tier3 --fast

# Run Tier 4 Multiclass Benchmark (Segment & Vehicle)
python main.py --tier4

# Run Tier 5 Clinical External Validation
python main.py --tier5

# Run Full Master Suite (All Tiers sequentially in 1 go)
python main.py --all --fast
```

---

## 📁 Repository Structure

```text
CCR-Tabular/
├── ccrtatex/                         # LaTeX manuscript source & figures
│   ├── figures/                     # Vector PDF & 600 DPI PNG publication figures
│   │   ├── fig1_schematic.pdf       # Figure 1: Pipeline schematic
│   │   ├── fig1a_macro_f1.pdf       # Figure 2a: Macro-F1 degradation
│   │   ├── fig1b_minority_recall.pdf# Figure 2b: Minority recall retention
│   │   ├── fig4_ccr_vs_xgboost.pdf  # Figure 3: Tree comparison
│   │   ├── fig6_ablation.pdf        # Figure 4: Component ablation
│   │   ├── fig5_gradient_attribution.pdf # Figure 5: Mechanistic gradient attribution
│   │   ├── fig6_optimizer_sensitivity.pdf# Figure 6: Optimizer study
│   │   ├── fig8a_k_beta_heatmap.pdf # Figure 7a: Hyperparameter heatmap
│   │   └── fig8b_beta_marginal.pdf  # Figure 7b: Marginal beta curve
│   ├── main.tex                     # Master LaTeX manuscript
│   └── references.bib               # BibTeX bibliography database
├── data/                            # Dataset cache (downloaded dynamically from OpenML)
├── experiments/                     # Execution tier entrypoints
├── scripts/                         # Master figure and table generation scripts
│   ├── generate_all_paper_figures.py# Master 7-Figure generator
│   ├── generate_all_paper_tables.py # Master 8-Table generator
│   └── generate_figure1_schematic.py# Dedicated Figure 1 schematic generator
├── src/                             # Core modular package
│   ├── analysis/                    # Statistical aggregation and validation
│   ├── data/                        # Leakage-free preprocessing & noise injection
│   ├── loss/                        # CCR loss and 9 baseline loss implementations
│   ├── models/                      # TabularMLP, TabularResNet, FT-Transformer, GBDT
│   ├── training/                    # Cross-validation engine and telemetry logger
│   └── utils/                       # Configs, metrics, and reproducibility seeds
├── tests/                           # Pytest unit and regression test suite (14 test modules)
├── main.py                          # Master heterogeneous parallel orchestrator CLI
├── requirements.txt                 # Project dependencies
└── README.md                        # Documentation
```

---

## 🧪 Testing & Verification

Run the full automated test suite (87 test cases across autograd detachment, noise generation, numerical bounds, and data leakage):

```bash
pytest tests/ -v
```

---

## 📜 Citation

If you find this codebase or method useful in your research, please cite:

```bibtex
@article{chanda2026ccr,
  title     = {Confidence-Calibrated Reweighting with Invariant Batch Normalization for Robust Tabular Deep Learning},
  author    = {Chanda, Md Shoaibuddin},
  journal   = {Neurocomputing},
  year      = {2026}
}
```

---

<div align="center">
<b>MIT License</b> &bull; CCR-Tabular Research Group
</div>
