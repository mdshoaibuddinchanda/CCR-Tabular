# When Reweighting Meets Scale Invariance: Robust Tabular Neural Classification Under Label Noise and Class Imbalance

> **Document Status**: Canonical Research Manuscript & Ground Truth Reference.  
> **Repository**: [mdshoaibuddinchanda/CCR-Tabular](https://github.com/mdshoaibuddinchanda/CCR-Tabular)  
> **Author**: Md Shoaib Uddin Chanda  
> **Affiliation**: Department of CSE (AI/ML), Lords Institute of Engineering and Technology, Hyderabad (Osmania University)

---

## 1. Executive Summary & Settled Scientific Thesis

> **Central Scientific Thesis**:  
> *Dynamic confidence-aware reweighting is the principal driver of predictive robustness under label noise, while per-batch weight normalization eliminates batch-dependent global gradient scaling and enhances optimization stability.*

### Decisive Empirical Findings:
1. **Minority Recall Retention**: Under severe 40% asymmetric corruption, standard Cross-Entropy (CE) and gradient-boosted trees collapse into majority-class bias ($\text{Recall}_1 < 0.40$). **CCR preserves minority recall at 0.6690** (a **+12.12 percentage point / 22.1% relative improvement** over CE).
2. **Macro-F1 Robustness**: CCR improves Macro-F1 by **4.86 percentage points (6.37% relative)** over CE under 40% asymmetric noise ($0.8116$ vs $0.7630$), with standout dataset improvements of **+15.17 percentage points (27.0% relative)** on Bank Marketing and **+10.90 percentage points (16.34% relative)** on Adult Census.
3. **Clean-Data Safety**: On 100% uncorrupted data, CCR achieves $0.8450$ vs $0.8435$ for CE, introducing **no meaningful clean-data penalty (+0.15 percentage points)**.
4. **Refutation of 3–4× Weight Inflation**: Empirical telemetry across 450 real-world training runs proves that real tabular training exhibits **batch deflation ($S/B \in [0.327, 1.022]$ with $P_{99} \le 0.96$)**, bounded by an analytical theoretical supremum $\sup(S/B) \le 2.125$.

---

## 2. Mathematical Formulations & Exact Analytical Proofs

### 2.1 Autograd-Detached Dynamic Sample Weighting
For mini-batch $\mathcal{B} = \{(x_i, y_i)\}_{i=1}^B$ with predicted probability vector $p_i = \text{softmax}(z_i) \in \Delta^{C-1}$:
$$w_i = \text{detach}\Big( (1 - p_{i, y_i}) + \beta \cdot \text{Var}_K(p_i) \cdot \mathbf{1}(p_{i, y_i} > \tau) + \gamma_{y_i} \Big)$$
where:
* $(1 - p_{i, y_i})$ is the dynamic confidence-inverse penalty.
* $\beta \cdot \text{Var}_K(p_i) \cdot \mathbf{1}(p_{i, y_i} > \tau)$ is the temporal prediction variance over $K=5$ epochs active above $\tau=0.30$.
* $\gamma_{y_i} = \frac{1/N_{y_i}}{\sum_c 1/N_c}$ is the normalized static inverse-class weight ($\sum_c \gamma_c = 1.0$).

### 2.2 Invariant Batch Normalization
$$\hat{w}_i = B \cdot \frac{w_i}{\sum_{j=1}^B w_j + \epsilon} \implies \frac{1}{B}\sum_{i=1}^B \hat{w}_i \equiv 1.0$$

### 2.3 Exact Analytical Gradient
$$\mathcal{L}_{\text{CCR}} = \frac{1}{B}\sum_{i=1}^B \hat{w}_i \, \ell_{\text{CE}}(z_i, y_i) \implies \frac{\partial \mathcal{L}_{\text{CCR}}}{\partial z_{ik}} = \frac{1}{B} \hat{w}_i \Big( p_{ik} - \mathbf{1}(y_i = k) \Big)$$

### 2.4 Theoretical Supremum Bound on $S/B$
$$\sup w_i \le \max(1 - p_y) + \beta \cdot \max(\text{Var}_K) + \max(\gamma_y) = 1.0 + (0.50 \times 0.25) + 1.0 = \mathbf{2.125}$$
$$\sup \left( \frac{S}{B} \right) = \frac{1}{B}\sum_{i=1}^B \sup w_i \le \mathbf{2.125}$$

---

## 3. The 14-Dataset Benchmark Taxonomy (10 + 2 + 2)

| Tier | Dataset | OpenML ID | Samples ($N$) | Features ($D$) | Classes ($C$) | Imbalance Ratio | Domain / Task |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Tier 1: Core 10** | **Adult** | 1590 | 48,842 | 14 | 2 | 3.17:1 | Socioeconomic Census Income |
| | **Bank Marketing** | 1461 | 45,211 | 16 | 2 | 7.55:1 | Finance Term Deposit Subscription |
| | **Electricity** | 151 | 45,312 | 8 | 2 | 1.36:1 | Energy Demand Market Pricing |
| | **MAGIC Gamma** | 1120 | 19,020 | 10 | 2 | 1.84:1 | Atmospheric Gamma Ray Physics |
| | **Customer Churn** | 40701 | 5,000 | 20 | 2 | 6.07:1 | Telecom Subscriber Retention |
| | **Phoneme** | 1489 | 5,404 | 5 | 2 | 2.41:1 | Acoustic Speech Signal |
| | **Spambase** | 44 | 4,601 | 57 | 2 | 1.54:1 | Email Text Classification |
| | **WILT** | 40983 | 4,839 | 5 | 2 | **17.54:1** | Forestry Remote Sensing |
| | **Credit-G** | 31 | 1,000 | 20 | 2 | 2.33:1 | Financial Credit Lending |
| | **Ionosphere** | 59 | 351 | 34 | 2 | 1.79:1 | High-Altitude Radar Signal |
| **Tier 4: Multiclass** | **Segment** | 36 | 2,310 | 19 | **7** | 1.00:1 | Image Vision Segmentation |
| | **Vehicle** | 54 | 846 | 18 | **4** | 1.10:1 | Silhouette Vision Geometry |
| **Tier 5: Clinical** | **Heart Disease** | 1498 | 462 | 9 | 2 | 1.89:1 | Clinical Cardiology Diagnosis |
| | **Breast Cancer** | 13 | 286 | 9 | 2 | 2.36:1 | Clinical Pathology Biopsy |

---

## 4. The 7 Manuscript Tables Summary

### Table 1: Dataset Taxonomy (10 + 2 + 2 Design)
See Section 3 above.

### Table 2: Primary Robustness Benchmark (Macro-F1 & Minority Recall)

| Model | Clean ($0\%$) F1 | Asym $20\%$ F1 | Asym $40\%$ F1 | Sym $20\%$ F1 | Clean ($0\%$) Rec | Asym $40\%$ Rec |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CCR (Proposed)** | **0.8450** | **0.8350** | **0.8116** | **0.8147** | **0.7836** | **0.6690** |
| CCR-NoNorm | 0.8431 | 0.8312 | 0.8074 | 0.8110 | 0.7790 | 0.6612 |
| Standard CE | 0.8435 | 0.8216 | 0.7630 | 0.8171 | 0.7504 | 0.5478 |
| Focal Loss | 0.8396 | 0.8156 | 0.7598 | 0.8075 | 0.7449 | 0.5393 |
| Early-Learning Reg (ELR) | 0.8420 | 0.8305 | 0.7980 | 0.8155 | 0.7640 | 0.6350 |
| XGBoost-Default | 0.8345 | 0.8071 | 0.7134 | 0.7721 | 0.7182 | 0.4156 |
| LightGBM-Default | 0.8375 | 0.7950 | 0.6931 | 0.8072 | 0.7292 | 0.3897 |
| FT-Transformer | 0.8148 | 0.7935 | 0.6988 | 0.7762 | 0.6762 | 0.3974 |

### Table 3: Statistical Significance (Paired Wilcoxon & BH-FDR)

| Comparison | Mean $\Delta$ Macro-F1 | 95% Conf. Interval | Wilcoxon $p$ | BH-FDR $q$ | Cohen's $d_z$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| CCR vs Standard CE | **+0.0486** | [+0.0312, +0.0660] | $0.0019$ | $0.0038$ | 1.42 (Large) |
| CCR vs Focal Loss | **+0.0518** | [+0.0345, +0.0691] | $0.0019$ | $0.0038$ | 1.51 (Large) |
| CCR vs GCE ($q=0.7$) | **+0.0296** | [+0.0142, +0.0450] | $0.0058$ | $0.0087$ | 1.05 (Large) |
| CCR vs SCE | **+0.0201** | [+0.0078, +0.0324] | $0.0137$ | $0.0164$ | 0.89 (Large) |
| CCR vs ELR | **+0.0136** | [+0.0031, +0.0241] | $0.0273$ | $0.0273$ | 0.71 (Medium) |
| CCR vs CCR-NoNorm | **+0.0042** | [+0.0011, +0.0073] | $0.0371$ | $0.0371$ | 0.58 (Medium) |

### Table 4: Batch Telemetry & Gradient Stability

| Regime | Measured $S/B$ Range | $P_{99}(S/B)$ | Grad CV ($\downarrow$) | Corrupted Mass $R_{\text{noise}}$ ($\downarrow$) |
| :--- | :---: | :---: | :---: | :---: |
| Clean ($0\%$) | $[0.842, 1.022]$ | $0.985$ | 0.342 | 0.000 |
| Asym $20\%$ | $[0.512, 0.945]$ | $0.912$ | 0.388 | 0.184 (vs 0.231 CE) |
| Asym $40\%$ | $[0.327, 0.884]$ | $0.856$ | 0.412 | 0.245 (vs 0.328 CE) |

### Table 5: Optimizer Dynamics (SGD vs Adam vs AdamW)

| Optimizer | CCR (Normalized) | CCR-NoNorm | Normalization Gain ($\Delta$) |
| :--- | :---: | :---: | :---: |
| SGD ($\text{lr}=0.01$) | **0.7842** | 0.7410 | **+0.0432 (+4.32 pp)** |
| Adam ($\text{lr}=0.001$) | **0.8095** | 0.8058 | +0.0037 (+0.37 pp) |
| AdamW ($\text{lr}=0.001$) | **0.8116** | 0.8074 | +0.0042 (+0.42 pp) |

### Table 6: Architecture Transferability

| Architecture | Standard CE | CCR (Ours) | Gain ($\Delta$) |
| :--- | :---: | :---: | :---: |
| TabularMLP | 0.7630 | **0.8116** | +0.0486 (+4.86 pp) |
| TabularResNet | 0.7712 | **0.8184** | +0.0472 (+4.72 pp) |
| TabularFTTransformer | 0.6988 | **0.7645** | **+0.0657 (+6.57 pp)** |

### Table 7: Multiclass ($C \ge 3$) & Clinical External Validation

| Tier | Dataset | Standard CE | CCR (Ours) | Macro-F1 Gain |
| :--- | :--- | :---: | :---: | :---: |
| **Tier 4: Multiclass** | **Segment ($C=7$)** | 0.8720 | **0.8945** | +0.0225 (+2.25 pp) |
| | **Vehicle ($C=4$)** | 0.7610 | **0.7830** | +0.0220 (+2.20 pp) |
| **Tier 5: Clinical** | **Heart Disease** | 0.7443 | **0.7463** | +0.0020 (+0.20 pp) |
| | **Breast Cancer** | 0.9497 | **0.9534** | +0.0037 (+0.37 pp) |

---

## 5. Declarations & Compliance

* **Generative AI Use**: Large language models (ChatGPT and Gemini) were utilized strictly for initial LaTeX typesetting and prose drafting. All mathematical proofs, software architecture, data audits, statistical testing, and scientific verifications were independently conducted and verified by the author.
* **Biography**: Removed.
