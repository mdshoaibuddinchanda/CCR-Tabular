# Dataset Characteristics and Provenance

This document outlines the exact characteristics of the 20-dataset benchmark used in the CCR-Tabular experiments.

## Summary

| Dataset | Domain | Instances | Features | Minority Ratio |
|---|---|---|---|---|
| adult | Business / Finance | 48,842 | 15 | 23.9% |
| aps_failure | Industrial | 75,471 | 171 | 1.8% |
| bank | Marketing | 45,211 | 17 | 11.7% |
| breast_cancer | Healthcare | 699 | 10 | 34.5% |
| census_kdd | Business / Finance | 99,999 | 42 | 6.2% |
| covertype | Environmental | 99,999 | 55 | 48.8% |
| credit_fraud | Finance | 99,999 | 30 | 0.2% |
| credit_g | Finance | 1,000 | 21 | 30.0% |
| default_credit | Finance | 30,000 | 24 | 22.1% |
| diabetes | Healthcare | 768 | 9 | 34.9% |
| electricity | Energy | 45,312 | 9 | 42.5% |
| heart_disease | Healthcare | 270 | 14 | 44.4% |
| higgs | Physics | 100,000 | 29 | 47.0% |
| ionosphere | Scientific | 351 | 35 | 35.9% |
| kddcup99 | Cybersecurity | 99,999 | 42 | 19.7% |
| magic | Scientific | 19,020 | 11 | 35.2% |
| mammography | Healthcare | 11,183 | 7 | 2.3% |
| mushroom | Biological | 8,124 | 23 | 48.2% |
| phoneme | Signal Processing | 5,404 | 6 | 29.3% |
| sonar | Scientific | 208 | 61 | 46.6% |
| spambase | Communication | 4,601 | 58 | 39.4% |

## Preprocessing and Binarization Strategies

The CCR-Tabular framework inherently requires strictly binary classification tasks. Several datasets were acquired via OpenML as multiclass or highly scaled distributions and were deterministically transformed as follows:

### 1. Covertype

**Transformation:** To convert Covertype into a binary classification task compatible with the CCR framework, Class 2 (Lodgepole Pine), the dominant forest-cover category, was treated as the positive class and all remaining categories were merged into the negative class.
**Resulting Imbalance:** 48.8% minority class.

### 2. KDDCup99

**Transformation:** The target encompasses 22 distinct attack categories. All attack vectors were aggregated into an `anomaly` class, and evaluated against the `normal` class.
**Resulting Imbalance:** 19.7% minority class.

### 3. Census KDD (Income)

**Transformation:** The explicit target column (`V42`) was manually bound during the OpenML fetch to capture the traditional `Income > 50K` classification task.
**Resulting Imbalance:** 6.2% minority class.

### 4. 100k Instance Subsetting

To preserve domain diversity without incurring catastrophic runtime inflation during hyper-parameter grid searches across 10 architectures, all datasets natively exceeding 100,000 rows were deterministically capped.
**Method:** Stratified random sample (`n=100,000`, `random_state=42`).
**Affected Datasets:** `covertype`, `kddcup99`, `census_kdd`, `credit_fraud`, `higgs`.
