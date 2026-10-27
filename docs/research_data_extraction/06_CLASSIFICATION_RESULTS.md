# FinFusionNet Research Extraction: Classification Performance Results

## 1. Consolidated Out-of-Sample Performance Table

All models are evaluated on the exact audited common evaluation cohort of **45,136 observations** across **916 trading dates** (2023-01-02 to 2026-09-29) and **56 unique constituents** across 45 monthly expanding folds (Protocol B).

| Model Configuration | Date Acc | Pooled Acc | Balanced Acc | Precision | Recall | F1 Score | ROC-AUC | Brier Score | Log Loss | Predicted Up % |
|---|---|---|---|---|---|---|---|---|---|---|
| **LSTM-F0 (Price-Only, Exp 1)** | **0.5108** | **0.5096** | **0.5091** | **0.5095** | 0.6297 | 0.5633 | **0.5149** | **0.2502** | **0.6935** | 62.07% |
| **LSTM-F1_VADER (Exp 3B)** | 0.5099 | 0.5088 | 0.5084 | 0.5092 | 0.6045 | 0.5528 | 0.5105 | 0.2504 | 0.6940 | 59.62% |
| **LR-F1_FinBERT (Exp 4)** | 0.5095 | 0.5083 | 0.5077 | 0.5081 | 0.6567 | 0.5729 | 0.5103 | 0.2505 | 0.6943 | 64.91% |
| **LR-F0 (Price-Only, Exp 1)** | 0.5094 | 0.5082 | 0.5075 | 0.5078 | 0.6724 | 0.5786 | 0.5102 | 0.2505 | 0.6942 | 66.49% |
| **XGB-F1_VADER (Exp 3B)** | 0.5054 | 0.5043 | 0.5040 | 0.5057 | 0.5622 | 0.5325 | 0.5090 | 0.2564 | 0.7069 | 55.82% |
| **LSTM-F1_FinBERT (Exp 4)** | 0.5053 | 0.5041 | 0.5038 | 0.5054 | 0.5842 | 0.5419 | 0.5092 | 0.2505 | 0.6942 | 58.04% |
| **XGB-F1_FinBERT (Exp 4)** | 0.5042 | 0.5042 | 0.5039 | 0.5057 | 0.5593 | 0.5311 | 0.5070 | 0.2568 | 0.7076 | 55.53% |
| *Baseline: Always-Up* | 0.5034 | 0.5021 | 0.5000 | 0.5021 | **1.0000** | **0.6686** | 0.5000 | 0.4979 | 8.0244 | 100.00% |
| **XGB-F0 (Price-Only, Exp 1)** | 0.5027 | 0.5015 | 0.4996 | 0.5019 | 0.9351 | 0.6532 | 0.5067 | 0.2503 | 0.6937 | 93.55% |
| **Cross-Attention Net (Exp 5)** | 0.5016 | 0.5004 | 0.4996 | 0.5019 | 0.6852 | 0.5794 | 0.5016 | 0.2506 | 0.6943 | 68.56% |
| *Baseline: Prev-Day Direction*| 0.4919 | 0.4930 | 0.4930 | 0.4952 | 0.4966 | 0.4959 | 0.4930 | 0.5070 | 8.1722 | 50.36% |

---

## 2. Key Empirical Findings on Direction Prediction

1. **Failure of Multimodal Lift**: Neither lexicon sentiment (VADER), domain-specific transformer sentiment (FinBERT early concatenation), nor Temporal Cross-Attention outperforms the simple price-only LSTM baseline (`LSTM-F0`).
2. **Top Performing Architecture**: The price-only 2-layer LSTM (`LSTM-F0`) achieved the highest date-level accuracy (**51.08%**), pooled accuracy (**50.96%**), ROC-AUC (**0.5149**), and lowest Brier score (**0.2502**).
3. **Underperformance of Cross-Attention Net**: Cross-Attention Net achieved **50.16%** date accuracy, trailing the naive Always-Up baseline (**50.34%**) and underperforming `LSTM-F0` by **-0.92%** ($p = 0.0430$ unadjusted Brier DM; $p = 0.0339$ 0-1 loss DM).
4. **Bull Bias Exploitation**: Most models exhibit heavy positive prediction bias (`predicted_up_share` ranging from 55.5% to 93.5%), exploiting the secular bull market of the Indian index over 2023-2026. XGB-F0 defaulted to predicting UP 93.55% of the time.

---

## 3. Subgroup and Market Regime Breakdown

Performance across distinct market environments reveals systemic fragility:

| Regime Category | Observation Count | Cross-Attention Net Date Acc | LSTM-F0 (Price-Only) Date Acc | LSTM-FinBERT Date Acc | XGB-F0 Date Acc |
|---|---|---|---|---|---|
| **High Volatility (Top 25% Vol)** | 11,303 (25.0%) | 48.41% | **49.63%** | 49.39% | 46.60% |
| **Normal Volatility (Mid 50% Vol)** | 22,603 (50.1%) | 50.74% | **51.20%** | **51.20%** | 50.24% |
| **Low Volatility (Bottom 25% Vol)** | 11,230 (24.9%) | 50.28% | 51.83% | 49.86% | **53.54%** |
| **Bull Sessions (Market Ret > 0)** | 24,619 (54.5%) | 50.57% | **51.30%** | 50.54% | 51.16% |
| **Bear Sessions (Market Ret < 0)** | 20,467 (45.3%) | 49.39% | **50.55%** | 50.25% | 48.94% |

**Key Insight**: All models experience substantial accuracy degradation in high-volatility regimes (dropping below 50.0%), and multimodal attention suffers worse degradation (48.41%) than price-only models.
