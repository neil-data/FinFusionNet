# FinFusionNet: Audited Research Track & Head-to-Head Report (v0.1-audited)

## 1. Executive Summary
This report presents the fully audited, leak-free empirical findings of FinFusionNet across both **XGBoost** and **PyTorch LSTM** model families for next-session directional prediction in the Indian equity market (NIFTY 50 top 10 heavyweights, 2020–2026).

All models were evaluated under strict parity using:
- **Identical Common Evaluation Cohort**: 10 heavyweights, 45 monthly expanding folds, exactly 915 unique trading dates (2023-01 to 2026-09).
- **Matched Feature Groups**: Group A (Technicals) + Group B (Market Context) present across all sentiment models (X3–X6 and L3–L6).
- **Leak-Free Partitioning**: Monotonic chronological boundaries, strictly enforced target-date purging.
- **Correlation-Safe Inference**: Paired date-block bootstrap (10,000 resamples, $L=10$ days) with Holm-Bonferroni correction and strict p/CI mathematical consistency.

---

## 2. Walk-Forward Performance Matrix (Protocol B: 45 Monthly Folds, 915 Test Dates)

### XGBoost Research Track (Audited)
| Model ID | Configuration | Date-Level Acc | Pooled Acc | Pooled F1 | ROC-AUC | Log Loss | Lift vs Always-Up | Lift vs Prev-Dir |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **X1** | Technical-only baseline | **0.4980** | 0.4975 | 0.6458 | 0.5038 | 0.6946 | +0.0008 | +0.0091 |
| **X2** | Technical + market context | **0.5002** | 0.4997 | 0.6256 | 0.5184 | 0.6940 | +0.0030 | +0.0113 |
| **X3** | Technical + market context + news availability control | **0.4993** | 0.4989 | 0.6277 | 0.5168 | 0.6942 | +0.0021 | +0.0104 |
| **X4** | Technical + market context + news availability + VADER | **0.5015** | 0.5010 | 0.6281 | 0.5171 | 0.6940 | +0.0043 | +0.0126 |
| **X5** | Technical + market context + news availability + FinBERT | **0.5004** | 0.4999 | 0.6263 | 0.5169 | 0.6939 | +0.0032 | +0.0115 |
| **X6** | Technical + market context + news availability + VADER + FinBERT | **0.4995** | 0.4990 | 0.6306 | 0.5152 | 0.6940 | +0.0022 | +0.0105 |

### LSTM Research Track (Audited)
| Model ID | Configuration | Date-Level Acc | Pooled Acc | Pooled F1 | ROC-AUC | Log Loss | Lift vs Always-Up | Lift vs Prev-Dir |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **L1** | Technical-only sequential baseline | **0.4984** | 0.4974 | 0.6213 | 0.5040 | 0.6947 | +0.0008 | +0.0089 |
| **L2** | Technical + market context | **0.4941** | 0.4931 | 0.5995 | 0.4930 | 0.6969 | -0.0035 | +0.0046 |
| **L3** | Technical + market context + news availability control | **0.4893** | 0.4883 | 0.5766 | 0.4922 | 0.6961 | -0.0083 | -0.0002 |
| **L4** | Technical + market context + news availability + VADER | **0.5019** | 0.5009 | 0.6219 | 0.5010 | 0.6949 | +0.0043 | +0.0124 |
| **L5** | Technical + market context + news availability + FinBERT | **0.4950** | 0.4940 | 0.6143 | 0.4975 | 0.6965 | -0.0026 | +0.0055 |
| **L6** | Technical + market context + news availability + VADER + FinBERT | **0.4926** | 0.4916 | 0.5948 | 0.4924 | 0.6961 | -0.0050 | +0.0031 |

---

## 3. Direct Head-to-Head Comparison: XGBoost vs LSTM (Parity Matched)

All evaluations conducted on identical out-of-sample trading dates ($N=915$). Difference is defined as $\Delta = 	ext{Acc}_{	ext{LSTM}} - 	ext{Acc}_{	ext{XGB}}$:

| Feature Tier | XGBoost ID | LSTM ID | XGB Date Acc | LSTM Date Acc | Diff (LSTM - XGB) | 95% Bootstrap CI | Raw p-val | Holm Adj p | Statistically Different? |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Technical Only Baseline | **X1** | **L1** | 0.4980 | 0.4984 | **+0.0003** | [-0.0102, +0.0092] | 0.9526 | 1.0000 | **NO** |
| Technical + Market Context | **X2** | **L2** | 0.5002 | 0.4941 | **-0.0061** | [-0.0219, +0.0074] | 0.3428 | 1.0000 | **NO** |
| Technical + Context + News Control | **X3** | **L3** | 0.4993 | 0.4893 | **-0.0101** | [-0.0272, +0.0037] | 0.1390 | 0.8340 | **NO** |
| Technical + Context + News Control + VADER | **X4** | **L4** | 0.5015 | 0.5019 | **+0.0003** | [-0.0142, +0.0117] | 0.8188 | 1.0000 | **NO** |
| Technical + Context + News Control + FinBERT | **X5** | **L5** | 0.5004 | 0.4950 | **-0.0055** | [-0.0210, +0.0064] | 0.2918 | 1.0000 | **NO** |
| Technical + Context + News Control + Combined Sentiment | **X6** | **L6** | 0.4995 | 0.4926 | **-0.0069** | [-0.0219, +0.0060] | 0.2386 | 1.0000 | **NO** |

---

## 4. XGBoost Incremental Feature Hypotheses (With Holm-Bonferroni Correction)

| Comparison | Description | Mean Diff | 95% Bootstrap CI | Raw p-value | Holm Adj p | Significant at $\alpha=0.05$? |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **X2_vs_X1** | H1: Market Context vs Technicals | **+0.0022** | [-0.0093, +0.0140] | 0.6920 | 1.0000 | **NO** |
| **X3_vs_X2** | H2: News Control vs Market Context | **-0.0009** | [-0.0083, +0.0056] | 0.7680 | 1.0000 | **NO** |
| **X4_vs_X3** | H3: VADER Sentiment vs News Control | **+0.0022** | [-0.0046, +0.0097] | 0.5212 | 1.0000 | **NO** |
| **X5_vs_X3** | H4: FinBERT Sentiment vs News Control (Primary) | **+0.0011** | [-0.0062, +0.0087] | 0.7292 | 0.7292 | **NO** |
| **X6_vs_X3** | H5: Combined Sentiment vs News Control | **+0.0001** | [-0.0072, +0.0081] | 0.9384 | 1.0000 | **NO** |
| **X6_vs_X4** | H6: Combined vs VADER | **-0.0021** | [-0.0116, +0.0072] | 0.6668 | 1.0000 | **NO** |
| **X6_vs_X5** | H7: Combined vs FinBERT | **-0.0010** | [-0.0104, +0.0085] | 0.8434 | 1.0000 | **NO** |

---

## 5. Primary Pre-Registered Hypothesis Testing: FinBERT (X5 vs X3)
- **Comparator**: Technical + Market Context + News Control (X3) vs + FinBERT Sentiment (X5)
- **Observed Mean Difference**: **+0.0011**
- **Block-Length Sensitivity ($L$ trading days)**:

| Block Length ($L$) | Mean Difference | 95% Bootstrap CI | Two-Sided p-value | Significant at $\alpha=0.05$? |
|:---:|:---:|:---:|:---:|:---:|
| 3 | +0.0011 | [-0.0061, +0.0085] | 0.7460 | NO |
| 5 | +0.0011 | [-0.0066, +0.0091] | 0.7636 | NO |
| 10 | +0.0011 | [-0.0063, +0.0089] | 0.7272 | NO |
| 15 | +0.0011 | [-0.0064, +0.0091] | 0.7312 | NO |
| 20 | +0.0011 | [-0.0064, +0.0094] | 0.7044 | NO |

---

## 6. Key Scientific Conclusions
1. **Resolution of the X2 Anomaly**:
   - The initial ~56% accuracy for XGBoost X2 was an artifact of unsorted `.iloc` integer slicing in walk-forward evaluation.
   - When fixed, XGBoost X2 achieves **0.5002**, which converges with LSTM L2 (**0.4941**). Market context alone provides no statistically defensible directional lift over pure technicals.
2. **Sentiment Value Under Leak-Free Parity**:
   - Neither VADER nor FinBERT sentiment provides statistically significant directional accuracy lift over news attention controls and market context under strict chronological walk-forward validation (Holm-adjusted $p > 0.05$).
3. **Tabular vs Sequential Models**:
   - Across all feature tiers, after correcting for multiple comparisons, there is no statistically significant difference between XGBoost and 2-layer LSTM classifiers for next-session directional prediction. Both model classes perform near the 49.2–50.2% range.
