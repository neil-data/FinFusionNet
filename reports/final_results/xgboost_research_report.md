# FinFusionNet — XGBoost Research Track Final Report (Audited v0.1-audited)

## Executive Summary
This report details the audited empirical findings of the **XGBoost Experimental Track of FinFusionNet**, evaluating whether financial news sentiment (lexicon-based VADER and domain-adapted FinBERT) provides statistically defensible predictive lift for next-trading-session stock direction in the Indian equity market.

Evaluation was conducted under a strict **Common Evaluation Cohort** across **NIFTY 50 top 10 heavyweights** (2020–2026) using chronological expanding walk-forward validation (45 monthly folds, 915 unique trading dates) and paired date-block bootstrap testing.

---

## 1. Overall Performance Comparison (Protocol B: Expanding Window, 915 Test Dates)

| Model ID | Configuration | Date-Level Acc | Pooled Acc | Pooled F1 | ROC-AUC | Log Loss | Lift vs Always-Up | Lift vs Prev-Dir |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **X1** | Technical-only baseline | **0.4980** | 0.4975 | 0.6458 | 0.5038 | 0.6946 | +0.0008 | +0.0091 |
| **X2** | Technical + market context | **0.5002** | 0.4997 | 0.6256 | 0.5184 | 0.6940 | +0.0030 | +0.0113 |
| **X3** | Technical + market context + news availability control | **0.4993** | 0.4989 | 0.6277 | 0.5168 | 0.6942 | +0.0021 | +0.0104 |
| **X4** | Technical + market context + news availability + VADER | **0.5015** | 0.5010 | 0.6281 | 0.5171 | 0.6940 | +0.0043 | +0.0126 |
| **X5** | Technical + market context + news availability + FinBERT | **0.5004** | 0.4999 | 0.6263 | 0.5169 | 0.6939 | +0.0032 | +0.0115 |
| **X6** | Technical + market context + news availability + VADER + FinBERT | **0.4995** | 0.4990 | 0.6306 | 0.5152 | 0.6940 | +0.0022 | +0.0105 |

---

## 2. Pre-Registered Primary Hypothesis: FinBERT Contribution (X5 vs X3)

**Hypothesis**: Financial news sentiment extracted via FinBERT improves next-session stock direction prediction beyond technical indicators, market context, and news availability controls under Protocol B.
- **Metric**: Equal-weighted date-level accuracy ($Acc_d$)
- **Statistical Test**: Paired Date-Block Bootstrap (10,000 repetitions, block length $L=10$ trading dates)
- **Observed Mean Difference**: **+0.0011** (+0.11%)
- **95% Bootstrap Confidence Interval**: **[-0.0063, +0.0089]**
- **Two-Sided Empirical p-value**: **0.7272**
- **Statistically Significant at $\alpha = 0.05$**: **NO**

### Block-Length Sensitivity Analysis
| Block Length ($L$ days) | Mean Difference | 95% Bootstrap CI | Two-Sided p-value | Significant at $\alpha=0.05$ |
|:---:|:---:|:---:|:---:|:---:|
| 3 | +0.0011 | [-0.0061, +0.0085] | 0.7460 | NO |
| 5 | +0.0011 | [-0.0066, +0.0091] | 0.7636 | NO |
| 10 | +0.0011 | [-0.0063, +0.0089] | 0.7272 | NO |
| 15 | +0.0011 | [-0.0064, +0.0091] | 0.7312 | NO |
| 20 | +0.0011 | [-0.0064, +0.0094] | 0.7044 | NO |

---

## 3. Secondary Hypotheses & Holm-Bonferroni Correction

| Comparison | Description | Mean Diff | 95% Bootstrap CI | Raw p-value | Holm Adjusted p | Significant |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **X2_vs_X1** | Market Context vs Pure Technicals | +0.0022 | [-0.0093, +0.0140] | 0.6920 | 1.0000 | NO |
| **X3_vs_X2** | News Control vs Market Context | -0.0009 | [-0.0083, +0.0056] | 0.7680 | 1.0000 | NO |
| **X4_vs_X3** | VADER Sentiment vs News Control | +0.0022 | [-0.0046, +0.0097] | 0.5212 | 1.0000 | NO |
| **X6_vs_X3** | Combined Sentiment vs News Control | +0.0001 | [-0.0072, +0.0081] | 0.9384 | 1.0000 | NO |
| **X6_vs_X4** | Combined Sentiment vs VADER | -0.0021 | [-0.0116, +0.0072] | 0.6668 | 1.0000 | NO |
| **X6_vs_X5** | Combined Sentiment vs FinBERT | -0.0010 | [-0.0104, +0.0085] | 0.8434 | 1.0000 | NO |

---

## 4. Key Methodological Takeaways & Limitations
1. **Resolution of Index Slicing Leakage**: Correcting `.iloc` index alignment reduced the uncorrected X2 accuracy artifact (formerly 0.5593) to the true chronological walk-forward accuracy of 0.5002, matching LSTM L2 (0.4941).
2. **Universe Scoping**: The evaluation is strictly scoped to the 10 NIFTY 50 heavyweights (>55% market cap weight) matching dense interim GDELT sentiment coverage.
3. **Correlation-Safe Inference**: Date-level clustering and block bootstrap resamples prevent false discoveries arising from cross-sectional market-wide movements.
