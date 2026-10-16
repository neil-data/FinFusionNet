# FinFusionNet — XGBoost Research Track Final Report

## Executive Summary
This report details the empirical findings of the **XGBoost Experimental Track of FinFusionNet**, evaluating whether financial news sentiment (lexicon-based VADER and domain-adapted FinBERT) provides statistically defensible predictive lift for next-trading-session stock direction in the Indian equity market.

Evaluation was conducted under a strict **Common Evaluation Cohort** across **point-in-time NIFTY 50 constituents** using chronological walk-forward validation and paired date-block bootstrap testing.

---

## 1. Overall Performance Comparison (Protocol B: Expanding Window)

| Model ID | Configuration | Date-Level Acc | Pooled Acc | Pooled F1 | ROC-AUC | Log Loss | Lift vs Always-Up | Lift vs Prev-Dir |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **X1** | Technical-only baseline | 0.5107 | 0.5100 | 0.5351 | 0.5145 | 0.6921 | +0.0047 | +0.0101 |
| **X2** | Technical + market context | 0.5593 | 0.5585 | 0.5518 | 0.5867 | 0.6797 | +0.0532 | +0.0585 |
| **X3** | Technical + news availability control | 0.5076 | 0.5067 | 0.5262 | 0.5111 | 0.6922 | +0.0014 | +0.0068 |
| **X4** | Technical + news availability + VADER | 0.5032 | 0.5024 | 0.5272 | 0.5051 | 0.6927 | -0.0030 | +0.0024 |
| **X5** | Technical + news availability + FinBERT | 0.5073 | 0.5065 | 0.5336 | 0.5105 | 0.6926 | +0.0012 | +0.0066 |
| **X6** | Technical + news availability + VADER + FinBERT | 0.5087 | 0.5072 | 0.5348 | 0.5103 | 0.6926 | +0.0019 | +0.0072 |

---

## 2. Pre-Registered Primary Hypothesis: FinBERT Contribution (X5 vs X3)

**Hypothesis**: Financial news sentiment extracted via FinBERT improves next-session stock direction prediction beyond technical indicators and news availability controls under Protocol B.
- **Metric**: Equal-weighted date-level accuracy ($Acc_d$)
- **Statistical Test**: Paired Date-Block Bootstrap (10,000 repetitions, block length $L=10$ trading days)
- **Observed Mean Difference**: **-0.0003** (-0.03%)
- **95% Bootstrap Confidence Interval**: **[-0.0102, +0.0091]**
- **Two-Sided Empirical p-value**: **0.9568**
- **One-Sided Empirical p-value ($H_1: \Delta > 0$)**: **0.5393**
- **Statistically Significant at $lpha = 0.05$**: **NO**

### Block-Length Sensitivity Analysis
| Block Length ($L$ days) | Mean Difference | 95% Bootstrap CI | Two-Sided p-value | Significant at $lpha=0.05$ |
|:---:|:---:|:---:|:---:|:---:|
| 3 | -0.0003 | [-0.0103, +0.0090] | 0.9522 | NO |
| 5 | -0.0003 | [-0.0107, +0.0095] | 0.9558 | NO |
| 10 | -0.0003 | [-0.0105, +0.0090] | 0.9572 | NO |
| 15 | -0.0003 | [-0.0111, +0.0092] | 0.9532 | NO |
| 20 | -0.0003 | [-0.0110, +0.0088] | 0.9578 | NO |

---

## 3. Secondary Hypotheses & Holm-Bonferroni Correction

| Comparison | Description | Mean Diff | 95% Bootstrap CI | Raw p-value | Holm Adjusted p | Significant |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **X2_vs_X1** | Market Context vs Pure Technicals | +0.0486 | [+0.0354, +0.0617] | 0.0000 | 0.0000 | YES |
| **X3_vs_X1** | News Attention vs Pure Technicals | -0.0032 | [-0.0119, +0.0060] | 0.4988 | 1.0000 | NO |
| **X4_vs_X3** | VADER Sentiment vs News Control | -0.0044 | [-0.0144, +0.0062] | 0.4014 | 1.0000 | NO |
| **X6_vs_X3** | Combined Sentiment vs News Control | +0.0011 | [-0.0089, +0.0103] | 0.8140 | 1.0000 | NO |
| **X6_vs_X4** | Combined Sentiment vs VADER | +0.0055 | [-0.0054, +0.0155] | 0.3090 | 1.0000 | NO |
| **X6_vs_X5** | Combined Sentiment vs FinBERT | +0.0014 | [-0.0090, +0.0117] | 0.7824 | 1.0000 | NO |

---

## 4. Key Methodological Takeaways & Limitations
1. **Survivorship Bias Protection**: Restricting evaluation to true point-in-time index constituents across 62 historical NIFTY 50 tickers prevented artificial selection bias.
2. **Correlation-Safe Inference**: By clustering at the date level and resampling contiguous trading blocks, the evaluation prevented false significance from market-wide cross-sectional co-movement.
3. **Execution Reality**: Predictive accuracy in directional classification must be paired with transaction costs and slippage before implying economic profitability.
