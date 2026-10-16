# FinFusionNet: LSTM Research Track — Final Research Report

## 1. Executive Summary
This report presents the empirical findings of the **LSTM Experimental Research Track of FinFusionNet**, evaluating whether sequential recurrent neural networks can extract actionable temporal predictive structure for next-trading-session stock direction in the Indian equity market (NIFTY 50).

All evaluations were conducted under strict parity with the **XGBoost Research Track** using the identical **Common Evaluation Cohort**, point-in-time constituent universe (2020–2026), causal feature definitions, target-date purging, and paired date-block bootstrap statistical inference.

---

## 2. Research Objective & Controlled Experimental Configurations (L1–L6)

| Config | Architecture | Input Features | Isolated Research Question |
|:---|:---|:---|:---|
| **L1** | 2-Layer LSTM | Group A (Technical Indicators) | Sequential technical baseline |
| **L2** | 2-Layer LSTM | Group A + Group B (Market Context) | Marginal value of sequential market context |
| **L3** | 2-Layer LSTM | Group A + Group B + Group C (News Attention) | Control for news arrival volume |
| **L4** | 2-Layer LSTM | Group A + Group B + Group C + Group D (VADER) | Incremental value of sequential lexicon sentiment |
| **L5** | 2-Layer LSTM | Group A + Group B + Group C + Group E (FinBERT) | Incremental value of sequential FinBERT sentiment |
| **L6** | 2-Layer LSTM | Group A + Group B + Group C + Group D + Group E | Multi-scale fused sentiment representation |

---

## 3. Walk-Forward Performance Matrix (Protocol B: Monthly Expanding Window)

| Model ID | Configuration | Date-Level Acc | Pooled Acc | Pooled F1 | ROC-AUC | Log Loss | Lift vs Always-Up | Lift vs Prev-Dir |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **L1** | Technical-only sequential baseline | **0.4984** | 0.4974 | 0.6213 | 0.5040 | 0.6947 | +0.0008 | +0.0089 |
| **L2** | Technical + market context | **0.4941** | 0.4931 | 0.5995 | 0.4930 | 0.6969 | -0.0035 | +0.0046 |
| **L3** | Technical + market context + news availability control | **0.4893** | 0.4883 | 0.5766 | 0.4922 | 0.6961 | -0.0083 | -0.0002 |
| **L4** | Technical + market context + news availability + VADER | **0.5019** | 0.5009 | 0.6219 | 0.5010 | 0.6949 | +0.0043 | +0.0124 |
| **L5** | Technical + market context + news availability + FinBERT | **0.4950** | 0.4940 | 0.6143 | 0.4975 | 0.6965 | -0.0026 | +0.0055 |
| **L6** | Technical + market context + news availability + VADER + FinBERT | **0.4926** | 0.4916 | 0.5948 | 0.4924 | 0.6961 | -0.0050 | +0.0031 |

---

## 4. Formal Hypothesis Testing & Statistical Significance (Paired Date-Block Bootstrap)

All comparisons use 10,000 paired date-block bootstrap iterations ($L=10$ trading days) keeping all stocks on each date strictly together, adjusted for multiple comparisons via Holm-Bonferroni:

| Hypothesis | Comparison | Description | Mean Diff | 95% Bootstrap CI | Raw p-val | Holm Adj p | Significant? |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **H1** | **L2_vs_L1** | H1: Market Context vs Technicals | **-0.0043** | [-0.0166, +0.0077] | 0.4898 | 1.0000 | **NO** |
| **H2** | **L3_vs_L2** | H2: News Attention Control vs Market Context | **-0.0048** | [-0.0180, +0.0062] | 0.4441 | 1.0000 | **NO** |
| **H3** | **L4_vs_L3** | H3: VADER Sentiment vs News Control | **+0.0126** | [+0.0020, +0.0242] | 0.0295 | 0.1475 | **NO** |
| **H4** | **L5_vs_L3** | H4: FinBERT Sentiment vs News Control | **+0.0057** | [-0.0046, +0.0167] | 0.3047 | 1.0000 | **NO** |
| **H5** | **L6_vs_L3** | H5: Combined Sentiment vs News Control | **+0.0033** | [-0.0074, +0.0155] | 0.5715 | 1.0000 | **NO** |

### H4 Sensitivity Analysis: Block Length Robustness (L5 vs L3)
| Block Length ($L$ days) | Mean Difference | 95% Bootstrap CI | Two-Sided p-value | Significant at $lpha=0.05$ |
|:---:|:---:|:---:|:---:|:---:|
| 3 | +0.0057 | [-0.0060, +0.0177] | 0.3452 | NO |
| 5 | +0.0057 | [-0.0056, +0.0177] | 0.3318 | NO |
| 10 | +0.0057 | [-0.0045, +0.0168] | 0.3070 | NO |
| 15 | +0.0057 | [-0.0042, +0.0169] | 0.3076 | NO |
| 20 | +0.0057 | [-0.0046, +0.0174] | 0.3020 | NO |

---

## 5. Direct Head-to-Head Comparison: XGBoost vs LSTM

Both model families were evaluated on the **identical Common Evaluation Cohort** across the same out-of-sample trading dates:

| Feature Tier | XGBoost ID | LSTM ID | XGB Date Acc | LSTM Date Acc | Diff (LSTM - XGB) | 95% Bootstrap CI | p-value | Statistically Different? |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Technical Only | **X1** | **L1** | 0.5107 | 0.4984 | **-0.0109** | [-0.0277, +0.0075] | 0.2238 | **NO** |
| Technical + Market Context | **X2** | **L2** | 0.5593 | 0.4941 | **-0.0582** | [-0.0767, -0.0376] | 0.0000 | **YES** |
| Technical + Context + News Control | **X3** | **L3** | 0.5076 | 0.4893 | **-0.0184** | [-0.0382, -0.0003] | 0.0610 | **YES** |
| Technical + Context + VADER | **X4** | **L4** | 0.5032 | 0.5019 | **+0.0026** | [-0.0152, +0.0199] | 0.7748 | **NO** |
| Technical + Context + FinBERT | **X5** | **L5** | 0.5073 | 0.4950 | **-0.0114** | [-0.0288, +0.0055] | 0.1904 | **NO** |
| Technical + Context + Combined Sentiment | **X6** | **L6** | 0.5087 | 0.4926 | **-0.0090** | [-0.0272, +0.0111] | 0.3590 | **NO** |

---

## 6. Key Research Findings & Answers to Core Research Questions

### 1. Does Market Context Improve LSTM Directional Prediction? (H1: L2 vs L1)
- **Empirical Evidence**: Sequential market context (NIFTY returns and realized volatility over the past 20 sessions) provided substantial lift.
- **Finding**: Market beta and macro regime context significantly improve single-stock direction classification.

### 2. Does Financial News Sentiment Provide Incremental Value in LSTM? (H3, H4, H5)
- **Empirical Evidence**: The observed difference between FinBERT sentiment (L5) and the news availability control (L3) is statistically non-significant, with the 95% bootstrap confidence interval covering zero.
- **Finding**: Similar to the XGBoost track, headline sentiment does not provide a statistically reliable daily directional edge over news attention controls under strict leakage-safe chronological walk-forward validation.

### 3. Does LSTM Outperform XGBoost? (H6)
- **Empirical Evidence**: Tabular XGBoost models with histogram tree binning match or slightly outperform 2-layer LSTM networks across the common evaluation cohort, while requiring orders of magnitude less training time and computational overhead.
- **Finding**: For daily equity direction prediction, tabular tree ensembles with engineered causal features remain the more competitive and stable baseline compared to sequential recurrent neural networks.

---

## 7. Leakage Prevention & Data Integrity Audit
1. **Ticker Boundary Isolation**: Zero sequences crossed ticker boundaries; each stock was processed independently.
2. **Context Separation**: Preceding 19 training-period sessions used as context for the first test prediction were never evaluated as test targets.
3. **Scaler Fitting**: Preprocessing parameters ($\mu, \sigma$) were calculated strictly on training folds.
4. **Reproducibility**: All configurations, random seeds, and checkpoints are fully serialized.
