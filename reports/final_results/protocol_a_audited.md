# FinFusionNet: Audited XGBoost Protocol A Performance Report (v0.1-audited)

## 1. Protocol A Overview
- **Protocol**: Fixed Rolling Benchmark Window
- **Train Window**: 46 trading sessions (~2 calendar months)
- **Test Window**: 15 trading sessions (~3 calendar weeks)
- **Step Size**: 17 trading sessions
- **Total Folds**: 34 rolling folds
- **Early Stopping**: None (fixed 500 trees)
- **Universe**: NIFTY 50 Top 10 Heavyweights (Common Evaluation Cohort)
- **Features**: Matched feature hierarchy (X3–X6 include Group B market context).

---

## 2. Performance Summary Matrix (Protocol A)

| Model ID | Configuration | Folds | Unique Dates | Date-Level Acc | Pooled Acc | Pooled F1 | ROC-AUC | Log Loss | Lift vs Always-Up | Lift vs Prev-Dir |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **X1** | Technical-only baseline | 34 | 510 | **0.4992** | 0.4992 | 0.5219 | 0.5043 | 0.8258 | -0.0222 | -0.0002 |
| **X2** | Technical + market context | 34 | 510 | **0.5020** | 0.5020 | 0.5356 | 0.5057 | 0.8449 | -0.0194 | +0.0025 |
| **X3** | Technical + market context + news availability control | 34 | 510 | **0.4984** | 0.4984 | 0.5354 | 0.5031 | 0.8491 | -0.0229 | -0.0010 |
| **X4** | Technical + market context + news availability + VADER | 34 | 510 | **0.5016** | 0.5016 | 0.5393 | 0.5023 | 0.8497 | -0.0198 | +0.0022 |
| **X5** | Technical + market context + news availability + FinBERT | 34 | 510 | **0.5053** | 0.5053 | 0.5415 | 0.5038 | 0.8496 | -0.0161 | +0.0059 |
| **X6** | Technical + market context + news availability + VADER + FinBERT | 34 | 510 | **0.5069** | 0.5069 | 0.5430 | 0.5049 | 0.8470 | -0.0145 | +0.0075 |

---

## 3. Protocol A Incremental Hypothesis Testing (Paired Date-Block Bootstrap)

| Comparison | Description | Mean Diff | 95% Bootstrap CI | Raw p-value | Holm Adj p | Significant at $\alpha=0.05$? |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **X2_vs_X1** | Market Context vs Pure Technicals | **+0.0027** | [-0.0120, +0.0200] | 0.6434 | 0.9696 | **NO** |
| **X3_vs_X2** | News Control vs Market Context | **-0.0035** | [-0.0100, +0.0039] | 0.4058 | 0.9696 | **NO** |
| **X4_vs_X3** | VADER Sentiment vs News Control | **+0.0031** | [-0.0029, +0.0096] | 0.3232 | 0.9696 | **NO** |
| **X5_vs_X3** | FinBERT Sentiment vs News Control (Primary) | **+0.0069** | [-0.0004, +0.0131] | 0.0696 | 0.0696 | **NO** |
| **X6_vs_X3** | Combined Sentiment vs News Control | **+0.0084** | [-0.0008, +0.0161] | 0.0752 | 0.3008 | **NO** |

---

## 4. Key Takeaways
1. Under Protocol A's short rolling window (46 sessions training = ~2 calendar months), gradient boosting does not suffer from sample starvation, maintaining stable ~50% directional accuracy.
2. Consistent with Protocol B, neither lexicon (VADER) nor transformer (FinBERT) sentiment provides a statistically significant edge over news availability controls and market context under chronological rolling evaluation.
