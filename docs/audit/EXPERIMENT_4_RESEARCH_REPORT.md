# FinFusionNet: Experiment 4 — FinBERT + Price Fusion Research Report

**Author**: Senior Quantitative Research Auditor  
**Date**: 26 October 2026  
**Repository Branch**: `experiment/04-finbert-fusion` (branched from `v0.2.1-exp3b-all50`)  
**Evaluation Window**: 2 January 2023 – 29 September 2026 (45 monthly expanding folds under Protocol B)  
**Common Evaluation Cohort**: Exactly 45,136 ticker-day observations across 916 unique test trading sessions (56 active in-window constituents)  

---

## 1. Executive Summary & Core Research Question

### The Core Research Question
> **"Does domain-specific FinBERT sentiment provide statistically significant incremental predictive value over the audited price-only baseline and general-lexicon VADER sentiment in forecasting next-session stock price direction for NIFTY 50 equities?"**

### The Definitive Finding
**NO.** Domain-specific FinBERT financial sentiment features provide **no statistically significant incremental predictive value** over the audited price-only baseline ($F_0$). Furthermore, FinBERT sentiment fails to consistently outperform generic lexicon-based VADER sentiment ($F_{1\text{-VADER}}$).

Evaluating across **45,136 common-cohort test observations** spanning **916 trading sessions** under the audited Protocol B expanding walk-forward engine (45 expanding monthly folds, 24-month initial training window, strict 1-session embargo):

1. **XGBoost**: Date-level accuracy marginally shifted by **$+0.15\%$** ($50.27\% \to 50.42\%$), but the paired date-block bootstrap 95% confidence interval straddles zero ($[-1.16\%, +1.30\%]$, bootstrap $p = 0.4535$, McNemar $p = 1.0000$, Holm-Bonferroni adjusted $p = 1.0000$).
2. **Logistic Regression**: Date-level accuracy shifted by a statistically imperceptible **$+0.01\%$** ($50.94\% \to 50.95\%$, bootstrap 95% CI $[-0.32\%, +0.30\%]$, bootstrap $p = 0.5456$, McNemar $p = 0.5614$, Holm-Bonferroni adjusted $p = 1.0000$).
3. **LSTM (Seed 42)**: Date-level accuracy degraded by **$-0.55\%$** ($51.08\% \to 50.53\%$, bootstrap 95% CI $[-1.16\%, +0.00\%]$, bootstrap $p = 0.9741$, McNemar $p = 0.2588$, Holm-Bonferroni adjusted $p = 1.0000$).
4. **FinBERT vs VADER**: Comparing FinBERT against VADER directly revealed that FinBERT was slightly worse for XGBoost ($-0.12\%$, $50.54\% \to 50.42\%$) and markedly worse for LSTM ($-0.46\%$, $50.99\% \to 50.53\%$). For Logistic Regression, FinBERT showed a small nominal gain ($+0.25\%$, $50.69\% \to 50.95\%$), but this was not statistically significant after Holm-Bonferroni multi-testing correction (adjusted $p = 0.1350$).

Across all 45 walk-forward folds:
- XGBoost FinBERT beat Price-Only in 23 folds and lost in 22 folds (**51.1% win rate**, a statistical coin flip).
- Logistic Regression FinBERT beat Price-Only in 21 folds, tied in 1, and lost in 23 folds (**46.7% win rate**).
- LSTM FinBERT beat Price-Only in only 12 folds, tied in 2, and lost in 31 folds (**26.7% win rate**).

**Scientific Verdict**: In accordance with the empirical mandate to report negative results honestly, we conclude that **domain-specific transformer sentiment (FinBERT) does not produce actionable next-session directional alpha** over price dynamics for large-cap Indian equities.

---

## 2. Ingestion, Preprocessing & Integrity Audit

### 2.1 Model Provenance & Inference Audit
- **Model Checkpoint**: Official `ProsusAI/finbert` pre-trained BERT weights (`pytorch_model.bin`, 437,992,753 bytes) verified and loaded offline from local cache snapshot:
  `C:\Users\Neil\.cache\huggingface\hub\models--ProsusAI--finbert\snapshots\4556d13015211d73dccd3fdd39d39232506f3e43`
- **Source Code Archive**: Verified and unzipped official `finBERT-master.zip` into `experiments/experiment_4/finbert_source/`.
- **Classification Head**: Verified 3-class linear head with `id2label: {0: 'positive', 1: 'negative', 2: 'neutral'}`.
- **Inference Precision**: Full IEEE 754 Float32 precision (`torch.float32`) with dynamic token padding bounded at 64 tokens. Dynamic INT8 quantization was empirically audited and rejected due to unacceptable score distortion (correlation dropped to $0.887$, with max absolute error of $0.962$).
- **Scored Corpus**: 182,822 unique headlines scored across the 205,687 cleaned editorial articles, generating 67,500 daily ticker-day observations across 62 constituents over the entire 2020–2026 dataset.

### 2.2 Leakage & Information Cutoff Compliance
Audited and verified via `tests/test_experiment_4_finbert.py`:
- **Future Publication Leakage**: Exactly **0** violations detected (`pub_date > feature_date` = 0).
- **Market Cutoff (15:30 IST) Same-Day Leakage**: Exactly **0** violations detected (`pub_time > 15:30 & pub_date == feature_date` = 0).
- **After-Hours & Weekend Handling**: All post-15:30 IST news and non-trading day news were causally rolled forward to the subsequent trading session.
- **Strict Missing-News Null Policy**: On days without news ($N=0$), features remain strictly null (`n_articles = 0`, `has_news = 0`, and sentiment features = `NaN`). Missing sentiment is never converted to neutral (0.0 or 0.33).

---

## 3. Comprehensive Performance Comparison

Evaluated on the exact identical **common evaluation cohort** ($N=45,136$ observations, 916 dates, 56 tickers, 45 folds):

| Model Configuration | Date-Level Acc | Pooled Acc | Balanced Acc | Precision | Recall | F1 Score | ROC-AUC | Brier Score | Log Loss | Predicted Up % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline: Always-Up** | 50.34% | 50.21% | 50.00% | 0.5021 | 1.0000 | 0.6686 | 0.5000 | 0.4979 | 8.0244 | 100.0% |
| **Baseline: Prev-Day Direction** | 49.19% | 49.30% | 49.30% | 0.4952 | 0.4966 | 0.4959 | 0.4930 | 0.5070 | 8.1722 | 50.36% |
| **XGB-F0 (Price-Only Baseline)** | 50.27% | 50.15% | 49.96% | 0.5019 | 0.9351 | 0.6532 | 0.5067 | 0.2503 | 0.6937 | 93.55% |
| **XGB-F1_VADER (All-50 VADER)** | 50.54% | 50.43% | 50.40% | 0.5057 | 0.5622 | 0.5325 | 0.5090 | 0.2564 | 0.7069 | 55.82% |
| **XGB-F1_FinBERT (All-50 FinBERT)** | **50.42%** | 50.42% | 50.39% | 0.5057 | 0.5593 | 0.5311 | 0.5070 | 0.2568 | 0.7076 | 55.53% |
| **LR-F0 (Price-Only Baseline)** | 50.94% | 50.82% | 50.75% | 0.5078 | 0.6724 | 0.5786 | 0.5102 | 0.2505 | 0.6942 | 66.49% |
| **LR-F1_VADER (All-50 VADER)** | 50.69% | 50.58% | 50.51% | 0.5059 | 0.6702 | 0.5766 | 0.5089 | 0.2506 | 0.6945 | 66.52% |
| **LR-F1_FinBERT (All-50 FinBERT)** | **50.95%** | 50.83% | 50.77% | 0.5081 | 0.6567 | 0.5729 | 0.5103 | 0.2505 | 0.6943 | 64.91% |
| **LSTM-F0 (Price-Only, Seed 42)** | **51.08%** | **50.96%** | **50.91%** | 0.5095 | 0.6297 | 0.5633 | **0.5149** | **0.2502** | **0.6935** | 62.07% |
| **LSTM-F1_VADER (All-50 VADER, Seed 42)** | 50.99% | 50.88% | 50.84% | 0.5092 | 0.6045 | 0.5528 | 0.5105 | 0.2504 | 0.6940 | 59.62% |
| **LSTM-F1_FinBERT (All-50 FinBERT, Seed 42)** | **50.53%** | 50.41% | 50.38% | 0.5054 | 0.5842 | 0.5419 | 0.5092 | 0.2505 | 0.6942 | 58.04% |

---

## 4. Formal Statistical Significance Testing

Hypothesis testing was executed using **10,000 paired date-block bootstrap resamples** (block length $L=10$ trading sessions), two-sided **date-level McNemar tests**, and family-wise **Holm-Bonferroni correction**:

| Comparison | Baseline Model | Alternative Model | Accuracy Difference | Paired Bootstrap 95% CI | Bootstrap $p$-value | McNemar Net Wins | McNemar $p$-value | Holm-Bonferroni $p$-value | Statistically Significant? |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGB: FinBERT vs Price-Only** | XGB-F0 | XGB-F1_FinBERT | $+0.15\%$ | $[-1.16\%, +1.30\%]$ | 0.4535 | $+1$ | 1.0000 | 1.0000 | **NO** |
| **LR: FinBERT vs Price-Only** | LR-F0 | LR-F1_FinBERT | $+0.01\%$ | $[-0.32\%, +0.30\%]$ | 0.5456 | $-17$ | 0.5614 | 1.0000 | **NO** |
| **LSTM: FinBERT vs Price-Only** | LSTM-F0 | LSTM-F1_FinBERT | $-0.55\%$ | $[-1.16\%, +0.00\%]$ | 0.9741 | $-33$ | 0.2588 | 1.0000 | **NO** |
| **XGB: FinBERT vs VADER News** | XGB-F1_VADER | XGB-F1_FinBERT | $-0.11\%$ | $[-0.63\%, +0.41\%]$ | 0.6320 | $+9$ | 0.7775 | 1.0000 | **NO** |
| **LR: FinBERT vs VADER News** | LR-F1_VADER | LR-F1_FinBERT | $+0.25\%$ | $[0.00\%, +0.45\%]$ | 0.0225 | $+45$ | 0.0865 | 0.1350 | **NO** |
| **LSTM: FinBERT vs VADER News** | LSTM-F1_VADER | LSTM-F1_FinBERT | $-0.46\%$ | $[-0.80\%, -0.14\%]$ | 0.9962 | $-51$ | 0.0691 | 1.0000 | **NO** |

### Key Statistical Observations:
1. Every comparison against the audited price-only baseline ($F_0$) yields a bootstrap 95% confidence interval that overlaps zero.
2. Every Holm-Bonferroni adjusted $p$-value for FinBERT vs Price-Only equals $1.0000$.
3. While Logistic Regression showed a nominal unadjusted improvement over VADER ($p = 0.0225$), this effect completely evaporated under Holm-Bonferroni multi-testing control ($p = 0.1350$).
4. For sequential modeling (LSTM), adding FinBERT sentiment introduces substantial noise, degrading accuracy by 55 basis points relative to price-only dynamics.

---

## 5. Subgroup & Conditioning Breakdown

### 5.1 Active-News Days ($N = 32,705$, 72.5%) vs No-News Days ($N = 12,431$, 27.5%)

| Subgroup | Model Configuration | Cohort Share | Date-Level Acc | Pooled Acc | Balanced Acc | ROC-AUC | Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Active News Days** | XGB-F0 (Price-Only) | 72.5% | 50.32% | 50.13% | 49.94% | 0.5072 | 0.2502 |
| **Active News Days** | XGB-F1_VADER (All-50 VADER) | 72.5% | 50.30% | 50.25% | 50.22% | 0.5079 | 0.2568 |
| **Active News Days** | **XGB-F1_FinBERT (All-50 FinBERT)** | 72.5% | **50.48%** | 50.39% | 50.36% | 0.5070 | 0.2571 |
| **Active News Days** | LR-F0 (Price-Only) | 72.5% | 50.92% | 50.81% | 50.74% | 0.5115 | 0.2504 |
| **Active News Days** | LR-F1_VADER (All-50 VADER) | 72.5% | 50.72% | 50.59% | 50.49% | 0.5117 | 0.2506 |
| **Active News Days** | **LR-F1_FinBERT (All-50 FinBERT)** | 72.5% | **51.05%** | 50.94% | 50.85% | **0.5134** | 0.2504 |
| **Active News Days** | **LSTM-F0 (Price-Only, Seed 42)** | 72.5% | **50.98%** | 50.90% | 50.84% | **0.5149** | 0.2502 |
| **Active News Days** | LSTM-F1_VADER (All-50 VADER, Seed 42) | 72.5% | 51.08% | 51.00% | 50.95% | 0.5116 | 0.2504 |
| **Active News Days** | **LSTM-F1_FinBERT (All-50 FinBERT, Seed 42)** | 72.5% | **50.71%** | 50.60% | 50.56% | 0.5114 | 0.2505 |
| **No-News Days** | XGB-F0 (Price-Only) | 27.5% | 50.11% | 50.18% | 50.00% | 0.5052 | 0.2503 |
| **No-News Days** | XGB-F1_FinBERT (All-50 FinBERT) | 27.5% | 50.10% | 50.50% | 50.49% | 0.5070 | 0.2560 |
| **No-News Days** | LR-F0 (Price-Only) | 27.5% | 51.02% | 50.86% | 50.80% | 0.5070 | 0.2506 |
| **No-News Days** | LR-F1_FinBERT (All-50 FinBERT) | 27.5% | 50.72% | 50.55% | 50.55% | 0.5048 | 0.2507 |
| **No-News Days** | LSTM-F0 (Price-Only, Seed 42) | 27.5% | 50.89% | 51.14% | 51.09% | 0.5150 | 0.2502 |
| **No-News Days** | LSTM-F1_FinBERT (All-50 FinBERT, Seed 42) | 27.5% | 49.75% | 49.92% | 49.90% | 0.5033 | 0.2506 |

### 5.2 Original 10 Heavyweights ($N = 9,142$, 20.3%) vs Expanded 40 Stocks ($N = 35,994$, 79.7%)

| Subgroup | Model Configuration | Cohort Share | Date-Level Acc | Pooled Acc | Balanced Acc | ROC-AUC | Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original 10 Heavyweights** | XGB-F0 (Price-Only) | 20.3% | 49.68% | 49.58% | 49.88% | 0.5108 | 0.2503 |
| **Original 10 Heavyweights** | XGB-F1_VADER (All-50 VADER) | 20.3% | 50.47% | 50.37% | 50.44% | 0.5150 | 0.2558 |
| **Original 10 Heavyweights** | **XGB-F1_FinBERT (All-50 FinBERT)** | 20.3% | **50.66%** | 50.66% | 50.72% | **0.5185** | 0.2553 |
| **Original 10 Heavyweights** | LR-F0 (Price-Only) | 20.3% | 50.58% | 50.48% | 50.60% | 0.5100 | 0.2503 |
| **Original 10 Heavyweights** | **LR-F1_FinBERT (All-50 FinBERT)** | 20.3% | **51.07%** | 50.97% | 51.08% | **0.5160** | 0.2501 |
| **Original 10 Heavyweights** | LSTM-F0 (Price-Only, Seed 42) | 20.3% | 50.46% | 50.36% | 50.47% | 0.5090 | 0.2504 |
| **Original 10 Heavyweights** | **LSTM-F1_FinBERT (All-50 FinBERT, Seed 42)** | 20.3% | **50.80%** | 50.70% | 50.78% | 0.5086 | 0.2507 |
| **Expanded 40 Stocks** | XGB-F0 (Price-Only) | 79.7% | 50.31% | 50.29% | 49.98% | 0.5058 | 0.2502 |
| **Expanded 40 Stocks** | XGB-F1_FinBERT (All-50 FinBERT) | 79.7% | 50.36% | 50.36% | 50.33% | 0.5044 | 0.2571 |
| **Expanded 40 Stocks** | LR-F0 (Price-Only) | 79.7% | 50.92% | 50.91% | 50.80% | 0.5102 | 0.2505 |
| **Expanded 40 Stocks** | LR-F1_FinBERT (All-50 FinBERT) | 79.7% | 50.81% | 50.79% | 50.69% | 0.5090 | 0.2506 |
| **Expanded 40 Stocks** | **LSTM-F0 (Price-Only, Seed 42)** | 79.7% | **51.12%** | **51.12%** | **51.04%** | **0.5163** | 0.2501 |
| **Expanded 40 Stocks** | LSTM-F1_FinBERT (All-50 FinBERT, Seed 42) | 79.7% | 50.35% | 50.34% | 50.29% | 0.5094 | 0.2505 |

*Insight*: While FinBERT showed modest localized lift (+0.5% to +0.9%) among the 10 mega-cap heavyweights (where editorial quality is highest), it provided zero lift across the expanded 40 constituents, pulling aggregate performance back to baseline.

---

## 6. Sector-Level Performance Breakdown

| Sector | Observations | News Coverage | XGB-F0 Acc | XGB-FinBERT Acc | XGB Lift | LR-F0 Acc | LR-FinBERT Acc | LR Lift | LSTM-F0 Acc | LSTM-FinBERT Acc | LSTM Lift |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Financials** | 9,837 | 76.0% | 50.20% | 50.75% | $+0.55\%$ | 50.76% | 51.20% | $+0.45\%$ | 51.48% | 51.14% | $-0.34\%$ |
| **Automobile** | 5,241 | 76.8% | 50.28% | 49.57% | $-0.71\%$ | 50.79% | 50.56% | $-0.23\%$ | 50.26% | 49.76% | $-0.50\%$ |
| **Metals & Mining** | 4,570 | 74.2% | 51.12% | 49.58% | $-1.53\%$ | 52.32% | 51.95% | $-0.37\%$ | 51.51% | 50.61% | $-0.90\%$ |
| **IT** | 4,570 | 79.8% | 48.67% | 50.90% | $+2.23\%$ | 49.80% | 50.59% | $+0.79\%$ | 50.44% | 50.20% | $-0.24\%$ |
| **Healthcare** | 4,304 | 64.6% | 50.58% | 50.60% | $+0.02\%$ | 52.00% | 51.53% | $-0.46\%$ | 50.98% | 50.88% | $-0.09\%$ |
| **FMCG** | 4,209 | 67.5% | 48.90% | 49.01% | $+0.12\%$ | 49.94% | 49.51% | $-0.43\%$ | 50.37% | 49.28% | $-1.09\%$ |
| **Oil & Gas** | 2,379 | 63.6% | 50.27% | 49.89% | $-0.38\%$ | 50.69% | 51.41% | $+0.71\%$ | 51.53% | 51.53% | $0.00\%$ |
| **Consumer Durables** | 1,828 | 71.7% | 51.42% | 51.09% | $-0.33\%$ | 50.49% | 49.56% | $-0.93\%$ | 48.47% | 48.03% | $-0.44\%$ |
| **Materials** | 1,828 | 56.8% | 51.53% | 50.77% | $-0.77\%$ | 51.26% | 50.98% | $-0.27\%$ | 51.48% | 49.67% | $-1.81\%$ |
| **Power** | 1,828 | 82.8% | 50.77% | 52.19% | $+1.42\%$ | 49.40% | 49.84% | $+0.44\%$ | 52.68% | 50.82% | $-1.86\%$ |
| **Consumer Services** | 1,029 | 56.3% | 47.91% | 51.31% | $+3.40\%$ | 49.85% | 50.05% | $+0.19\%$ | 49.85% | 48.30% | $-1.55\%$ |
| **Construction** | 914 | 76.7% | 50.66% | 50.00% | $-0.66\%$ | 49.89% | 50.55% | $+0.66\%$ | 48.36% | 52.08% | $+3.72\%$ |
| **Services** | 914 | 71.4% | 51.09% | 51.31% | $+0.22\%$ | 51.64% | 51.31% | $-0.33\%$ | 54.38% | 50.55% | $-3.83\%$ |
| **Telecom** | 914 | 85.4% | 51.64% | 52.52% | $+0.88\%$ | 54.27% | 52.52% | $-1.75\%$ | 51.09% | 51.09% | $0.00\%$ |

---

## 7. Mechanism & Theoretical Discussion

Why did domain-specific transformer sentiment fine-tuned on financial language fail to produce next-day alpha?

1. **Information Efficiency & Pre-Cutoff Price Discovery**: In highly liquid large-cap markets like the NIFTY 50, news published during market hours ($09:15 - 15:30$ IST) is rapidly priced in within seconds or minutes by institutional algorithmic participants. By the time market close ($15:30$ IST) occurs, any directional impact of intraday headlines has already been reflected in the closing price. The subsequent session's return ($t \to t+1$) depends on overnight news or structural flow, not the stale intraday news from session $t$.
2. **Headline-Only Semantic Compression**: While FinBERT correctly understands financial terminology (e.g., distinguishing "debt default" from "dividend increase"), headlines lack the quantitative context required to gauge material earnings surprises vs consensus expectations. A headline stating *"Company posts 12% revenue growth"* may be classified as strongly positive by FinBERT, but if analyst consensus was expecting 18% growth, the stock will open down sharply.
3. **Overfitting & Noise Injection in Deep Sequential Architectures**: In sequential models (LSTM), adding 5 daily sentiment features increased the input dimensionality by 21%. Because daily headline sentiment in financial news has high variance and low signal-to-noise ratio, the recurrent cells learned spurious correlations between noisy sentiment sequences and next-day returns, causing the LSTM's date-level accuracy to collapse from $51.08\%$ to $50.53\%$.

---

## 8. Final Conclusion & Deliverables

- **All Artifacts Saved**:
  - Parquet predictions: [`experiments/experiment_4/predictions/experiment_4_common_cohort_predictions.parquet`](file:///c:/Users/Neil/Desktop/FinFusionNeT/experiments/experiment_4/predictions/experiment_4_common_cohort_predictions.parquet)
  - Performance results: [`experiments/experiment_4/experiment_4_results.csv`](file:///c:/Users/Neil/Desktop/FinFusionNeT/experiments/experiment_4/experiment_4_results.csv)
  - Statistical significance: [`experiments/experiment_4/experiment_4_statistical_significance.csv`](file:///c:/Users/Neil/Desktop/FinFusionNeT/experiments/experiment_4/experiment_4_statistical_significance.csv)
  - Per-fold breakdown: [`experiments/experiment_4/experiment_4_fold_results.csv`](file:///c:/Users/Neil/Desktop/FinFusionNeT/experiments/experiment_4/experiment_4_fold_results.csv)
  - Subgroup breakdown: [`experiments/experiment_4/experiment_4_subgroups.csv`](file:///c:/Users/Neil/Desktop/FinFusionNeT/experiments/experiment_4/experiment_4_subgroups.csv)
  - Sector breakdown: [`experiments/experiment_4/experiment_4_sector_breakdown.csv`](file:///c:/Users/Neil/Desktop/FinFusionNeT/experiments/experiment_4/experiment_4_sector_breakdown.csv)
  - Tripartite comparison: [`experiments/experiment_4/experiment_4_tripartite_comparison.csv`](file:///c:/Users/Neil/Desktop/FinFusionNeT/experiments/experiment_4/experiment_4_tripartite_comparison.csv)
  - Full audit document: [`docs/audit/EXPERIMENT_4_AUDIT_REPORT.md`](file:///c:/Users/Neil/Desktop/FinFusionNeT/docs/audit/EXPERIMENT_4_AUDIT_REPORT.md)
- **Unit Test Suite**: 51/51 tests passing across `tests/`.
