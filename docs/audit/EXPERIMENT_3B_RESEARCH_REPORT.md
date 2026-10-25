# FinFusionNet: Experiment 3B — All-50 Company News Coverage Expansion Research Report

**Author**: Senior Quantitative Research Auditor  
**Date**: 25 October 2026  
**Repository Branch**: `experiment/03b-all50-news` (branched from `v0.2.0-exp3-vader`)  
**Evaluation Window**: 2 January 2023 – 29 September 2026 (45 monthly expanding folds under Protocol B)  
**Common Evaluation Cohort**: 45,136 ticker-day observations across 916 unique test trading sessions (56 active in-window constituents)  

---

## 1. Executive Summary & Core Research Question

### The Core Research Question
> **"Does expanding historical financial-news coverage from the initial 10 heavyweights to all 50 NIFTY 50 companies (all 62 in-window constituents) improve next-trading-session stock direction prediction over price-only baselines and over the 10-company news fusion model?"**

### The Definitive Finding
**NO.** Expanding historical news coverage to all 50 companies provides **no statistically significant predictive edge** over price-only baselines (F0) or over the 10-stock news model (F1_10).

Across **45,136 common-cohort observations** and **916 trading sessions**, expanding news coverage increased news-day representation by **4.4x** (from 16.38% to 72.46% of observations), but:
1. **XGBoost**: Date-level accuracy increased by a negligible $+0.26\%$ ($50.27\% \to 50.54\%$, bootstrap $p = 0.3849$, Holm-Bonferroni adjusted $p = 1.0000$).
2. **Logistic Regression**: Date-level accuracy decreased by $-0.24\%$ ($50.94\% \to 50.69\%$, bootstrap $p = 0.9423$, Holm-Bonferroni adjusted $p = 1.0000$).
3. **LSTM (Seed 42)**: Date-level accuracy decreased by $-0.09\%$ ($51.08\% \to 50.99\%$, bootstrap $p = 0.6495$, Holm-Bonferroni adjusted $p = 1.0000$).

Furthermore, comparing All-50 News directly against the 10-Stock News model yielded:
- **XGBoost Lift**: $+0.24\%$ ($50.30\% \to 50.54\%$, bootstrap $p = 0.2091$, adjusted $p = 1.0000$).
- **Logistic Regression Lift**: $-0.12\%$ ($50.81\% \to 50.69\%$, bootstrap $p = 0.7802$, adjusted $p = 1.0000$).
- **LSTM Lift**: $+0.26\%$ ($50.72\% \to 50.99\%$, bootstrap $p = 0.1029$, adjusted $p = 0.6174$).

**Conclusion**: The lack of predictive signal observed in Experiment 3 was **not** an artifact of sparse ticker coverage. Even when 72.46% of all daily ticker observations have active headline news sentiment, off-the-shelf general lexicon sentiment (VADER) aggregated at the daily level fails to generate alpha in predicting next-day direction for NIFTY 50 equities.

---

## 2. Ingestion & Preprocessing Audit

### 2.1 Universe & Article Volume Expansion
- **Raw Articles Ingested**: 208,524 raw article records across all 62 in-window constituents over the 81-month window (2020-01-01 to 2026-09-30).
- **Syndicated Deduplication & Social Filtering**:
  - Dropped 1,822 non-editorial/social domain records.
  - Dropped 1,015 syndicated duplicates matching `(ticker, feature_date, normalized_title)`.
  - Retained **205,687 unique, cleaned editorial articles**.
- **Daily Ticker-Day Aggregation**:
  - Generated **67,500 daily ticker-day sentiment records** across 62 tickers (compared to 12,828 in Experiment 3, a **5.26x expansion**).
  - Average articles per active news day: **3.05 articles**.

### 2.2 Leakage & Information Cutoff Compliance
All causal integrity checks were enforced and audited via `scripts/validate_experiment_3b_data.py`:
- **Future Publication Leakage**: **0** violations detected (`pub_date > feature_date` = 0).
- **Market Cutoff (15:30 IST) Same-Day Leaks**: **0** violations detected (`pub_time > 15:30 & pub_date == feature_date` = 0).
- **Weekend / Non-Trading Day Leakage**: **0** violations detected. All 35,199 post-cutoff and non-trading articles rolled forward to the next authenticated trading session.
- **Missing Sentiment Policy**: For days without news, features strictly remain `n_articles = 0`, `has_news = 0`, and `vader_mean = NaN` (never imputed as neutral 0.0).

---

## 3. Tripartite Model Performance Comparison

The evaluation was conducted on the exact identical **common cohort** of **45,136 observations across 916 test dates** (2023-01-02 to 2026-09-29) across 45 monthly expanding folds under Protocol B:

| Model Configuration | Date-Level Acc | Pooled Acc | Balanced Acc | ROC-AUC | Brier Score | Log Loss | Predicted Up Share |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Baseline: Always-Up** | 50.34% | 50.21% | 50.00% | 0.5000 | 0.4979 | 8.0244 | 1.0000 |
| **Naive Baseline: Prev-Day Direction** | 49.19% | 49.30% | 49.30% | 0.4930 | 0.5070 | 8.1722 | 0.5036 |
| **XGB-F0 (Price-Only Baseline)** | 50.27% | 50.15% | 49.96% | 0.5067 | 0.2503 | 0.6937 | 0.9355 |
| **XGB-F1_10 (10-Stock News)** | 50.30% | 50.39% | 50.37% | 0.5086 | 0.2562 | 0.7064 | 0.5542 |
| **XGB-F1_All50 (All-50 News)** | **50.54%** | 50.43% | 50.40% | 0.5090 | 0.2564 | 0.7069 | 0.5582 |
| **LR-F0 (Price-Only Baseline)** | **50.94%** | 50.82% | 50.75% | 0.5102 | 0.2505 | 0.6942 | 0.6649 |
| **LR-F1_10 (10-Stock News)** | 50.81% | 50.70% | 50.64% | 0.5091 | 0.2506 | 0.6944 | 0.6490 |
| **LR-F1_All50 (All-50 News)** | 50.69% | 50.58% | 50.51% | 0.5089 | 0.2506 | 0.6945 | 0.6652 |
| **LSTM-F0 (Price-Only Baseline, Seed 42)** | **51.08%** | 50.96% | 50.91% | **0.5149** | **0.2502** | **0.6935** | 0.6207 |
| **LSTM-F1_10 (10-Stock News, Seed 42)** | 50.72% | 50.61% | 50.56% | 0.5091 | 0.2505 | 0.6941 | 0.6130 |
| **LSTM-F1_All50 (All-50 News, Seed 42)** | 50.99% | 50.88% | 50.84% | 0.5105 | 0.2504 | 0.6940 | 0.5962 |

---

## 4. Formal Statistical Significance Testing

Evaluated using **10,000 paired date-block bootstrap resamples** ($L=10$ trading days), two-sided **date-level McNemar tests**, and family-wise **Holm-Bonferroni correction**:

| Comparison | Baseline Model | Evaluated Model | Accuracy Diff | Paired Bootstrap 95% CI | Bootstrap $p$ | McNemar Win Diff | McNemar $p$ | Holm-Bonferroni $p$ | Significant? |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGB: All-50 News vs Price-Only** | XGB-F0 | XGB-F1_All50 | $+0.26\%$ | $[-0.98\%, +1.30\%]$ | 0.3849 | $+20$ | 0.5214 | 1.0000 | **NO** |
| **LR: All-50 News vs Price-Only** | LR-F0 | LR-F1_All50 | $-0.24\%$ | $[-0.55\%, +0.06\%]$ | 0.9423 | $-51$ | 0.0629 | 1.0000 | **NO** |
| **LSTM: All-50 News vs Price-Only** | LSTM-F0 | LSTM-F1_All50 | $-0.09\%$ | $[-0.68\%, +0.47\%]$ | 0.6495 | $-4$ | 0.9155 | 1.0000 | **NO** |
| **XGB: All-50 News vs 10-Stock News** | XGB-F1_10 | XGB-F1_All50 | $+0.24\%$ | $[-0.27\%, +0.78\%]$ | 0.2091 | $+25$ | 0.3959 | 1.0000 | **NO** |
| **LR: All-50 News vs 10-Stock News** | LR-F1_10 | LR-F1_All50 | $-0.12\%$ | $[-0.43\%, +0.18\%]$ | 0.7802 | $-32$ | 0.2532 | 1.0000 | **NO** |
| **LSTM: All-50 News vs 10-Stock News** | LSTM-F1_10 | LSTM-F1_All50 | $+0.26\%$ | $[-0.14\%, +0.66\%]$ | 0.1029 | $+18$ | 0.5412 | 0.6174 | **NO** |

*All 6 bootstrap 95% confidence intervals straddle zero, and every Holm-Bonferroni adjusted $p$-value fails to achieve statistical significance at the $\alpha = 0.05$ level.*

---

## 5. Subgroup & Conditioning Analysis

### 5.1 Active-News Days vs No-News Days in All-50 Universe
Does the model perform better on days where news is actively present ($N=32,705, 72.46\%$) versus days where news is absent ($N=12,431, 27.54\%$)?

| Subgroup | Model Configuration | Observations | Cohort Share | Date-Level Acc | Pooled Acc | ROC-AUC | Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Active-News Days** | XGB-F0 (Price-Only) | 32,705 | 72.46% | 50.32% | 50.13% | 0.5072 | 0.2502 |
| **Active-News Days** | XGB-F1_10 (10-Stock News) | 32,705 | 72.46% | 50.12% | 50.17% | 0.5082 | 0.2564 |
| **Active-News Days** | XGB-F1_All50 (All-50 News) | 32,705 | 72.46% | 50.30% | 50.25% | 0.5079 | 0.2568 |
| **Active-News Days** | LR-F0 (Price-Only) | 32,705 | 72.46% | 50.92% | 50.81% | 0.5115 | 0.2504 |
| **Active-News Days** | LR-F1_All50 (All-50 News) | 32,705 | 72.46% | 50.72% | 50.59% | 0.5117 | 0.2506 |
| **Active-News Days** | LSTM-F0 (Price-Only) | 32,705 | 72.46% | 50.98% | 50.90% | 0.5149 | 0.2502 |
| **Active-News Days** | LSTM-F1_All50 (All-50 News) | 32,705 | 72.46% | **51.08%** | **51.00%** | 0.5116 | 0.2504 |
| **No-News Days** | XGB-F0 (Price-Only) | 12,431 | 27.54% | 50.11% | 50.18% | 0.5052 | 0.2503 |
| **No-News Days** | XGB-F1_All50 (All-50 News) | 12,431 | 27.54% | 50.61% | 50.88% | 0.5121 | 0.2555 |
| **No-News Days** | LR-F0 (Price-Only) | 12,431 | 27.54% | 51.02% | 50.86% | 0.5070 | 0.2506 |
| **No-News Days** | LR-F1_All50 (All-50 News) | 12,431 | 27.54% | 50.71% | 50.54% | 0.5048 | 0.2507 |
| **No-News Days** | LSTM-F0 (Price-Only) | 12,431 | 27.54% | 50.89% | 51.14% | 0.5150 | 0.2502 |
| **No-News Days** | LSTM-F1_All50 (All-50 News) | 12,431 | 27.54% | 50.33% | 50.54% | 0.5078 | 0.2504 |

**Key Subgroup Insight**: Even when conditioned exclusively on days with active editorial headlines, the lift of `LSTM-F1_All50` over `LSTM-F0` is only $+0.10\%$ ($50.98\% \to 51.08\%$), while for Logistic Regression it degrades accuracy by $-0.20\%$ ($50.92\% \to 50.72\%$). The presence of news does not unlock predictive power.

### 5.2 Original 10 Heavyweights vs Expanded Constituents
Does news help the expanded 40 constituents more than the top 10?

| Subgroup | Model Configuration | Observations | Cohort Share | Date-Level Acc | Pooled Acc | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Original 10 Heavyweights** | XGB-F0 (Price-Only) | 9,142 | 20.25% | 49.68% | 49.58% | 0.5108 |
| **Original 10 Heavyweights** | XGB-F1_10 (10-Stock News) | 9,142 | 20.25% | 50.62% | 50.72% | 0.5174 |
| **Original 10 Heavyweights** | XGB-F1_All50 (All-50 News) | 9,142 | 20.25% | 50.47% | 50.37% | 0.5150 |
| **Original 10 Heavyweights** | LR-F0 (Price-Only) | 9,142 | 20.25% | 50.58% | 50.48% | 0.5100 |
| **Original 10 Heavyweights** | LR-F1_All50 (All-50 News) | 9,142 | 20.25% | 50.44% | 50.34% | 0.5110 |
| **Original 10 Heavyweights** | LSTM-F0 (Price-Only) | 9,142 | 20.25% | 50.46% | 50.36% | 0.5090 |
| **Original 10 Heavyweights** | LSTM-F1_All50 (All-50 News) | 9,142 | 20.25% | 50.79% | 50.69% | 0.5108 |
| **Expanded Constituents** | XGB-F0 (Price-Only) | 35,994 | 79.75% | 50.31% | 50.29% | 0.5058 |
| **Expanded Constituents** | XGB-F1_All50 (All-50 News) | 35,994 | 79.75% | 50.44% | 50.44% | 0.5078 |
| **Expanded Constituents** | LR-F0 (Price-Only) | 35,994 | 79.75% | 50.92% | 50.91% | 0.5102 |
| **Expanded Constituents** | LR-F1_All50 (All-50 News) | 35,994 | 79.75% | 50.65% | 50.64% | 0.5084 |
| **Expanded Constituents** | LSTM-F0 (Price-Only) | 35,994 | 79.75% | **51.12%** | **51.12%** | **0.5163** |
| **Expanded Constituents** | LSTM-F1_All50 (All-50 News) | 35,994 | 79.75% | 50.93% | 50.93% | 0.5105 |

On the 40 expanded stocks (representing ~80% of the market universe), `LSTM-F0` (Price-Only) remains superior at **51.12%** accuracy versus **50.93%** for the All-50 news model.

---

## 6. Fold-Level Stability Analysis (45 Protocol B Folds)

Evaluating the distribution of test accuracies across all 45 monthly expanding folds:

| Model Configuration | Mean Fold Acc | Std Dev | Min Fold Acc | 25th Pct | Median | 75th Pct | Max Fold Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGB-F0 (Price-Only)** | 50.17% | 0.0525 | 40.90% | 46.33% | 50.20% | 52.91% | 61.34% |
| **XGB-F1_10 (10-Stock News)** | 50.41% | 0.0220 | 46.00% | 49.17% | 50.10% | 51.95% | 55.53% |
| **XGB-F1_All50 (All-50 News)** | 50.46% | **0.0208** | 46.90% | 48.80% | 50.19% | 51.58% | 55.10% |
| **LR-F0 (Price-Only)** | 50.82% | 0.0316 | 43.60% | 49.30% | 51.14% | 52.42% | 57.96% |
| **LR-F1_10 (10-Stock News)** | 50.70% | 0.0318 | 43.33% | 49.36% | 50.92% | 51.74% | 57.55% |
| **LR-F1_All50 (All-50 News)** | 50.57% | 0.0295 | 43.00% | 49.30% | 50.39% | 52.21% | 56.84% |
| **LSTM-F0 (Price-Only, Seed 42)** | **51.01%** | 0.0324 | 44.90% | 49.26% | 51.41% | 52.76% | 59.78% |
| **LSTM-F1_10 (10-Stock News, Seed 42)** | 50.66% | 0.0330 | 43.47% | 48.69% | 50.42% | 51.95% | 58.34% |
| **LSTM-F1_All50 (All-50 News, Seed 42)** | 50.93% | 0.0309 | 43.89% | 48.88% | 50.82% | 52.21% | 57.50% |

**Variance Regularization Effect**: Notice that for XGBoost, adding news features reduced fold variance ($\sigma = 0.0525 \to 0.0208$) by discouraging extreme all-up predictions, but this stabilization did not translate to superior accuracy over linear or LSTM price baselines.

---

## 7. Sector-Level Breakdown

| Sector | Ticker Count | Test Observations | Top Model | Best Accuracy | F0 Baseline Acc | Lift ($\Delta$) |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| **Financials** | 13 | 12,060 | LR-F0 | 51.44% | 51.44% | $+0.00\%$ |
| **IT** | 5 | 4,570 | LSTM-F0 | 51.58% | 51.58% | $+0.00\%$ |
| **Automobile** | 6 | 5,241 | LR-F1_10 | 50.84% | 50.79% | $+0.05\%$ |
| **FMCG** | 4 | 3,656 | LSTM-F1_All50 | 51.50% | 51.10% | $+0.40\%$ |
| **Metals & Mining** | 5 | 4,570 | LR-F0 | 52.32% | 52.32% | $+0.00\%$ |
| **Healthcare** | 5 | 4,132 | LR-F0 | 51.62% | 51.62% | $+0.00\%$ |
| **Oil & Gas** | 3 | 2,379 | LSTM-F1_All50 | 52.41% | 51.66% | $+0.75\%$ |
| **Power** | 2 | 1,828 | LSTM-F0 | 52.68% | 52.68% | $+0.00\%$ |
| **Services** | 1 | 914 | LSTM-F0 | 54.38% | 54.38% | $+0.00\%$ |
| **Telecom** | 1 | 914 | LR-F0 | 54.27% | 54.27% | $+0.00\%$ |

In 7 out of 10 sectors, the price-only model (either LR-F0 or LSTM-F0) matched or outperformed all news-augmented models. Only Oil & Gas and FMCG showed minor raw lifts ($+0.75\%$ and $+0.40\%$, respectively), which are statistically indistinguishable from noise across the 10-sector multiple-testing surface.

---

## 8. Final Research Conclusion & Strategic Recommendations

### Core Scientific Takeaway
Expanding news coverage from 10 heavyweights to the complete 50-stock NIFTY universe **exhaustively confirms the null hypothesis**:
> **VADER lexicon-based financial news sentiment, whether restricted to 10 heavyweights or expanded to all 50 constituents, does NOT provide statistically significant predictive power for next-session directional stock movement over audited price-only features.**

### Why Expanding News Coverage Did Not Help
1. **Lexical Mismatch**: VADER is a general social-media lexicon designed for Twitter/informal text. Financial headlines require domain-specific financial semantics (e.g., "rate hike paused", "margin compression", "EBITDA expansion").
2. **High-Efficiency NIFTY 50 Market**: The NIFTY 50 represents India's most liquid, institutionalized large-cap equities. Material corporate announcements are priced in within minutes; daily aggregated sentiment arrives too late to offer an informational advantage for next-session direction.
3. **Signal Dilution**: Aggregating diverse daily headlines into an arithmetic mean washes out idiosyncratic, high-impact news catalysts into near-neutral numbers ($\mu = +0.1182$).

### Recommendations for Future Experiments
1. **Transition to Financial Domain LLM / FinBERT Embeddings**: Replace rule-based lexicon scoring with contextual financial language models (FinBERT or specialized financial LLMs) that understand financial nuance and entity-specific exposure.
2. **Event-Driven Filtering**: Rather than evaluating all headlines indiscriminately, focus on high-impact corporate filings (earnings surprises, regulatory actions, M&A) where price absorption may exhibit drift.
3. **Preserve Price-Only Architecture as Benchmark**: Any proposed text or alternative data signal must be strictly benchmarked against the verified price-only baseline ($51.08\%$ date-level accuracy on LSTM) using Protocol B and paired block bootstrapping.
