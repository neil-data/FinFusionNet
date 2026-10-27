# FinFusionNet: Experiment 3 Research Report
## Lexicon-Based News Sentiment Fusion (VADER)

**Project**: FinFusionNet — Multimodal Stock Direction Prediction  
**Evaluation Protocol**: Protocol B (45 Monthly Expanding Folds, 24-Month Initial Window, 1-Session Embargo)  
**Evaluation Window**: 2023-01-02 to 2026-09-29 (916 Out-of-Sample Test Dates, 45,136 Observations)  
**Branch**: `experiment/03-vader-fusion`  
**Auditor / Quant Research Engineer**: Antigravity  

---

## 1. Executive Summary & Research Decision

### Core Research Question
> **Does VADER-based financial news sentiment fusion provide statistically significant predictive improvement over corresponding price-only baselines for next-trading-session direction prediction?**

### Definitive Empirical Answer
**NO.** Across all three evaluated model families—gradient boosted trees (XGBoost), linear models (Logistic Regression), and sequential deep networks (LSTM)—the inclusion of VADER lexicon-based financial news sentiment (Group C availability controls + Group D VADER polarity features) **fails to produce statistically significant predictive improvement** over pure historical price features.

### Key Empirical Findings:
1. **XGBoost (XGB-F1 vs XGB-F0)**:
   - Date-Level Accuracy: **50.30%** (F1) vs **50.27%** (F0) ($\Delta = \mathbf{+0.0002}$, $+0.02\%$).
   - 95% Paired Date-Block Bootstrap Confidence Interval: **`[-0.0130, +0.0117]`** (spans zero).
   - One-sided Bootstrap $p$-value: **0.5339** | Holm-Bonferroni Adjusted $p$-value: **1.0000** (Not Significant).
   - McNemar Test $p$-value: **0.8657** (No significant classification discordance).
2. **Logistic Regression (LR-F1 vs LR-F0)**:
   - Date-Level Accuracy: **50.81%** (F1) vs **50.94%** (F0) ($\Delta = \mathbf{-0.0013}$, $-0.13\%$).
   - 95% Paired Date-Block Bootstrap Confidence Interval: **`[-0.0035, +0.0011]`** (spans zero).
   - One-sided Bootstrap $p$-value: **0.8468** | Holm-Bonferroni Adjusted $p$-value: **1.0000** (Not Significant).
   - McNemar Test $p$-value: **0.6690**.
3. **LSTM Sequential Model (LSTM-F1 vs LSTM-F0)**:
   - Date-Level Accuracy: **50.72%** (F1) vs **51.08%** (F0) ($\Delta = \mathbf{-0.0035}$, $-0.35\%$).
   - 95% Paired Date-Block Bootstrap Confidence Interval: **`[-0.0100, +0.0024]`** (spans zero).
   - One-sided Bootstrap $p$-value: **0.8741** | Holm-Bonferroni Adjusted $p$-value: **1.0000** (Not Significant).
   - McNemar Test $p$-value: **0.6217**.
4. **Behavioral De-Biasing Effect**:
   - While VADER features do not add predictive edge, they significantly altered XGBoost's prediction distribution: XGB-F0 exhibited near-total behavioral collapse (predicting "Up" 93.55% of the time), whereas XGB-F1 moderated to a balanced **55.42%** predicted-up share.

---

## 2. Ingestion Pipeline & News Provider Pilot

### 2.1 Provider Pilot Summary (GNews vs Event Registry)
Prior to historical data collection, a controlled 30-day pilot (2024-09-01 to 2024-09-30) was executed on 3 representative constituents (`RELIANCE`, `TCS`, `HDFCBANK`):
- **Event Registry**: Enforced a commercial paywall (`HTTP 401 Unauthorized` without paid key). A multi-year historical pull was projected to cost **$150 to $350+ USD** in commercial API credits.
- **GNews (via RSS Syndication)**: Delivered 189 high-quality Indian financial news articles, verified multi-year historical depth back to January 2020, authentic RFC 822 UTC timestamps, 79 distinct publishers, and **$0.00 cost**.
- **Formal Approval**: Approved for production historical collection on 2026-10-25.

### 2.2 Historical News Collection & Ingestion Stats
- **Historical Window**: January 1, 2020 through September 30, 2026 (81 monthly slices per ticker).
- **Target Universe**: 10 NIFTY 50 Heavyweights (`BHARTIARTL`, `HDFCBANK`, `HINDUNILVR`, `ICICIBANK`, `INFY`, `ITC`, `LT`, `RELIANCE`, `SBIN`, `TCS`).
- **Raw Articles Collected**: **46,809 records** logged in `data/raw/gnews/manifest.jsonl`.
- **Domain Cleaning & Pruning**: 768 social/non-editorial posts stripped; 261 syndicated duplicate headlines removed.
- **Cleaned Article Corpus**: **45,780 articles** (`data/processed/gnews/gnews_articles_clean.parquet`).
- **Daily Aggregated Ticker-Days**: **12,828 active news days** (`data/processed/gnews/daily_gnews_vader_features.parquet`).
- **Mean Articles per Active News Day**: **3.57**.

### 2.3 Strict Causal Cutoff & Zero-Leakage Verification
- **15:30 IST Cutoff**: Any article published on trading day $T$ before or at 15:30 IST attached to day $T$ (known at close of $T$ to predict $T \to T+1$).
- **Post-Close Roll Forward**: Any article published after 15:30 IST on day $T$, or on weekends/holidays, rolled forward to day $T+1$ (predicting $T+1 \to T+2$).
- **Empirical Proof**:
  - Future publication leaks (`pub_date > feature_date`): **0 (0.00%)**.
  - Post-15:30 same-day leaks (`pub_time > 15:30` on day $T$): **0 (0.00%)**.
  - Rolled forward articles: **7,899 (17.25%)**.
  - Documented in [`docs/audit/LEAKAGE_AND_TIMESTAMP_VALIDATION_REPORT.md`](file:///c:/Users/Neil/Desktop/FinFusionNeT/docs/audit/LEAKAGE_AND_TIMESTAMP_VALIDATION_REPORT.md).

---

## 3. Experimental Setup & Feature Sets

### 3.1 Feature Configurations
- **F0 (Historical Price Features Only)**:
  - 24 Group A Technical Indicators: 5 returns (`return_1d`, `return_2d`, `return_5d`, `return_10d`, `return_20d`), 5 moving average ratios (`close_sma_5_ratio`, `close_sma_10_ratio`, `close_sma_20_ratio`, `close_sma_50_ratio`, `sma_5_sma_20_ratio`), 4 oscillators & bands (`rsi_14`, `macd`, `macd_signal`, `macd_histogram`, `atr_14_pct`, `bb_width_20`), 4 volatility & volume metrics (`volatility_10`, `volatility_20`, `volume_change_1d`, `volume_ratio_5`, `volume_ratio_20`), and 3 intraday price action features (`high_low_range_pct`, `close_open_return`, `close_position_in_range`).
- **F1 (Price Features + VADER News Sentiment Fusion)**:
  - All 24 Group A Price Features.
  - 2 Group C News Availability Controls: `n_articles`, `has_news`.
  - 5 Group D VADER Sentiment Features: `vader_mean`, `vader_positive_mean`, `vader_negative_mean`, `vader_neutral_mean`, `vader_dispersion`.
  - Total: **31 features**.

### 3.2 Mandatory Missing Data Policy
On days where a stock has zero news coverage:
- `n_articles = 0`, `has_news = 0`.
- All `vader_*` sentiment features are strictly populated with `np.nan` (null).
- Missing news is **never** imputed as neutral (0.0) polarity. XGBoost handles missingness natively; linear and sequential models impute strictly using in-fold training means.

---

## 4. Comprehensive Empirical Results

All evaluations were executed across all **45 Protocol B monthly expanding folds** on a strictly aligned **common evaluation cohort of 45,136 observations across 916 out-of-sample trading dates**.

### 4.1 Model Classification Metrics

| Configuration | Date-Level Acc | Pooled Acc | Balanced Acc | Precision | Recall | F1-Score | ROC-AUC | Brier Score | Log Loss | Pred Up Share |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGB-F0 (Price-Only)** | 50.27% | 50.15% | 49.96% | 0.5019 | 0.9351 | 0.6532 | 0.5067 | 0.2503 | 0.6937 | 93.55% |
| **XGB-F1 (Price + VADER)** | **50.30%** | **50.39%** | **50.37%** | 0.5055 | 0.5579 | 0.5304 | **0.5086** | 0.2562 | 0.7064 | **55.42%** |
| **LR-F0 (Price-Only)** | **50.94%** | **50.82%** | **50.75%** | 0.5078 | 0.6724 | 0.5786 | **0.5102** | 0.2505 | 0.6942 | 66.49% |
| **LR-F1 (Price + VADER)** | 50.81% | 50.70% | 50.64% | 0.5070 | 0.6553 | 0.5717 | 0.5091 | 0.2506 | 0.6944 | 64.90% |
| **LSTM-F0 (Price-Only)** | **51.08%** | **50.96%** | **50.91%** | 0.5095 | 0.6297 | 0.5633 | **0.5149** | 0.2502 | 0.6935 | 62.07% |
| **LSTM-F1 (Price + VADER)** | 50.72% | 50.61% | 50.56% | 0.5067 | 0.6186 | 0.5571 | 0.5091 | 0.2505 | 0.6941 | 61.30% |
| *Baseline: Always-Up* | 50.34% | 50.21% | 50.00% | 0.5021 | 1.0000 | 0.6686 | 0.5000 | 0.4979 | 8.0244 | 100.00% |
| *Baseline: Prev-Day Dir* | 49.19% | 49.30% | 49.30% | 0.4952 | 0.4966 | 0.4959 | 0.4930 | 0.5070 | 8.1722 | 50.36% |

---

## 5. Statistical Significance & Inferential Hypothesis Testing

To test whether the performance differences between F1 and F0 are statistically meaningful or merely stochastic sampling variations, three complementary inferential procedures were applied:
1. **Paired Date-Block Bootstrapping**: 10,000 resamples using non-overlapping blocks of $L = 10$ trading days to preserve serial correlation in daily macro returns.
2. **Date-Level McNemar Tests**: Testing for discordance in daily win/loss classifications.
3. **Holm-Bonferroni Correction**: Correcting for family-wise multi-hypothesis testing across model architectures.

### 5.1 Hypothesis Testing Results

| Model Family | Baseline ($F0$) | Fusion ($F1$) | Date Acc Diff ($\Delta$) | 95% Block Bootstrap CI | Bootstrap $p$-value | McNemar $\Delta_{\text{wins}}$ | McNemar $p$-value | Holm-Bonferroni Adjusted $p$ | Significant at $\alpha=0.05$? |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGBoost** | XGB-F0 | XGB-F1 | **+0.0002** (+0.02%) | `[-0.0130, +0.0117]` | **0.5339** | -6 | 0.8657 | **1.0000** | **NO (Fail to Reject $H_0$)** |
| **Logistic Reg** | LR-F0 | LR-F1 | **-0.0013** (-0.13%) | `[-0.0035, +0.0011]` | **0.8468** | -12 | 0.6690 | **1.0000** | **NO (Fail to Reject $H_0$)** |
| **LSTM** | LSTM-F0 | LSTM-F1 | **-0.0035** (-0.35%) | `[-0.0100, +0.0024]` | **0.8741** | +15 | 0.6217 | **1.0000** | **NO (Fail to Reject $H_0$)** |

**Statistical Conclusion**: In all three model comparisons, the 95% bootstrap confidence interval spans zero, and all adjusted $p$-values are 1.0000. There is **zero statistically significant evidence** that VADER news sentiment improves predictive accuracy over price-only models.

---

## 6. Deep Forensic Analysis & Ablation Insights

### 6.1 Subgroup Analysis: Active News Days vs No-News Days
To isolate whether sentiment had localized power on days when news was actually published, the 45,136 test predictions were stratified into:
- **Active News Days (N = 7,394 observations, 16.4%)**:
  - XGB-F0: 50.00% $\to$ XGB-F1: **50.73%** ($\Delta = \mathbf{+0.0073}$, $+0.73\%$)
  - LR-F0: 50.74% $\to$ LR-F1: **50.61%** ($\Delta = -0.0014$, $-0.14\%$)
  - LSTM-F0: 50.34% $\to$ LSTM-F1: **50.57%** ($\Delta = +0.0023$, $+0.23\%$)
- **No-News Days (N = 37,742 observations, 83.6%)**:
  - XGB-F0: 50.17% $\to$ XGB-F1: **50.33%** ($\Delta = +0.0015$)
  - LR-F0: 50.84% $\to$ LR-F1: **50.72%** ($\Delta = -0.0012$)
  - LSTM-F0: 51.09% $\to$ LSTM-F1: **50.62%** ($\Delta = -0.0047$)

**Insight**: Even when conditioned strictly on active news days, the directional gain from VADER for XGBoost is minor (+0.73%) and for LSTM is marginal (+0.23%), while Logistic Regression degrades. When diluted across the 83.6% no-news trading days, the net system-wide impact is imperceptible.

### 6.2 Fold-Level Stability Across 45 Walk-Forward Folds
Evaluating fold-by-fold accuracy variance (`experiments/experiment_3/experiment_3_fold_results.csv`):
- **XGB-F0 Fold Accuracy**: Mean = 0.5017, Standard Deviation = **0.0525**
- **XGB-F1 Fold Accuracy**: Mean = 0.5041, Standard Deviation = **0.0220** (Stability improved)
- **LR-F0 Fold Accuracy**: Mean = 0.5082, Standard Deviation = **0.0316**
- **LR-F1 Fold Accuracy**: Mean = 0.5070, Standard Deviation = **0.0318**
- **LSTM-F0 Fold Accuracy**: Mean = 0.5101, Standard Deviation = **0.0324**
- **LSTM-F1 Fold Accuracy**: Mean = 0.5066, Standard Deviation = **0.0330**

### 6.3 Why Does VADER Fail in Short-Term Financial Direction Prediction?
Forensic inspection of headlines and scores reveals three fundamental linguistic failure modes of rule-based lexicons in financial markets:
1. **Financial Polysemy**: VADER was built on social media text (Twitter/Amazon reviews). Financial terms with severe market implications are scored inappropriately:
   - *"Company increases crude oil reserves"* $\to$ positive operational news, but often scored as neutral.
   - *"Interest rate hike"* $\to$ negative for equities, but VADER interprets "interest" as curiosity/engagement (positive).
   - *"Share buyback approved"* $\to$ strong bullish signal, but "buyback" is unlisted in standard VADER sentiment lexicons.
2. **Context-Insensitivity to Market Expectations**: A headline announcing *"Company Q3 profit rises 10%"* appears positive to VADER (positive compound score $\approx +0.45$). However, if consensus analyst estimates expected a 20% rise, the stock drops -4% at market open. VADER cannot capture divergence from expectations.
3. **Low Signal-to-Noise Ratio**: General news headlines contain high volume of peripheral corporate announcements (CSR events, sponsorships, routine management quotes) that carry zero price-formation signal.

---

## 7. Artifacts & Deliverables Summary

All scripts, datasets, predictions, and reports have been produced and serialized:
1. **Provider Pilot Report**: [`docs/audit/NEWS_PROVIDER_PILOT_REPORT.md`](file:///c:/Users/Neil/Desktop/FinFusionNeT/docs/audit/NEWS_PROVIDER_PILOT_REPORT.md)
2. **Leakage & Cutoff Validation Report**: [`docs/audit/LEAKAGE_AND_TIMESTAMP_VALIDATION_REPORT.md`](file:///c:/Users/Neil/Desktop/FinFusionNeT/docs/audit/LEAKAGE_AND_TIMESTAMP_VALIDATION_REPORT.md)
3. **Raw News Dataset**: `data/raw/gnews/raw_articles.parquet` (46,809 records across 10 heavyweights).
4. **Collection Manifest**: `data/raw/gnews/manifest.jsonl` (810 monthly chunks logged).
5. **Cleaned Article Dataset**: `data/processed/gnews/gnews_articles_clean.parquet` (45,780 articles).
6. **Daily VADER Features Dataset**: `data/processed/gnews/daily_gnews_vader_features.parquet` (12,828 ticker-days).
7. **Experiment 3 Common Cohort Predictions**: `experiments/experiment_3/predictions/experiment_3_common_cohort_predictions.parquet` (45,136 observations).
8. **Classification Metrics Table**: `experiments/experiment_3/experiment_3_results.csv`.
9. **Statistical Significance Table**: `experiments/experiment_3/experiment_3_statistical_significance.csv`.
10. **Fold-Level Results**: `experiments/experiment_3/experiment_3_fold_results.csv`.
11. **Unit Test Suite**: `tests/test_experiment_3_vader.py` (7/7 passed).
12. **Reproducible Pipeline Scripts**:
    - `scripts/collect_historical_gnews.py`: Resumable historical collector.
    - `scripts/process_gnews_vader.py`: Cleaning, cutoff alignment, and VADER extraction.
    - `scripts/run_experiment_3_vader.py`: Walk-forward Protocol B model evaluation.
    - `scripts/compute_fold_stability.py`: Stability and active news day breakdown.

---

## 8. Conclusion & Academic Positioning

Experiment 3 delivers an unambiguous, methodologically rigorous, and scientifically honest outcome:
**Simple lexicon-based news sentiment (VADER) combined with basic feature concatenation does not improve short-term stock direction forecasting in the Indian stock market over price-only baselines.**

This null result confirms the central thesis hypothesis of FinFusionNet: simple dictionary sentiment counts are insufficient to overcome market noise. This directly motivates **Experiment 4**, where domain-specific financial language representations (**FinBERT**) are evaluated under the identical controlled ablation framework.
