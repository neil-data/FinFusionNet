# FinFusionNet — XGBoost Track Methodology

## 1. Executive Summary & Research Objective
FinFusionNet's XGBoost experimental track investigates whether financial news sentiment (lexicon-based VADER and transformer-based FinBERT) provides statistically defensible predictive lift for next-trading-session Indian equity price direction beyond structured technical indicators, market context, and news attention controls.

The research prioritizes:
1. **Data Integrity & Point-in-Time Realism**: Survivorship-bias-free dynamic NIFTY 50 membership (2020-2026).
2. **Leakage-Safe Feature Construction**: Causal rolling windows; strict NSE 15:30 IST market close cutoff; target-date purging.
3. **Fair Evaluation**: Identical common evaluation cohort across all experimental variants (X1–X6).
4. **Statistical Rigor**: Preserving cross-sectional correlation and temporal dependence via paired date-block bootstrap rather than row-level i.i.d. assumptions.

---

## 2. Mathematical Target Formulation

### Primary Target: Next-Session Direction (Binary)
Let $t$ index valid trading sessions according to the National Stock Exchange of India (NSE) trading calendar.
Let $Close_{i,t}$ be the official closing price of stock $i$ at the end of session $t$ (15:30 IST).
Let $t+1$ be the *next valid trading session* (accounting for weekends, exchange holidays, and special sessions like Muhurat trading).

The forward return is defined as:
$$R_{i, t+1} = \frac{Close_{i, t+1}}{Close_{i, t}} - 1$$

The primary binary target $Y_{i, t}$ is formulated as:
$$Y_{i, t} = \begin{cases} 1 & \text{if } R_{i, t+1} > 0 \\ 0 & \text{if } R_{i, t+1} \le 0 \end{cases}$$

*Note*: Flat sessions ($R_{i, t+1} = 0$) are strictly mapped to class 0 to ensure conservative classification.

### Secondary Target: Market-Relative Direction
To determine whether sentiment identifies stock-specific alpha rather than broad market beta, the secondary relative return target is defined relative to the NIFTY 50 index return ($R_{NIFTY, t+1}$):
$$R^{rel}_{i, t+1} = R_{i, t+1} - R_{NIFTY, t+1}$$
$$Y^{rel}_{i, t} = \begin{cases} 1 & \text{if } R^{rel}_{i, t+1} > 0 \\ 0 & \text{if } R^{rel}_{i, t+1} \le 0 \end{cases}$$
*The secondary target never replaces the primary target; it serves as an auxiliary robustness track.*

---

## 3. Data Strategy & Point-in-Time Universe

### Temporal Horizons
- **Warm-up Period**: `2020-01-01` to `2020-12-31` (used strictly to prime 50-day rolling indicators like SMA-50, MACD, ATR-14, and volatility).
- **Active Modeling Period**: `2021-01-01` to `2026-09-30` (~1,430 trading days).
- **Freeze Date**: `2026-09-30`.

### Point-in-Time Constituent Management
Using a static, modern list of NIFTY 50 stocks introduces acute survivorship bias (selecting historical winners). FinFusionNet tracks point-in-time constituent status using documented semi-annual index rebalancings (62 distinct tickers across the modeling window):
- A stock observation $(i, t)$ enters the dataset if and only if:
  1. Stock $i$ was an active, listed constituent of NIFTY 50 on date $t$.
  2. Required 50-day warm-up price series exists.
  3. Valid $Close_{i, t+1}$ is available on the next trading session.
- Stocks exiting the index cease generating observations after their effective replacement date.

---

## 4. Feature Engineering Architecture

Features are engineered into five distinct, decoupled modular groups:

### Group A: Technical Indicators (Causal Rolling Windows)
All indicators are computed strictly backward-looking from session $t$ close ($Close_t, High_t, Low_t, Open_t, Volume_t$):
- **Returns**: 1-day, 2-day, 5-day, 10-day, 20-day percentage returns.
- **Trend Ratios**: $Close / SMA_k$ for $k \in \{5, 10, 20, 50\}$; $SMA_5 / SMA_{20}$.
- **Momentum**: 14-period RSI; standard MACD (12, 26, 9) line, signal, and histogram.
- **Volatility**: 14-period ATR percentage ($ATR / Close$); 20-period Bollinger Band width; 10-day and 20-day close-to-close annualized realized volatility.
- **Volume Dynamics**: 1-day volume percentage change; Volume relative to 5-day and 20-day moving average.
- **Intraday Price Action**: High-Low range percentage ($(High - Low) / Close$); Close-Open return ($(Close - Open) / Open$); Close position within range ($(Close - Low) / (High - Low + \epsilon)$).

### Group B: Market Context
Broad market conditions derived from the NIFTY 50 index series on date $t$:
- `nifty_return_1d`, `nifty_return_5d`, `nifty_return_20d`
- `nifty_volatility_10`, `nifty_volatility_20`

### Group C: News Availability Controls
To separate true sentiment content from simple media attention/chatter volume:
- `n_articles`: Total relevant deduplicated articles mapped to ticker $i$ on date $t$.
- `has_news`: Binary indicator ($1$ if $n\_articles > 0$, else $0$).

### Group D: VADER Sentiment Features
Lexicon-based sentiment scores aggregated across all articles assigned to ticker $i$ in session $t$:
- `vader_mean`: Mean compound score.
- `vader_positive_mean`, `vader_negative_mean`, `vader_neutral_mean`
- `vader_dispersion`: Standard deviation of compound scores across articles.

### Group E: FinBERT Sentiment Features
Domain-adapted transformer probability distributions across positive, negative, and neutral classes:
- `finbert_positive_mean`, `finbert_negative_mean`, `finbert_neutral_mean`
- `finbert_dispersion`: Standard deviation of net sentiment across articles.
- `finbert_confidence`: Mean maximum class probability across articles.

### Ticker Categorical Encoding
- Ticker identity is encoded via `OneHotEncoder(handle_unknown="ignore")`.
- **Strict Leakage Prevention**: The encoder is fit *strictly* on the training fold and transforms validation/test folds out-of-sample.

---

## 5. Missing Data & Null Value Policy
1. **Technical Warm-up**: Rows prior to 2021-01-01 or with incomplete 50-day warm-up are dropped.
2. **Legitimate Sentiment Missingness**:
   - If `has_news == 0`: `n_articles = 0`, `has_news = 0`, and all VADER / FinBERT sentiment features are set to `np.nan` (null).
   - **Prohibition**: Missing sentiment is NEVER replaced with 0.0 or interpreted as neutral sentiment (which would corrupt neutral probability distributions).
   - XGBoost natively handles `NaN` values by learning optimal default branch directions during split finding (`tree_method="hist"`).
3. **Data Quality Failures**: If `n_articles > 0` but sentiment values are null, records are flagged, logged, and isolated for pipeline inspection.
4. **Infinite Values**: Any infinite values resulting from division by zero are converted to null and logged.

---

## 6. Model Training & Evaluation Protocols

### XGBoost Hyperparameter Baseline
```yaml
objective: binary:logistic
eval_metric: [auc, logloss]
n_estimators: 500
learning_rate: 0.03
max_depth: 4
min_child_weight: 5
subsample: 0.8
colsample_bytree: 0.8
reg_alpha: 0.1
reg_lambda: 2.0
gamma: 0.0
tree_method: hist
n_jobs: 4
random_state: 42
```
Classification decisions use a fixed out-of-sample threshold of $0.50$ ($P(Y=1) > 0.50$). Threshold tuning against test partitions is strictly prohibited.

### Protocol A: Historical Rolling Benchmark
- Train window: 46 trading sessions (~2 months).
- Test window: 15 trading sessions (~3 weeks).
- Step size: 17 trading sessions.
- Fixed 500 estimators (no early stopping).
- Evaluates consistency against historical benchmarks across up to 34 rolling folds (622 total sessions).

### Protocol B: Monthly Expanding Window (Primary Track)
- Initial training window: 24 months (2021-01-01 to 2022-12-31).
- Out-of-sample test window: 1 calendar month.
- Step size: 1 calendar month (~45 test folds through 2026-09-30).
- Early stopping: 50 rounds on internal validation partition (last 15% chronological training dates, minimum 60 sessions), optimizing `logloss`.
- **Target-Date Purging**: A training record at date $t$ is retained only if its label date $t+1 < \min(t_{val})$. Similarly, validation records must purge overlapping horizon dates before the test fold.

---

## 7. Statistical Inference Protocol

### Equal-Weighted Date-Level Accuracy
Because multiple stocks on the same trading day are cross-sectionally correlated (market beta shock), individual stock-day observations are not independent. The primary evaluation metric is date-level accuracy:
$$Acc_d = \frac{1}{N_d} \sum_{i=1}^{N_d} \mathbb{I}(\hat{Y}_{i,d} = Y_{i,d})$$
Each date $d$ receives equal weight:
$$Acc_{mean} = \frac{1}{D} \sum_{d=1}^D Acc_d$$

### Paired Date-Block Bootstrap
To test the pre-registered primary hypothesis ($H_1: Acc(X5) > Acc(X3)$ under Protocol B):
1. Compute daily paired differences $\Delta_d = Acc_{X5, d} - Acc_{X3, d}$ for each test date $d \in \{1, \dots, D\}$.
2. Resample contiguous calendar/trading blocks of length $L$ (e.g., $L=5$ or $10$ trading days) with replacement for 10,000 bootstrap iterations.
3. Compute 95% bootstrap confidence intervals $[CI_{0.025}, CI_{0.975}]$ and two-sided empirical p-values.
4. Conduct block-length sensitivity analysis across $L \in \{3, 5, 10, 15, 20\}$.
5. Secondary multiple hypothesis comparisons apply the Holm-Bonferroni step-down correction.
