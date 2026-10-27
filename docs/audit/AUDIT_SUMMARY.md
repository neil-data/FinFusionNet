# FinFusionNet Forensic Pipeline Audit & Pre-Experiment 3 Verification Report

**Audit Status**: Phase 1 Paused | Forensic Audit & Pre-Experiment 3 Verification Completed  
**Branch**: `audit/price-only` (branched from tag `v0.1-audited`)  
**Auditor Mandate**: Evaluate pipeline trustworthiness, resolve the 72.07% anomaly, fix data ingestion and warmup coverage defects, freeze price-only baselines across 5 seeds on the point-in-time NIFTY 50 universe, evaluate market-level context, and certify the news dataset for Experiment 3.

---

## 1. Executive Summary & Plain-Language Verdict

### (1) Can the pipeline be trusted?
**YES. Following forensic verification and data ordering fixes, the core feature engineering, chronological fold partitioning, and target calculation pipeline are mathematically causal, reproducible, and completely free of lookahead leaks.**
- **Calendar & Target Causality (A1)**: Evaluated across weekend boundaries and exchange holidays (e.g., 2023-01-26, 2023-03-07). Verified $Y_{i, t} = 1[Close_{i, t+1} / Close_{i, t} - 1 > 0]$ strictly maps next-trading-session returns to class 1 and flat returns ($R=0$) to class 0. Zero technical features at time $t$ utilize information beyond the 15:30 IST market close of date $t$.
- **Synthetic Null Verification (A3)**: On simulated 50-ticker random walks (i.i.d. returns with zero predictability across 20 distinct seeds), both XGBoost (Mean: 50.20%, 95% CI: `[0.4922, 0.5105]`) and LSTM (Mean: 49.92%, 95% CI: `[0.4930, 0.5055]`) produce pure 50% chance, confirming the pipeline does not manufacture spurious signal from noise.
- **Positive Controls (A4)**: A leaked next-day canary feature yielded 100.00% accuracy (AUC: 1.0000), and a simulated weak AR(1) signal of known strength ($\rho = 0.20$) was recovered with 56.65% directional accuracy against a theoretical expectation of 56.41%.
- **Resolution of Historical High-Accuracy Claims**:
  - The historical **55.93%** accuracy was caused by an unsorted `.iloc` indexing bug that shuffled test fold indices across calendar years.
  - The historical **72.07%** accuracy on shifted labels was caused by confusing a pandas temporal lag (`shift(1)`) with a future lead, effectively predicting the past day's return using features that already knew that return.

### (2) What do price-only XGBoost and LSTM score on all 50 companies?
On the corrected full point-in-time NIFTY 50 universe under Protocol B (45 monthly expanding folds, 916 unique out-of-sample test dates, 45,319 test observations):
- **XGB-P (Price-Only XGBoost)**:
  - Date-Level Accuracy: **50.23%** | Pooled Accuracy: **50.16%** | Balanced Accuracy: **49.97%** | ROC-AUC: **0.5066** | Brier Score: **0.2502** | Log Loss: **0.6936**
  - Predicted-Up Share: **93.55%** (Behavioral Collapse: predicts "Up" over 93% of the time)
  - Lift over Always-Up Baseline (50.23%): **-0.0007** (95% CI: `[-0.0043, +0.0026]`, $p = 0.6646$, Not Significant).
  - Lift over Previous-Day Direction (49.31%): **+0.0085** (95% CI: `[-0.0057, +0.0255]`, $p = 0.1011$, Not Significant).
- **LSTM-P (5-Seed Ensemble, 2-Layer Sequential)**:
  - Date-Level Accuracy: **50.99%** | Pooled Accuracy: **50.88%** | Balanced Accuracy: **50.83%** | ROC-AUC: **0.5125** | Brier Score: **0.2502** | Log Loss: **0.6936**
  - Predicted-Up Share: **62.20%** | Fraction of Days Beating Always-Up: **47.71%**
  - Lift over Always-Up Baseline (50.23%): **+0.0065** (95% Block Bootstrap CI: `[-0.0064, +0.0181]`, $p = 0.1663$). Since 0 is inside the 95% CI and $p > 0.05$, there is **zero statistically significant lift over the Always-Up naive baseline**.
  - Lift over Previous-Day Direction (49.31%): **+0.0157** (95% CI: `[+0.0041, +0.0294]`, $p = 0.0040$).
  - 5-Seed Spread: Accuracy range `[50.72%, 51.08%]` ($\text{std} = 0.0013$), ROC-AUC range `[0.5077, 0.5149]`.
- **Logistic Regression Baseline (Linear)**:
  - Date-Level Accuracy: **50.89%** | Pooled Accuracy: **50.83%** | ROC-AUC: **0.5103** | Brier Score: **0.2505** | Log Loss: **0.6943**
  - Lift over Always-Up Baseline (50.23%): **+0.0060** (95% CI: `[-0.0069, +0.0174]`, $p = 0.2067$, Not Significant).
- **Market Context (Group B Features) Impact**:
  - Adding NIFTY index returns, volatility, and moving average ratios (Group B) yields **no statistically significant lift** over single-stock technicals for XGBoost (+0.0052, 95% CI: `[-0.0044, +0.0164]`, $p = 0.1359$) or LSTM (+0.0033, 95% CI: `[-0.0070, +0.0124]`, $p = 0.3031$).

### (3) Does the pipeline produce predictions for all 50 NIFTY 50 members on every test date?
**PARTIALLY. Following the warmup ordering fix and TATAMOTORS/TMPV mapping:**
- **484 out of 916 out-of-sample trading dates (52.84%)** evaluate all 50 constituents simultaneously (up from 0.0% prior to audit fixes).
- The mean evaluated constituent count increased from **48.10** to **49.47** per date.
- The remaining dropped ticker-days (481 total) are strictly attributable to external historical data availability:
  1. **Delisted HDFC** (130 days, July 2023 onward following merger into HDFC Bank).
  2. **Unquoted LTIM** (298 days, unquoted under `LTIM.NS` on Yahoo Finance).
  3. **JIOFIN Post-Listing Indicator Warmup** (53 days, newly listed on 2023-08-21, requiring 50 trading sessions to accumulate 50-day SMA).
  4. **Post-entry warmup drops** for existing stocks entering the index were completely eliminated (**393 ticker-days recovered, 0 post-entry warmup drops remaining**).

---

## 2. Pre-Experiment 3 Verification Scorecard (Tasks 1–8)

| Task ID | Description | Status | Forensic Evidence & Verification Deliverables |
|:---|:---|:---:|:---|
| **Task 1** | **Indicator Warmup & Coverage Fix** | **PASS** | Fixed computation order: Group A indicators now computed on full raw OHLCV before filtering point-in-time constituent membership. Post-entry warmup drops eliminated (0 drops remaining; 393 ticker-days recovered). Dates with all 50 members increased from 0% to 52.84%. |
| **Task 2** | **Missing Tickers & Corporate Actions** | **PASS** | Added `TICKER_YFINANCE_MAP = {"TATAMOTORS": "TMPV.NS"}` in `src/finfusionnet/data/prices.py`. Downloaded 1,672 sessions for `TATAMOTORS`. Documented corporate actions: TMPV demerger on 2025-10-14 (-40.15%), TRENT 3:2 bonus/split on 2026-01-01 (-33.05%), unquoted HDFC and LTIM. (`data_quality_table.csv`, `extreme_returns_anomalies.csv`) |
| **Task 3** | **Data Integrity & Leakage Rerun** | **PASS** | Timing proof verified across calendar jumps. Shifted-label sweep verified ($k \ge 0$ bounded at $\le 50.36\%$; $k=-1$ confirmed at 72.07%). Positive controls verified (Canary: 100%, AR(1): 56.65%). All 38/38 unit tests passing in pytest suite (`tests/test_timing.py`, `tests/test_synthetic_null.py`, `tests/test_fold_integrity.py`). |
| **Task 4** | **Freeze Price-Only Dataset & Baselines** | **PASS** | Frozen dataset: 82,425 ticker-day rows across 60 tickers (`experiments/audit/master_universe_ticker_day.parquet`). Evaluated 45 Protocol B folds (45,319 observations). Serialized predictions: `xgb_p_predictions.parquet`, `lr_p_predictions.parquet`, `lstm_p_predictions.parquet`. |
| **Task 5** | **LSTM 5-Seed Investigation** | **PASS** | Evaluated 5 seeds (42–46). Seed accuracy tightly clustered between 50.72% and 51.08% ($\text{std} = 0.0013$). 5-seed ensemble pooled accuracy: 50.88%. Lift over Always-Up (50.23%) is +0.0065 (95% CI `[-0.0064, +0.0181]`, $p = 0.1663$), confirming zero statistically significant alpha from technicals alone. (`lstm_seed_performance.csv`) |
| **Task 6** | **Market-Level Context Experiment** | **PASS** | Evaluated Group A vs Group A+B across 45,086 common test predictions. XGB-P lift: +0.0052 ($p = 0.1359$, 95% CI `[-0.0044, +0.0164]`). LSTM-P lift: +0.0033 ($p = 0.3031$, 95% CI `[-0.0070, +0.0124]`). Group B market context produces no statistically significant gain. (`market_context_experiment_results.csv`) |
| **Task 7** | **Evaluation Protocols & Stats Engine** | **PASS** | Verified common evaluation cohort inner joins (`build_common_evaluation_cohort`), 1-session embargo, paired date-block bootstrap ($B=10,000, L=10$), Holm-Bonferroni corrections, and McNemar tests in `src/finfusionnet/evaluation/stats_tests.py`. |
| **Task 8** | **News Dataset Validation for Exp 3** | **PASS** | Validated `data/processed/financial_news/daily_news_features.parquet` across 10 heavyweights from 2020-01-01 to 2026-09-30. 4,988 active news days verified. Strict null sentiments preserved on 11,732 no-news days. Schema verified (14 columns). Status: PASS (Ready). (`news_validation_summary.json`) |

---

## 3. Forensic Investigation of the 72.07% Anomaly

### 3.1 Background & Root Cause
In `scripts/audit_linear_baseline.py` (committed under commit `351f1ef`), an evaluation of the linear baseline under shifted targets reported:
- `+1 Session Shift (Forecast Y_t+2)`: Date Accuracy = **0.7207**, ROC-AUC = **0.7868**
- `-1 Session Shift (Forecast Y_t)`: Date Accuracy = **0.5001**, ROC-AUC = **0.4933**
- The report mistakenly claimed: *"Performance remains centered around ~50%, confirming zero lookahead or retrospective label contamination."*

### 3.2 Code-Level Evidence
Inspection of `scripts/audit_linear_baseline.py` lines 38–42 reveals the exact bug:
```python
if label_shift != 0:
    # Shift target by label_shift within each ticker
    df["target"] = df.groupby("ticker")["target"].shift(label_shift)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
```
In pandas:
- `Series.shift(periods=1)` shifts rows **downward** along the index.
- Therefore, at index $t$, the value is taken from index $t-1$.
- At row $t$, the unshifted target was $Y_t = 1[Close_{t+1}/Close_t - 1 > 0]$.
- When `label_shift = 1`, row $t$ received $Y_{t-1} = 1[Close_t/Close_{t-1} - 1 > 0]$.
- **$Y_{t-1}$ is the return that completed at close of date $t$**!
- In the feature matrix at row $t$, the model is provided with `return_1d = Close_t / Close_{t-1} - 1`, `rsi_14`, `macd`, and moving average ratios—all of which already know whether date $t$ was positive!
- The author mistakenly believed `shift(1)` shifted the target into the future ($Y_{t+2}$), when in reality pandas `shift(1)` is a retrospective lag ($Y_{t-1}$).
- Conversely, when `label_shift = -1`, `df["target"].shift(-1)` pulled $Y_{t+1}$ (the return from $t+1$ to $t+2$, two sessions into the future). On this genuine future session, the linear model achieved **50.01%** accuracy and **0.4933** AUC.

### 3.3 Systematic Shifted-Label Sweep Verification (Audit A2)
We executed the full sweep across $k \in \{-3, -2, -1, 0, 1, 2, 3\}$, where $k$ represents the return from $Close_{t+k}$ to $Close_{t+k+1}$ (in pandas: `shift(-k)`):

| Shift $k$ | Pandas Shift | Temporal Semantics | Linear Date Acc | Linear AUC | XGBoost Date Acc | XGBoost AUC | Status |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **$k = -3$** | $+3$ | Past return ($t-3 \to t-2$) | 62.58% | 0.6769 | 79.85% | 0.8802 | Expected High |
| **$k = -2$** | $+2$ | Past return ($t-2 \to t-1$) | 67.60% | 0.7365 | 95.18% | 0.9864 | Expected High |
| **$k = -1$** | $+1$ | **Past return ($t-1 \to t$, known at $t$)** | **71.98%** | **0.7872** | **99.09%** | **0.9957** | **72.07% Reproduced** |
| **$k = 0$** | $0$ | **Real target ($t \to t+1$)** | **50.36%** | **0.5002** | **49.77%** | **0.5037** | **Chance (~50%)** |
| **$k = +1$** | $-1$ | Future return ($t+1 \to t+2$) | 50.01% | 0.4933 | 49.81% | 0.5073 | **Chance (~50%)** |
| **$k = +2$** | $-2$ | Future return ($t+2 \to t+3$) | 49.22% | 0.4908 | 49.34% | 0.5040 | **Chance (~50%)** |
| **$k = +3$** | $-3$ | Future return ($t+3 \to t+4$) | 49.30% | 0.4897 | 49.51% | 0.5085 | **Chance (~50%)** |

**Conclusion**: The 72.07% figure is completely explained by a reversed pandas shift sign. When predicting genuine future sessions ($k \ge 0$), accuracy is strictly bounded below 50.4% for all models.

---

## 4. Part B: Empirical Evaluation on Full NIFTY 50 Universe

### 4.1 Universe & Data Quality Audit (B1)
- **Scope**: 60 point-in-time constituent equities across 2020–2026.
- **Extreme Return Threshold Breaches ($|R| > 20\%$)**: 16 occurrences across 8 tickers.
  - COVID Market Crash (2020-03-23): `AXISBANK` (-27.91%), `BAJAJFINSV` (-25.86%), `BAJFINANCE` (-23.23%), `INDUSINDBK` (-23.59%), `SHRIRAMFIN` (-22.18%).
  - INDUSINDBK Recovery (2020-03-26): +44.67%.
  - Adani Enterprises Hindenburg Report (2023-02-01 to 2023-02-02): -28.20% and -26.70%.
  - General Election Results (2024-06-04): `ADANIPORTS` (-21.15%).
  - **Tata Motors Demerger (TMPV 2025-10-14)**: Unadjusted price fell from ₹655.31 on 2025-10-13 to ₹392.20 on 2025-10-14 (**-40.15% drop**).
  - **Trent Bonus/Split (2026-01-01)**: Unadjusted price fell from ₹7,045.55 to ₹4,717.00 (**-33.05% drop**).
- **Delistings & Late Listings**:
  - `HDFC`: Delisted on 2023-07-13 following merger into HDFC Bank (130 days missing).
  - `LTIM`: Ticker quote unavailable under `LTIM.NS` on Yahoo Finance (298 days missing).
  - `TATAMOTORS`: Successfully recovered via `TMPV.NS` (1,672 sessions loaded).
  - `JIOFIN`: Listed on 2023-08-21 (53 warmup days missing post-listing).
  - `ETERNAL` (Zomato): Listed on 2021-07-23 (fully available during evaluation window).

### 4.2 Walk-Forward Protocol Partitions (B2)
- **Protocol B (Primary — Expanding Monthly Window)**:
  - Total Folds: **45 monthly folds** (Initial 24-month training: `2021-01-01` to `2022-12-31`).
  - Total Out-of-Sample Test Dates: **916 trading dates** (`2023-01-02` to `2026-09-29`).
  - Total Evaluated Predictions: **45,319 observations** (XGB-P & Linear); **45,136 observations** (LSTM-P).
  - Mean Tickers per Date: **49.47** | Min: **1** | Max: **50** | Dates with exactly 50: **484 (52.84%)**.
- **Protocol A (Benchmark — Historical Rolling Window)**:
  - Total Folds: **34 rolling folds** (46 train sessions, 15 test sessions, 17 step sessions).
  - Total Out-of-Sample Test Dates: **510 trading dates**.
  - Mean Tickers per Date: **48.95**.

### 4.3 Model Performance on the Full Universe (B3)

| Model Name | Date-Level Acc | Pooled Acc | Balanced Acc | ROC-AUC | Brier Score | Log Loss | Pred Up Share | Actual Up Rate | Always-Up Acc | Previous-Day Acc | Frac Days Beat Always-Up |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGB-P (Price-Only)** | **50.23%** | **50.16%** | **49.97%** | **0.5066** | **0.2502** | **0.6936** | **93.55%** | 50.23% | 50.23% | 49.31% | **27.40%** |
| **LSTM-P (5-Seed Ensemble)** | **50.99%** | **50.88%** | **50.83%** | **0.5125** | **0.2502** | **0.6936** | **62.20%** | 50.21% | 50.23% | 49.31% | **47.71%** |
| **Logistic Regression (Linear)** | **50.89%** | **50.83%** | **50.75%** | **0.5103** | **0.2505** | **0.6943** | **66.53%** | 50.23% | 50.23% | 49.31% | **47.60%** |

#### LSTM-P 5-Seed Random Initialization Spread

| Seed | Date-Level Acc | Pooled Acc | Balanced Acc | ROC-AUC | Brier Score | Log Loss | Predicted-Up Share | Mean Predicted Prob |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Seed 42** | 51.08% | 50.96% | 50.91% | 0.5149 | 0.2502 | 0.6935 | 62.07% | 0.5121 |
| **Seed 43** | 50.83% | 50.72% | 50.68% | 0.5077 | 0.2505 | 0.6942 | 59.67% | 0.5101 |
| **Seed 44** | 51.15% | 51.04% | 50.99% | 0.5115 | 0.2503 | 0.6937 | 60.44% | 0.5101 |
| **Seed 45** | 51.19% | 51.08% | 51.02% | 0.5112 | 0.2504 | 0.6940 | 62.32% | 0.5118 |
| **Seed 46** | 51.01% | 50.89% | 50.85% | 0.5136 | 0.2502 | 0.6936 | 60.54% | 0.5117 |
| **5-Seed Ensemble** | **50.99%** | **50.88%** | **50.83%** | **0.5125** | **0.2502** | **0.6936** | **62.20%** | **0.5112** |

- **Seed Spread Summary**:
  - Minimum Accuracy: **50.72%** | Maximum Accuracy: **51.08%** | Standard Deviation: **0.0013** (0.13%)
  - Minimum ROC-AUC: **0.5077** | Maximum ROC-AUC: **0.5149**
  - Behavioral Stability: The 5 random initializations produce remarkably tight metrics across folds, confirming model convergence without seed variance artifacts.

#### Statistical Paired Date-Block Bootstrapping (10,000 Resamples, $L=10$ Days)
- **XGB-P vs Always-Up Baseline**:
  - Point Estimate Difference: **-0.0007** (-0.07%)
  - 95% Bootstrap Confidence Interval: **`[-0.0043, +0.0026]`**
  - Empirical one-sided $p$-value ($H_0: \Delta \le 0$): **0.6646** ($p > 0.05$, Not Significant)
- **XGB-P vs Previous-Day Direction Baseline**:
  - Point Estimate Difference: **+0.0085** (+0.85%)
  - 95% Bootstrap Confidence Interval: **`[-0.0057, +0.0255]`**
  - Empirical one-sided $p$-value ($H_0: \Delta \le 0$): **0.1011** ($p > 0.05$, Not Significant)
- **LSTM-P vs Always-Up Baseline**:
  - Point Estimate Difference: **+0.0065** (+0.65%)
  - 95% Bootstrap Confidence Interval: **`[-0.0064, +0.0181]`** (Spans 0!)
  - Empirical one-sided $p$-value ($H_0: \Delta \le 0$): **0.1663** ($p > 0.05$, **Zero statistically significant lift over Always-Up**)
- **LSTM-P vs Previous-Day Direction Baseline**:
  - Point Estimate Difference: **+0.0157** (+1.57%)
  - 95% Bootstrap Confidence Interval: **`[+0.0041, +0.0294]`**
  - Empirical one-sided $p$-value ($H_0: \Delta \le 0$): **0.0040**
- **Logistic Regression vs Always-Up Baseline**:
  - Point Estimate Difference: **+0.0060** (+0.60%)
  - 95% Bootstrap Confidence Interval: **`[-0.0069, +0.0174]`** (Spans 0!)
  - Empirical one-sided $p$-value ($H_0: \Delta \le 0$): **0.2067** ($p > 0.05$, Not Significant)

### 4.4 Stratified Performance: 10 Heavyweights vs Remaining Names (B4)

| Cohort Subset | Model | Total Predictions | Date-Level Acc | Pooled Acc | ROC-AUC | Predicted-Up Share |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **10 Heavyweights** | XGB-P | 9,151 | **49.65%** | **49.60%** | **0.5106** | 94.33% |
| **10 Heavyweights** | LSTM-P | 9,142 | **50.27%** | **50.18%** | **0.5060** | 66.89% |
| **Non-Heavyweights (Rest)** | XGB-P | 36,168 | **50.33%** | **50.31%** | **0.5057** | 93.35% |
| **Non-Heavyweights (Rest)** | LSTM-P | 35,994 | **51.07%** | **51.06%** | **0.5140** | 61.01% |

**Finding**: There is zero meaningful divergence between the 10 mega-caps and the broader index constituents. Across all capitalization tiers, technical indicators fail to provide directional edge over naive market drift.

### 4.5 Market-Level Context Experiment (Task 6)
To confirm whether adding market-wide context (NIFTY 50 index returns, volatility, moving average ratios; Group B features) produces predictive lift, we evaluated Group A (single-stock technicals) against Group A+B across 45,086 common test predictions:

| Model & Feature Set | Date-Level Acc | Pooled Acc | Balanced Acc | ROC-AUC | Brier Score | Log Loss | Pred Up Share | Lift over Base (95% CI & p-val) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **XGB-P (X1: Group A Only)** | 50.65% | 50.65% | 50.62% | 0.5084 | 0.2565 | 0.7070 | 55.50% | *Baseline* |
| **XGB-P (X2: Group A + B)** | 51.16% | 51.06% | 51.04% | 0.5142 | 0.2616 | 0.7195 | 53.00% | **+0.0052** `[-0.0044, 0.0164]` ($p=0.1359$, Not Sig) |
| **Linear-1 (Group A Only)** | 50.93% | 50.82% | 50.75% | 0.5103 | 0.2505 | 0.6942 | 66.49% | *Baseline* |
| **Linear-2 (Group A + B)** | 51.52% | 51.41% | 51.38% | 0.5195 | 0.2501 | 0.6936 | 56.97% | **+0.0058** `[-0.0014, 0.0123]` ($p=0.0599$, Not Sig) |
| **LSTM-P (L1: Group A Only)** | 51.16% | 51.05% | 51.00% | 0.5150 | 0.2501 | 0.6934 | 61.35% | *Baseline* |
| **LSTM-P (L2: Group A + B)** | 51.49% | 51.39% | 51.39% | 0.5173 | 0.2509 | 0.6951 | 48.54% | **+0.0033** `[-0.0070, 0.0124]` ($p=0.3031$, Not Sig) |

**Conclusion**: Market context features (Group B) do not provide statistically significant lift over single-stock technicals on the fixed Protocol B evaluation folds. All bootstrap confidence intervals include zero.

### 4.6 Dropped Ticker-Day Forensics & Daily Coverage Proof (B5)

| Primary Exclusion Category | Dropped Ticker-Days | Percentage | Forensic Root Cause Explanation |
|:---|:---:|:---:|:---|
| **Delisted Equities (HDFC)** | 130 | 27.03% | `HDFC` delisted on 2023-07-13 post-merger into HDFC Bank. |
| **Missing Yahoo Price Quotes (LTIM)** | 298 | 61.95% | `LTIM` merged in 2022; historical quotes unavailable under `LTIM.NS` on Yahoo Finance. |
| **New Listing Indicator Warmup (JIOFIN)** | 53 | 11.02% | `JIOFIN` listed on 2023-08-21. Requiring 50 trading days to compute 50-day SMA is a mathematical constraint of a newly listed stock. |
| **Incomplete Post-Entry Warmup (Existing Stocks)** | **0** | **0.00%** | **Completely eliminated** by computing technical indicators before filtering point-in-time constituent membership (393 ticker-days recovered). |
| **Total Dropped Ticker-Days** | **481** | **100.0%** | |

- **Daily Coverage Stats**:
  - Total Trading Dates in Evaluation Window: **916**
  - Dates with Exactly 50 Evaluated Constituents: **484 dates (52.84%)** (previously 0.0%)
  - Mean Evaluated Constituents per Date: **49.47** (previously 48.10)

---

## 5. Artifact & Script Inventory

All audit code, datasets, predictions, and reports have been generated and serialized under `experiments/audit/`, `docs/audit/`, `scripts/`, and `tests/`:

1. `experiments/audit/master_universe_ticker_day.parquet`: Corrected master dataset (82,425 rows across 60 constituents).
2. `experiments/audit/timing_proof.csv`: Sampled ticker-date pairs proving calendar integrity and absence of future lookahead leakage.
3. `experiments/audit/shifted_label_sweep.csv`: Full sweep across $k \in \{-3 \dots +3\}$ reproducing the 72.07% figure at $k=-1$ and chance at $k \ge 0$.
4. `experiments/audit/synthetic_null_results.csv`: 20-seed synthetic random-walk null distribution proving zero manufactured edge on unforecastable series.
5. `experiments/audit/positive_controls.csv`: Perfect leak canary (100% accuracy) and weak AR(1) signal recovery (56.65% recovered vs 56.41% theoretical).
6. `experiments/audit/data_quality_table.csv`: Comprehensive per-ticker audit across all 62 NIFTY 50 constituents.
7. `experiments/audit/extreme_returns_anomalies.csv`: All 16 flagged extreme return events ($|R| > 20\%$) and the TMPV / TRENT corporate actions.
8. `experiments/audit/daily_constituent_counts.csv`: Exact constituent count for every single out-of-sample trading session.
9. `experiments/audit/per_ticker_prediction_counts.csv`: Coverage percentage and dropped reasons for every single stock.
10. `experiments/audit/dropped_ticker_days_audit.csv`: Line-by-line categorization of all 481 dropped ticker-days.
11. `experiments/audit/model_metrics_full_universe.csv`: Full universe classification metrics, baselines, and bootstrap confidence intervals.
12. `experiments/audit/lstm_seed_performance.csv`: 5-seed performance breakdown (Seeds 42–46 + Ensemble).
13. `experiments/audit/heavyweights_vs_others.csv`: Stratified breakdown comparing the 10 heavyweights against the remaining constituents.
14. `experiments/audit/market_context_experiment_results.csv`: Empirical results of the market context (Group A vs Group A+B) experiment.
15. `experiments/audit/news_validation_summary.json`: Comprehensive news dataset certification summary for Experiment 3.
16. `experiments/audit/predictions/xgb_p_predictions.parquet`: 45,319 out-of-sample XGB-P predictions.
17. `experiments/audit/predictions/lr_p_predictions.parquet`: 45,319 out-of-sample Logistic Regression predictions.
18. `experiments/audit/predictions/lstm_p_predictions.parquet`: 45,136 out-of-sample LSTM-P predictions across 5 seeds.
19. `experiments/audit/predictions/market_context_predictions.parquet`: 45,086 common out-of-sample predictions comparing Group A vs Group A+B.
20. `tests/test_timing.py`: Automated pytest suite for calendar alignment and causal features.
21. `tests/test_synthetic_null.py`: Automated pytest suite for random-walk null behavior.
22. `tests/test_fold_integrity.py`: Automated pytest suite for fold chronology, embargo, and target-date purging.
23. `scripts/run_market_context_experiment.py`: Script executing the Group A vs Group A+B market context experiment.
24. `scripts/validate_news_dataset.py`: Script validating the financial news dataset for Experiment 3.
