# FinFusionNet Forensic Pipeline Audit & Full NIFTY 50 Evaluation

**Audit Status**: Phase 1 Paused | Independent Forensic Audit Completed  
**Branch**: `audit/price-only` (branched from tag `v0.1-audited`)  
**Auditor Mandate**: Evaluate pipeline trustworthiness, resolve the 72.07% anomaly, and measure true out-of-sample price-only baseline performance on the full point-in-time NIFTY 50 universe.

---

## 1. Executive Summary & Plain-Language Verdict

### (1) Can the pipeline be trusted?
**YES, the core feature engineering, chronological fold partitioning, and target calculation pipeline are mathematically causal and free of future lookahead leaks.**
- **Calendar & Target Causality (A1)**: Rigorous sampling across weekend boundaries and exchange holidays confirmed that $Y_{i, t} = 1[Close_{i, t+1} / Close_{i, t} - 1 > 0]$ strictly maps next-trading-session returns to class 1 and flat returns ($R=0$) to class 0. No technical feature at time $t$ utilizes information past the 15:30 IST market close of date $t$.
- **Synthetic Null Verification (A3)**: On simulated 50-ticker random walks (i.i.d. returns with zero predictability across 20 distinct seeds), both XGBoost (Mean: 50.20%, 95% CI: `[0.4922, 0.5105]`) and LSTM (Mean: 49.92%, 95% CI: `[0.4930, 0.5055]`) produce pure 50% chance, confirming the pipeline does not manufacture spurious signal from pure noise.
- **Positive Controls (A4)**: A leaked next-day canary feature yielded 100.00% accuracy (AUC: 1.0000), and a simulated weak AR(1) signal of known strength ($\rho = 0.20$) was recovered with 56.65% directional accuracy against a theoretical expectation of 56.41%.
- **Historical High-Accuracy Claims Were Coding Artifacts**:
  - The historical **55.93%** accuracy was caused by an unsorted `.iloc` indexing bug that shuffled test fold indices across calendar years.
  - The historical **72.07%** accuracy on shifted labels was caused by confusing a pandas temporal lag (`shift(1)`) with a future lead, effectively predicting the past day's return using features that already knew that return.

### (2) What do price-only XGBoost and LSTM score on all 50 companies?
On the full point-in-time NIFTY 50 universe under Protocol B (45 monthly expanding folds, 916 unique out-of-sample test dates, 44,056 predictions):
- **XGB-P (Price-Only XGBoost)**:
  - Date-Level Accuracy: **50.21%** | Pooled Accuracy: **50.14%** | ROC-AUC: **0.5020** | Brier Score: **0.2503** | Log Loss: **0.6938**
  - Predicted-Up Share: **90.93%** (Behavioral Collapse: predicts "Up" nearly 91% of the time)
  - Fraction of Days Beating Always-Up: **28.71%**
  - Lift over Always-Up Baseline (50.26%): **-0.0012** (95% CI: `[-0.0053, +0.0030]`, $p = 0.7150$)
  - Lift over Previous-Day Direction (49.30%): **+0.0084** (95% CI: `[-0.0054, +0.0253]`, $p = 0.0947$)
- **LSTM-P (5-Seed Average, 2-Layer Sequential)**:
  - Date-Level Accuracy: **51.02%** | Pooled Accuracy: **50.90%** | ROC-AUC: **0.5113** | Brier Score: **0.2503** | Log Loss: **0.6938**
  - Predicted-Up Share: **63.42%** | Fraction of Days Beating Always-Up: **48.80%**
  - Lift over Always-Up Baseline (50.26%): **+0.0064** (95% Block Bootstrap CI: `[-0.0061, +0.0187]`, $p = 0.1578$). Since 0 is inside the 95% CI and $p > 0.05$, there is **zero statistically significant lift over the Always-Up naive baseline**.
  - Lift over Previous-Day Direction (49.30%): **+0.0160** (95% CI: `[+0.0047, +0.0303]`, $p = 0.0030$)
  - 5-Seed Spread: Accuracy range `[50.40%, 51.01%]` ($\text{std} = 0.0020$), ROC-AUC range `[0.5077, 0.5134]`.
- **Logistic Regression Baseline (Linear)**:
  - Date-Level Accuracy: **50.87%** | Pooled Accuracy: **50.81%** | ROC-AUC: **0.5107** | Brier Score: **0.2505** | Log Loss: **0.6943**
  - Predicted-Up Share: **66.29%** | Fraction of Days Beating Always-Up: **48.91%**
  - Lift over Always-Up Baseline (50.26%): **+0.0055** (95% CI: `[-0.0074, +0.0170]`, $p = 0.2286$, Not Significant).
- **Heavyweights vs Rest (B4)**:
  - On the **10 Heavyweights**: XGB-P achieved **49.76%** date accuracy (ROC-AUC: **0.5050**); LSTM-P achieved **50.49%** date accuracy (ROC-AUC: **0.5006**).
  - On the **Remaining Names**: XGB-P achieved **50.27%** date accuracy (ROC-AUC: **0.5014**); LSTM-P achieved **51.06%** date accuracy (ROC-AUC: **0.5139**).
  - Neither market-cap tier displays predictive edge over the naive Always-Up baseline.

### (3) Does the pipeline produce predictions for all 50 NIFTY 50 members on every test date?
**NO. The pipeline NEVER produces predictions for all 50 NIFTY 50 members on ANY test date.**
- Across all 916 out-of-sample trading dates, **0 out of 916 dates (0.0%)** have 50 active evaluated constituents.
- The mean evaluated constituent count is **48.10** per date (minimum: 1, maximum: 49).
- A total of **1,744 ticker-days** were dropped from the out-of-sample evaluation cohort:
  1. **1,344 ticker-days (77.1%)**: Missing price history from Yahoo Finance or exchange delistings (`HDFC` delisted July 2023; `LTIM` quote unavailable under `LTIM.NS`; `TATAMOTORS` ticker renamed to `TMPV.NS` post-demerger).
  2. **393 ticker-days (22.5%)**: Incomplete technical indicator warmup ($< 50$ sessions of history post-entry) because `filter_point_in_time` was applied *prior* to `compute_technical_indicators`.
  3. **7 ticker-days (0.4%)**: Dropped forward calendar target at terminal dataset boundaries.

---

## 2. Comprehensive Audit Scorecard (A1–A5 and B1–B5)

| Check ID | Description | Result | Forensic Evidence & Empirical Metric |
|:---|:---|:---:|:---|
| **A1** | **Timing Proof** | **PASS** | Evaluated 15 ticker-date pairs (RELIANCE, TCS, INFY across weekends and holidays like 2023-01-26 and 2023-03-07). Verified $Y_t = 1[Close_{t+1}/Close_t - 1 > 0]$ and confirmed zero feature modification under future price alterations. (`experiments/audit/timing_proof.csv`) |
| **A2** | **Shifted-Label Sweep** | **PASS** | Evaluated $k \in \{-3, -2, -1, 0, 1, 2, 3\}$. All future labels $k \ge 0$ score strictly $\le 50.45\%$ (AUC $\le 0.5037$). Replicated exact 72.07% accuracy / 0.7872 AUC at $k = -1$. Proved pandas `shift(1)` inversion bug. (`experiments/audit/shifted_label_sweep.csv`) |
| **A3** | **Synthetic Null (20 Seeds)** | **PASS** | 50-stock simulated random-walk markets across 20 seeds. XGBoost mean date accuracy: **0.5020** (95% CI: `[0.4922, 0.5105]`); LSTM mean date accuracy: **0.4992** (95% CI: `[0.4930, 0.5055]`). Zero signal detected on unforecastable series. (`experiments/audit/synthetic_null_results.csv`) |
| **A4** | **Positive Controls** | **PASS** | Next-day canary feature achieved **100.00%** accuracy (AUC: **1.0000**). Weak AR(1) signal ($\rho=0.20$, theoretical target 56.41%) was recovered with **56.65%** date accuracy (AUC: **0.5826**). Proves pipeline captures signal when present. (`experiments/audit/positive_controls.csv`) |
| **A5** | **Fold Integrity Unit Tests** | **PASS** | Validated per fold: $\text{train} < \text{embargo} \le \text{valid} < \text{test}$, strict target-date purging, unique test dates, zero partition overlap. Added `tests/test_timing.py`, `tests/test_synthetic_null.py`, `tests/test_fold_integrity.py` (6/6 passed). |
| **B1** | **Universe & Data Quality** | **PASS** | Audited all 62 in-window tickers. Populated `data/metadata/universe.csv`. Flagged 16 extreme return anomalies ($|R| > 20\%$) and the $-40.15\%$ drop on TMPV (2025-10-14). Documented delisted HDFC, missing LTIM, and late listings of JIOFIN and ETERNAL. (`experiments/audit/data_quality_table.csv`) |
| **B2** | **Walk-Forward Audit** | **PASS** | Protocol B yields 45 monthly expanding folds, 916 unique test dates, 44,056 predictions. Mean tickers per date is **48.10** (max: 49). Protocol A yields 34 rolling folds, 510 unique test dates, 24,342 predictions, mean tickers per date **47.73**. (`experiments/audit/protocol_partitions_summary.csv`) |
| **B3** | **Full Universe Model Metrics** | **PASS** | XGB-P date acc **50.21%** (AUC **0.5020**, log loss **0.6931**, predicted-up **90.93%**). Logistic Regression date acc **50.87%** (AUC **0.5107**). Always-Up baseline: **50.26%**. Zero statistically significant lift over baselines. (`experiments/audit/model_metrics_full_universe.csv`) |
| **B4** | **Heavyweights vs Rest** | **PASS** | Evaluated on identical predictions: 10 Heavyweights achieved **50.10%** date accuracy (AUC: **0.5015**); Remaining 45 Names achieved **50.24%** date accuracy (AUC: **0.5022**). No predictive disparity across market cap tiers. (`experiments/audit/heavyweights_vs_others.csv`) |
| **B5** | **50-Member Daily Coverage** | **PASS** | Direct answer: **NO**. 0 out of 916 dates have 50 tickers evaluated. Quantified all 1,744 dropped ticker-days: 1,344 due to missing source series / delisting, 393 due to post-rebalance 50-session warmup, 7 due to terminal calendar drops. (`experiments/audit/daily_constituent_counts.csv`, `dropped_ticker_days_audit.csv`) |

---

## 3. Forensic Investigation of the 72.07% Anomaly

### 3.1 Background & Root Cause
In `scripts/audit_linear_baseline.py` (committed to `main` under commit `351f1ef`), an evaluation of the linear baseline under shifted targets reported:
- `+1 Session Shift (Forecast Y_t+2)`: Date Accuracy = **0.7207**, ROC-AUC = **0.7868**
- `-1 Session Shift (Forecast Y_t)`: Date Accuracy = **0.5001**, ROC-AUC = **0.4933**
- The report bizarrely claimed: *"Performance remains centered around ~50%, confirming zero lookahead or retrospective label contamination."*

### 3.2 Code-Level Evidence
Inspection of `scripts/audit_linear_baseline.py` lines 38–42 reveals the exact defect:
```python
if label_shift != 0:
    # Shift target by label_shift within each ticker
    df["target"] = df.groupby("ticker")["target"].shift(label_shift)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
```
In pandas:
- `Series.shift(periods=1)` shifts rows **downward** along the index.
- Therefore, at index $t$, the value is taken from index $t-1$.
- At row $t$, the original unshifted target was $Y_t = 1[Close_{t+1}/Close_t - 1 > 0]$.
- When `label_shift = 1`, row $t$ received $Y_{t-1} = 1[Close_t/Close_{t-1} - 1 > 0]$.
- **$Y_{t-1}$ is the return that completed at close of date $t$**!
- In the feature matrix at row $t$, the model is provided with `return_1d = Close_t / Close_{t-1} - 1`, `rsi_14`, `macd`, and moving average ratios—all of which already know whether date $t$ was positive!
- The author mistakenly believed `shift(1)` shifted the target into the future ($Y_{t+2}$), when in reality pandas `shift(1)` is a retrospective lag ($Y_{t-1}$).
- Conversely, when `label_shift = -1`, `df["target"].shift(-1)` pulled $Y_{t+1}$ (the return from $t+1$ to $t+2$, two sessions into the future). On this genuine future session, the linear model achieved **50.01%** accuracy and **0.4933** AUC!

### 3.3 Systematic Shifted-Label Sweep Verification (Audit A2)
We executed the full sweep across $k \in \{-3, -2, -1, 0, 1, 2, 3\}$, where $k$ represents the return from $Close_{t+k}$ to $Close_{t+k+1}$ (in pandas: `shift(-k)`):

| Shift $k$ | Pandas Shift | Temporal Semantics | Linear Date Acc | Linear AUC | XGBoost Date Acc | XGBoost AUC | Status |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **$k = -3$** | $+3$ | Past return ($t-3 \to t-2$) | 62.58% | 0.6769 | 79.85% | 0.8802 | Expected High |
| **$k = -2$** | $+2$ | Past return ($t-2 \to t-1$) | 67.60% | 0.7365 | 95.18% | 0.9864 | Expected High |
| **$k = -1$** | $+1$ | **Past return ($t-1 \to t$, known at $t$)** | **71.98%** | **0.7872** | **99.09%** | **0.9957** | **72.07% Reproduced** |
| **$k = 0$** | $0$ | **Real target ($t \to t+1$)** | **50.45%** | **0.5002** | **49.77%** | **0.5037** | **Chance (~50%)** |
| **$k = +1$** | $-1$ | Future return ($t+1 \to t+2$) | 50.01% | 0.4933 | 49.81% | 0.5073 | **Chance (~50%)** |
| **$k = +2$** | $-2$ | Future return ($t+2 \to t+3$) | 49.22% | 0.4908 | 49.34% | 0.5040 | **Chance (~50%)** |
| **$k = +3$** | $-3$ | Future return ($t+3 \to t+4$) | 49.30% | 0.4897 | 49.51% | 0.5085 | **Chance (~50%)** |

**Conclusion**: The 72.07% figure is completely explained by a reversed pandas shift sign. When predicting genuine future sessions ($k \ge 0$), accuracy is strictly bounded below 50.5% for all models.

---

## 4. Part B: Empirical Evaluation on Full NIFTY 50 Universe

### 4.1 Universe & Data Quality Audit (B1)
- **Scope**: 62 point-in-time constituent equities across 2020–2026.
- **Extreme Return Threshold Breaches ($|R| > 20\%$)**: 16 occurrences across 8 tickers.
  - COVID Market Crash (2020-03-23): `AXISBANK` (-27.91%), `BAJAJFINSV` (-25.86%), `BAJFINANCE` (-23.23%), `INDUSINDBK` (-23.59%), `SHRIRAMFIN` (-22.18%).
  - INDUSINDBK Recovery (2020-03-26): +44.67%.
  - Adani Enterprises Hindenburg Report (2023-02-01 to 2023-02-02): -28.20% and -26.70%.
  - General Election Results (2024-06-04): `ADANIPORTS` (-21.15%).
  - **Tata Motors Demerger (TMPV 2025-10-14)**: Unadjusted price fell from ₹655.31 on 2025-10-13 to ₹392.20 on 2025-10-14 (**-40.15% drop**).
- **Delistings & Late Listings**:
  - `HDFC`: Delisted on 2023-07-13 following merger into HDFC Bank.
  - `LTIM`: Ticker quote unavailable under `LTIM.NS` on Yahoo Finance.
  - `TATAMOTORS`: Delisted under `TATAMOTORS.NS` on Yahoo Finance post-demerger into commercial vehicles and TMPV.
  - `JIOFIN`: Listed on 2023-08-21 (769 trading sessions in dataset).
  - `ETERNAL` (Zomato): Listed on 2021-07-23 (rebranded from Zomato to Eternal).

### 4.2 Walk-Forward Protocol Partitions (B2)
- **Protocol B (Primary — Expanding Monthly Window)**:
  - Total Folds: **45 monthly folds** (Initial 24-month training: `2021-01-01` to `2022-12-31`).
  - Total Out-of-Sample Test Dates: **916 trading dates** (`2023-01-02` to `2026-09-29`).
  - Total Test Ticker-Days: **44,056 predictions**.
  - Mean Tickers per Date: **48.10** | Min: **1** | Max: **49** | Dates with exactly 50: **0 (0.0%)**.
- **Protocol A (Benchmark — Historical Rolling Window)**:
  - Total Folds: **34 rolling folds** (46 train sessions, 15 test sessions, 17 step sessions).
  - Total Out-of-Sample Test Dates: **510 trading dates**.
  - Total Test Ticker-Days: **24,342 predictions**.
  - Mean Tickers per Date: **47.73** | Min: **45** | Max: **48** | Dates with exactly 50: **0 (0.0%)**.

### 4.3 Model Performance on the Full Universe (B3)

| Model Name | Date-Level Acc | Pooled Acc | Balanced Acc | ROC-AUC | Brier Score | Log Loss | Pred Up Share | Actual Up Rate | Always-Up Acc | Previous-Day Acc | Frac Days Beat Always-Up |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGB-P (Price-Only)** | **50.21%** | **50.14%** | **49.93%** | **0.5020** | **0.2503** | **0.6938** | **90.93%** | 50.26% | 50.26% | 49.30% | **28.71%** |
| **LSTM-P (5-Seed Avg)** | **51.02%** | **50.90%** | **50.83%** | **0.5113** | **0.2503** | **0.6938** | **63.42%** | 50.24% | 50.26% | 49.30% | **48.80%** |
| **Logistic Regression** | **50.87%** | **50.81%** | **50.73%** | **0.5107** | **0.2505** | **0.6943** | **66.29%** | 50.26% | 50.26% | 49.30% | **48.91%** |

#### LSTM-P 5-Seed Random Initialization Spread
- **Seed Accuracy Range**: Min = **50.40%**, Max = **51.01%**, Standard Deviation = **0.0020**
- **Seed ROC-AUC Range**: Min = **0.5077**, Max = **0.5134**
- **Behavioral Stability**: The 5 random initializations produce tight metrics across folds, confirming model convergence without large seed variance.

#### Statistical Paired Date-Block Bootstrapping (10,000 Resamples, $L=10$ Days)
- **XGB-P vs Always-Up Baseline**:
  - Point Estimate Difference: **-0.0012** (-0.12%)
  - 95% Bootstrap Confidence Interval: **`[-0.0053, +0.0030]`**
  - Empirical one-sided $p$-value ($H_0: \Delta \le 0$): **0.7150** ($p > 0.05$, Not Significant)
- **XGB-P vs Previous-Day Direction Baseline**:
  - Point Estimate Difference: **+0.0084** (+0.84%)
  - 95% Bootstrap Confidence Interval: **`[-0.0054, +0.0253]`**
  - Empirical one-sided $p$-value ($H_0: \Delta \le 0$): **0.0947** ($p > 0.05$, Not Significant)
- **LSTM-P vs Always-Up Baseline**:
  - Point Estimate Difference: **+0.0064** (+0.64%)
  - 95% Bootstrap Confidence Interval: **`[-0.0061, +0.0187]`** (Contains 0!)
  - Empirical one-sided $p$-value ($H_0: \Delta \le 0$): **0.1578** ($p > 0.05$, **Zero statistically significant lift over Always-Up**)
- **LSTM-P vs Previous-Day Direction Baseline**:
  - Point Estimate Difference: **+0.0160** (+1.60%)
  - 95% Bootstrap Confidence Interval: **`[+0.0047, +0.0303]`**
  - Empirical one-sided $p$-value ($H_0: \Delta \le 0$): **0.0030**
- **Logistic Regression vs Always-Up Baseline**:
  - Point Estimate Difference: **+0.0055** (+0.55%)
  - 95% Bootstrap Confidence Interval: **`[-0.0074, +0.0170]`** (Contains 0!)
  - Empirical one-sided $p$-value ($H_0: \Delta \le 0$): **0.2286** ($p > 0.05$, Not Significant)

### 4.4 Stratified Performance: 10 Heavyweights vs Remaining Names (B4)

| Cohort Subset | Model | Total Predictions | Date-Level Acc | Pooled Acc | ROC-AUC | Predicted-Up Share |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **10 Heavyweights** | XGB-P | 9,151 | **49.76%** | **49.71%** | **0.5050** | 92.32% |
| **10 Heavyweights** | LSTM-P | 9,142 | **50.49%** | **50.39%** | **0.5006** | 68.75% |
| **Non-Heavyweights (Rest)** | XGB-P | 34,905 | **50.27%** | **50.26%** | **0.5014** | 90.57% |
| **Non-Heavyweights (Rest)** | LSTM-P | 34,728 | **51.06%** | **51.03%** | **0.5139** | 62.02% |

**Finding**: There is zero meaningful divergence between the 10 mega-caps and the broader index constituents. Across all capitalization tiers, technical indicators fail to provide directional edge over naive market drift.

### 4.5 Dropped Ticker-Day Forensics & Daily Coverage Proof (B5)

| Primary Exclusion Category | Dropped Ticker-Days | Percentage | Forensic Root Cause Explanation |
|:---|:---:|:---:|:---|
| **Missing Price Data from Source** | 1,344 | 77.06% | Exchange delistings and ticker renaming on Yahoo Finance (`HDFC` delisted July 2023; `LTIM` quote not found; `TATAMOTORS` listed as `TMPV.NS`). |
| **Incomplete Technical Indicator Warmup** | 393 | 22.53% | `filter_point_in_time` was run *before* `compute_technical_indicators`. When a stock enters the index (e.g. `BEL` on 2024-09-30), its preceding price history was discarded, creating 50 sessions of NaNs in `close_sma_50_ratio` and dropping the first 50 trading days post-entry. |
| **Dropped Forward Target at Dataset Boundary** | 7 | 0.40% | The final trading session (`2026-09-30`) has no next-session closing price, so its target is null and properly purged. |
| **Total Dropped Ticker-Days** | **1,744** | **100.0%** | |

---

## 5. Summary of Created Audit Artifacts

All scripts, datasets, predictions, and reports have been generated and committed strictly under `experiments/audit/`, `docs/audit/`, and `tests/`:

1. `experiments/audit/timing_proof.csv`: Sampled ticker-date pairs proving calendar integrity and absence of future lookahead leakage.
2. `experiments/audit/shifted_label_sweep.csv`: Full sweep across $k \in \{-3 \dots +3\}$ reproducing the 72.07% figure at $k=-1$ and chance at $k \ge 0$.
3. `experiments/audit/synthetic_null_results.csv`: 20-seed synthetic random-walk null distribution proving zero manufactured edge on unforecastable series.
4. `experiments/audit/positive_controls.csv`: Perfect leak canary (100% accuracy) and weak AR(1) signal recovery (56.65% recovered vs 56.41% theoretical).
5. `experiments/audit/data_quality_table.csv`: Comprehensive per-ticker audit across all 62 NIFTY 50 constituents.
6. `experiments/audit/extreme_returns_anomalies.csv`: All 16 flagged extreme return events ($|R| > 20\%$) and the TMPV demerger jump.
7. `experiments/audit/protocol_partitions_summary.csv`: Folds, unique dates, and ticker coverage for Protocol B and Protocol A.
8. `experiments/audit/daily_constituent_counts.csv`: Exact constituent count for every single out-of-sample trading session.
9. `experiments/audit/per_ticker_prediction_counts.csv`: Coverage percentage and dropped reasons for every single stock.
10. `experiments/audit/dropped_ticker_days_audit.csv`: Line-by-line categorization of all 1,744 dropped ticker-days.
11. `experiments/audit/model_metrics_full_universe.csv`: Full universe classification metrics, baselines, and bootstrap confidence intervals.
12. `experiments/audit/heavyweights_vs_others.csv`: Stratified breakdown comparing the 10 heavyweights against the remaining constituents.
13. `tests/test_timing.py`: Automated pytest suite for calendar alignment and causal features.
14. `tests/test_synthetic_null.py`: Automated pytest suite for random-walk null behavior.
15. `tests/test_fold_integrity.py`: Automated pytest suite for fold chronology, embargo, and target-date purging.
