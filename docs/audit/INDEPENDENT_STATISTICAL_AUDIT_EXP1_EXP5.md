# FinFusionNet: Independent Statistical Audit Report (Experiments 1–5)
## Rigorous Econometric Verification, Benchmark Fairness Analysis, and Final Reproducibility Audit

**Audit Date:** 2026-10-27  
**Auditor:** Independent Quantitative Research & Econometric Validation Team  
**Evaluation Cohort:** Audited Common Cohort ($N = 45,136$ observations, 916 dates, 56 unique NIFTY constituents)  
**Protocol:** Protocol B (45 Monthly Expanding Folds, 24-Month Initial Window, 1-Session Embargo)  
**Verification Scope:** Exp1 (Price-Only), Exp2 (Market Context), Exp3 (VADER Pilot), Exp3B (All-50 VADER), Exp4 (FinBERT Concatenation), Exp5 (Temporal Cross-Attention Dual-Head Fusion)  

---

## 1. Executive Audit Summary

This independent statistical audit re-examined all predictions, targets, features, and metric computations across Experiments 1 through 5 in FinFusionNet without modifying previous experiment artifacts or retraining models. 

### Key Audit Findings:
1. **Prediction-Target Alignment & Cohort Parity (VERIFIED):**
   - The common evaluation cohort contains exactly **45,136 observations** across **916 trading sessions** (2023-01-02 to 2026-09-29) and **56 constituents** across all 45 Protocol B expanding folds.
   - Zero label contamination or future date leakage was detected. Test dates are strictly monotonically increasing, and a 1-session causal embargo was consistently enforced across all folds.
2. **Next-Session Direction Prediction (HONEST NEGATIVE VERDICT ACROSS ALL TRACKS):**
   - **Audited Performance Ranking (Date-Level Accuracy):**
     $$\text{LSTM-F0 (Price-Only)}: 51.08\% > \text{LR-F1\_FinBERT}: 50.95\% \approx \text{LR-F0}: 50.94\% > \text{LSTM-F1\_VADER}: 50.99\% >$$
     $$\text{LSTM-F1\_FinBERT}: 50.53\% \approx \text{XGB-F1\_VADER}: 50.54\% > \text{XGB-F1\_FinBERT}: 50.42\% >$$
     $$\text{Baseline: Always-Up}: 50.34\% > \text{XGB-F0}: 50.27\% > \mathbf{\text{Cross-Attention (Exp 5)}}: \mathbf{50.16\%} > \text{Prev-Day Direction}: 49.19\%$$
   - **Statistical Significance (Bootstrap, McNemar, & Direction DM Tests):**
     - Cross-Attention significantly *underperformed* the audited Price-Only LSTM baseline on Brier score loss ($DM = -2.0261, p = 0.0430$) and 0-1 misclassification loss ($DM = -2.1250, p = 0.0339$).
     - After Holm-Bonferroni correction, **zero sentiment or fusion configurations (Exp3, Exp3B, Exp4, Exp5) provide statistically significant incremental directional alpha** over the price-only baseline.
3. **Special Audit: Exp5's Reported Volatility Improvement (CRITICAL METHODOLOGICAL CORRECTION):**
   - **Flaw 1 — Benchmark Metric Asymmetry:** Exp5 evaluated next-day *Parkinson (1980) High-Low realized volatility* (sample mean $\mu = 0.2024$), but compared it against `volatility_20`, which is *close-to-close rolling standard deviation* (sample mean $\mu = 0.2355$). Close-to-close volatility includes overnight jump variance, creating a $+0.0331$ (+3.31 percentage point) systematic bias against Parkinson volatility that artificially severely penalized the baseline ($R^2 = -35.58\%$).
   - **Flaw 2 — Pooled vs Panel-Corrected Diebold-Mariano:** Applying univariate Diebold-Mariano directly to 45,129 stacked panel rows violates independent time-series assumptions due to strong cross-sectional market correlation on common calendar dates, inflating the DM statistic ($DM = +19.49$).
   - **The True Like-for-Like Benchmark Test:** When evaluated against a fair, like-for-like trailing 20-day Parkinson realized volatility benchmark (`parkinson_20`), the benchmark actually achieves:
     - **Lower RMSE:** **0.1052** (Fair Benchmark) vs **0.1075** (Cross-Attention Net)
     - **Higher Correlation:** **r = 0.3979** (Fair Benchmark) vs **0.3369** (Cross-Attention Net)
     - **Higher Out-of-Sample $R^2$:** **+14.24%** (Fair Benchmark) vs **+10.38%** (Cross-Attention Net)
     - **Date-Level Diebold-Mariano (MSE):** $DM = -1.9515, p = 0.0513$ (Fair Benchmark has lower squared error, borderline statistically significant).
     - On MAE alone, Cross-Attention slightly edges the trailing benchmark ($0.0686$ vs $0.0695$, $\Delta = 0.0009$) because Huber loss explicitly penalizes absolute deviations.

---

## 2. Comprehensive Classification Performance Recheck

Every observation in [`experiments/experiment_5/predictions/experiment_5_common_cohort_predictions.parquet`](file:///c:/Users/Neil/Desktop/FinFusionNeT/experiments/experiment_5/predictions/experiment_5_common_cohort_predictions.parquet) was re-evaluated:

| Model Track | Date-Level Acc | Pooled Acc | Balanced Acc | Precision | Recall | F1-Score | ROC-AUC | Brier Score | Log Loss | Pred Up % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LSTM-F0 (Price-Only, Exp 1)** | **0.5108** | **0.5096** | **0.5091** | **0.5095** | 0.6297 | 0.5633 | **0.5149** | **0.2502** | **0.6935** | 62.07% |
| LR-F1_FinBERT (Exp 4) | 0.5095 | 0.5083 | 0.5077 | 0.5081 | 0.6567 | 0.5729 | 0.5103 | 0.2505 | 0.6943 | 64.91% |
| LR-F0 (Price-Only, Exp 1) | 0.5094 | 0.5082 | 0.5075 | 0.5078 | 0.6724 | 0.5786 | 0.5102 | 0.2505 | 0.6942 | 66.49% |
| LSTM-F1_VADER (Exp 3B) | 0.5099 | 0.5088 | 0.5084 | 0.5092 | 0.6045 | 0.5528 | 0.5105 | 0.2504 | 0.6940 | 59.62% |
| LSTM-F1_FinBERT (Exp 4 Concat) | 0.5053 | 0.5041 | 0.5038 | 0.5054 | 0.5842 | 0.5419 | 0.5092 | 0.2505 | 0.6942 | 58.04% |
| XGB-F1_VADER (Exp 3B) | 0.5054 | 0.5043 | 0.5040 | 0.5057 | 0.5622 | 0.5325 | 0.5090 | 0.2564 | 0.7069 | 55.82% |
| XGB-F1_FinBERT (Exp 4) | 0.5042 | 0.5042 | 0.5039 | 0.5057 | 0.5593 | 0.5311 | 0.5070 | 0.2568 | 0.7076 | 55.53% |
| Baseline: Always-Up | 0.5034 | 0.5021 | 0.5000 | 0.5021 | 1.0000 | 0.6686 | 0.5000 | 0.4979 | 8.0244 | 100.0% |
| XGB-F0 (Price-Only, Exp 1) | 0.5027 | 0.5015 | 0.4996 | 0.5019 | 0.9351 | 0.6532 | 0.5067 | 0.2503 | 0.6937 | 93.55% |
| **Cross-Attention Net (Exp 5)** | **0.5016** | **0.5004** | **0.4996** | **0.5019** | 0.6852 | 0.5794 | 0.5016 | 0.2506 | 0.6943 | 68.56% |
| Baseline: Prev-Day Direction | 0.4919 | 0.4930 | 0.4930 | 0.4952 | 0.4966 | 0.4959 | 0.4930 | 0.5070 | 8.1722 | 50.36% |

---

## 3. Direction Statistical Testing & Diebold-Mariano Analysis

To resolve potential cross-sectional correlation between stocks on the same date, two complementary statistical frameworks were executed:
1. **Paired Date-Block Bootstrap (10,000 resamples, Block Length $L=10$):** Resamples blocks of entire market trading days to preserve simultaneous cross-sectional correlation and autoregressive structure.
2. **Date-Level Diebold-Mariano Tests with Newey-West HAC Variance Estimation:** Aggregates cross-sectional losses $\bar{L}_t = \frac{1}{N_t} \sum_{i=1}^{N_t} L(y_{i, t}, \hat{p}_{i, t})$ to form a genuine daily time-series loss differential $d_t = \bar{L}_{A, t} - \bar{L}_{B, t}$, with Newey-West autocorrelation weighting ($h=1, \text{lag}=5$).

| Comparison (Base vs Alternative) | $\Delta$ Date Acc | 95% Bootstrap CI | Boot $p$ | McNemar Wins (Alt vs Base) | McNemar $p$ | DM Brier Stat ($d_t$) | DM Brier $p$ | DM 0-1 Stat ($d_t$) | DM 0-1 $p$ | Holm-Bonferroni $p$ | Stat. Sig.? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cross-Attn vs LSTM-F0 (Price-Only)** | -0.0092 (-0.92%) | [-0.0174, -0.0001] | 0.9758 | 390 vs 453 (Ties 73) | **0.0327** | **-2.0261** | **0.0430** | **-2.1250** | **0.0339** | 1.0000 | **False** |
| Cross-Attn vs LSTM-F1_FinBERT | -0.0037 (-0.37%) | [-0.0114, +0.0061] | 0.7414 | 417 vs 434 (Ties 65) | 0.5834 | -0.4806 | 0.6309 | -0.8388 | 0.4018 | 1.0000 | **False** |
| Cross-Attn vs LSTM-F1_VADER | -0.0083 (-0.83%) | [-0.0161, +0.0012] | 0.9525 | 398 vs 441 (Ties 77) | 0.1470 | -0.8245 | 0.4099 | -1.8971 | 0.0581 | 1.0000 | **False** |
| Cross-Attn vs XGB-F0 (Price-Only) | -0.0011 (-0.11%) | [-0.0115, +0.0086] | 0.6002 | 412 vs 427 (Ties 77) | 0.6289 | -1.4122 | 0.1582 | -0.2104 | 0.8334 | 1.0000 | **False** |
| Cross-Attn vs XGB-F1_FinBERT | -0.0026 (-0.26%) | [-0.0117, +0.0076] | 0.6527 | 422 vs 439 (Ties 55) | 0.5856 | **+11.6587** | < 1e-15 | -0.5382 | 0.5906 | 1.0000 | **False** |
| LSTM-F1_FinBERT vs LSTM-F0 | -0.0055 (-0.55%) | [-0.0117, -0.0002] | 0.9800 | 385 vs 418 (Ties 113) | 0.2588 | -1.8802 | 0.0604 | -1.7991 | 0.0723 | 0.9800 | **False** |
| XGB-F1_FinBERT vs XGB-F0 | +0.0015 (+0.15%) | [-0.0116, +0.0129] | 0.4595 | 440 vs 439 (Ties 37) | 1.0000 | -12.4165 | < 1e-15 | +0.2336 | 0.8153 | 1.0000 | **False** |
| LR-F1_FinBERT vs LR-F0 | +0.0001 (+0.01%) | [-0.0034, +0.0030] | 0.5445 | 371 vs 388 (Ties 157) | 0.5614 | -0.3954 | 0.6926 | +0.0679 | 0.9459 | 1.0000 | **False** |

> [!CAUTION]
> **Direction Invalidation:** In every planned comparison against Price-Only baselines, the cross-attention network and concatenation models failed to generate positive excess accuracy. On date-level McNemar and Diebold-Mariano tests, the price-only LSTM baseline significantly beats the multimodal cross-attention network ($p = 0.033$).

---

## 4. Special Audit: Realized Volatility Forecasting

### 4.1 Root Cause of Flawed Baseline Comparison in Experiment 5
Experiment 5 reported that Cross-Attention achieved an out-of-sample $R^2$ of **+10.38%** while the historical benchmark `volatility_20` yielded **-35.58%**. 
Our audit decomposed the exact source of this discrepancy:
1. **Target:** $\sigma_{\text{Parkinson}, t+1} = \sqrt{\frac{(\ln(H/L))^2}{4\ln 2}} \sqrt{252}$, which measures continuous intraday diffusion. Empirical mean = **0.2024 (20.24%)**.
2. **Exp5 Baseline (`volatility_20`):** $\text{Std}(\text{return}_{\text{1d}}) \times \sqrt{252}$, which measures close-to-close variance. Empirical mean = **0.2355 (23.55%)**.
3. **The Bias Penalty:** Close-to-close returns include discrete overnight gap risk ($C_t \to O_t$). In Indian equities, this introduces an inherent $+0.0331$ (+3.31 percentage point) systematic bias. When predicting a 20.2% target with a 23.5% proxy, the squared bias alone adds $0.0331^2 \approx 0.0011$ to MSE, converting an otherwise informative historical signal into a negative $R^2$ model!

### 4.2 Fair Like-for-Like Volatility Benchmarking
We reconstructed the true, fair benchmarks: rolling historical moving averages of **Parkinson realized volatility** strictly on past sessions up to date $t$:

| Volatility Model / Benchmark | Target Scale Matching | RMSE | MAE | Bias ($\bar{\hat{y}} - \bar{y}$) | Pearson $r$ | Out-of-Sample $R^2$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trailing Parkinson 20d (`parkinson_20`)** | **Exact Fair Match** | **0.1052** | **0.0695** | **+0.0003** | **0.3979** | **+14.24%** | **Strongest Benchmark** |
| Trailing Parkinson 10d (`parkinson_10`) | Exact Fair Match | 0.1058 | 0.0700 | +0.0003 | 0.4084 | +13.13% | Robust Fast Benchmark |
| Trailing Parkinson 5d (`parkinson_5`) | Exact Fair Match | 0.1066 | 0.0708 | +0.0002 | 0.4301 | +11.89% | Short Horizon Benchmark |
| **Cross-Attention Net (Exp 5)** | Model Predicted | **0.1075** | **0.0686** | **-0.0089** | **0.3369** | **+10.38%** | **Lower RMSE than Mean, but Trails P20** |
| Expanding Historical Mean Vol | Historical Mean | 0.1135 | 0.0751 | 0.0000 | 0.0000 | 0.00% | Neutral Constant Anchor |
| Trailing Parkinson 1d (`parkinson_t`) | 1-Day Lag | 0.1241 | 0.0839 | -0.0001 | 0.4057 | -19.38% | High Daily Noise |
| Trailing Close-to-Close 20d (`volatility_20`) | Mismatched (C-to-C) | 0.1322 | 0.0900 | +0.0331 | 0.3225 | -35.58% | Flawed Exp5 Baseline |
| Trailing Close-to-Close 10d (`volatility_10`) | Mismatched (C-to-C) | 0.1397 | 0.0927 | +0.0259 | 0.3188 | -51.29% | Flawed Exp5 Baseline |

### 4.3 Diebold-Mariano Tests: Corrected Date-Level HAC vs Naive Pooled
When Diebold-Mariano is correctly evaluated on daily cross-sectional averaged loss series with Newey-West HAC standard errors:

| Comparison (Benchmark vs Cross-Attn) | Loss Function | Mean Daily Loss Diff ($\bar{d}_t$) | Date-Level HAC DM Stat | Date-Level $p$-value | Truly Superior Model | Naive Pooled DM Stat |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trailing Parkinson 20d (Fair)** | **MSE (Squared Error)** | **-0.000497** | **-1.9515** | **0.0513** | **Benchmark (P20)** | -4.3941 |
| **Trailing Parkinson 20d (Fair)** | **MAE (Absolute Error)** | **+0.000920** | **+2.1118** | **0.0350** | **Cross-Attention** | — |
| Trailing Parkinson 10d (Fair) | MSE (Squared Error) | -0.000354 | -0.9768 | 0.3289 | Benchmark (P10) | -2.3198 |
| Trailing Parkinson 10d (Fair) | MAE (Absolute Error) | +0.001421 | +2.1022 | 0.0358 | Cross-Attention | — |
| Trailing Parkinson 5d (Fair) | MSE (Squared Error) | -0.000199 | -0.3994 | 0.6897 | Benchmark (P5) | -0.9891 |
| Trailing Parkinson 5d (Fair) | MAE (Absolute Error) | +0.002253 | +2.9819 | 0.0029 | Cross-Attention | — |
| Trailing Close-to-Close 20d (Flawed) | MSE (Squared Error) | +0.005903 | +6.4544 | < 1e-9 | Cross-Attention (Artificial) | +19.4873 |
| Trailing Close-to-Close 20d (Flawed) | MAE (Absolute Error) | +0.021291 | +13.3333 | < 1e-15 | Cross-Attention (Artificial) | — |

### 4.4 Audited Volatility Verdict
1. **The reported "18.7% volatility error reduction and $R^2$ lift from -35.5% to +10.4%" was an artifact of comparing a high-low target against a close-to-close baseline.**
2. When compared to the true like-for-like trailing Parkinson volatility benchmark (`parkinson_20`), **the simple trailing benchmark achieves lower RMSE (0.1052 vs 0.1075) and higher $R^2$ (+14.24% vs +10.38%)**.
3. Cross-Attention achieves a slight edge in MAE (0.0686 vs 0.0695, $\Delta = 0.0009$) because the neural network was trained with Huber loss, which directly targets median / $L_1$ deviations.

---

## 5. Market Regime & Sector Audit

### 5.1 Performance across Volatility and Directional Regimes

| Market Regime | Observations ($N$) | Cohort Share | Cross-Attn Dir Acc | LSTM-F0 Dir Acc | XGB-F0 Dir Acc | Cross-Attn Vol RMSE | Parkinson 20d Vol RMSE | Vol RMSE Gain (Attn - P20) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **High Volatility (Top 25% Vol Days)** | 11,303 | 25.0% | 48.41% | 49.63% | 46.60% | 0.1571 | **0.1485** | -0.0086 (P20 Wins) |
| **Normal Volatility (Middle 50% Vol)** | 22,603 | 50.1% | 50.74% | 51.20% | 50.24% | 0.0897 | 0.0903 | +0.0006 (Attn Wins) |
| **Low Volatility (Bottom 25% Vol)** | 11,230 | 24.9% | 50.28% | 51.83% | 53.54% | **0.0735** | 0.0765 | **+0.0030** (Attn Wins) |
| **Bull Days (Market Return > 0)** | 24,619 | 54.5% | 50.57% | 51.30% | 51.16% | 0.1073 | 0.1064 | -0.0009 (P20 Wins) |
| **Bear Days (Market Return < 0)** | 20,467 | 45.3% | 49.39% | 50.55% | 48.94% | 0.1077 | 0.1036 | -0.0041 (P20 Wins) |

> [!NOTE]
> In high volatility regimes and down-market regimes, the simple trailing Parkinson 20d moving average is strictly superior to the neural network on RMSE. The neural network only improves slightly on quiet, low-volatility days where it shrinks aggressively towards the historical mean.

---

## 6. Audit Artifacts & Generated Files

All audit tables, corrected metrics, and statistical outputs have been saved to [`docs/audit/exp1_exp5_statistical_audit/`](file:///c:/Users/Neil/Desktop/FinFusionNeT/docs/audit/exp1_exp5_statistical_audit):
- [`audited_direction_performance.csv`](file:///c:/Users/Neil/Desktop/FinFusionNeT/docs/audit/exp1_exp5_statistical_audit/audited_direction_performance.csv): Comprehensive rechecked classification metrics across all models.
- [`audited_direction_statistical_significance.csv`](file:///c:/Users/Neil/Desktop/FinFusionNeT/docs/audit/exp1_exp5_statistical_audit/audited_direction_statistical_significance.csv): Date-block bootstrap, McNemar, Brier DM, 0-1 DM, and Holm-Bonferroni corrections.
- [`audited_volatility_performance.csv`](file:///c:/Users/Neil/Desktop/FinFusionNeT/docs/audit/exp1_exp5_statistical_audit/audited_volatility_performance.csv): Fair like-for-like volatility benchmarking with bias decomposition.
- [`audited_volatility_diebold_mariano.csv`](file:///c:/Users/Neil/Desktop/FinFusionNeT/docs/audit/exp1_exp5_statistical_audit/audited_volatility_diebold_mariano.csv): Panel-corrected date-level HAC Diebold-Mariano tests vs pooled naive tests.
- [`audited_market_regimes.csv`](file:///c:/Users/Neil/Desktop/FinFusionNeT/docs/audit/exp1_exp5_statistical_audit/audited_market_regimes.csv): Performance across volatility and trend regimes.

---

## 7. Conclusions & Scientific Summary

1. **Protocol and Pipeline Integrity:** The FinFusionNet evaluation infrastructure is verified to be leak-free, strictly chronological, and executed with mathematical rigor.
2. **Direction Classification Verdict:** Neither domain-specific sentiment (FinBERT) nor sophisticated temporal cross-attention architectures provide statistically significant incremental predictive power for next-session equity direction in the NIFTY 50 universe. Price-only baselines remain dominant.
3. **Volatility Forecasting Verdict:** The reported massive outperformance of Cross-Attention over historical volatility was an artifact of metric asymmetry (comparing high-low Parkinson target with close-to-close return variance). When compared to an identical trailing Parkinson benchmark, the simple moving average benchmark achieves lower RMSE and higher $R^2$.
