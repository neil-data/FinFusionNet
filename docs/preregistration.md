# FinFusionNet: Formal Pre-Registration Plan & Preregistered Design (v1.0)

**Project**: FinFusionNet — Multi-Modal Directional Equity Forecasting Under Chronological Parity  
**Pre-Registration Date**: October 17, 2026  
**Repository State**: Commit tag `v0.1-audited`  
**License**: MIT  

---

## 1. Study Information & Objectives

FinFusionNet evaluates whether financial news sentiment (lexicon-based VADER and domain-adapted transformer FinBERT) provides statistically defensible predictive lift for next-trading-session equity direction in the Indian equity market (NIFTY 50), comparing tabular gradient-boosted decision trees (XGBoost) against recurrent sequential neural networks (2-layer LSTM).

---

## 2. Universe & Data Specifications

1. **Universe**: 10 NIFTY 50 Heavyweights (RELIANCE, TCS, HDFCBANK, ICICIBANK, BHARTIARTL, INFY, ITC, LT, SBIN, HINDUNILVR).
   - *Justification*: Captures >55% of NIFTY 50 capitalization and >60% of trading volume; strictly aligned with the dense interim GDELT news sentiment parquet (4,988 records, 11,882 articles across 2020–2026).
   - *Scoping*: All claims are formally restricted to the 10 NIFTY 50 mega-cap equities.
2. **Date Boundaries**:
   - Technical Warmup: `2020-01-01` to `2020-12-31` (used strictly to initialize causal indicators like SMA-50, ATR-14, MACD, and rolling volatility).
   - Common Evaluation Cohort: `2021-01-01` to `2026-09-30` (14,101 ticker-day observations across 1,411 dates).
   - Out-of-Sample Test Window (Protocol B): `2023-01-02` to `2026-09-29` (45 monthly folds, exactly 915 unique trading dates).
3. **Target Formulation**:
   - Primary Binary Target: $Y_{i, t} = 1$ if $Close_{i, t+1} > Close_{i, t}$ else $0$. Flat sessions ($R=0$) map strictly to 0.

---

## 3. Pre-Registered Hypotheses & Model Configurations

All sentiment configurations (X3–X6 and L3–L6) include Group A (Technicals) + Group B (Market Context) to guarantee feature parity:

| Model ID | Feature Groups Included | Research Question / Isolated Variable |
|:---|:---|:---|
| **X1 / L1** | Group A (Technicals: 24 features) | Pure technical baseline |
| **X2 / L2** | Group A + Group B (Market Context: 5 features) | Marginal value of macro index returns and volatility |
| **X3 / L3** | Group A + Group B + Group C (News Attention: 2 features) | Control for news arrival volume ($n\_articles$, $has\_news$) |
| **X4 / L4** | Group A + Group B + Group C + Group D (VADER: 5 features) | Incremental value of lexicon sentiment |
| **X5 / L5** | Group A + Group B + Group C + Group E (FinBERT: 5 features) | Incremental value of domain-adapted FinBERT sentiment |
| **X6 / L6** | Group A + Group B + Group C + Group D + Group E | Multi-scale combined sentiment representation |

### Pre-Registered Primary Hypothesis
- **Hypothesis**: $H_0: \Delta \le 0$ vs $H_1: \Delta > 0$, where $\Delta = \text{Acc}_{date}(\text{X5}) - \text{Acc}_{date}(\text{X3})$.
- **Evaluation Metric**: Equal-weighted date-level accuracy across common out-of-sample trading dates ($N=915$).
- **Statistical Test**: Paired date-block bootstrap (10,000 resamples, primary block length $L=10$ trading sessions).
- **Decision Rule**: Significant at $\alpha=0.05$ if and only if two-sided $p < 0.05$ and $95\%$ bootstrap CI strictly excludes zero.

### Secondary Hypotheses (Holm-Bonferroni Family)
1. **H1 (Market Context)**: X2 vs X1
2. **H2 (News Control)**: X3 vs X2
3. **H3 (VADER Sentiment)**: X4 vs X3
4. **H5 (Combined Sentiment)**: X6 vs X3
5. **H6 (Combined vs VADER)**: X6 vs X4
6. **H7 (Combined vs FinBERT)**: X6 vs X5
7. **Head-to-Head Model Family Comparisons**: L1 vs X1, L2 vs X2, L3 vs X3, L4 vs X4, L5 vs X5, L6 vs X6.

---

## 4. Evaluation Protocols

1. **Protocol B (Primary — Expanding Monthly Window)**:
   - Initial training period: 24 months (`2021-01-01` to `2022-12-31`).
   - Test period: 1 month; Step: 1 month.
   - Validation split: Chronological latest 15% of training pool (minimum 60 sessions).
   - Early stopping: 50 rounds on validation log-loss.
   - Target-date purging: Train target-dates must precede first validation date; validation target-dates must precede first test date.
   - Total Folds: **45 folds**, **915 unique test dates**.
2. **Protocol A (Benchmark — Historical Rolling Window)**:
   - Fixed window: 46 training sessions, 15 test sessions, 17 step sessions.
   - Early stopping: None (fixed 500 trees for XGBoost, fixed 15 epochs for LSTM).
   - Total Folds: 34 rolling folds.

---

## 5. Statistical Inference Protocol

1. **Bootstrap Unit**: Contiguous trading day blocks ($L \in \{3, 5, 10, 15, 20\}$ days), keeping all stocks on date $d$ strictly clustered.
2. **P-Value Inversion**: Two-sided empirical p-values are inverted directly from bootstrap distribution tails:
   $$p = \min\left(1.0, 2 \times \min(P(\Delta^* \le 0), P(\Delta^* \ge 0))\right)$$
   Guarantees zero discrepancy between $p < 0.05$ and the $95\%$ percentile confidence interval excluding zero.
3. **Multiplicity Correction**: Holm-Bonferroni step-down adjustment applied to all secondary hypothesis families at $\alpha=0.05$.

---

## 6. Already Observed Section (Prior Data & Pre-Audit Exposure)

In accordance with open-science pre-registration standards, we transparently declare all data analysis, pre-audit inspections, and prior experimental runs observed prior to this pre-registration commitment:

1. **Prior Protocol B Audit Observations**:
   - Initial XGBoost runs suffered from an unsorted `.iloc` indexing bug where fold test indices from a date-sorted view were indexed into a `[ticker, date]` dataframe, resulting in cross-year leakage and an artificial 55.93% accuracy on X2.
   - After repairing the indexing bug and aligning feature groups, the observed Protocol B date-level accuracies on the 915 test dates are:
     - XGBoost: X1 = 0.4980, X2 = 0.5002, X3 = 0.4993, X4 = 0.5015, X5 = 0.5004, X6 = 0.4995.
     - PyTorch LSTM: L1 = 0.4984, L2 = 0.4941, L3 = 0.4893, L4 = 0.5019, L5 = 0.4950, L6 = 0.4926.
   - Under these leak-free Protocol B runs, the primary hypothesis (X5 vs X3) showed a difference of +0.0011 ($p = 0.7272$, not significant).
2. **Prior Linear Baseline Observation**:
   - An interim script evaluated a sequence-pooled linear model across test sets without boundary context expansion, observing an artifactual 54.12% accuracy on the ~85 month-end dates. This anomaly is subjected to formal verification (canary, permutation, and shifted-label tests).
3. **Pre-Registration Boundary**:
   - **Protocol A has NOT been evaluated on the fixed fold engine** since the indexing fix.
   - All subsequent runs of Protocol A and follow-up robustness checks are pre-registered under the protocols herein, evaluated blindly against these pre-registered criteria without hyperparameter alterations.
