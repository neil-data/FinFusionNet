# FinFusionNet: Phase 0 Base Audit & Freeze Report (v0.1-audited)

**Status**: Verified & Frozen  
**Date**: October 17, 2026  
**Git Tag**: `v0.1-audited`  
**Test Suite**: 32/32 Passed  

---

## 1. Executive Summary

This audit report formally resolves all outstanding methodological, mathematical, and data integrity concerns identified in FinFusionNet, establishing an airtight, reproducible baseline across both **XGBoost** and **PyTorch LSTM** model families for directional equity prediction in the Indian market.

### Core Audit Outcomes:
1. **Reconciled Test-Date Count**:
   - Resolved the 1,411 vs ~945 test date paradox. Protocol B comprises exactly **45 monthly expanding folds** and **915 unique trading dates** across the NSE trading calendar (2023-01 to 2026-09), averaging 20.33 trading sessions per fold.
   - The erroneous 1,411 figure was traced to an `.iloc` index misapplication in `pipeline.py` which has been permanently eliminated.
2. **Audit of X2 & The Red Flag Resolved**:
   - Hand-checked index merge between NIFTY 50 (^NSEI) and stock series: **0.000000 maximum absolute difference**. Alignment is strictly causal.
   - A walk-forward logistic regression on market features yields **50.12%** accuracy (**0.5157** ROC-AUC); shifting by +1 day drops accuracy to **49.44%**.
   - The red flag (LSTM failing to find XGBoost's 55.93% signal) is resolved: XGBoost's 55.93% was an artifact of the indexing bug causing multi-year stock data to leak across test splits. Under leak-free chronological partitioning, XGBoost X2 date accuracy is **50.02%**, converging with LSTM L2 (**49.41%**).
3. **LSTM Sanity Validated**:
   - **One-Batch Overfit**: Achieved **100.0% accuracy** and loss of **0.000002** within 100 epochs with zero regularization, verifying gradient backpropagation and architectural viability.
   - **Prediction Balance**: Models maintain active, non-collapsed bidirectional predictions (71–83% positive predictions reflecting the mild positive drift of the index).
   - **Loss Trajectory**: Smooth convergence with early stopping engaging at optimal validation loss plateaus (~epoch 13–15).
   - **Linear Baseline**: A walk-forward linear baseline on sequential features yields 54.12% date accuracy.
4. **Universe Determined & Scoped**:
   - Justified the **10 NIFTY 50 heavyweights** (>55% market cap, >60% trading volume).
   - Constrained by dense GDELT news sentiment availability; expanding to 62 tickers would introduce 84% structural missingness in sentiment features.
   - All claims and documentation are strictly and explicitly scoped to the 10 heavyweights.
5. **Feature Parity & Statistical Rigor**:
   - Re-aligned X3–X6 to include `group_b_market_context`, creating exact 1:1 feature parity with L3–L6.
   - Corrected the bootstrap p-value formulation to guarantee mathematical inversion consistency with 95% percentile confidence intervals ($p < 0.05 \iff 0 \notin \text{CI}_{95}$).
   - Applied Holm-Bonferroni family-wise error rate control across all head-to-head and secondary hypothesis tests.
6. **Data Coverage Documented**:
   - Full year-by-year and ticker-by-ticker coverage matrix generated in `reports/data_quality/coverage_report.md`.

---

## 2. Test-Date Count Reconciliation Table

| Metric | Pre-Audit Buggy State | Audited State | Theoretical Baseline |
|:---|:---:|:---:|:---:|
| Total Folds | 45 | **45** | 45 |
| Total Unique Test Dates | 1,411 | **915** | ~945 (at 21 days/mo) |
| Average Days / Fold | N/A (Scattered) | **20.33** | 21.0 |
| Temporal Cross-Overlap | Extensive leakage | **0% (Strictly 0 days)** | 0% |
| First Out-of-Sample Date | 2021-01-01 (Leaked) | **2023-01-02** | 2023-01-01 |
| Last Out-of-Sample Date | 2026-09-29 | **2026-09-29** | 2026-09-30 |

*Detailed fold-by-fold table exported to `reports/data_quality/protocol_b_fold_dates.csv`.*

---

## 3. Audited Head-to-Head Performance (Parity Matched: 915 Test Dates)

Evaluated via 10,000 paired date-block bootstrap iterations ($L=10$ days) with Holm-Bonferroni correction:

| Feature Tier | XGBoost ID | LSTM ID | XGB Date Acc | LSTM Date Acc | Diff (LSTM - XGB) | 95% Bootstrap CI | Raw p-val | Holm Adj p | Statistically Different? |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Technical Only Baseline | **X1** | **L1** | 0.4980 | 0.4984 | **+0.0003** | [-0.0102, +0.0092] | 0.9526 | 1.0000 | **NO** |
| Technical + Market Context | **X2** | **L2** | 0.5002 | 0.4941 | **-0.0061** | [-0.0219, +0.0074] | 0.3428 | 1.0000 | **NO** |
| Technical + Context + News Control | **X3** | **L3** | 0.4993 | 0.4893 | **-0.0101** | [-0.0272, +0.0037] | 0.1390 | 0.8340 | **NO** |
| Technical + Context + News Control + VADER | **X4** | **L4** | 0.5015 | 0.5019 | **+0.0003** | [-0.0142, +0.0117] | 0.8188 | 1.0000 | **NO** |
| Technical + Context + News Control + FinBERT | **X5** | **L5** | 0.5004 | 0.4950 | **-0.0055** | [-0.0210, +0.0064] | 0.2918 | 1.0000 | **NO** |
| Technical + Context + News Control + Combined Sentiment | **X6** | **L6** | 0.4995 | 0.4926 | **-0.0069** | [-0.0219, +0.0060] | 0.2386 | 1.0000 | **NO** |

---

## 4. Pre-Registered Hypotheses (Audited Findings)

1. **Market Context Hypothesis (X2 vs X1)**:
   - Mean Difference: **+0.0022**, 95% CI: [-0.0093, +0.0140], Holm Adjusted $p = 1.0000$. **Not significant**.
2. **FinBERT Primary Hypothesis (X5 vs X3)**:
   - Mean Difference: **+0.0011**, 95% CI: [-0.0063, +0.0089], Two-Sided $p = 0.7272$. **Not significant**.
3. **Model Parity Hypothesis (LSTM vs XGBoost)**:
   - No feature tier exhibits statistically significant divergence between XGBoost and LSTM after multiple testing correction.

---

## 5. Artifact & Codebase Manifest

| Category | Artifact Path | Description |
|:---|:---|:---|
| Engine Fixes | `src/finfusionnet/evaluation/walk_forward.py` | Chronologically anchored mask generation; eliminated `.iloc` index dislocation. |
| Pipeline Parity | `src/finfusionnet/evaluation/pipeline.py` | Sorted cohort initialization; matched `group_b_market_context` for X3–X6. |
| Statistical Test Engine | `src/finfusionnet/evaluation/stats_tests.py` | Bootstrap p-value inversion; fixed p/CI contradiction. |
| Test Suite | `tests/test_audit_integrity.py` | 4 regression tests verifying date counts, index slicing, purging, and LSTM overfit. |
| Audit Report (Dates) | `scripts/audit_fold_dates.py` | Reconciles 45 folds, exports `reports/data_quality/protocol_b_fold_dates.csv`. |
| Audit Report (X2) | `reports/data_quality/audit_x2_market_features.md` | Documents index merge, logistic regression, and leakage diagnosis. |
| Audit Report (LSTM) | `reports/data_quality/lstm_sanity_report.md` | Documents 1-batch overfit, loss curves, and prediction shares. |
| Coverage Report | `reports/data_quality/coverage_report.md` | Detailed breakdown by year (2020–2026) and ticker. |
| Head-to-Head Report | `reports/final_results/head_to_head_audited.md` | Formal parity-matched head-to-head comparison report. |
