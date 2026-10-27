# Experiment 3B: Data Integrity, Leakage, and Coverage Validation Report

**Date of Audit**: October 2026  
**Scope**: Full Point-in-Time NIFTY 50 Universe (62 in-window constituents)  
**Historical News Window**: 2020-01-01 to 2026-09-30  
**Source Provider**: GNews RSS Financial News Pipeline  

---

## 1. Executive Summary & Audit Verification
- **Future Publication Leakage**: **0** violations detected. (PASS)
- **Market Cutoff (15:30 IST) Leakage**: **0** same-day post-cutoff violations detected. (PASS)
- **Weekend / Non-Trading Day Leakage**: **0** non-trading day assignments detected. (PASS)
- **Trading Date Validity**: All `205,687` clean articles map to authenticated NSE trading dates. (PASS)
- **Missing News Policy**: Confirmed strictly `NaN` across all sentiment dimensions for days without news. No neutral zero-imputation. (PASS)

---

## 2. Temporal & Information Cutoff Compliance

| Audit Check | Requirement | Count / Value | Status |
| :--- | :--- | :--- | :--- |
| **Future Publication Leaks** | `pub_date > feature_date` = 0 | 0 | **PASS** |
| **Post-15:30 IST Same-Day Leaks** | `pub_time > 15:30 & pub_date == feature_date` = 0 | 0 | **PASS** |
| **Weekend / Holiday Assignments** | `feature_date in trading_calendar` = 100% | 100% (205,687 / 205,687) | **PASS** |
| **Articles Rolled Forward** | Rolled to next active trading session | 35,199 (17.11%) | **VERIFIED** |
| **Missing Sentiment Representation** | `has_news == 0 ==> sentiment is NaN` | Strictly Enforced | **PASS** |

---

## 3. Dataset Volume & Universe Coverage

- **Total Cleaned Articles**: `205,687` (deduplicated editorial articles)
- **Unique Tickers with News**: `62` / 62 constituents
- **Total Master Ticker-Day Observations**: `82,425`
- **Universe Ticker-Days with News**: `56,155` (`68.13%`)
- **Evaluation Window (2023–2026) Coverage**: `32,921` / `45,422` (`72.48%`)

### Comparison: Experiment 3 (10 Heavyweights) vs Experiment 3B (All-50)

| Metric | Experiment 3 (10 Stocks) | Experiment 3B (All-50 Stocks) | Delta |
| :--- | :--- | :--- | :--- |
| **Tickers Covered** | 10 | 62 | +52 |
| **Total Cleaned Articles** | 45,780 | 205,687 | +159,907 |
| **Active Ticker-Days with News** | 12,828 | 67,500 | +54,672 |
| **Evaluation Cohort Coverage %** | 16.38% | 72.48% | +56.10% |

---

## 4. Per-Ticker Coverage Breakdown

| Rank | Ticker | Total Trading Days | Days with News | Coverage % | Total Articles | Mean Articles / Active Day | Mean VADER Compound |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `SBIN` | 1672 | 1508 | 90.2% | 6,700 | 4.44 | +0.0778 |
| 2 | `TATAMOTORS` | 1672 | 1476 | 88.3% | 7,579 | 5.13 | +0.0792 |
| 3 | `COALINDIA` | 1672 | 1418 | 84.8% | 4,809 | 3.39 | +0.0277 |
| 4 | `TATASTEEL` | 1672 | 1404 | 84.0% | 5,511 | 3.93 | +0.1186 |
| 5 | `MARUTI` | 1672 | 1396 | 83.5% | 7,138 | 5.11 | +0.1015 |
| 6 | `M&M` | 1672 | 1385 | 82.8% | 7,074 | 5.11 | +0.0776 |
| 7 | `ICICIBANK` | 1672 | 1383 | 82.7% | 4,753 | 3.44 | +0.1362 |
| 8 | `BHARTIARTL` | 1672 | 1362 | 81.5% | 5,064 | 3.72 | +0.1087 |
| 9 | `INFY` | 1672 | 1349 | 80.7% | 4,910 | 3.64 | +0.1312 |
| 10 | `TECHM` | 1672 | 1345 | 80.4% | 3,964 | 2.95 | +0.1081 |
| 11 | `HDFCBANK` | 1672 | 1343 | 80.3% | 5,201 | 3.87 | +0.1481 |
| 12 | `NTPC` | 1672 | 1326 | 79.3% | 4,433 | 3.34 | +0.1088 |
| 13 | `AXISBANK` | 1672 | 1324 | 79.2% | 4,224 | 3.19 | +0.1383 |
| 14 | `BAJAJ-AUTO` | 1672 | 1320 | 79.0% | 5,451 | 4.13 | +0.1189 |
| 15 | `POWERGRID` | 1672 | 1318 | 78.8% | 3,427 | 2.60 | +0.1323 |
| 16 | `WIPRO` | 1671 | 1289 | 77.1% | 4,043 | 3.14 | +0.1252 |
| 17 | `TCS` | 1672 | 1284 | 76.8% | 4,349 | 3.39 | +0.1331 |
| 18 | `RELIANCE` | 1672 | 1279 | 76.5% | 5,472 | 4.28 | +0.1152 |
| 19 | `TITAN` | 1672 | 1263 | 75.5% | 3,554 | 2.81 | +0.1073 |
| 20 | `SUNPHARMA` | 1672 | 1225 | 73.3% | 3,386 | 2.76 | +0.0991 |
| 21 | `KOTAKBANK` | 1672 | 1204 | 72.0% | 3,543 | 2.94 | +0.1398 |
| 22 | `ASIANPAINT` | 1672 | 1169 | 69.9% | 2,954 | 2.53 | +0.1338 |
| 23 | `NESTLEIND` | 1672 | 1167 | 69.8% | 2,957 | 2.53 | +0.0922 |
| 24 | `HCLTECH` | 1672 | 1157 | 69.2% | 3,070 | 2.65 | +0.1411 |
| 25 | `LT` | 1672 | 1140 | 68.2% | 3,526 | 3.09 | +0.1622 |
| 26 | `BAJFINANCE` | 1672 | 1137 | 68.0% | 3,131 | 2.75 | +0.1424 |
| 27 | `ITC` | 1672 | 1129 | 67.5% | 2,709 | 2.40 | +0.1221 |
| 28 | `JSWSTEEL` | 1672 | 1109 | 66.3% | 2,890 | 2.61 | +0.0878 |
| 29 | `HEROMOTOCO` | 1424 | 1070 | 75.1% | 4,239 | 3.96 | +0.5756 |
| 30 | `HDFCLIFE` | 1672 | 1055 | 63.1% | 2,266 | 2.15 | +0.1322 |
| 31 | `ADANIPORTS` | 1672 | 1054 | 63.0% | 3,325 | 3.15 | +0.0836 |
| 32 | `HINDUNILVR` | 1672 | 1002 | 59.9% | 2,849 | 2.84 | +0.1383 |
| 33 | `ULTRACEMCO` | 1672 | 988 | 59.1% | 2,334 | 2.36 | +0.1005 |
| 34 | `TATACONSUM` | 1361 | 960 | 70.5% | 2,450 | 2.55 | +0.1411 |
| 35 | `BAJAJFINSV` | 1672 | 957 | 57.2% | 2,216 | 2.32 | +0.1467 |
| 36 | `SBILIFE` | 1672 | 952 | 56.9% | 1,897 | 1.99 | +0.1367 |
| 37 | `CIPLA` | 1672 | 911 | 54.5% | 1,940 | 2.13 | +0.0668 |
| 38 | `ONGC` | 1672 | 909 | 54.4% | 2,282 | 2.51 | +0.0594 |
| 39 | `DRREDDY` | 1672 | 875 | 52.3% | 1,920 | 2.19 | +0.0561 |
| 40 | `HINDALCO` | 1672 | 859 | 51.4% | 1,803 | 2.10 | +0.0987 |
| 41 | `GRASIM` | 1672 | 840 | 50.2% | 1,549 | 1.84 | +0.0974 |
| 42 | `APOLLOHOSP` | 1113 | 816 | 73.3% | 2,406 | 2.95 | +0.0861 |
| 43 | `EICHERMOT` | 1672 | 810 | 48.4% | 1,802 | 2.22 | +0.1248 |
| 44 | `INDUSINDBK` | 1424 | 757 | 53.2% | 1,706 | 2.25 | +0.1345 |
| 45 | `ADANIENT` | 988 | 686 | 69.4% | 2,709 | 3.95 | +0.0930 |
| 46 | `DIVISLAB` | 1175 | 595 | 50.6% | 963 | 1.62 | +0.0801 |
| 47 | `BRITANNIA` | 1299 | 513 | 39.5% | 871 | 1.70 | +0.1416 |
| 48 | `BPCL` | 1299 | 471 | 36.3% | 736 | 1.56 | +0.0813 |
| 49 | `SHRIRAMFIN` | 621 | 455 | 73.3% | 1,329 | 2.92 | +0.1732 |
| 50 | `BEL` | 497 | 397 | 79.9% | 1,434 | 3.61 | +0.1879 |
| 51 | `ETERNAL` | 373 | 308 | 82.6% | 1,631 | 5.30 | +0.0828 |
| 52 | `UPL` | 1051 | 277 | 26.4% | 344 | 1.24 | +0.0517 |
| 53 | `IOC` | 559 | 276 | 49.4% | 456 | 1.65 | +0.0477 |
| 54 | `JIOFIN` | 373 | 275 | 73.7% | 970 | 3.53 | +0.1425 |
| 55 | `MAXHEALTH` | 248 | 199 | 80.2% | 648 | 3.26 | +0.0780 |
| 56 | `SHREECEM` | 684 | 194 | 28.4% | 263 | 1.36 | +0.0976 |
| 57 | `TRENT` | 497 | 178 | 35.8% | 464 | 2.61 | +0.1574 |
| 58 | `GAIL` | 311 | 159 | 51.1% | 271 | 1.70 | +0.0756 |
| 59 | `INDIGO` | 248 | 147 | 59.3% | 530 | 3.61 | +0.0457 |
| 60 | `BSE` | 1 | 1 | 100.0% | 3 | 3.00 | +0.6710 |

---
*Report automatically generated by `scripts/validate_experiment_3b_data.py` on FinFusionNet.*
