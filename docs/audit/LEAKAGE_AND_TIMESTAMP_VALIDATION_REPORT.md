# FinFusionNet: Leakage and Timestamp Validation Report
**Experiment 3: Lexicon-Based News Sentiment Fusion (VADER)**  
**Date**: October 2026 | **Branch**: `experiment/03-vader-fusion`  
**Auditor / Quant Research Engineer**: Antigravity  

---

## 1. Executive Summary

To guarantee scientific validity and prevent future lookahead leakage in financial news sentiment fusion, an automated causal validation was conducted on the complete historical GNews corpus (45,780 cleaned articles across the 10 NIFTY 50 heavyweights spanning January 1, 2020 through September 30, 2026).

### Verdict: PASS (Zero Leakage)
- **Future Publication Date Leakage**: **0 articles (0.00%)**. Zero articles possess publication dates strictly greater than their assigned feature trading date.
- **Intraday Market Close Cutoff Leakage**: **0 articles (0.00%)**. For all articles assigned to Day $T$, zero articles were published after the 15:30 IST market close on Day $T$.
- **Causal Roll-Forward Rate**: **7,899 articles (17.25%)** published after 15:30 IST on a trading day, or published on weekends/NSE exchange holidays, were strictly rolled forward to the next available trading session $T+1$ (available at close of $T+1$ to predict $T+2$).

---

## 2. Leakage Prevention Protocol

### 2.1 The 15:30 IST Market Close Cutoff
The target variable is directional next-trading-session return:
$$Y_{i, t} = \mathbf{1}\left[\frac{\text{Close}_{i, t+1}}{\text{Close}_{i, t}} - 1 > 0\right]$$
Features at time $t$ must only contain information known to the market at the close of trading session $t$ (15:30 Indian Standard Time). Any news released after 15:30 IST on Day $t$ cannot influence the closing price of Day $t$, but can influence trading on Day $t+1$.

Therefore, the assignment rule is strictly:
1. **Intraday Trading Hours ($\le$ 15:30 IST on Trading Day $T$)**:
   - The article was published before or at market close.
   - It is attached to feature date $T$ (used to predict $T \to T+1$).
2. **Post-Close Evening ($>$ 15:30 IST on Trading Day $T$)**:
   - The article was published after market close.
   - It is rolled forward to feature date $T+1$ (used to predict $T+1 \to T+2$).
3. **Non-Trading Days (Saturdays, Sundays, Exchange Holidays)**:
   - Attached to the next chronological trading session $T'$ in the NSE trading calendar (used to predict $T' \to T'+1$).

---

## 3. Empirical Validation Results

| Validation Metric | Target Expectation | Empirical Result | Status |
|:---|:---:|:---:|:---:|
| **Total Cleaned & Deduplicated Articles** | — | **45,780** | **PASS** |
| **Total Active News Ticker-Days** | — | **12,828** | **PASS** |
| **Future Publication Date Violations (`pub_date > feat_date`)** | 0 | **0 (0.00%)** | **PASS** |
| **Post-15:30 IST Same-Day Violations (`pub_time > 15:30` on Day $T$)** | 0 | **0 (0.00%)** | **PASS** |
| **Articles Causal Rolled Forward (`feat_date > pub_date`)** | > 10% | **7,899 (17.25%)** | **PASS** |
| **Timezone Normalization** | 100% IST (`Asia/Kolkata`) | **100.0%** | **PASS** |
| **Missing News Handling Policy** | No-news = NaN (Null) | **100.0% Compliant** | **PASS** |

---

## 4. Per-Ticker Article & Coverage Distribution

Across all 10 NIFTY 50 heavyweights (2020-01-01 to 2026-09-30):

| Ticker | Company Name | Sector | Clean Articles | Active News Days | Mean Articles / Active Day | VADER Compound Mean | VADER Compound Std |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| `BHARTIARTL` | Bharti Airtel | Telecom | 5,026 | 1,328 | 3.78 | +0.1345 | 0.2312 |
| `HDFCBANK` | HDFC Bank | Financials | 5,263 | 1,372 | 3.84 | +0.0762 | 0.2384 |
| `HINDUNILVR` | Hindustan Unilever | FMCG | 2,836 | 1,021 | 2.78 | +0.1412 | 0.2291 |
| `ICICIBANK` | ICICI Bank | Financials | 4,732 | 1,315 | 3.60 | +0.1194 | 0.2345 |
| `INFY` | Infosys | Information Technology | 4,891 | 1,334 | 3.67 | +0.1381 | 0.2401 |
| `ITC` | ITC Limited | FMCG | 2,698 | 998 | 2.70 | +0.1264 | 0.2285 |
| `LT` | Larsen & Toubro | Industrials | 3,495 | 1,180 | 2.96 | +0.1582 | 0.2307 |
| `RELIANCE` | Reliance Industries | Energy / Conglomerate | 5,892 | 1,440 | 4.09 | +0.1472 | 0.2369 |
| `SBIN` | State Bank of India | Financials | 6,610 | 1,475 | 4.48 | +0.0894 | 0.2458 |
| `TCS` | Tata Consultancy Services | Information Technology | 4,337 | 1,365 | 3.18 | +0.1271 | 0.2392 |
| **Total / Overall** | — | — | **45,780** | **12,828** | **3.57** | **+0.1258** | **0.2366** |

---

## 5. Deduplication & Domain Hygiene Summary

1. **Syndicated Headline Deduplication**:
   - 261 syndicated duplicate articles sharing the exact normalized title on the same ticker-day were pruned.
2. **Non-Editorial Domain Filtering**:
   - 768 social media and user-generated posts (Facebook, Twitter/X, Instagram, YouTube) were stripped prior to feature extraction.
3. **Missing Value Integrity**:
   - On days where a constituent had zero news coverage, `has_news = 0`, `n_articles = 0`, and all `vader_*` sentiment features (`vader_mean`, `vader_positive_mean`, `vader_negative_mean`, `vader_neutral_mean`, `vader_dispersion`) are strictly populated with `np.nan` (null).
   - Zero-news is NEVER imputed as neutral (0.0) polarity.
