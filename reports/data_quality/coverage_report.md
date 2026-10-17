# FinFusionNet: Universe Scoping & Data Coverage Report

## 1. Executive Summary & Universe Decision

### Universe Determination: Justification of the 10 Heavyweights
After conducting a thorough audit of raw price series, news extraction pipelines, and model validity, we formally **decide and justify the 10-heavyweight universe**, scoping all experimental claims to them:

1. **Empirical News Availability Constraint**:
   - The pre-extracted GDELT financial news sentiment dataset (`ticker_day_news_sentiment.parquet`) contains dense, high-confidence news sentiment records **strictly for the 10 heavyweights** (4,988 ticker-day records, 11,768 total articles).
   - If the universe were expanded to the historical 62-ticker point-in-time constituent list without corresponding news ingestion, 52 out of 62 tickers would have exactly zero news records (100% missingness).
   - This would dilute the news sentiment hypotheses (H3, H4, H5) by introducing 84% structural nulls, destroying experimental test power.
2. **Economic & Liquidity Representation**:
   - The 10 heavyweights—**RELIANCE, TCS, HDFCBANK, ICICIBANK, BHARTIARTL, INFY, ITC, LT, SBIN, and HINDUNILVR**—consistently represent **over 55% of the total NIFTY 50 index market capitalization** and **over 60% of daily trading turnover**.
   - These 10 mega-caps represent the core drivers of Indian equity market momentum and macroeconomic beta.
3. **Formal Scoping of Claims**:
   - Every empirical claim, baseline comparison, and hypothesis conclusion in FinFusionNet is explicitly and transparently scoped to:
     > *"NIFTY 50 Top 10 Mega-Cap Equities (2020–2026)"*.

---

## 2. Coverage by Ticker (2020 – 2026)

| Ticker | Date Range | Total Observations | Trading Sessions | Total Articles | Sessions with News | News Density (%) | Up Target (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **BHARTIARTL** | 2020-01-01 to 2026-09-30 | 1,672 | 1,672 | 1,213 | 510 | 30.5% | 50.7% |
| **HDFCBANK** | 2020-01-01 to 2026-09-30 | 1,672 | 1,672 | 1,138 | 501 | 30.0% | 50.1% |
| **HINDUNILVR** | 2020-01-01 to 2026-09-30 | 1,672 | 1,672 | 1,154 | 493 | 29.5% | 47.6% |
| **ICICIBANK** | 2020-01-01 to 2026-09-30 | 1,672 | 1,672 | 1,280 | 513 | 30.7% | 51.8% |
| **INFY** | 2020-01-01 to 2026-09-30 | 1,672 | 1,672 | 1,039 | 475 | 28.4% | 50.0% |
| **ITC** | 2020-01-01 to 2026-09-30 | 1,672 | 1,672 | 1,164 | 511 | 30.6% | 48.7% |
| **LT** | 2020-01-01 to 2026-09-30 | 1,672 | 1,672 | 1,131 | 473 | 28.3% | 50.8% |
| **RELIANCE** | 2020-01-01 to 2026-09-30 | 1,672 | 1,672 | 1,261 | 505 | 30.2% | 51.0% |
| **SBIN** | 2020-01-01 to 2026-09-30 | 1,672 | 1,672 | 1,205 | 496 | 29.7% | 53.0% |
| **TCS** | 2020-01-01 to 2026-09-30 | 1,672 | 1,672 | 1,183 | 511 | 30.6% | 48.9% |

---

## 3. Coverage by Year (2020 – 2026)

| Year | Total Ticker-Days | Index Trading Days | Total News Articles | Sessions with News | News Density (%) | Market Up Days (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **2020.0** | 2,510.0 | 251.0 | 1,864.0 | 772.0 | 30.8% | 51.3% |
| **2021.0** | 2,480.0 | 248.0 | 1,564.0 | 700.0 | 28.2% | 51.5% |
| **2022.0** | 2,480.0 | 248.0 | 1,829.0 | 755.0 | 30.4% | 51.7% |
| **2023.0** | 2,450.0 | 245.0 | 1,685.0 | 728.0 | 29.7% | 52.4% |
| **2024.0** | 2,460.0 | 246.0 | 1,743.0 | 733.0 | 29.8% | 50.7% |
| **2025.0** | 2,490.0 | 249.0 | 1,744.0 | 739.0 | 29.7% | 49.2% |
| **2026.0** | 1,850.0 | 185.0 | 1,339.0 | 561.0 | 30.3% | 43.4% |

---

## 4. News Availability Matrix (% of Sessions with News by Ticker and Year)

| Ticker | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **BHARTIARTL** | 30.3% | 29.8% | 30.6% | 31.4% | 30.1% | 30.1% | 31.4% |
| **HDFCBANK** | 29.9% | 24.6% | 32.7% | 28.2% | 32.1% | 32.9% | 29.2% |
| **HINDUNILVR** | 29.5% | 29.4% | 30.6% | 27.3% | 26.4% | 31.7% | 31.9% |
| **ICICIBANK** | 33.1% | 31.9% | 34.7% | 31.4% | 26.0% | 28.1% | 29.2% |
| **INFY** | 33.5% | 26.6% | 25.8% | 29.8% | 27.2% | 30.9% | 23.8% |
| **ITC** | 29.5% | 30.2% | 29.8% | 29.0% | 31.7% | 28.5% | 36.8% |
| **LT** | 31.9% | 25.0% | 29.8% | 26.1% | 28.0% | 25.3% | 33.0% |
| **RELIANCE** | 31.9% | 32.3% | 27.0% | 31.8% | 31.3% | 29.3% | 27.0% |
| **SBIN** | 29.9% | 26.6% | 30.2% | 31.0% | 32.1% | 27.7% | 30.3% |
| **TCS** | 28.3% | 25.8% | 33.1% | 31.0% | 32.9% | 32.1% | 30.8% |

---

## 5. Data Quality & Missingness Policy
1. **Technical Warmup (2020)**: The first 50 sessions of 2020 serve as the causal warmup period for 50-day moving averages and 20-day volatility windows. All observations in the common cohort (2021–2026) have 0% missingness across technical indicators.
2. **Market Context**: NIFTY 50 index prices have 0% missingness across the entire 2020–2026 span.
3. **No-News Policy**: Sessions with zero news articles are preserved with `has_news = 0`, `n_articles = 0`, and null sentiment features, natively handled by XGBoost and mean-imputed in LSTM scaling.
