# FinFusionNet Research Extraction: Dataset and Feature Engineering

## 1. Market Data Universe & Ticker Specifications

The market dataset encompasses the constituents of the **Nifty 50 Index** (National Stock Exchange of India) with point-in-time constituent handling to prevent survivorship bias across the historical research window (2019-01-01 to 2026-09-30).

- **Data Provider**: Yahoo Finance (`yfinance` API) with split and dividend adjustments (`auto_adjust=True`).
- **Total Historical Trading Sessions**: 1,920 trading days.
- **Constituent Representation**: 56 unique equity tickers (accounting for index additions and deletions such as BEL, SHRIRAMFIN, TRENT, etc., and historical deletions like UPL, SHREECEM).
- **Benchmark / Context Tickers**: `^NSEI` (Nifty 50 Index) and `^INDIAVIX` (India Volatility Index).
- **Temporal Integrity**: Zero weekend trading days; zero duplicate dates per ticker. Strict forward calendar alignment.

### Evaluated Constituent Universe (56 Tickers)
`ADANIENT`, `ADANIPORTS`, `APOLLOHOSP`, `ASIANPAINT`, `AXISBANK`, `BAJAJ-AUTO`, `BAJAJFINSV`, `BAJFINANCE`, `BEL`, `BHARTIARTL`, `BPCL`, `BRITANNIA`, `CIPLA`, `COALINDIA`, `DIVISLAB`, `DRREDDY`, `EICHERMOT`, `GRASIM`, `HCLTECH`, `HDFCBANK`, `HDFCLIFE`, `HEROMOTOCO`, `HINDALCO`, `HINDUNILVR`, `ICICIBANK`, `INDUSINDBK`, `INFY`, `ITC`, `JSWSTEEL`, `KOTAKBANK`, `LT`, `M&M`, `MARUTI`, `NESTLEIND`, `NTPC`, `ONGC`, `POWERGRID`, `RELIANCE`, `SBILIFE`, `SBIN`, `SUNPHARMA`, `TATACONSUM`, `TATAMOTORS`, `TATASTEEL`, `TCS`, `TECHM`, `TITAN`, `TRENT`, `ULTRACEMCO`, `WIPRO`, `SHRIRAMFIN`, `IOC`, `GAIL`, `UPL`, `SHREECEM`, `INDIGO`.

---

## 2. Financial News Dataset & Preprocessing

- **News Aggregator**: Google News RSS feeds queried per constituent using canonical entity alias lists.
- **Corpus Scale**: 205,687 cleaned news articles spanning 2020-01-01 through 2026-09-30 across 62 search queries/tickers.
- **Storage Architecture**: Raw articles collected in JSON chunks (`data/raw/gnews_all50/chunks/`, 5,022 files) and consolidated into compressed parquet (`data/processed/gnews_all50/gnews_all50_articles_clean.parquet`).
- **Deduplication**: Strict title fuzzy matching and URL hash canonicalization. 

### The 15:30 IST Causal Cutoff Protocol
To strictly guarantee that news sentiment is causally prior to the next trading session and prevent lookahead bias:
1. Indian equity markets trade between **09:15 IST** and **15:30 IST**.
2. For any trading session $t$, the associated news window is:
   $$\mathcal{W}_t = (15:30\text{ IST on } t-1, \quad 15:30\text{ IST on } t]$$
3. Any article published **after 15:30 IST on day $t$** is shifted to feature date $t+1$ (available for predicting return from $t+1$ close to $t+2$ close).
4. Weekend and market-holiday news articles are aggregated into the next active trading session's feature basket.

### Missing-News Representation
- In equity markets, news is naturally sparse. Converting missing news to "neutral" (e.g., VADER 0.0 or FinBERT neutral probability 1.0) introduces severe distribution distortion.
- FinFusionNet enforces **Missing-News Preservation**:
  - `has_news = 0`, `n_articles = 0`.
  - Continuous sentiment metrics remain `NaN` / null during feature creation, and are imputed via explicit missing indicator mechanics or sentinel zero-masking in the cross-attention module.

---

## 3. Exhaustive Feature Dictionary

The master dataset (`data/processed/master_dataset/master_ticker_day.parquet`) contains 54 columns, including 39 predictive input features, 7 raw price fields, and 8 target/audit fields.

| Feature Name | Category | Lookback | Mathematical Formula / Implementation | Available From |
|---|---|---|---|---|
| `return_1d` | Price / Momentum | 1 day | $(C_t - C_{t-1}) / C_{t-1}$ | Inception |
| `return_2d` | Price / Momentum | 2 days | $(C_t - C_{t-2}) / C_{t-2}$ | Inception |
| `return_5d` | Price / Momentum | 5 days | $(C_t - C_{t-5}) / C_{t-5}$ | Inception |
| `return_10d` | Price / Momentum | 10 days | $(C_t - C_{t-10}) / C_{t-10}$ | Inception |
| `return_20d` | Price / Momentum | 20 days | $(C_t - C_{t-20}) / C_{t-20}$ | Inception |
| `close_open_return` | Price / Intraday | 1 day | $(C_t - O_t) / O_t$ | Inception |
| `high_low_range_pct` | Volatility | 1 day | $(H_t - L_t) / L_t$ | Inception |
| `close_position_in_range`| Technical | 1 day | $(C_t - L_t) / (H_t - L_t + \epsilon)$ | Inception |
| `close_sma_5_ratio` | Technical / Trend | 5 days | $C_t / \text{SMA}_5(C_t) - 1.0$ | Day 5 |
| `close_sma_10_ratio` | Technical / Trend | 10 days | $C_t / \text{SMA}_{10}(C_t) - 1.0$ | Day 10 |
| `close_sma_20_ratio` | Technical / Trend | 20 days | $C_t / \text{SMA}_{20}(C_t) - 1.0$ | Day 20 |
| `close_sma_50_ratio` | Technical / Trend | 50 days | $C_t / \text{SMA}_{50}(C_t) - 1.0$ | Day 50 |
| `sma_5_sma_20_ratio` | Technical / Trend | 20 days | $\text{SMA}_5 / \text{SMA}_{20} - 1.0$ | Day 20 |
| `rsi_14` | Technical / Momentum| 14 days | Wilder's 14-day Relative Strength Index (0-100) | Day 15 |
| `macd` | Technical / Trend | 26 days | $\text{EMA}_{12}(C) - \text{EMA}_{26}(C)$ | Day 26 |
| `macd_signal` | Technical / Trend | 9 days | $\text{EMA}_9(\text{MACD})$ | Day 35 |
| `macd_histogram` | Technical / Trend | - | $\text{MACD} - \text{MACD}_{signal}$ | Day 35 |
| `atr_14_pct` | Volatility | 14 days | $\text{ATR}_{14} / C_t$ | Day 15 |
| `bb_width_20` | Volatility | 20 days | $(U_{20} - L_{20}) / \text{SMA}_{20}(C)$ | Day 20 |
| `volatility_10` | Volatility (Close-Close)| 10 days | $\text{Std}(\text{returns}_{10}) \times \sqrt{252}$ | Day 11 |
| `volatility_20` | Volatility (Close-Close)| 20 days | $\text{Std}(\text{returns}_{20}) \times \sqrt{252}$ | Day 21 |
| `volume_change_1d` | Volume | 1 day | $(V_t - V_{t-1}) / V_{t-1}$ | Day 2 |
| `volume_ratio_5` | Volume | 5 days | $V_t / \text{SMA}_5(V_t)$ | Day 5 |
| `volume_ratio_20` | Volume | 20 days | $V_t / \text{SMA}_{20}(V_t)$ | Day 20 |
| `nifty_return_1d` | Context / Macro | 1 day | $(I_t - I_{t-1}) / I_{t-1}$ of Nifty 50 | Inception |
| `nifty_return_5d` | Context / Macro | 5 days | $(I_t - I_{t-5}) / I_{t-5}$ of Nifty 50 | Inception |
| `nifty_return_20d` | Context / Macro | 20 days | $(I_t - I_{t-20}) / I_{t-20}$ of Nifty 50 | Inception |
| `nifty_volatility_10` | Context / Macro | 10 days | 10-day rolling annualized volatility of Nifty 50 | Day 11 |
| `nifty_volatility_20` | Context / Macro | 20 days | 20-day rolling annualized volatility of Nifty 50 | Day 21 |
| `n_articles` | News Control | Day $t$ window | Integer count of articles in $(15:30_{t-1}, 15:30_t]$ | Day 1 |
| `has_news` | News Control | Day $t$ window | Binary indicator $\mathbb{I}(n\_articles > 0)$ | Day 1 |
| `vader_mean` | VADER Sentiment | Day $t$ window | Mean compound score $\in [-1, 1]$ | Day 1 |
| `vader_positive_mean`| VADER Sentiment | Day $t$ window | Mean positive score $\in [0, 1]$ | Day 1 |
| `vader_negative_mean`| VADER Sentiment | Day $t$ window | Mean negative score $\in [0, 1]$ | Day 1 |
| `vader_neutral_mean` | VADER Sentiment | Day $t$ window | Mean neutral score $\in [0, 1]$ | Day 1 |
| `vader_dispersion` | VADER Sentiment | Day $t$ window | Standard deviation of compound scores | Day 1 |
| `finbert_positive_mean`| FinBERT Sentiment | Day $t$ window | Mean positive softmax probability | Day 1 |
| `finbert_negative_mean`| FinBERT Sentiment | Day $t$ window | Mean negative softmax probability | Day 1 |
| `finbert_neutral_mean` | FinBERT Sentiment | Day $t$ window | Mean neutral softmax probability | Day 1 |
| `finbert_sentiment_score`| FinBERT Sentiment | Day $t$ window | $P(\text{pos}) - P(\text{neg}) \in [-1, 1]$ | Day 1 |
| `finbert_dispersion` | FinBERT Sentiment | Day $t$ window | Std dev of sentiment scores across articles | Day 1 |
| `finbert_confidence` | FinBERT Sentiment | Day $t$ window | Mean maximum softmax probability $\max(P)$ | Day 1 |

---

## 4. Master Dataset Summary Statistics

- **Total Rows**: 82,425 daily stock observations across 2020-01-01 to 2026-09-30.
- **Audit Evaluation Cohort Rows**: Exactly **45,136 observations** (45 monthly folds spanning 2023-01-02 to 2026-09-29).
- **Target Distribution**:
  - `target = 1` (Next-session return $> 0$): 22,662 observations (**50.21%**).
  - `target = 0` (Next-session return $\le 0$): 22,474 observations (**49.79%**).
- **News Availability**:
  - Days with news (`has_news = 1`): 31,458 observations (**69.70%**).
  - Days without news (`has_news = 0`): 13,678 observations (**30.30%**).
  - Average articles on news days: 3.42 articles/day.
