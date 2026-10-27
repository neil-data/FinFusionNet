# FinFusionNet Research Extraction: Source File Cross-Reference Directory

## 1. Primary Source Code Modules

| Subsystem | File Path | Core Functionality & Primary Classes |
|---|---|---|
| **Data Pipelines** | `src/data/download_prices.py` | Historical OHLCV fetching via `yfinance` for Nifty 50 universe. |
| **Data Pipelines** | `src/data/gnews_scraper.py` | Google News RSS querying, pagination, rate limiting, and chunk storage. |
| **Feature Engineering**| `src/features/technical_indicators.py` | Calculation of returns, SMAs, RSI, MACD, Bollinger Bands, ATR. |
| **Feature Engineering**| `src/features/market_context.py` | Nifty 50 index and India VIX macro feature generation. |
| **Feature Engineering**| `src/features/sentiment_vader.py` | Article-level VADER scoring and daily aggregations. |
| **Feature Engineering**| `src/features/sentiment_finbert.py` | Tokenization, inference with ProsusAI FinBERT, softmax probability pooling. |
| **Model Architectures**| `src/models/cross_attention_net.py` | `CrossAttentionNet`, `BiLSTMSentinelEncoder`, `MultiheadCrossAttention`, dual prediction heads. |
| **Model Architectures**| `src/models/lstm_baseline.py` | 2-layer LSTM price-only baseline (`LSTM-F0`) and concatenation (`LSTM-F1`). |
| **Cross-Validation** | `src/validation/walk_forward.py` | Protocol B expanding monthly fold generator with 1-session embargo. |
| **Statistical Testing**| `src/evaluation/diebold_mariano.py` | Diebold-Mariano test with Newey-West HAC spectral variance estimator. |
| **Statistical Testing**| `src/evaluation/bootstrap.py` | Date-block paired bootstrap with replacement ($B = 10,000$, block length = 10). |

---

## 2. Core Datasets & Prediction Parquets

| Dataset Description | File Path | Records | Research Significance |
|---|---|---:|---|
| **Master Feature Dataset** | `data/processed/master_dataset/master_ticker_day.parquet` | 82,425 | Consolidated feature dataset containing price, macro, VADER, and FinBERT. |
| **Clean News Corpus** | `data/processed/gnews_all50/gnews_all50_articles_clean.parquet` | 205,687 | Cleaned, deduplicated financial news articles with IST timestamps. |
| **Daily FinBERT Features** | `data/processed/gnews_all50/daily_gnews_all50_finbert_features.parquet` | 67,500 | Constituent-day aggregated FinBERT probabilities and dispersion. |
| **Exp 1 LSTM Predictions** | `experiments/audit/predictions/lstm_p_predictions.parquet` | 45,136 | Out-of-sample predictions for price-only LSTM baseline (`LSTM-F0`). |
| **Exp 1 XGB Predictions** | `experiments/audit/predictions/xgb_p_predictions.parquet` | 45,136 | Out-of-sample predictions for price-only XGBoost baseline (`XGB-F0`). |
| **Exp 1 LR Predictions** | `experiments/audit/predictions/lr_p_predictions.parquet` | 45,136 | Out-of-sample predictions for price-only Logistic Regression (`LR-F0`). |
| **Exp 3B VADER Predictions**| `experiments/experiment_3b/predictions/experiment_3b_common_cohort_predictions.parquet` | 45,136 | Out-of-sample predictions for VADER sentiment models. |
| **Exp 4 FinBERT Predictions**| `experiments/experiment_4/predictions/experiment_4_common_cohort_predictions.parquet` | 45,136 | Out-of-sample predictions for FinBERT early concatenation models. |
| **Exp 5 Cross-Attention Preds**| `experiments/experiment_5/predictions/experiment_5_common_cohort_predictions.parquet` | 45,136 | Out-of-sample predictions for `CrossAttentionNet` (Direction + Volatility). |

---

## 3. Audited Evaluation Artifacts

| Audit Report / Table | File Path | Contents |
|---|---|---|
| **Consolidated Audit Script** | `docs/audit/exp1_exp5_statistical_audit/audit_exp1_exp5.py` | Master audit execution script validating cohorts, DM tests, and bootstrap. |
| **Audited Direction Performance**| `docs/audit/exp1_exp5_statistical_audit/audited_direction_performance.csv` | Exact classification metrics across all models and naive baselines. |
| **Audited Statistical Significance**| `docs/audit/exp1_exp5_statistical_audit/audited_direction_statistical_significance.csv` | Pairwise bootstrap CIs, McNemar tests, DM tests, Holm corrections. |
| **Audited Volatility Performance**| `docs/audit/exp1_exp5_statistical_audit/audited_volatility_performance.csv` | Like-for-like Parkinson comparisons, RMSE, MAE, $R^2$, and bias. |
| **Audited Volatility DM Tests**| `docs/audit/exp1_exp5_statistical_audit/audited_volatility_diebold_mariano.csv` | Date-level and pooled HAC DM tests across volatility benchmarks. |
| **Audited Market Regimes** | `docs/audit/exp1_exp5_statistical_audit/audited_market_regimes.csv` | Performance breakdown by volatility quartiles and market bull/bear days. |
