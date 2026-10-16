# FinFusionNet

### Multi-Modal Financial Market Prediction Using Technical Indicators, Deep Learning, and Financial News Sentiment

<p align="center">
  <strong>Research Framework for Indian Equity Market Analysis</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/Domain-Financial%20Machine%20Learning-green" alt="Domain">
  <img src="https://img.shields.io/badge/Models-XGBoost%20%7C%20LSTM-orange" alt="Models">
  <img src="https://img.shields.io/badge/Status-Research%20in%20Progress-yellow" alt="Status">
  <img src="https://img.shields.io/badge/Market-India%20%7C%20NIFTY%2050-blue" alt="Market">
</p>

## 1. Overview
FinFusionNet is a research-oriented financial machine learning framework designed to investigate whether combining historical market behaviour with financial news sentiment can improve next-trading-day stock direction prediction.

Traditional technical-analysis-based prediction systems primarily rely on historical price and volume information. However, financial markets also respond to company announcements, economic developments, industry trends, and market-moving news.

FinFusionNet explores a multi-modal approach by integrating:
- **Technical indicators**: Extracting market trends, momentum, volatility, and volume-related patterns.
- **Machine learning**: Using XGBoost to model structured financial features.
- **Deep learning**: Using LSTM networks to investigate sequential market behaviour.
- **News sentiment**: Extracting sentiment signals from historical financial news using VADER and FinBERT.
- **Feature fusion**: Investigating whether combining technical and sentiment information provides additional predictive value.

The initial research focuses on NIFTY 50 constituent stocks and historical Indian equity market data.

The objective is not to claim that markets can be predicted with certainty, but to systematically evaluate the predictive contribution of different information sources under a realistic experimental setup.

## 2. Research Objectives
FinFusionNet is built around the following research questions:
- How effectively can technical indicators capture patterns associated with next-day stock direction?
- Can LSTM models extract useful temporal relationships from historical stock data?
- Does financial news sentiment provide information beyond conventional technical indicators?
- How do lexicon-based sentiment models compare with transformer-based financial sentiment models?
- Does combining technical and sentiment features improve predictive performance?
- Are observed improvements statistically meaningful and consistent across different evaluation periods?

### Core research hypothesis
> Combining technical market information with financial news sentiment may provide additional predictive information compared with using technical indicators alone.

This hypothesis will be evaluated through controlled experiments and out-of-sample validation.

## 3. System Architecture
```text
FINFUSIONNET
│
├── MARKET DATA                     NEWS DATA
│   ├── Historical Prices           ├── GDELT GKG 2.0
│   ├── Data Validation             ├── News Cleaning
│   └── Technical Indicators        ├── Entity Matching
│                                   └── Historical Alignment
│                                            │
│                                            ▼
│                                   ┌─────────────────┐
│                                   │ VADER | FinBERT │
│                                   └─────────────────┘
│                                            │
└───┬────────────────────────────────────────┘
    │
    ▼
Feature Aggregation
    │
    ▼
Master Dataset
    │
    ▼
Model Experiments
    ├── XGBoost
    ├── LSTM
    └── FinFusionNet
    │
    ▼
Walk-Forward Testing
    │
    ▼
Statistical Evaluation
    │
    ▼
Research Reports
```

FinFusionNet's architecture is modular. Individual components can be tested independently before being combined.

## 4. Data Strategy

### 4.1 Historical price data
The project uses historical daily stock market data for Indian equities.

| Parameter | Configuration |
| :--- | :--- |
| **Market** | Indian equity market |
| **Primary universe** | Point-in-time NIFTY 50 constituents |
| **Price warm-up period** | 2020-01-01 to 2026-09-30 |
| **Modeling period** | 2021-01-01 to 2026-09-30 |
| **Frequency** | Daily |
| **Initial data source** | Yahoo Finance |
| **Validation** | Corporate actions, missing dates, ticker continuity |

The additional warm-up period allows technical indicators to be calculated before the modeling window begins.

> [!IMPORTANT]
> Historical index membership is used to reduce survivorship bias. A stock should only enter the dataset during periods when it was both listed and an index constituent.

A frozen current-constituent universe may also be evaluated separately as a robustness comparison.

### 4.2 Financial news data
The historical news pipeline uses GDELT Global Knowledge Graph (GKG) 2.0 as its primary source.

The pipeline is designed to:
- Extract historical news records.
- Identify relevant companies through entity aliases.
- Clean and deduplicate news records.
- Filter relevant Indian-market coverage.
- Extract sentiment information.
- Align news with actual NSE trading sessions.
- Aggregate news information into ticker-day features.

Historical news extraction will be validated through sample-period pilots before full-scale processing.

### 4.3 Company entity matching
Company names and aliases are used to associate news articles with stock tickers.

For example:

| Ticker | Company / alias examples |
| :--- | :--- |
| `RELIANCE` | Reliance Industries, Reliance |
| `INFY` | Infosys, Infosys Limited |
| `TCS` | Tata Consultancy Services, TCS |
| `ETERNAL` | Eternal, Zomato |
| `TMPV` | Tata Motors, TMPV |

Ambiguous names require stricter matching rules to reduce incorrect company associations. Alias validity periods will account for company renames and other relevant changes.

## 5. News Sentiment Analysis
FinFusionNet investigates two complementary approaches to financial sentiment extraction.

### VADER
VADER is a lexicon-based sentiment analysis model. It provides a lightweight baseline for measuring positive, negative, and neutral language in news text.

### FinBERT
FinBERT is a transformer-based model adapted for financial language. It estimates financial text sentiment across three categories:
- Positive
- Negative
- Neutral

### Sentiment aggregation
Article-level sentiment is aggregated into ticker-day features. Planned features include:

| Feature | Description |
| :--- | :--- |
| `n_articles` | Number of relevant articles |
| `has_news` | Whether relevant news is available |
| `vader_mean` | Average VADER compound sentiment |
| `finbert_positive_mean` | Average positive probability |
| `finbert_negative_mean` | Average negative probability |
| `finbert_neutral_mean` | Average neutral probability |
| `sentiment_dispersion` | Variation in article sentiment |

Days without news are represented explicitly using `n_articles = 0` and `has_news = 0`. Missing sentiment values are not automatically treated as neutral sentiment.

## 6. Feature Engineering

### Technical indicators
The technical feature pipeline is designed to capture several dimensions of market behaviour.

| Category | Example indicators |
| :--- | :--- |
| **Trend** | SMA, EMA |
| **Momentum** | RSI, MACD |
| **Volatility** | Bollinger Bands, ATR |
| **Returns** | Daily returns, rolling returns |
| **Volume** | Volume change, volume moving averages |
| **Price behaviour** | High-low range, close-open movement |

All indicators must be calculated using information available at the relevant prediction time.

### Feature fusion
The research investigates combining:
- Historical price-derived features.
- Technical indicators.
- VADER sentiment.
- FinBERT sentiment.

Sector information may be evaluated as an additional feature group after the core experiments are established.

## 7. Prediction Target
The initial prediction task is binary next-trading-day direction classification.

The next-session return is defined as:
$$R_{t+1} = \frac{\text{Close}_{t+1}}{\text{Close}_t} - 1$$

The target variable is:
$$Y_t = \begin{cases} 1 & \text{if } R_{t+1} > 0 \\ 0 & \text{otherwise} \end{cases}$$

Where:
- $\text{Close}_t$ is the closing price of the current trading session.
- $\text{Close}_{t+1}$ is the closing price of the next trading session.
- $Y_t = 1$ represents an upward move.
- $Y_t = 0$ represents a non-positive move.

The target is based on the next valid trading session, not simply the next calendar day.

## 8. Model Development
FinFusionNet uses modular model implementations so that each information source can be evaluated independently.

- **XGBoost**: A gradient-boosted tree model used for structured technical and sentiment features.
- **LSTM**: A recurrent neural network used to investigate temporal dependencies in historical financial sequences.
- **FinFusionNet**: The proposed multi-modal model, designed to investigate the fusion of technical market information and financial news sentiment.

The exact fusion mechanism and architecture will be documented alongside the implementation and experiments.

### Controlled experiments
The experimental design compares progressively richer feature configurations.

| Experiment | Purpose |
| :--- | :--- |
| **E1** | Technical-feature baseline |
| **E2** | Additional model or feature baseline, defined in experiment configuration |
| **E3** | Technical features with VADER sentiment |
| **E4** | Technical features with FinBERT sentiment |
| **E5** | Multi-modal FinFusionNet fusion model |

The precise model, feature set, and evaluation configuration for every experiment will be recorded in version-controlled configuration files. The experiment definitions must remain consistent across comparisons.

## 9. Evaluation Methodology
Financial time series require careful validation because randomly splitting observations can introduce temporal leakage. FinFusionNet therefore emphasizes chronological evaluation.

### Walk-forward validation
Models are trained on historical observations and evaluated on later unseen periods. The existing experimental baseline uses chronological train/test windows. Alternative expanding-window configurations will be evaluated separately rather than silently replacing the baseline.

### Evaluation metrics
The following metrics are planned:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Balanced accuracy
- Directional prediction performance

### Statistical evaluation
Predictive improvements should not be judged using accuracy alone. The project includes statistical evaluation such as:
- Binomial testing against a clearly specified baseline.
- McNemar's test for paired classification comparisons.
- Appropriate consideration of serial dependence and repeated testing.

### Leakage prevention
The evaluation pipeline will test for:
- Future information entering technical features.
- News published after the prediction cutoff entering current-session features.
- Incorrect weekend and holiday alignment.
- Train/test temporal overlap.
- Incorrect historical index membership.
- Target information entering model inputs.

## 10. News and Market Time Alignment
Correct timestamp alignment is essential for realistic financial prediction.

The primary market reference is the NSE closing time of 15:30 Indian Standard Time. The pipeline converts news timestamps from UTC to Asia/Kolkata and maps each article to the appropriate valid trading session.

The intended convention is:
- News available by the prediction cutoff may contribute to that session's features.
- News published after the cutoff is assigned to the next valid trading session.
- Weekend and holiday news is assigned to the next valid trading session.
- The prediction target remains the following valid trading session.

The treatment of an article timestamped exactly at 15:30 will be explicitly defined and tested. A real trading calendar is used instead of assuming that the next calendar day is always a trading day.

## 11. Repository Structure
```text
FinFusionNet/
│
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── requirements.txt
├── pyproject.toml
│
├── configs/
│   ├── dataset.yaml
│   ├── universe.yaml
│   ├── features.yaml
│   ├── models.yaml
│   └── evaluation.yaml
│
├── docs/
│   ├── literature_review.md
│   ├── dataset_design.md
│   ├── methodology.md
│   ├── experimental_design.md
│   └── decision_log.md
│
├── metadata/
│   └── india_domains.txt
│
├── data/
│   ├── raw/
│   │   ├── prices/
│   │   ├── gdelt/
│   │   │   └── manifest.jsonl
│   │   └── constituent_history/
│   │
│   ├── interim/
│   │   ├── prices/
│   │   ├── news/
│   │   └── aligned_data/
│   │
│   ├── processed/
│   │   ├── ticker_day/
│   │   ├── master_dataset/
│   │   └── model_ready/
│   │
│   └── metadata/
│       ├── universe.csv
│       ├── sectors.csv
│       ├── aliases.csv
│       └── trading_calendar.csv
│
├── src/
│   └── finfusionnet/
│       ├── __init__.py
│       │
│       ├── data/
│       │   ├── universe.py
│       │   ├── prices.py
│       │   ├── gdelt.py
│       │   ├── news_cleaner.py
│       │   ├── trading_calendar.py
│       │   └── validation.py
│       │
│       ├── features/
│       │   ├── technical.py
│       │   ├── sentiment.py
│       │   ├── aggregation.py
│       │   └── targets.py
│       │
│       ├── models/
│       │   ├── xgboost_model.py
│       │   ├── lstm_model.py
│       │   └── finfusionnet.py
│       │
│       ├── evaluation/
│       │   ├── walk_forward.py
│       │   ├── metrics.py
│       │   ├── stats_tests.py
│       │   ├── backtesting.py
│       │   └── reports.py
│       │
│       └── utils/
│           ├── logging.py
│           ├── reproducibility.py
│           └── paths.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_price_analysis.ipynb
│   ├── 03_news_analysis.ipynb
│   ├── 04_feature_analysis.ipynb
│   └── 05_model_experiments.ipynb
│
├── scripts/
│   ├── fetch_prices.py
│   ├── validate_universe.py
│   ├── extract_gdelt.py
│   ├── build_features.py
│   ├── train.py
│   └── evaluate.py
│
├── experiments/
│   └── runs/
│       └── .gitkeep
│
├── artifacts/
│   └── final/
│
├── reports/
│   ├── data_quality/
│   ├── statistical_tests/
│   └── final_results/
│
└── tests/
    ├── test_universe.py
    ├── test_prices.py
    ├── test_gdelt.py
    ├── test_features.py
    ├── test_no_leakage.py
    ├── test_walk_forward.py
    └── test_stats_tests.py
```

## 12. Reproducibility
Every experiment should record:
- Experiment ID and timestamp.
- Configuration snapshot.
- Git commit hash.
- Dataset version or hash.
- Random seed.
- Software and dependency versions.
- Training and testing periods.
- Model parameters.
- Predictions and evaluation metrics.

Each experiment receives its own run directory:
```text
experiments/
└── runs/
    └── e3_technical_vader/
        └── 2026-10-16_120000/
            ├── config.yaml
            ├── metrics.json
            ├── predictions.csv
            └── run_metadata.json
```

Final trained models are stored separately under `artifacts/final/`.
Raw and processed datasets are not committed to Git unless they are small, redistributable, and explicitly intended for version control.

## 13. Installation

### Requirements
- Python 3.10 or higher
- Git
- Internet connection for data extraction
- Sufficient disk space for historical data
- Optional Google Cloud BigQuery access for GDELT extraction

### Clone the repository
```bash
git clone https://github.com/neil-data/FinFusionNet.git
cd FinFusionNet
```

### Create a virtual environment

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Configure environment variables
```bash
copy .env.example .env
```
Add any required credentials locally. Never commit API keys, service-account credentials, or other secrets.

## 14. Development Roadmap

| Phase | Work | Status |
| :---: | :--- | :---: |
| **1** | Project architecture and research design | In progress |
| **2** | Historical universe and metadata validation | Planned |
| **3** | Historical price extraction and validation | Planned |
| **4** | GDELT historical news extraction | Planned |
| **5** | News cleaning and company entity matching | Planned |
| **6** | Sentiment extraction and ticker-day aggregation | Planned |
| **7** | Technical feature engineering | Planned |
| **8** | Master dataset construction | Planned |
| **9** | Baseline XGBoost and LSTM experiments | Planned |
| **10** | FinFusionNet fusion architecture | Planned |
| **11** | Walk-forward and statistical evaluation | Planned |
| **12** | Final research analysis and documentation | Planned |

Project status should be updated as each phase is implemented and validated.

## 15. Research Limitations
FinFusionNet is an experimental research framework. Its results may be affected by:
- Incomplete or inaccurate historical news coverage.
- Incorrect company-name matching.
- Corporate actions and historical ticker changes.
- Survivorship bias in constituent selection.
- Market regime changes.
- Data-source limitations.
- Class imbalance and noisy market labels.
- Transaction costs and execution assumptions.
- Temporal dependence in financial observations.

Predictive performance in historical data does not guarantee future performance.

## 16. Future Scope
Potential extensions include:
- More advanced multimodal fusion architectures.
- Company-level event detection.
- Sector-level sentiment analysis.
- Additional financial news sources.
- Explainable AI for financial predictions.
- Probability calibration and uncertainty estimation.
- Transaction-cost-aware backtesting.
- Broader Indian equity market coverage.

These extensions will be considered after the core experimental framework has been validated.

## 17. Disclaimer
FinFusionNet is developed strictly for educational, analytical, and research purposes.
It does not provide financial advice, investment recommendations, or guaranteed market predictions.
Financial markets involve risk. Users should not make investment decisions solely on the basis of this project's outputs.

## 18. Author
**Neil**  
B.Tech Computer Engineering | Data Science & Machine Learning  
GitHub: [@neil-data](https://github.com/neil-data)

<p align="center">
  <strong>FinFusionNet — Researching the intersection of financial markets, machine learning, and news intelligence.</strong>
</p>
