# FinFusionNet: Multimodal Financial Machine Learning & Temporal Cross-Attention Framework

<p align="center">
  <strong>An Empirical Investigation into Multimodal Fusion, Financial NLP, and Realized Volatility Forecasting in the Indian Equity Market</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/Framework-PyTorch%20%7C%20XGBoost-orange" alt="Framework">
  <img src="https://img.shields.io/badge/Market-India%20%7C%20Nifty%2050-green" alt="Market">
  <img src="https://img.shields.io/badge/NLP-ProsusAI%20FinBERT%20%7C%20VADER-purple" alt="NLP">
  <img src="https://img.shields.io/badge/Validation-Protocol%20B%20Expanding%20Walk--Forward-red" alt="Validation">
  <img src="https://img.shields.io/badge/Status-Audited%20Research%20Extraction-success" alt="Status">
</p>

---

## 1. Executive Summary & Core Research Questions

**FinFusionNet** is an academic-grade financial machine learning framework developed to evaluate whether integrating domain-specific financial news representations with sequential market microstructure features delivers statistically significant out-of-sample predictive power in Indian equities (**Nifty 50 index** universe).

The project investigates two fundamental empirical research questions:
1. **Direction Classification (RQ1)**: Does the multimodal fusion of daily financial news headlines (via lexicon scoring or transformer sentiment) with technical indicators produce statistically significant incremental predictive alpha over audited price-only baselines?
2. **Realized Volatility Regression (RQ2)**: Does a temporal cross-attention mechanism allow a neural network to attend to sparse news shocks and forecast next-day intraday realized volatility (**Parkinson 1980 High-Low estimator**) more accurately than standard autoregressive or moving-average benchmarks?

### Definitive Empirical Findings (Audited)
Across **5 experimental phases**, **205,687 cleaned news articles**, **56 unique constituents**, and **45 expanding monthly walk-forward folds** evaluated on an audited common evaluation cohort of **45,136 out-of-sample observations** (916 trading dates, 2023–2026):

1. **Multimodal Alpha Deficit**: Domain-specific sentiment (ProsusAI FinBERT) and temporal cross-attention **fail to produce statistically significant directional alpha** over price-only baselines. The simple price-only 2-layer LSTM (`LSTM-F0`) achieved the highest overall date-level accuracy (**51.08%**). The multimodal `CrossAttentionNet` achieved **50.16%** date accuracy, trailing the naive Always-Up baseline (**50.34%**) and underperforming `LSTM-F0` by **-0.92%** ($p = 0.0430$ date-level Brier DM; $p = 0.0339$ 0-1 loss DM).
2. **Benchmark Asymmetry Resolution**: Early reports indicating that Cross-Attention Net achieved a "+44.8% error reduction" over baseline volatility were an **artifact of benchmark asymmetry**—the model's intraday Parkinson high-low forecasts were compared against a close-to-close return volatility baseline carrying an intrinsic $+0.0331$ (+16.3%) overnight jump bias. When evaluated against a **fair like-for-like trailing 20-day Parkinson benchmark (`parkinson_20`)**, the simple benchmark outperforms the neural network on squared error ($R^2 = +14.24\%$ vs $+10.38\%$, $DM = -1.9515, p = 0.0513$).
3. **Probability Calibration vs Decision Accuracy**: Cross-Attention Net with sigmoid output shrinks probabilities tightly around 0.50 ($\sigma = 0.027, \text{Brier} = 0.2506$), achieving massive statistical superiority on probability calibration over tree-based XGBoost ($\sigma = 0.158, \text{Brier} = 0.2568, DM = +11.66, p < 10^{-15}$), despite having identical discrete directional accuracy (50.16% vs 50.42%, $p = 0.5906$).

---

## 2. Experimental Roadmap

```
+---------------------------------------------------------------------------------------------------+
|                                       FINFUSIONNET PIPELINE                                       |
+---------------------------------------------------------------------------------------------------+
  |
  +--> Exp 1: Audited Price-Only Baselines (F0)
  |    Models: XGBoost, Logistic Regression, 2-Layer LSTM
  |    Features: 24 technical indicators (returns, SMA ratios, RSI, MACD, Bollinger Bands, ATR, Vol)
  |    Result: LSTM-F0 establishes state-of-the-art benchmark (51.08% date acc, 0.5149 ROC-AUC).
  |
  +--> Exp 2: Market Context Augmentation
  |    Features: Price features + 5 Macro covariates (Nifty 50 return/volatility, India VIX)
  |    Result: Marginal +0.05% accuracy shift; firm-specific dynamics dominate broad macro beta.
  |
  +--> Exp 3 & 3B: Lexicon News Sentiment Fusion (VADER)
  |    News Source: Google News RSS feeds (Exp 3: 10-stock pilot; Exp 3B: Full All-50 expansion)
  |    Features: Price + 5 VADER sentiment features + 2 news controls (has_news, n_articles)
  |    Result: LSTM-F1_VADER achieved 50.99%; failed to beat price-only LSTM.
  |
  +--> Exp 4: Domain-Specific Transformer Concatenation (FinBERT)
  |    Sentiment Model: ProsusAI FinBERT (softmax probabilities, sentiment score, dispersion)
  |    Architecture: Early feature concatenation into XGBoost, Logistic Regression, and LSTM
  |    Result: LSTM-F1_FinBERT reached 50.53%; XGB-F1_FinBERT reached 50.42% (no incremental alpha).
  |
  +--> Exp 5: Temporal Cross-Attention Multimodal Network (CrossAttentionNet)
       Architecture: BiLSTM Price Encoder + Linear News Projection + Sentinel-Masked Cross-Attention
       Prediction Heads: Dual-task (Direction Classification + Parkinson Realized Volatility Regression)
       Result: Direction acc 50.16%; Volatility RMSE 0.1075 (fair benchmark parkinson_20 achieves 0.1052).
```

---

## 3. Audited Performance Results

All models are evaluated on the identical, audited common evaluation cohort of **45,136 observations** across **916 trading dates** (2023-01-02 to 2026-09-29) and **56 unique constituents** across 45 monthly expanding folds.

### Direction Classification Benchmark

| Model Configuration | Experiment | Date Acc | Pooled Acc | Balanced Acc | Precision | Recall | F1 Score | ROC-AUC | Brier Score | Log Loss | Predicted Up % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **LSTM-F0 (Price-Only)** | Exp 1 | **0.5108** | **0.5096** | **0.5091** | **0.5095** | 0.6297 | 0.5633 | **0.5149** | **0.2502** | **0.6935** | 62.07% |
| **LSTM-F1_VADER** | Exp 3B | 0.5099 | 0.5088 | 0.5084 | 0.5092 | 0.6045 | 0.5528 | 0.5105 | 0.2504 | 0.6940 | 59.62% |
| **LR-F1_FinBERT** | Exp 4 | 0.5095 | 0.5083 | 0.5077 | 0.5081 | 0.6567 | 0.5729 | 0.5103 | 0.2505 | 0.6943 | 64.91% |
| **LR-F0 (Price-Only)** | Exp 1 | 0.5094 | 0.5082 | 0.5075 | 0.5078 | 0.6724 | 0.5786 | 0.5102 | 0.2505 | 0.6942 | 66.49% |
| **XGB-F1_VADER** | Exp 3B | 0.5054 | 0.5043 | 0.5040 | 0.5057 | 0.5622 | 0.5325 | 0.5090 | 0.2564 | 0.7069 | 55.82% |
| **LSTM-F1_FinBERT** | Exp 4 | 0.5053 | 0.5041 | 0.5038 | 0.5054 | 0.5842 | 0.5419 | 0.5092 | 0.2505 | 0.6942 | 58.04% |
| **XGB-F1_FinBERT** | Exp 4 | 0.5042 | 0.5042 | 0.5039 | 0.5057 | 0.5593 | 0.5311 | 0.5070 | 0.2568 | 0.7076 | 55.53% |
| *Baseline: Always-Up* | Baseline | 0.5034 | 0.5021 | 0.5000 | 0.5021 | **1.0000** | **0.6686** | 0.5000 | 0.4979 | 8.0244 | 100.00% |
| **XGB-F0 (Price-Only)** | Exp 1 | 0.5027 | 0.5015 | 0.4996 | 0.5019 | 0.9351 | 0.6532 | 0.5067 | 0.2503 | 0.6937 | 93.55% |
| **Cross-Attention Net** | Exp 5 | 0.5016 | 0.5004 | 0.4996 | 0.5019 | 0.6852 | 0.5794 | 0.5016 | 0.2506 | 0.6943 | 68.56% |
| *Baseline: Prev-Day Direction* | Baseline | 0.4919 | 0.4930 | 0.4930 | 0.4952 | 0.4966 | 0.4959 | 0.4930 | 0.5070 | 8.1722 | 50.36% |

### Volatility Regression Benchmark (Parkinson Target)

The volatility target is the intraday high-low realized volatility estimator (**Parkinson 1980**):
$$\sigma_{P, t+1} = \sqrt{\frac{1}{4 \ln 2} \left( \ln \frac{H_{t+1}}{L_{t+1}} \right)^2 \times 252}$$

| Model / Benchmark | RMSE | MAE | Bias | Pearson $r$ | Out-of-Sample $R^2$ (%) | Date DM vs Cross-Attn (MSE) |
|---|---|---|---|---|---|---|
| **Trailing Parkinson 20d (Fair Benchmark)** | **0.1052** | 0.0695 | **+0.0003** | 0.3979 | **+14.24%** | $DM = -1.9515$ ($p = 0.0513$) |
| **Trailing Parkinson 10d (Fair Benchmark)** | 0.1058 | 0.0700 | +0.0003 | 0.4084 | +13.13% | $DM = -0.9768$ ($p = 0.3289$) |
| **Trailing Parkinson 5d (Fair Benchmark)** | 0.1066 | 0.0708 | +0.0002 | **0.4301** | +11.89% | $DM = -0.3994$ ($p = 0.6897$) |
| **Cross-Attention Net (Exp 5)** | 0.1075 | **0.0686** | -0.0089 | 0.3369 | +10.38% | — |
| *Expanding Historical Mean Vol* | 0.1135 | 0.0751 | 0.0000 | 0.0000 | 0.00% | — |
| *Trailing Parkinson 1d (Yesterday Vol)* | 0.1241 | 0.0839 | -0.0001 | 0.4057 | -19.38% | $DM = +8.1879$ ($p < 10^{-15}$) |
| *Trailing Close-to-Close 20d (Exp 5 Baseline)* | 0.1322 | 0.0900 | +0.0331 | 0.3225 | -35.58% | $DM = +6.4544$ ($p = 1.8 \times 10^{-10}$) |

---

## 4. Methodological Integrity & Leakage Prevention

1. **The 15:30 IST Causal Cutoff Protocol**: Indian equities trade from 09:15 to 15:30 IST. For trading session $t$, the allowed news window is strictly $(15:30\text{ IST on } t-1, 15:30\text{ IST on } t]$. Any headline published after 15:30 IST is assigned to session $t+1$. Weekend/holiday news is mapped to the next trading day.
2. **Missing-News Preservation**: Sparse news is natural in equity markets. Rather than imputing missing days with "neutral" sentiment (which distorts distribution), days without news have `has_news = 0`, `n_articles = 0`, and are masked using dedicated sentinel representations in the attention layer.
3. **Protocol B Expanding Walk-Forward**: 45 monthly expanding folds with a 24-month initial training window (~500 sessions) and a strict 1-session embargo between training and testing.
4. **Scaler Fitting**: Preprocessing (`StandardScaler`) is fitted strictly on the expanding training fold and applied out-of-sample to test folds.
5. **Point-in-Time Universe**: Includes all 56 historical Nifty 50 constituents across the research window (2019–2026), accounting for index additions and deletions without survivorship bias.

---

## 5. Research Data Extraction Deliverables

The complete project data extraction, audit trail, and supporting research reports are cataloged in [`docs/research_data_extraction/`](docs/research_data_extraction/):

- [`00_MASTER_RESEARCH_REPORT.md`](docs/research_data_extraction/00_MASTER_RESEARCH_REPORT.md): Comprehensive executive research synthesis and academic paper drafting guide.
- [`01_PROJECT_INVENTORY.csv`](docs/research_data_extraction/01_PROJECT_INVENTORY.csv): Full inventory of all 6,578 project files with metadata and research relevance.
- [`02_DATASET_AND_FEATURES.md`](docs/research_data_extraction/02_DATASET_AND_FEATURES.md): Constituent specifications, news corpus, 15:30 IST cutoff logic, and 39-feature dictionary.
- [`03_PREPROCESSING_AND_LEAKAGE_AUDIT.md`](docs/research_data_extraction/03_PREPROCESSING_AND_LEAKAGE_AUDIT.md): 10-point data leakage audit matrix.
- [`04_EXPERIMENT_CONFIGURATIONS.md`](docs/research_data_extraction/04_EXPERIMENT_CONFIGURATIONS.md): Hyperparameters, architectures, and loss functions for Exp 1 through Exp 5.
- [`05_ALL_EXPERIMENT_RESULTS.csv`](docs/research_data_extraction/05_ALL_EXPERIMENT_RESULTS.csv): Consolidated machine-readable table of audited metrics across all 19 model/benchmark configurations.
- [`06_CLASSIFICATION_RESULTS.md`](docs/research_data_extraction/06_CLASSIFICATION_RESULTS.md): Detailed classification metrics, confusion matrices, and market regime analyses.
- [`07_VOLATILITY_RESULTS.md`](docs/research_data_extraction/07_VOLATILITY_RESULTS.md): Parkinson volatility target formulation and like-for-like benchmark audit.
- [`08_STATISTICAL_AUDIT.md`](docs/research_data_extraction/08_STATISTICAL_AUDIT.md): In-depth mathematical resolution of mandatory inconsistencies (A, B, C, D).
- [`09_ARCHITECTURE_AND_MODEL_DETAILS.md`](docs/research_data_extraction/09_ARCHITECTURE_AND_MODEL_DETAILS.md): `CrossAttentionNet` PyTorch specifications, tensor shapes, and sentinel token equations.
- [`10_FIGURE_INVENTORY.md`](docs/research_data_extraction/10_FIGURE_INVENTORY.md): Inventory of visual assets (1 PDF) and 7 required publication figure blueprints.
- [`11_RESEARCH_TIMELINE.md`](docs/research_data_extraction/11_RESEARCH_TIMELINE.md): Reconstructed 6-phase research timeline.
- [`12_CONTRADICTIONS_AND_MISSING_DATA.md`](docs/research_data_extraction/12_CONTRADICTIONS_AND_MISSING_DATA.md): Severity-ranked contradiction matrix (C1–C4) and project gaps.
- [`13_SOURCE_FILE_REFERENCES.md`](docs/research_data_extraction/13_SOURCE_FILE_REFERENCES.md): Direct mapping of all pipelines, modules, predictions, and audit scripts.

---

## 6. Repository Structure

```
FinFusionNet/
├── configs/                             # Experiment configurations & hyperparameters
├── data/
│   ├── raw/                             # Historical prices & raw scraped news chunks
│   ├── processed/                       # Master feature datasets & cleaned news corpus
│   └── interim/                         # Ticker-day sentiment aggregations
├── docs/
│   ├── audit/                           # Historical experiment audits & statistical logs
│   │   └── exp1_exp5_statistical_audit/ # Independent statistical audit CSVs & scripts
│   └── research_data_extraction/        # Master extraction deliverables (00 through 13)
├── experiments/
│   ├── audit/                           # Exp 1 price-only fold predictions
│   ├── experiment_3/                    # Exp 3 VADER 10-stock pilot
│   ├── experiment_3b/                   # Exp 3B VADER All-50 predictions
│   ├── experiment_4/                    # Exp 4 FinBERT early concatenation
│   └── experiment_5/                    # Exp 5 Temporal Cross-Attention
├── notebooks/                           # Exploratory data analysis notebooks
├── scripts/                             # CLI execution scripts for training & audit
├── src/
│   └── finfusionnet/                    # Core production Python package
│       ├── data/                        # Ingestion & sequence generators
│       ├── evaluation/                  # Stats tests (DM, Bootstrap, McNemar)
│       ├── features/                    # Technical, context & sentiment features
│       └── models/                      # LSTM, XGBoost, and CrossAttentionNet
├── tests/                               # Comprehensive unit and integration test suite
└── README.md                            # Project documentation and audit summary
```

---

## 7. Quickstart & Verification

### Installation
```bash
git clone https://github.com/neil-data/FinFusionNet.git
cd FinFusionNet
pip install -e .
```

### Run Full Test Suite
```bash
pytest tests/
```

### Reproduce Statistical Audit & Cohort Parity
```bash
python docs/audit/exp1_exp5_statistical_audit/audit_exp1_exp5.py
```

---

## 8. License & Academic Citation

This project is licensed under the MIT License. If you use FinFusionNet code, datasets, or audit methodologies in your academic research, please cite:

```bibtex
@misc{finfusionnet2026,
  author = {Neil},
  title = {FinFusionNet: Multimodal Financial Machine Learning and Temporal Cross-Attention Framework},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/neil-data/FinFusionNet}}
}
```
