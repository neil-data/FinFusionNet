# FinFusionNet — LSTM Research Track Methodology

## 1. Executive Summary & Research Objective
The LSTM Research Track of FinFusionNet investigates whether sequential deep learning models (Long Short-Term Memory networks) can extract meaningful temporal patterns from historical financial time series to improve next-trading-session stock direction prediction in the Indian equity market (NIFTY 50).

The research enforces:
1. **Methodological Consistency**: Directly mirrors the data integrity, point-in-time constituent universe, target definitions, and walk-forward protocols established in the XGBoost track.
2. **Strict Ticker Isolation**: 20-session rolling sequence lookback windows ($T-19$ to $T$) generated independently per ticker. Sequences never cross ticker boundaries.
3. **Leakage-Safe Preprocessing**: Feature scaling (e.g., StandardScaler) is fitted strictly on training sequences within each fold. Test data never enters preprocessing.
4. **Target-Date Purging**: Strict separation at train/val and val/test boundaries ensuring overlapping forward prediction horizons do not leak into evaluation.
5. **Statistical Defensibility**: Paired date-block bootstrap (10,000 repetitions, block length $L=10$ trading days) to account for cross-sectional and temporal correlation.

---

## 2. Sequence Construction & Lookback Window

### Sequence Geometry
- **Lookback Window ($W$)**: Fixed at 20 consecutive trading sessions ($T-19, T-18, \dots, T$).
- **Prediction Target ($Y_T$)**: Next-session direction at $T+1$ ($1$ if $Close_{T+1} > Close_T$ else $0$).
- **Ticker Isolation**: For every ticker $i$, sequences are generated chronologically. Under no circumstances are observations from different stocks mixed or concatenated within a sequence.

### Boundary Context Preservation
At the start of any validation or test fold:
- The preceding 19 sessions from the end of the chronological training/validation partition are used strictly as input feature context for the initial test sequence.
- These context sessions are never evaluated as test targets; only predictions for dates in the test partition are logged and evaluated.
- This ensures zero loss of test dates due to sequence warm-up, maintaining 100% common evaluation cohort alignment with the tabular XGBoost track.

---

## 3. Preprocessing & Missing Data Policy

### In-Fold Feature Scaling
Unlike gradient-boosted trees, neural networks require normalized numerical inputs.
- Scalers are fitted **strictly on the training sequences** within each walk-forward fold:
  $$\mu_{train} = \frac{1}{N_{train} \cdot W} \sum_{n=1}^{N_{train}} \sum_{t=1}^W x_{n, t}$$
- Validation and test sequences are transformed out-of-sample using $\mu_{train}$ and $\sigma_{train}$.

### Missing Sentiment Handling
- On trading sessions without qualifying financial news (`has_news == 0` and `n_articles == 0`), sentiment features are filled with $0.0$.
- Because `has_news` is included as an explicit binary attention feature ($0$ for no-news, $1$ for news), the LSTM network can natively disambiguate between true neutral sentiment and no-news days.

---

## 4. LSTM Architecture Specification

The network architecture is kept controlled, reproducible, and parsimonious:
```text
Input Sequence [Batch Size, 20 Sessions, N Features]
       │
       ▼
  LSTM Layer 1 (hidden_dim = 64, return_sequences = True)
       │
       ▼
  Dropout (p = 0.20)
       │
       ▼
  LSTM Layer 2 (hidden_dim = 32, return_sequences = False)
       │
       ▼
  Dropout (p = 0.20)
       │
       ▼
  Dense Hidden Layer (Linear 32 -> 32, ReLU)
       │
       ▼
  Dropout (p = 0.10)
       │
       ▼
  Output Layer (Linear 32 -> 1, Sigmoid)
```

- **Loss Function**: Binary Cross-Entropy (`BCELoss`)
- **Optimizer**: Adam (`lr = 0.001`, `weight_decay = 1e-5`)
- **Batch Size**: 64
- **Epochs**: Fixed 20 (Protocol A) / Up to 30 with early stopping patience = 5 on validation loss (Protocol B).
- **Classification Threshold**: Fixed 0.50 ($P(Y=1) > 0.50$).
