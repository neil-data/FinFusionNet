# FinFusionNet Research Extraction: Experiment Configurations & Hyperparameters

## 1. Architectural Overview Across Experiments

FinFusionNet was developed through five iterative experiment phases to test whether multimodal fusion (combining financial price series with sentiment representations) yields statistically significant out-of-sample alpha in Indian equities.

```
Exp 1 (Price-Only Baselines)       --> XGB-F0, LR-F0, LSTM-F0 (24 Technical Features)
Exp 2 (Macro Context Augmentation) --> Price + 5 Macro Context Features (Nifty 50, India VIX)
Exp 3 & 3B (Lexicon Sentiment)     --> Price + VADER Sentiment (10-Stock Pilot -> All-50 Full Universe)
Exp 4 (Domain Transformer Concat)  --> Price + FinBERT Probabilities & Dispersions (Early Concatenation)
Exp 5 (Temporal Cross-Attention)   --> Temporal Cross-Attention Multimodal Network (Dual Heads: Direction + Parkinson Volatility)
```

---

## 2. Model Hyperparameters and Training Configurations

### Experiment 1: Price-Only Baselines (F0)
- **Feature Set**: 24 technical indicators (returns, SMA ratios, RSI, MACD, ATR, Bollinger Bands, Volume ratios).
- **Models**:
  1. **Logistic Regression (LR-F0)**:
     - Regularization: L2 ($C = 1.0$), solver: `lbfgs`, max iterations: 1,000.
     - Preprocessing: `StandardScaler` fitted on train fold.
  2. **XGBoost (XGB-F0)**:
     - `n_estimators`: 100, `max_depth`: 4, `learning_rate`: 0.05.
     - `subsample`: 0.8, `colsample_bytree`: 0.8, `random_state`: 42.
     - Objective: `binary:logistic`, evaluation metric: `logloss`.
  3. **LSTM (LSTM-F0)**:
     - Sequence Length: 20 trading days ($T=20$).
     - Input Dimension: 24 features.
     - Hidden Dimension: 64, Layers: 2, Dropout: 0.2.
     - Fully Connected Head: $64 \to 32 \to 1$ (Sigmoid).
     - Optimizer: Adam ($lr = 10^{-3}$, weight decay $= 10^{-4}$), Loss: Binary Cross-Entropy.
     - Batch Size: 64, Epochs: 30, Early Stopping patience: 5 on validation split.

### Experiment 2: Market Context Augmentation
- **Feature Set**: 24 technical features + 5 Macro/Market Context features (`nifty_return_1d`, `nifty_return_5d`, `nifty_return_20d`, `nifty_volatility_10`, `nifty_volatility_20`).
- Model parameters identical to Exp 1 with updated input dimension ($D=29$).

### Experiment 3 & 3B: Lexicon Sentiment Fusion (VADER)
- **Feature Set**: Price features + 5 VADER sentiment features + 2 news controls (`n_articles`, `has_news`).
- Exp 3: Initial 10-stock exploratory pilot.
- Exp 3B: Full 50-constituent expansion. Evaluated using XGB-F1_VADER and LSTM-F1_VADER.

### Experiment 4: FinBERT Early Concatenation (F1_FinBERT)
- **Sentiment Model**: Local ProsusAI FinBERT transformer (`ProsusAI/finbert`).
- **Extracted Sentiment Dimensions**: 6 daily aggregated features:
  - `finbert_positive_mean`, `finbert_negative_mean`, `finbert_neutral_mean`
  - `finbert_sentiment_score` ($P(\text{pos}) - P(\text{neg})$)
  - `finbert_dispersion` (article-level standard deviation)
  - `finbert_confidence` (mean maximum probability)
- **Models**:
  - `LR-F1_FinBERT`: L2 penalized logistic regression ($D = 32$).
  - `XGB-F1_FinBERT`: Gradient boosted trees ($D = 32$).
  - `LSTM-F1_FinBERT`: 2-layer LSTM with concatenated technical and sentiment vectors ($D = 32$).

### Experiment 5: Temporal Cross-Attention Multimodal Net (`CrossAttentionNet`)
- **Architecture**:
  - **Temporal Price Encoder**: 2-layer BiLSTM ($D_{in} = 24$, $D_{hidden} = 64$, bidirectional $\implies 128$).
  - **Temporal News Feature Encoder**: Linear projection layer mapping 8 FinBERT/news features to $D_{model} = 128$.
  - **Cross-Attention Mechanism**:
    - Multi-Head Attention: 4 attention heads ($d_k = 32$).
    - Queries ($Q$): Derived from Price representations ($X_{price} W_Q$).
    - Keys ($K$) & Values ($V$): Derived from News representations ($X_{news} W_K, X_{news} W_V$).
    - Sentinel Masking: Softmax masked for missing news dates with additive large negative bias ($-10^9$).
  - **Fusion Layer**: Concatenation of Price representation, Attended News representation, and Gated Residual Connection ($128 + 128 = 256$).
  - **Dual Prediction Heads**:
    1. **Direction Classification Head**: MLP ($256 \to 64 \to \text{ReLU} \to \text{Dropout}(0.2) \to 1 \to \text{Sigmoid}$).
       - Loss: Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`).
    2. **Volatility Regression Head**: MLP ($256 \to 64 \to \text{ReLU} \to \text{Dropout}(0.2) \to 1$).
       - Target: Parkinson High-Low Realized Volatility $\sigma_{P, t+1}$.
       - Loss: Smooth L1 / Huber Loss (`SmoothL1Loss`, $\beta = 1.0$).
  - **Joint Multi-Task Objective**:
    $$\mathcal{L}_{total} = \mathcal{L}_{BCE}(\hat{y}_{dir}, y_{dir}) + \lambda \cdot \mathcal{L}_{SmoothL1}(\hat{\sigma}, \sigma_P), \quad \lambda = 1.0$$
  - **Training Setup**:
    - Optimizer: AdamW ($lr = 5 \times 10^{-4}$, weight decay $= 10^{-3}$).
    - Learning rate scheduler: Cosine annealing with 3-epoch warmup.
    - Batch Size: 64 sequences. Epochs: 25.
