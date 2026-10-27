# FinFusionNet: Experiment 4 — FinBERT Feature Extraction & Pipeline Audit Report

**Date**: 26 October 2026  
**Auditor**: Lead Quantitative Systems Auditor  
**Scope**: Ingestion, Tokenization, Inference Pipeline, Information Cutoff Compliance, Feature Extraction, and Reproducibility for Experiment 4 (FinBERT + Price Fusion).

---

## 1. Executive Audit Summary

Experiment 4 evaluates whether domain-specific transformer sentiment features extracted via FinBERT (`ProsusAI/finbert`) provide statistically significant incremental predictive value over the audited price-only baseline ($F_0$) and over general-lexicon VADER sentiment ($F_{1\text{-VADER}}$).

This document audits the integrity of the data pipeline, the local model provenance, inference parameters, trading-calendar alignment, missing data handling, and leakage controls.

### Summary of Audit Findings
1. **Model Source & Offline Operation**: FinBERT weights (`pytorch_model.bin`, 438 MB) and tokenizer configuration were loaded strictly from the pre-downloaded local Hugging Face cache snapshot (`4556d13015211d73dccd3fdd39d39232506f3e43`). The official `finBERT-master.zip` source archive was extracted and inspected under `experiments/experiment_4/finbert_source/`. Zero external network calls were executed.
2. **Causal Information Cutoff**: Cleaned articles reuse the audited 15:30 IST market close cutoff. Zero future publications or same-day post-cutoff publications are accessible to feature extraction.
3. **Missing Data Policy**: Strict non-imputation is enforced. Days without news receive $n\_articles = 0$, $has\_news = 0$, and null ($NaN$) for all five FinBERT sentiment features. Zero no-news days are imputed with neutral sentiment (0.0 or 0.33).
4. **Cohort Parity**: The evaluation cohort strictly mirrors Protocol B ($N = 45,136$ observations, 916 trading dates, 56 active in-window constituents across 45 monthly expanding folds).
5. **Unit & Integration Test Suite**: 100% of pipeline tests passed in `tests/test_experiment_4_finbert.py`.

---

## 2. Model Provenance, Architecture & Inference Settings

### 2.1 Model Specification
- **Base Architecture**: BERT-base uncased (12 transformer layers, 768 hidden dimension, 12 attention heads, 110M parameters).
- **Domain Fine-Tuning**: Pre-trained on financial communication corpora and fine-tuned on Financial PhraseBank for 3-class sentiment classification.
- **Local Checkpoint Path**: `C:\Users\Neil\.cache\huggingface\hub\models--ProsusAI--finbert\snapshots\4556d13015211d73dccd3fdd39d39232506f3e43`
- **Model Checkpoint SHA**: `4556d13015211d73dccd3fdd39d39232506f3e43`
- **Model Weight Size**: 437,992,753 bytes (`pytorch_model.bin`).
- **Label Mapping (`id2label`)**:
  - `0`: `positive`
  - `1`: `negative`
  - `2`: `neutral`

### 2.2 Source ZIP Archive
- **Uploaded Archive**: `media_1793012683887_ffb4daf8.zip` (`finBERT-master.zip`, 31,992 bytes).
- **Extracted Location**: `experiments/experiment_4/finbert_source/`
- **Key Modules**:
  - `finbert/finbert.py`: Prediction logic, dataset tokenization, classification heads.
  - `finbert/utils.py`: Text chunking, input example structures.
  - `scripts/predict.py`: CLI inference interface.

### 2.3 Inference Settings & Precision
- **Precision**: Unquantized IEEE 754 Float32 (`torch.float32`).
  - *Audit Note*: Dynamic INT8 quantization was evaluated during benchmarking; although yielding a speedup, it exhibited a maximum absolute probability error of $0.962$ and a score correlation of only $0.887$ relative to Float32. To maintain uncompromising scientific fidelity and avoid quantization distortion, full Float32 inference was mandated.
- **Execution Mode**: `torch.inference_mode()` with `torch.set_num_threads(8)`.
- **Batch Size**: 64 with dynamic batch padding (sequences grouped and sorted by character/token length to eliminate redundant padding computation).
- **Truncation**: Maximum sequence length bounded at 64 tokens (99th percentile of cleaned headlines is 36 tokens; 0 headlines exceeded 64 tokens).
- **Checkpointing Frequency**: Checkpointed every 5,000 unique headlines to `data/processed/gnews_all50/finbert_title_cache.parquet`.

---

## 3. Mathematical Definition of Extracted Features

For each headline $T_i$ on a given trading session for ticker $k$, FinBERT outputs class probabilities via softmax:
$$P_{\text{pos}}(T_i), \quad P_{\text{neg}}(T_i), \quad P_{\text{neu}}(T_i)$$
where $P_{\text{pos}} + P_{\text{neg}} + P_{\text{neu}} = 1.0$.

Individual article net sentiment is defined as:
$$S_i = P_{\text{pos}}(T_i) - P_{\text{neg}}(T_i) \in [-1.0, +1.0]$$

For a ticker-day observation $(k, d)$ with $N$ articles:
1. **Positive Probability (`finbert_positive_mean`)**:
   $$\bar{P}_{\text{pos}} = \frac{1}{N} \sum_{i=1}^N P_{\text{pos}}(T_i)$$
2. **Negative Probability (`finbert_negative_mean`)**:
   $$\bar{P}_{\text{neg}} = \frac{1}{N} \sum_{i=1}^N P_{\text{neg}}(T_i)$$
3. **Neutral Probability (`finbert_neutral_mean`)**:
   $$\bar{P}_{\text{neu}} = \frac{1}{N} \sum_{i=1}^N P_{\text{neu}}(T_i)$$
4. **Sentiment Score (`finbert_sentiment_score` / `finbert_mean`)**:
   $$\bar{S} = \frac{1}{N} \sum_{i=1}^N S_i = \bar{P}_{\text{pos}} - \bar{P}_{\text{neg}}$$
5. **Sentiment Dispersion (`finbert_dispersion`)**:
   $$\sigma_S = \begin{cases} \sqrt{\frac{1}{N-1} \sum_{i=1}^N (S_i - \bar{S})^2} & \text{if } N > 1 \\ 0.0 & \text{if } N = 1 \end{cases}$$
6. **Sentiment Confidence (`finbert_confidence`)**:
   $$\bar{C} = \frac{1}{N} \sum_{i=1}^N \max\left(P_{\text{pos}}(T_i), P_{\text{neg}}(T_i), P_{\text{neu}}(T_i)\right)$$

On days without news ($N=0$):
- $n\_articles = 0$
- $has\_news = 0$
- $\bar{P}_{\text{pos}} = \bar{P}_{\text{neg}} = \bar{P}_{\text{neu}} = \bar{S} = \sigma_S = \bar{C} = \text{NaN}$

---

## 4. Verification of Leakage and Timestamp Integrity

The article-level dataset (`gnews_all50_articles_clean.parquet`) was audited against the trading calendar with a strict 15:30 IST cutoff:
- **Rule 1 (Calendar Order)**: $\text{pub\_date\_ist} \le \text{feature\_date}$. Verified: 0 violations across all 205,687 records.
- **Rule 2 (Same-Day Cutoff)**: If $\text{pub\_date\_ist} == \text{feature\_date}$, then $\text{pub\_time\_ist} \le 15:30:00$. Verified: 0 violations.
- **Rule 3 (After-Hours Roll-Forward)**: All articles published on weekend/holidays or after 15:30 IST on trading days are assigned to the immediate subsequent trading session. Verified: 35,199 records correctly rolled forward.
- **Rule 4 (No Future Lookahead)**: Models evaluated at trading session $t$ utilize features calculated strictly from news published prior to 15:30 IST of session $t$ to predict direction over session $t \to t+1$.

---

## 5. Protocol B Walk-Forward Specification

1. **Expanding Window**: 45 monthly expanding folds starting in January 2023 and concluding in September 2026.
2. **Initial Training Window**: 24 calendar months (2021-01-01 to 2022-12-30).
3. **Embargo Period**: 1-session embargo between training and test sets to eliminate boundary label overlap.
4. **Preprocessing Policy**:
   - StandardScaler fits strictly on the training partition of each fold.
   - Test data is transformed using training statistics.
   - Missing news features are imputed with in-fold training means for LR and LSTM; XGBoost utilizes native missingness branch routing.
5. **Statistical Rigor**:
   - Paired date-block bootstrap (10,000 resamples, block length $L=10$ trading days).
   - Date-level McNemar test.
   - Holm-Bonferroni family-wise error rate control ($\alpha = 0.05$).
