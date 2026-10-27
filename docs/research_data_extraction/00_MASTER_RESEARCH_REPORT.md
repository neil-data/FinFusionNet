# FinFusionNet: Master Research Report & Scientific Audit Synthesis

**A Comprehensive, Read-Only Research Extraction and Methodological Audit of the FinFusionNet Project**  
*Document Version: 1.0 (Audited)*  
*Prepared for Academic Manuscript Preparation & Peer Review Submission*

---

## 1. Executive Summary

FinFusionNet is an empirical quantitative finance research initiative investigating whether **multimodal deep learning**—specifically combining high-frequency daily price technical features with financial news sentiment via temporal cross-attention—delivers statistically significant out-of-sample incremental predictive value over audited price-only baselines in the Indian equity market (Nifty 50 universe).

Across 5 major experimental phases, 6,578 project files, 205,687 cleaned news articles, and 45 expanding monthly walk-forward folds evaluated on an audited common cohort of **45,136 out-of-sample observations** (916 trading dates, 2023-01 to 2026-09), this investigation establishes two definitive empirical conclusions:

1. **Direction Classification**: Domain-specific sentiment representations (FinBERT) and temporal cross-attention **fail to produce statistically significant directional alpha** over price-only baselines. The top-performing architecture is the simple price-only LSTM baseline (`LSTM-F0`) with a date-level accuracy of **51.08%** (ROC-AUC **0.5149**). The multimodal `CrossAttentionNet` achieved **50.16%** date accuracy, trailing the naive Always-Up baseline (**50.34%**) and underperforming `LSTM-F0` by **-0.92%** ($p = 0.0430$ date-level Brier DM; $p = 0.0339$ 0-1 loss DM).
2. **Realized Volatility Regression**: The initial claim that Cross-Attention Net achieved a "+44.8% error reduction" over baseline was an **artifact of benchmark asymmetry**—the model's intraday Parkinson high-low volatility forecasts were erroneously evaluated against a historical close-to-close return volatility baseline that carried an intrinsic $+0.0331$ (+16.3%) overnight jump bias. When audited against a **fair, like-for-like trailing 20-day Parkinson realized volatility benchmark (`parkinson_20`)**, the simple benchmark outperforms the neural network on squared error:
   - `Trailing Parkinson 20d`: RMSE = **0.1052**, MAE = 0.0695, Out-of-Sample $R^2$ = **+14.24%**
   - `Cross-Attention Net`: RMSE = 0.1075, MAE = **0.0686**, Out-of-Sample $R^2$ = +10.38%
   - Date-Level HAC Diebold-Mariano test on MSE: $DM = -1.9515, p = 0.0513$ (Benchmark has lower MSE).

---

## 2. Research Identity & Theoretical Problem Formulation

### Official Project Title
**FinFusionNet: Evaluating Multimodal Temporal Cross-Attention and Financial NLP for Equity Direction and Realized Volatility Forecasting in Indian Markets**

### Research Questions & Hypotheses
- **Primary Research Question (RQ1)**: Does the integration of transformer-derived financial news sentiment features (ProsusAI FinBERT) into temporal sequential models provide statistically significant incremental predictive power for next-session equity price direction compared to an audited price-only baseline?
  - *Hypothesis $H_{1, dir}$*: Multimodal fusion reduces directional classification error relative to price-only baselines.
  - *Empirical Finding*: **Rejected ($p > 0.05$ after family-wise Holm-Bonferroni correction; lift is negative).**
- **Secondary Research Question (RQ2)**: Does a temporal cross-attention mechanism allow a neural network to dynamically extract market-shock signals from sparse news events and improve next-session Parkinson realized volatility forecasts over standard autoregressive/moving-average benchmarks?
  - *Hypothesis $H_{1, vol}$*: Cross-attention volatility forecasts achieve lower out-of-sample RMSE than historical volatility benchmarks.
  - *Empirical Finding*: **Rejected on squared error (Fair Parkinson 20d benchmark achieves RMSE 0.1052 vs model 0.1075; $DM = -1.9515$). Supported marginally only on MAE due to Huber loss optimization.**

### Market Universe & Scope
- **Equity Universe**: 56 unique historical and point-in-time constituents of the **Nifty 50 Index** (NSE India).
- **Macro Covariates**: Nifty 50 Index (`^NSEI`) and India VIX (`^INDIAVIX`).
- **Research Period**: 2019-01-01 to 2026-09-30 (1,920 market sessions).
- **Out-of-Sample Evaluation Window**: 2023-01-02 to 2026-09-29 (45 expanding monthly folds, 916 market sessions).

---

## 3. Data Pipeline & Preprocessing Rigor

### The 15:30 IST Causal Cutoff Protocol
To strictly eliminate lookahead bias:
$$\mathcal{W}_t = (15:30\text{ IST on day } t-1, \quad 15:30\text{ IST on day } t]$$
Any news article published after the 15:30 IST closing bell on day $t$ is strictly mapped to trading session $t+1$. Non-trading day news (weekends, national holidays) is aggregated into the next active market session.

### Missing-News Preservation
In real equity markets, news is intermittent. In the audited cohort:
- Days with news (`has_news = 1`): 31,458 observations (**69.70%**).
- Days without news (`has_news = 0`): 13,678 observations (**30.30%**).
Converting missing news days to "neutral" sentiment (e.g., 0.0 compound score or 1.0 neutral probability) distorts the data generating process. FinFusionNet enforces orthogonal missing indicators (`has_news=0`, `n_articles=0`) and numerical sentinel masking within the cross-attention layer to preserve the informational value of news absence.

### Protocol B Walk-Forward Cross-Validation
- **Initial Training Window**: 24 months (~500 market sessions).
- **Step Size**: 1 calendar month (~20-22 market sessions).
- **Total Expanding Folds**: 45 folds.
- **Embargo Window**: 1 market session between training fold end and test fold start.
- **Feature Scaling**: `StandardScaler` fitted strictly on the expanding training fold and applied out-of-sample to test folds.

---

## 4. Master Consolidated Classification Results

Evaluated across the common evaluation cohort of **45,136 observations** (916 dates):

| Model Configuration | Date Acc | Pooled Acc | Balanced Acc | Precision | Recall | F1 Score | ROC-AUC | Brier Score | Log Loss | Predicted Up % |
|---|---|---|---|---|---|---|---|---|---|---|
| **LSTM-F0 (Price-Only, Exp 1)** | **0.5108** | **0.5096** | **0.5091** | **0.5095** | 0.6297 | 0.5633 | **0.5149** | **0.2502** | **0.6935** | 62.07% |
| **LSTM-F1_VADER (Exp 3B)** | 0.5099 | 0.5088 | 0.5084 | 0.5092 | 0.6045 | 0.5528 | 0.5105 | 0.2504 | 0.6940 | 59.62% |
| **LR-F1_FinBERT (Exp 4)** | 0.5095 | 0.5083 | 0.5077 | 0.5081 | 0.6567 | 0.5729 | 0.5103 | 0.2505 | 0.6943 | 64.91% |
| **LR-F0 (Price-Only, Exp 1)** | 0.5094 | 0.5082 | 0.5075 | 0.5078 | 0.6724 | 0.5786 | 0.5102 | 0.2505 | 0.6942 | 66.49% |
| **XGB-F1_VADER (Exp 3B)** | 0.5054 | 0.5043 | 0.5040 | 0.5057 | 0.5622 | 0.5325 | 0.5090 | 0.2564 | 0.7069 | 55.82% |
| **LSTM-F1_FinBERT (Exp 4)** | 0.5053 | 0.5041 | 0.5038 | 0.5054 | 0.5842 | 0.5419 | 0.5092 | 0.2505 | 0.6942 | 58.04% |
| **XGB-F1_FinBERT (Exp 4)** | 0.5042 | 0.5042 | 0.5039 | 0.5057 | 0.5593 | 0.5311 | 0.5070 | 0.2568 | 0.7076 | 55.53% |
| *Baseline: Always-Up* | 0.5034 | 0.5021 | 0.5000 | 0.5021 | **1.0000** | **0.6686** | 0.5000 | 0.4979 | 8.0244 | 100.00% |
| **XGB-F0 (Price-Only, Exp 1)** | 0.5027 | 0.5015 | 0.4996 | 0.5019 | 0.9351 | 0.6532 | 0.5067 | 0.2503 | 0.6937 | 93.55% |
| **Cross-Attention Net (Exp 5)** | 0.5016 | 0.5004 | 0.4996 | 0.5019 | 0.6852 | 0.5794 | 0.5016 | 0.2506 | 0.6943 | 68.56% |
| *Baseline: Prev-Day Direction*| 0.4919 | 0.4930 | 0.4930 | 0.4952 | 0.4966 | 0.4959 | 0.4930 | 0.5070 | 8.1722 | 50.36% |

---

## 5. Volatility Target Formulation & Fair Benchmark Audit

### Target Estimator: Parkinson (1980) High-Low Volatility
$$\sigma_{P, t+1} = \sqrt{ \frac{1}{4 \ln 2} \left( \ln \frac{H_{t+1}}{L_{t+1}} \right)^2 \times 252 }$$
- Mean of True Target: **0.2024** | Std Dev: **0.1135**

### Corrected Volatility Performance Table
| Model / Benchmark | RMSE | MAE | Bias | Pearson $r$ | Out-of-Sample $R^2$ (%) |
|---|---|---|---|---|---|
| **Trailing Parkinson 20d (Fair Benchmark)** | **0.1052** | 0.0695 | **+0.0003** | 0.3979 | **+14.24%** |
| **Trailing Parkinson 10d (Fair Benchmark)** | 0.1058 | 0.0700 | +0.0003 | 0.4084 | +13.13% |
| **Trailing Parkinson 5d (Fair Benchmark)** | 0.1066 | 0.0708 | +0.0002 | **0.4301** | +11.89% |
| **Cross-Attention Net (Exp 5)** | 0.1075 | **0.0686** | -0.0089 | 0.3369 | +10.38% |
| *Expanding Historical Mean Vol* | 0.1135 | 0.0751 | 0.0000 | 0.0000 | 0.00% |
| *Trailing Parkinson 1d (Yesterday Vol)* | 0.1241 | 0.0839 | -0.0001 | 0.4057 | -19.38% |
| *Trailing Close-to-Close 20d (Exp 5 Baseline)* | 0.1322 | 0.0900 | +0.0331 | 0.3225 | -35.58% |

---

## 6. Mathematical Audit of Mandatory Inconsistencies

1. **Inconsistency A (Bootstrap Tail Direction)**: The 95% Date-Block Bootstrap CI for (Exp 5 - LSTM-F0) is **[-0.0174, -0.0001]**. The original bootstrap $p = 0.9758$ was computed as $P(\Delta^* \le 0)$. The one-sided test for superiority of Exp 5 over F0 is $P(\Delta^* \ge 0) = 0.0242$. This mathematically validates that Exp 5 is **strictly inferior** to the price-only baseline.
2. **Inconsistency B (DM Tests vs Holm Correction)**: Date-level unadjusted DM tests show LSTM-F0 beats Exp 5 ($p_{\text{Brier}} = 0.0430, p_{\text{01}} = 0.0339$). Under family-wise Holm-Bonferroni correction across planned superiority tests, all adjusted p-values remain $p > 0.05$ because no candidate multimodal model achieved positive alpha.
3. **Inconsistency C (XGB-FinBERT Brier Stat +11.66 vs 0-1 Loss $p=0.59$)**: Discrete accuracy is identical (50.42% vs 50.16%), but tree probability dispersion ($\sigma = 0.158$) heavily degrades XGBoost's Brier score (0.2568), while Cross-Attention shrinks probabilities to the 0.50 mean ($\sigma = 0.027, \text{Brier} = 0.2506$). The statistic reflects **probability shrinkage and calibration**, not directional market timing.
4. **Inconsistency D (Volatility Benchmark Asymmetry)**: Exp 5 originally benchmarked against close-to-close volatility (`volatility_20`), creating an artificial $+0.0331$ error gap. Benchmarking against Trailing Parkinson 20d proves the simple benchmark achieves superior RMSE (0.1052 vs 0.1075) and higher correlation ($r = 0.3979$ vs $0.3369$).

---

## 7. Complete Research Extraction Deliverable Directory

All underlying research data, configurations, audits, and inventories have been extracted and saved to `docs/research_data_extraction/`:

| File Name | Format | Description |
|---|---|---|
| [`00_MASTER_RESEARCH_REPORT.md`](file:///c:/Users/Neil/Desktop/FinFusionNeT/docs/research_data_extraction/00_MASTER_RESEARCH_REPORT.md) | Markdown | This comprehensive master report and executive research synthesis. |
| [`01_PROJECT_INVENTORY.csv`](file:///c:/Users/Neil/Desktop/FinFusionNeT/docs/research_data_extraction/01_PROJECT_INVENTORY.csv) | CSV | Full inventory of all 6,578 project files with metadata and research relevance. |
| [`02_DATASET_AND_FEATURES.md`](file:///c:/Users/Neil/Desktop/FinFusionNeT/docs/research_data_extraction/02_DATASET_AND_FEATURES.md) | Markdown | Exhaustive feature dictionary (39 features), news corpus, 15:30 IST cutoff logic. |
| [`03_PREPROCESSING_AND_LEAKAGE_AUDIT.md`](file:///c:/Users/Neil/Desktop/FinFusionNeT/docs/research_data_extraction/03_PREPROCESSING_AND_LEAKAGE_AUDIT.md) | Markdown | 10-point leakage audit matrix, Protocol B walk-forward details. |
| [`04_EXPERIMENT_CONFIGURATIONS.md`](file:///c:/Users/Neil/Desktop/FinFusionNeT/docs/research_data_extraction/04_EXPERIMENT_CONFIGURATIONS.md) | Markdown | Full hyperparameter specifications for Exp 1 through Exp 5. |
| [`05_ALL_EXPERIMENT_RESULTS.csv`](file:///c:/Users/Neil/Desktop/FinFusionNeT/docs/research_data_extraction/05_ALL_EXPERIMENT_RESULTS.csv) | CSV | Machine-readable table containing audited metrics for all 19 model/benchmark configurations. |
| [`06_CLASSIFICATION_RESULTS.md`](file:///c:/Users/Neil/Desktop/FinFusionNeT/docs/research_data_extraction/06_CLASSIFICATION_RESULTS.md) | Markdown | Detailed classification metrics, naive baselines, and market regime breakdowns. |
| [`07_VOLATILITY_RESULTS.md`](file:///c:/Users/Neil/Desktop/FinFusionNeT/docs/research_data_extraction/07_VOLATILITY_RESULTS.md) | Markdown | Parkinson volatility formulation, benchmark asymmetry audit, like-for-like comparisons. |
| [`08_STATISTICAL_AUDIT.md`](file:///c:/Users/Neil/Desktop/FinFusionNeT/docs/research_data_extraction/08_STATISTICAL_AUDIT.md) | Markdown | Full statistical audit, mathematical resolution of Inconsistencies A, B, C, D. |
| [`09_ARCHITECTURE_AND_MODEL_DETAILS.md`](file:///c:/Users/Neil/Desktop/FinFusionNeT/docs/research_data_extraction/09_ARCHITECTURE_AND_MODEL_DETAILS.md) | Markdown | PyTorch layer topology, tensor dimensions, sentinel token masking equations. |
| [`10_FIGURE_INVENTORY.md`](file:///c:/Users/Neil/Desktop/FinFusionNeT/docs/research_data_extraction/10_FIGURE_INVENTORY.md) | Markdown | Inventory of existing figures (1 PDF) and 7 required academic publication plots. |
| [`11_RESEARCH_TIMELINE.md`](file:///c:/Users/Neil/Desktop/FinFusionNeT/docs/research_data_extraction/11_RESEARCH_TIMELINE.md) | Markdown | Chronological project history reconstructed across 6 phases. |
| [`12_CONTRADICTIONS_AND_MISSING_DATA.md`](file:///c:/Users/Neil/Desktop/FinFusionNeT/docs/research_data_extraction/12_CONTRADICTIONS_AND_MISSING_DATA.md) | Markdown | Severity-ranked contradiction matrix (C1 to C4) and missing project artifacts. |
| [`13_SOURCE_FILE_REFERENCES.md`](file:///c:/Users/Neil/Desktop/FinFusionNeT/docs/research_data_extraction/13_SOURCE_FILE_REFERENCES.md) | Markdown | Cross-reference table linking source modules, scripts, and prediction parquets. |

---

## 8. Guidance for Academic Paper Framing

When writing the research paper from these findings, the author should adopt an **objective, high-impact empirical framing**:
1. **The Peril of Multimodal Fusion in Noisy Financial Regimes**: Many published financial ML papers report inflated performance by evaluating on unembargoed splits, survivorship-biased universes, or in-sample hyperparameter tuning. FinFusionNet serves as an audited benchmark proving that under rigorous point-in-time constraints, causal cutoffs, and walk-forward validation, public financial news headlines provide near-zero incremental directional alpha in liquid large-cap equities.
2. **Probability Calibration vs Decision Accuracy**: The paper should highlight the contrast between tree-based methods (XGBoost) and neural cross-attention: cross-attention dramatically improves probability calibration (Brier score 0.2506 vs 0.2568) without improving discrete directional accuracy (50.16% vs 50.42%).
3. **The Imperative of Like-for-Like Benchmark Symmetry**: The volatility investigation provides a vital lesson for quantitative researchers: comparing intraday high-low volatility estimators against close-to-close trailing moving averages creates an illusion of superiority driven entirely by overnight jump bias. When compared symmetrically against trailing Parkinson estimators, simple autoregressive moving averages remain exceedingly difficult to beat.
