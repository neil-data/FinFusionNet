# FinFusionNet — LSTM Experimental Design

## 1. Controlled Experimental Configurations (L1–L6)

To test the incremental predictive contribution of each information layer within a sequential deep learning framework, six controlled configurations are evaluated:

| ID | Feature Groups Included | Research Question / Isolated Factor |
|:---|:---|:---|
| **L1** | Group A (Technical Indicators) | Sequential technical-only baseline |
| **L2** | Group A + Group B (Market Context) | Marginal predictive power of macro market regime & NIFTY beta |
| **L3** | Group A + Group B + Group C (News Availability Controls) | Control for media attention volume ($n\_articles$, $has\_news$) |
| **L4** | Group A + Group B + Group C + Group D (VADER Sentiment) | Incremental predictive value of lexicon sentiment |
| **L5** | Group A + Group B + Group C + Group E (FinBERT Sentiment) | Incremental predictive value of domain-adapted transformer sentiment |
| **L6** | Group A + Group B + Group C + Group D + Group E (Combined Sentiment) | Combined multi-scale sentiment representation |

---

## 2. Research Hypotheses & Statistical Testing Framework

The track investigates six formal empirical hypotheses:

- **H1 (Market Context)**: $H_1: Acc_{date}(L2) > Acc_{date}(L1)$ — Does sequential market context improve direction prediction over technicals alone?
- **H2 (News Attention)**: $H_1: Acc_{date}(L3) > Acc_{date}(L2)$ — Does knowing news occurred provide predictive signal?
- **H3 (VADER Sentiment)**: $H_1: Acc_{date}(L4) > Acc_{date}(L3)$ — Does lexicon sentiment add incremental value beyond news availability?
- **H4 (FinBERT Sentiment)**: $H_1: Acc_{date}(L5) > Acc_{date}(L3)$ — Does transformer sentiment add incremental value beyond news availability?
- **H5 (Combined Sentiment)**: $H_1: Acc_{date}(L6) > Acc_{date}(L3)$ — Does combining lexicon and transformer sentiment provide aggregate lift?
- **H6 (Temporal Relationships)**: Does LSTM sequential modeling outperform tabular XGBoost on the identical evaluation cohort?

### Statistical Inference Protocol
- **Primary Metric**: Equal-weighted date-level accuracy ($Acc_d$).
- **Bootstrap Engine**: Paired Date-Block Bootstrap (10,000 repetitions, block length $L=10$ trading days).
- **Multi-Comparison Adjustment**: Holm-Bonferroni step-down correction for secondary comparisons.

---

## 3. Naive Baselines & Benchmark Comparison
Models are compared against:
1. **Always-Up**: Predicts $\hat{Y} = 1$ for all observations.
2. **Previous-Direction (Momentum)**: Predicts $\hat{Y} = 1$ if $Return_{1d} > 0$ else $0$.
3. **XGBoost Counterparts (X1–X6)**: Direct head-to-head performance across the identical evaluation dates.
