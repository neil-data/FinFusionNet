# FinFusionNet — XGBoost Experimental Design

## 1. Experiment Matrix (X1–X6)
The experimental design decomposes the prediction system into six isolated, strictly controlled feature configurations:

| ID | Feature Groups Included | Isolated Variable / Research Purpose |
|:---|:---|:---|
| **X1** | Group A (Technical Indicators) | Technical-only baseline |
| **X2** | Group A + Group B (Market Context) | Marginal predictive power of macro market regime & NIFTY beta |
| **X3** | Group A + Group C (News Availability Controls) | Control for news attention/volume effect ($n\_articles$, $has\_news$) |
| **X4** | Group A + Group C + Group D (VADER Sentiment) | Incremental value of traditional lexicon sentiment |
| **X5** | Group A + Group C + Group E (FinBERT Sentiment) | Incremental value of financial domain-adapted transformer sentiment |
| **X6** | Group A + Group C + Group D + Group E (VADER + FinBERT) | Combined multi-scale sentiment representation |

---

## 2. Hypothesis Testing & Formal Comparisons

### Pre-Registered Primary Hypothesis
- **Hypothesis**: $H_0: Acc_{date}(X5) \le Acc_{date}(X3)$ vs $H_1: Acc_{date}(X5) > Acc_{date}(X3)$
- **Protocol**: Protocol B (Monthly Expanding Window)
- **Primary Metric**: Equal-weighted date-level accuracy
- **Test**: Paired Date-Block Bootstrap (10,000 repetitions, block length $L=10$ trading days, $\alpha=0.05$)
- **Significance Criteria**: 95% Bootstrap CI strictly excludes 0, with empirical $p < 0.05$.

### Secondary Model Comparisons (Holm-Bonferroni Adjusted)
1. **X2 vs X1**: Does macro market context improve pure single-stock technical indicators?
2. **X3 vs X1**: Does knowing that news occurred (media attention) improve over pure technicals?
3. **X4 vs X3**: Does lexicon sentiment (VADER) provide signal beyond news availability?
4. **X6 vs X3**: Does fusing both lexicon and transformer sentiment provide aggregate lift?
5. **X6 vs X4**: Does FinBERT provide incremental value over VADER?
6. **X6 vs X5**: Does VADER add anything once FinBERT is present?

All secondary comparisons form a multiple testing family adjusted via the step-down Holm method.

---

## 3. Evaluation Cohort Management

### Common Evaluation Cohort (Strict Parity)
To ensure absolute experimental fairness, all models X1–X6 are evaluated on **the exact same set of ticker-date rows**:
- Each row $(i, t)$ must meet all eligibility filters:
  - Active NIFTY 50 constituent at $t$.
  - Valid forward target at $t+1$.
  - Fully primed technical features.
  - Valid trading calendar and exchange status.
  - Successfully processed in the news alignment pipeline (days with no news have $has\_news=0$ and null sentiment; they remain valid in the cohort).
- **Prohibition**: An experiment with fewer features (e.g., X1) must never be evaluated on dates where sentiment data was missing due to pipeline crashes. A formal cohort manifest records total rows, eligible test rows, and exclusion reasons.

### Secondary Cohort: News-Days Only (`has_news == 1`)
A secondary evaluation evaluates X3, X4, X5, and X6 strictly on observations where $has\_news == 1$:
- Tests whether sentiment models achieve concentrated lift on days where market-moving news was actively published.
- Evaluated on an identical subset across all models.

---

## 4. Naive Baselines
Models are compared against two deterministic baselines:
1. **Always-Up**: Predicts $\hat{Y}_{i,t} = 1$ for all observations. Reflects long-term market upward drift.
2. **Previous-Direction (Momentum)**: Predicts $\hat{Y}_{i,t} = 1$ if $Return_{1d}(i,t) > 0$ else $0$.
*Lift over both naive baselines is calculated and tested for statistical significance.*

---

## 5. Artifact & Run Organization
Runs are saved under `experiments/runs/<experiment_id>/<timestamp>/` containing:
- `config.yaml`: Snapshot of exact active configuration.
- `metadata.json`: Git commit hash, random seed, hardware specs, date boundaries.
- `predictions.parquet`: Out-of-sample predictions ($P(Y=1)$, $\hat{Y}$, true $Y$, date, ticker).
- `fold_metrics.csv`: Performance metrics per fold.
- `date_metrics.csv`: Daily equal-weighted metrics.
- `stock_metrics.csv`: Per-ticker performance breakdown.
- `feature_importance.csv`: XGBoost gain, cover, and weight importance.
- `statistical_tests.json`: Bootstrap results, p-values, confidence intervals.
- `summary.json`: High-level executive performance summary.
