# FinFusionNet Research Extraction: Volatility Regression Results & Benchmark Audit

## 1. Volatility Target Formulation

In Experiment 5, FinFusionNet introduced a secondary regression head to predict next-day realized volatility simultaneously with direction classification.

The target is defined using the **Parkinson (1980) High-Low Volatility Estimator**:
$$
\sigma_{P, t+1} = \sqrt{ \frac{1}{4 \ln 2} \left( \ln \frac{H_{t+1}}{L_{t+1}} \right)^2 \times 252 }
$$
- **Empirical Target Distribution**:
  - Sample Mean: **0.2024** (20.24% annualized volatility)
  - Standard Deviation: **0.1135**
  - Median: **0.1741**
  - Interquartile Range: [0.1284, 0.2435]

---

## 2. The Benchmark Asymmetry Audit

### The Original Historical Claim
In the initial Experiment 5 report, Cross-Attention Net was reported as achieving a dramatic **+44.8% error reduction** over baseline volatility.

### Audit Investigation & Root Cause Discovery
Independent audit revealed that the original Exp 5 code evaluated the model against `volatility_20` (a 20-day rolling close-to-close return standard deviation: $\text{Std}(\Delta \ln C) \times \sqrt{252}$):
- `volatility_20` Mean: **0.2355**
- Parkinson Target Mean: **0.2024**
- **Systematic Bias**: $+0.0331$ (+16.3% relative over-estimation).

Because close-to-close volatility incorporates overnight jump variances and discrete closing jumps, it systematically trades at a premium to intraday Parkinson high-low volatility. When compared against `volatility_20`, the baseline suffered an artificially degraded $R^2$ of **-35.58%**.

---

## 3. Audited Volatility Performance Against Like-for-Like Benchmarks

When Cross-Attention Net is benchmarked against **fair, like-for-like historical trailing Parkinson volatility estimators** (`parkinson_20`, `parkinson_10`, `parkinson_5`), the reported outperformance vanishes on squared error:

| Model / Benchmark | RMSE | MAE | Bias | Pearson $r$ | Out-of-Sample $R^2$ (%) |
|---|---|---|---|---|---|
| **Trailing Parkinson 20d (Fair Benchmark)** | **0.1052** | 0.0695 | **+0.0003** | 0.3979 | **+14.24%** |
| **Trailing Parkinson 10d (Fair Benchmark)** | 0.1058 | 0.0700 | +0.0003 | 0.4084 | +13.13% |
| **Trailing Parkinson 5d (Fair Benchmark)** | 0.1066 | 0.0708 | +0.0002 | **0.4301** | +11.89% |
| **Cross-Attention Net (Exp 5)** | 0.1075 | **0.0686** | -0.0089 | 0.3369 | +10.38% |
| *Expanding Historical Mean Vol* | 0.1135 | 0.0751 | 0.0000 | 0.0000 | 0.00% |
| *Trailing Parkinson 1d (Yesterday Vol)* | 0.1241 | 0.0839 | -0.0001 | 0.4057 | -19.38% |
| *Trailing Close-to-Close 20d (Exp 5 Baseline)* | 0.1322 | 0.0900 | +0.0331 | 0.3225 | -35.58% |
| *Trailing Close-to-Close 10d (Exp 5 Baseline)* | 0.1397 | 0.0927 | +0.0259 | 0.3188 | -51.29% |

---

## 4. Diebold-Mariano Tests for Volatility Loss Differentials

Date-level Diebold-Mariano tests with Newey-West HAC lag adjustment ($h=1, \text{lag}=5$) confirm:

1. **Mean Squared Error (MSE)**:
   - Trailing Parkinson 20d beats Cross-Attention Net on Date-Level MSE: $DM = -1.9515$, $p = 0.0513$ (two-sided), $p = 0.0256$ (one-sided in favor of Benchmark).
   - Trailing Parkinson 20d beats Cross-Attention Net on Pooled MSE: $DM = -4.3941$, $p = 1.1 \times 10^{-5}$.
2. **Mean Absolute Error (MAE)**:
   - Cross-Attention Net slightly beats Trailing Parkinson 20d on Date-Level MAE ($\Delta = 0.0009$): $DM = +2.1118$, $p = 0.0350$.
   - **Reason**: Cross-Attention Net was optimized using **Smooth L1 (Huber) Loss**, which is linear in absolute deviations, naturally optimizing MAE at the expense of MSE.
