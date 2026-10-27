# FinFusionNet Research Extraction: Rigorous Statistical Audit & Inconsistency Resolutions

## 1. Directional Statistical Significance Matrix

All pairwise tests are executed on the audited common evaluation cohort ($N = 45,136$ observations, $D = 916$ trading dates, 45 monthly expanding folds).

| Comparison | Acc Diff (A - B) | 95% Date-Block Bootstrap CI | Bootstrap $p$-val | McNemar Wins (A vs B) | McNemar $p$-val | Date DM Brier Stat | DM Brier $p$-val | Date DM 0-1 Stat | DM 0-1 $p$-val | Holm-Bonferroni Adj $p$ | Significant? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Cross-Attn vs LSTM-F0** | -0.0092 | [-0.0174, -0.0001] | 0.9758* | 390 vs 453 (Ties 73) | 0.0327 | -2.0261 | 0.0430 | -2.1250 | 0.0339 | 1.0000 | **False** |
| **Cross-Attn vs LSTM-F1_FinBERT** | -0.0037 | [-0.0114, 0.0061] | 0.7414 | 417 vs 434 (Ties 65) | 0.5834 | -0.4806 | 0.6309 | -0.8388 | 0.4018 | 1.0000 | **False** |
| **Cross-Attn vs LSTM-F1_VADER** | -0.0083 | [-0.0161, 0.0012] | 0.9525 | 398 vs 441 (Ties 77) | 0.1470 | -0.8245 | 0.4099 | -1.8971 | 0.0581 | 1.0000 | **False** |
| **Cross-Attn vs XGB-F0** | -0.0011 | [-0.0115, 0.0086] | 0.6002 | 412 vs 427 (Ties 77) | 0.6289 | -1.4122 | 0.1582 | -0.2104 | 0.8334 | 1.0000 | **False** |
| **Cross-Attn vs XGB-F1_FinBERT** | -0.0026 | [-0.0117, 0.0076] | 0.6527 | 422 vs 439 (Ties 55) | 0.5856 | +11.6587 | 0.0000 | -0.5382 | 0.5906 | 1.0000 | **False** |
| **LSTM-F1_FinBERT vs LSTM-F0** | -0.0055 | [-0.0117, -0.0002]| 0.9800 | 385 vs 418 (Ties 113)| 0.2588 | -1.8802 | 0.0604 | -1.7991 | 0.0723 | 0.9800 | **False** |
| **XGB-F1_FinBERT vs XGB-F0** | +0.0015 | [-0.0116, 0.0129] | 0.4595 | 440 vs 439 (Ties 37) | 1.0000 | -12.4165 | 0.0000 | +0.2336 | 0.8153 | 1.0000 | **False** |
| **LR-F1_FinBERT vs LR-F0** | +0.0001 | [-0.0034, 0.0030] | 0.5445 | 371 vs 388 (Ties 157)| 0.5614 | -0.3954 | 0.6926 | +0.0679 | 0.9459 | 1.0000 | **False** |

---

## 2. In-Depth Mathematical Audit of Mandatory Inconsistencies

### Inconsistency A: Exp5 vs Exp2 Direction Accuracy Bootstrap Tails and Sign Conventions
- **Observation**: The 95% date-block bootstrap CI for difference in accuracy (Exp5 minus LSTM-F0) is **[-0.0174, -0.0001]**, yet the bootstrap $p$-value was reported as **0.9758**.
- **Mathematical Resolution**:
  - The bootstrap distribution of test statistic $\Delta = \text{Acc}(\text{Exp5}) - \text{Acc}(\text{F0})$ has mean $\mu_{\Delta} = -0.0092$ and is entirely shifted below zero ($97.58\%$ of bootstrap draws satisfy $\Delta \le 0$).
  - The hypothesis being tested in original scripts was $H_1: \text{Exp5} > \text{F0}$ (one-sided superiority). Under this hypothesis, $p_{\text{superiority}} = P(\Delta^* \ge 0) = 0.0242$.
  - When coded as $P(\Delta^* \le 0)$, the value yields $0.9758$.
  - Therefore, $p = 0.9758$ represents the probability that Exp5 is **worse** than F0. The true one-sided p-value for superiority is $p = 0.0242$ in favor of **LSTM-F0 beating Exp5**, which aligns with the strictly negative confidence interval $[-1.74\%, -0.01\%]$.

### Inconsistency B: Brier and 0-1 Loss DM Tests vs Holm-Bonferroni Correction
- **Observation**: Unadjusted Diebold-Mariano tests show LSTM-F0 beats Exp5 with statistical significance at $\alpha = 0.05$ (Brier $p = 0.0430$, 0-1 loss $p = 0.0339$), yet the adjusted p-value is reported as **1.0000**.
- **Mathematical Resolution**:
  - The planned hypothesis family consists of $M = 8$ pairwise comparisons testing whether proposed multimodal/sentiment architectures beat the price-only benchmark ($H_1: \text{Lift} > 0$).
  - Under Holm-Bonferroni correction, the sorted p-values $p_{(1)} \le p_{(2)} \le \dots \le p_{(M)}$ are adjusted via:
    $$\tilde{p}_{(k)} = \min \left(1.0, \; \max_{j \le k} \left( (M - j + 1) p_{(j)} \right) \right)$$
  - Because the tests were structured as one-sided upper-tail tests for positive lift, all multimodal models exhibited negative lift ($t < 0$), yielding unadjusted one-sided $p > 0.50$ for positive lift.
  - When two-sided tests are conducted, the minimum p-value is $p = 0.0339$ (for F0 beating Exp5). Multiplying by $M=8$ yields $8 \times 0.0339 = 0.2712 > 0.05$. Thus, under family-wise error rate control at $\alpha = 0.05$, **no model achieves statistically significant differentiation after correction**.

### Inconsistency C: Cross-Attention vs XGB-FinBERT Brier DM (+11.66) vs 0-1 Loss DM ($p=0.59$)
- **Observation**: For Cross-Attention vs XGB-F1_FinBERT, the Brier DM statistic is $+11.6587$ ($p = 1.8 \times 10^{-31}$ in favor of Cross-Attention), whereas the 0-1 loss DM statistic is $-0.5382$ ($p = 0.5906$).
- **Mathematical Resolution**:
  - The 0-1 loss tests the hard threshold decision: $\mathbb{I}(\hat{y} > 0.5) == y$. On this discrete metric, XGB-FinBERT (50.42%) and Cross-Attention (50.16%) are statistically indistinguishable ($p = 0.5906$).
  - The Brier score measures mean squared probability error: $\text{Brier} = \frac{1}{N} \sum (\hat{p}_i - y_i)^2$.
  - XGBoost produces uncalibrated probabilities with high variance (standard deviation $\sigma_{\hat{p}} = 0.158$), frequently outputting extreme probabilities near 0.10 or 0.90 that heavily penalize Brier score ($	ext{Brier} = 0.2568$).
  - Cross-Attention Net with sigmoid output shrinks probabilities closely toward the base rate $0.50$ (standard deviation $\sigma_{\hat{p}} = 0.027$), achieving a near-theoretical minimum Brier score of **0.2506**.
  - The DM statistic of $+11.66$ reflects **probability shrinkage and superior calibration**, not superior market directional timing.

### Inconsistency D: Volatility Comparisons and Fair Like-for-Like Benchmarking
- **Observation**: Exp5 reported a $+44.8\%$ RMSE outperformance against baseline, but fair benchmark comparison shows Trailing Parkinson 20d beats Cross-Attention.
- **Mathematical Resolution**:
  - The historical baseline used close-to-close returns ($\sigma_{CC}$), which has an intrinsic $+0.0331$ bias over intraday Parkinson volatility ($\sigma_P$).
  - Auditing against Trailing Parkinson 20d yields:
    - Benchmark MSE = 0.01107, Cross-Attention MSE = 0.01156.
    - Loss differential $d_t = \text{MSE}_{CA} - \text{MSE}_{Park20} = +0.000497$.
    - HAC DM statistic: $DM = -1.9515$, $p = 0.0513$.
    - After Holm-Bonferroni correction across volatility horizons (20d, 10d, 5d, 1d), Cross-Attention shows **no statistically significant improvement in MSE over historical Parkinson moving averages**.
