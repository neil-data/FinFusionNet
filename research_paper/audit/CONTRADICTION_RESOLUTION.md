# FinFusionNet Paper Audit: Resolution of Inconsistencies and Contradictions

This document establishes the resolution of the four mandatory inconsistencies investigated during the research audit:

## 1. Contradiction C1: Volatility Benchmark Asymmetry
- **Historical Claim**: Experiment 5 initially reported a dramatic **+44.8% error reduction** over baseline volatility.
- **Audited Reality**: The baseline was calculated using trailing close-to-close returns (`volatility_20`), which has an average annualized value of **0.2355**, compared to the true Parkinson High-Low target average of **0.2024**. This produced a systematic $+0.0331$ (+16.3%) overnight gap bias, penalizing the baseline $R^2$ to $-35.58\%$.
- **Resolution**: Recomputed against a fair like-for-like trailing 20-day Parkinson benchmark (`parkinson_20`). The fair benchmark outperforms Cross-Attention Net on squared error:
  $$\text{RMSE}(\text{parkinson\_20}) = 0.1052 < \text{RMSE}(\text{CrossAttention}) = 0.1075 \quad (DM = -1.9515, p = 0.0513)$$
  The paper reports this finding transparently as a quantitative cautionary lesson on benchmark asymmetry.

## 2. Contradiction C2: Direction Accuracy Bootstrap Tails
- **Historical Discrepancy**: Difference in accuracy (Exp 5 - LSTM-F0) has a 95% Date-Block Bootstrap CI of **[-0.0174, -0.0001]**, yet bootstrap $p$-value was logged as **0.9758**.
- **Mathematical Cause**: The bootstrap empirical distribution had $97.58\%$ of draws $\le 0$. The original script tested one-sided superiority $H_1: \text{Exp 5} > \text{LSTM-F0}$, for which $p_{\text{superiority}} = P(\Delta^* \ge 0) = 1.0 - 0.9758 = 0.0242$.
- **Resolution**: The value $0.9758$ represents the probability that Exp 5 is worse than F0. The one-sided test for superiority of LSTM-F0 over Exp 5 is $p = 0.0242$, exactly matching the strictly negative confidence interval $[-1.74\%, -0.01\%]$.

## 3. Contradiction C3: XGB-FinBERT Brier DM Stat (+11.66) vs 0-1 Loss DM ($p=0.59$)
- **Historical Discrepancy**: Cross-Attention vs XGB-F1_FinBERT has a Brier DM statistic of $+11.6587$ ($p < 10^{-15}$ in favor of Cross-Attention), but a 0-1 loss DM of $-0.5382$ ($p = 0.5906$).
- **Mathematical Cause**: 0-1 loss evaluates discrete thresholded classification $\mathbb{I}(\hat{p} > 0.5)$, where both models hover near 50.4% and 50.1% ($p = 0.5906$). Brier score measures mean squared error of probability forecasts: $(\hat{p} - y)^2$. XGBoost outputs extreme, uncalibrated tree probabilities (std = 0.158), yielding high Brier penalty ($0.2568$). Cross-Attention outputs shrink to the empirical mean ($0.50$, std = 0.027), yielding near-optimal Brier score ($0.2506$).
- **Resolution**: The paper explains that this divergence is an artifact of **probability calibration and variance shrinkage**, not directional market timing.

## 4. Contradiction C4: Missing Rendered Figures
- **Historical Discrepancy**: The original repository contained only 1 PDF plan and zero image files.
- **Resolution**: All 9 essential publication figures have been rendered into `research_paper/figures/` directly from audited parquet and CSV data.
