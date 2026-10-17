# FinFusionNet: Audit of X2 Market Context Features & Leakage Red Flag

## 1. Executive Summary
This audit addresses the discrepancy where XGBoost X2 originally reported **55.93%** date accuracy, while LSTM L2 reported **49.41%**. The LSTM's inability to extract this edge was identified as a critical red flag.

### Core Findings:
1. **Index Merge Integrity**:
   - The merge between NIFTY 50 benchmark prices (^NSEI) and individual stock records was verified.
   - Max absolute difference between merged `nifty_return_1d` and direct calculation on raw NIFTY prices: **0.000000**.
   - Timing alignment is strictly causal: NIFTY close at session $t$ is aligned with stock close at session $t$ to forecast stock direction at session $t+1$.
2. **Logistic Regression Benchmark**:
   - A walk-forward logistic regression model trained exclusively on market context features yields **50.12%** date-level accuracy and **0.5157** ROC-AUC.
   - Adding an extra 1-day lag (`nifty_return_1d.shift(1)`) reduces performance to **49.44%** date-level accuracy and **0.5047** ROC-AUC.
   - There is no linear predictive edge in market returns for single-stock direction.
3. **The Red Flag Explained (Index Slicing Bug)**:
   - The artificial 55.93% accuracy in XGBoost X2 was caused by an indexing mismatch in `pipeline.py`: integer fold indices generated from a date-sorted view were used with `.iloc` on an unsorted `[ticker, date]` dataframe.
   - When the indexing bug is resolved, XGBoost X2 date accuracy drops to **49.26%**, closely matching LSTM L2 (**49.41%**).
   - Neither model finds an edge; market context features do not provide directional predictability for next-session returns.
