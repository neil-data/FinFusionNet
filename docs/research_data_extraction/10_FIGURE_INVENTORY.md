# FinFusionNet Research Extraction: Figure & Visualization Inventory

## 1. Current Repository Visualization Status

An exhaustive search across the entire repository for image and visualization assets (`.png`, `.jpg`, `.jpeg`, `.svg`, `.pdf`) revealed:

| File Path | File Type | Size | Description | Status in Repository |
|---|---|---|---|---|
| `docs/finfusionnet plan.pdf` | PDF Document | 42,757 bytes | Architectural overview and experiment staging plan | **PRESENT** |

**Zero image files (`.png`, `.jpg`, `.svg`) exist in the repository.** All historical visualizations were either logged to temporary display terminals or documented purely as tabular ASCII / Markdown data.

---

## 2. Required Figures for Academic Publication

To prepare the manuscript for academic peer review (e.g., IEEE Transactions on Neural Networks, Journal of Financial Data Science, or ACM ICAIF), the following 7 figures must be generated from the audited predictions parquets:

1. **Figure 1: Multimodal Cross-Attention Architecture Diagram**: High-resolution vector diagram illustrating BiLSTM price encoding, FinBERT news projection, sentinel masking, and dual prediction heads.
2. **Figure 2: Receiver Operating Characteristic (ROC) & Precision-Recall Curves**: Comparison across LSTM-F0, XGB-F0, LSTM-FinBERT, and Cross-Attention Net.
3. **Figure 3: Reliability & Calibration Curves (Brier Score Decomposition)**: Visual demonstration of why Cross-Attention Net achieves Brier $0.2506$ (probability shrinkage around 0.50) while XGBoost yields $0.2568$ (overconfident tree leaves).
4. **Figure 4: Cumulative Directional Walk-Forward Return (Equity Curves)**: Simulating a long-flat trading strategy based on model probabilities across the 45 monthly test folds (2023-2026).
5. **Figure 5: Volatility Scatter & Prediction Tracking**: Predicted Parkinson volatility vs Actual Parkinson volatility vs Trailing 20d Parkinson benchmark.
6. **Figure 6: Cross-Attention Weight Heatmap**: Attention weight matrix $\alpha_{i, j}$ over the 20-day lookback window during high-news vs zero-news market shocks (e.g., earnings releases, macro events).
7. **Figure 7: Regime Breakdown Bar Plot**: Direction accuracy degradation across High Volatility, Normal Volatility, Low Volatility, Bull, and Bear market regimes.
