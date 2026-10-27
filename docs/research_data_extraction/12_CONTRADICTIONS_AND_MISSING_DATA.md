# FinFusionNet Research Extraction: Contradictions, Discrepancies, and Missing Artifacts

## 1. Severity-Ranked Contradiction Matrix

| ID | Issue Description | Severity | Historical Claim | Audited Reality | Root Cause & Resolution |
|---|---|---|---|---|---|
| **C1** | **Volatility Benchmark Asymmetry** | **HIGH** | Exp 5 claimed a **+44.8% RMSE error reduction** over baseline volatility. | When benchmarked against Trailing Parkinson 20d, the model **underperforms** ($R^2 = +10.38\%$ vs $+14.24\%$, $DM = -1.9515, p = 0.0513$). | Baseline was incorrectly computed using close-to-close returns, introducing $+0.0331$ overnight gap bias. Corrected to fair like-for-like Parkinson benchmark. |
| **C2** | **Direction Accuracy Bootstrap $p$-value Tail Inversion** | **MEDIUM** | Reported bootstrap $p = 0.9758$ for difference (Exp 5 - LSTM-F0). | Exp 5 is strictly inferior to LSTM-F0 ($\Delta = -0.92\%$, 95% CI: [-1.74%, -0.01%]). Superiority $p = 0.0242$ for F0 beating Exp 5. | Sign convention in one-sided bootstrap test was coded as $P(\Delta^* \le 0)$ rather than $P(\Delta^* \ge 0)$. Negative CI correctly indicates inferiority. |
| **C3** | **XGB-FinBERT Brier Score Discrepancy** | **MEDIUM** | Brier DM stat showed overwhelming significance ($+11.66, p < 10^{-15}$), but 0-1 loss DM showed no significance ($p = 0.5906$). | Discrete directional accuracy is identical (50.42% vs 50.16%), but tree probability dispersion penalizes Brier score heavily. | XGBoost outputs uncalibrated probabilities with high variance ($\sigma = 0.158$), whereas Cross-Attention shrinks probabilities toward 0.50 ($\sigma = 0.027$). |
| **C4** | **Absence of Rendered Publication Visualizations** | **LOW** | Research paper figures ready for submission. | Zero `.png`, `.jpg`, or `.svg` files exist in the repository; only 1 PDF plan document. | Visualizations must be programmatically generated from the audited prediction parquets. |

---

## 2. Missing Expected Project Artifacts

1. **Publication Figure Scripts**: Automated plotting script to render vector figures directly into `docs/figures/`.
2. **Tick-Level Order Book / High-Frequency Data**: The repository operates exclusively on daily OHLCV and daily news aggregations. Intraday price reaction dynamics are not captured.
3. **Alternative Large Language Models (LLMs)**: Experiments utilized FinBERT and VADER; modern decoder LLMs (e.g., Llama-3-Fin, Mistral-7B, GPT-4 embeddings) were not evaluated.
