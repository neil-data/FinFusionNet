# FinFusionNet Research Extraction: Preprocessing and Data Leakage Audit

## 1. Data Leakage Assessment Matrix

An exhaustive 10-point audit was conducted on the entire preprocessing and cross-validation pipeline.

| Item # | Audit Checkpoint | Verification Status | Implementation & Code Evidence | Verdict / Finding |
|---|---|---|---|---|
| 1 | **Temporal Walk-Forward Ordering** | **VERIFIED** | Protocol B: 45 expanding monthly folds (24-month initial window). Folds ordered strictly chronologically from 2023-01 to 2026-09. Test folds never precede training folds. | Clean. No temporal shuffling. |
| 2 | **Embargo Period** | **VERIFIED** | One-session embargo enforced between train end date $T_{train}$ and test start date $T_{test}$. Prevents overlapping 1-day return targets. | Clean. No boundary spillover. |
| 3 | **Feature Normalization Fitting** | **VERIFIED** | `StandardScaler` is instantiated and fitted strictly inside the fold loop: `scaler.fit(X_train)`, then `scaler.transform(X_test)`. Never fitted globally. | Clean. Zero out-of-fold distribution leakage. |
| 4 | **15:30 IST Causal Cutoff** | **VERIFIED** | `scripts/process_gnews_all50.py` and `scripts/generate_finbert_features.py` map article publish timestamps strictly using $(15:30_{t-1}, 15:30_t] \to t$. Post-15:30 news assigned to $t+1$. | Clean. Verified against live timestamps. |
| 5 | **Survivorship Bias Handling** | **VERIFIED** | Master universe includes all 56 historical Nifty 50 constituents across the period. Stocks deleted or added during the window are tracked accurately point-in-time. | Clean. Survivorship bias minimized. |
| 6 | **Missing-News Imputation** | **VERIFIED** | Missing news days are not imputed with neutral (0.0). `has_news=0` explicitly retained as an orthogonal control feature. Missing continuous features zeroed only after sentinel masking. | Clean. Preserves information value of silence. |
| 7 | **Dual-Target Formulations** | **VERIFIED** | Direction target: $\mathbb{I}(C_{t+1} > C_t)$. Parkinson volatility: $\sigma_{P, t+1} = \sqrt{\frac{(\ln(H_{t+1}/L_{t+1}))^2}{4\ln 2} \times 252}$. Targets use only $t+1$ data. | Clean. No lookahead in target creation. |
| 8 | **Attention Masking & Sentinels** | **VERIFIED** | In `CrossAttentionNet`, when a stock has no news on day $t$, attention weights are masked using a dedicated sentinel representation, avoiding division-by-zero or NaN gradient propagation. | Clean. Numerical stability verified. |
| 9 | **Common Evaluation Cohort** | **VERIFIED** | All models (Exp 1 through Exp 5) evaluated on identical 45,136 observations, 916 dates, and 56 tickers. Exact row-level index alignment verified via `scratch/inspect_experiments_data.py`. | Clean. 100% paired comparison parity. |
| 10 | **Volatility Benchmark Comparability** | **ISSUE IDENTIFIED & AUDITED** | Original Exp5 compared predicted Parkinson volatility against trailing close-to-close volatility (`volatility_20`), producing an artificial +44.8% error reduction due to overnight jump bias (+0.0331). Audited against fair like-for-like Parkinson benchmark (`parkinson_20`). | **Defect Corrected in Audit.** |

---

## 2. In-Depth Proof of 15:30 IST Timestamp Enforcement

Let $T_p$ be the published timestamp of news article $i$, recorded in Indian Standard Time (IST, UTC+5:30).
Let $D(T_p)$ be the calendar date of $T_p$, and $\tau(T_p)$ be the time of day.

The feature assignment function $f_{date}(T_p)$ is formally implemented as:
$$
f_{date}(T_p) = \begin{cases}
D(T_p) & \text{if } \tau(T_p) \le 15:30:00 \text{ and } D(T_p) \in \mathcal{T} \\
\text{NextTradingDay}(D(T_p)) & \text{if } \tau(T_p) > 15:30:00 \text{ or } D(T_p) \notin \mathcal{T}
\end{cases}
$$
where $\mathcal{T}$ represents the set of official NSE trading days.

All articles published during trading hours before 15:30 IST on day $t$ are packaged into the feature vector $X_t$, which is fed to the model at 15:30 IST on day $t$ to forecast the price change from $C_t$ to $C_{t+1}$. Articles published after 15:30 IST are impossible to execute before market close; thus, they are mapped to $X_{t+1}$.

---

## 3. Preprocessing Fit Protocol (Protocol B)

Walk-forward validation follows **Protocol B**:
- Expanding training window: Initial training period = 24 months (2021-01 to 2022-12, ~500 trading days).
- Fold step = 1 calendar month (~20-22 trading days).
- Total folds = 45 folds (January 2023 through September 2026).
- Test set size per fold = 1 month of unseen future data.
- Feature Scaling:
  $$z = \frac{x - \hat{\mu}_{\text{train}}}{\hat{\sigma}_{\text{train}}}$$
  The parameters $\hat{\mu}_{\text{train}}$ and $\hat{\sigma}_{\text{train}}$ are re-estimated at the beginning of each monthly fold using exclusively the expanding training window. Zero future information is utilized in test transformations.
