# FinFusionNet Research Extraction: Chronological Research Timeline

## 1. Project Evolution & Milestone History

The FinFusionNet research initiative proceeded through six distinct chronological phases, reconstructed from Git commit logs, experiment directories, and prompt records:

| Phase / Milestone | Estimated Date | Core Objective | Key Deliverables & Code Artifacts | Primary Finding |
|---|---|---|---|---|
| **Phase 1: Project Conception & Price Baselines (Exp 1)** | Q1 2024 / Q2 2024 | Establish price-only directional baselines on Nifty 50 constituents. | `data/raw/prices/`, `src/features/`, `experiments/audit/`, `scripts/run_audit_protocol_b.py`. | LSTM-F0 achieved 51.08% date accuracy, establishing the audited benchmark. |
| **Phase 2: Macro Context Augmentation (Exp 2)** | Q3 2024 | Test whether adding index-level returns and India VIX improves stock direction prediction. | `src/features/market_context.py`, `configs/market_context.yaml`, `experiments/audit/predictions/market_context_folds/`. | Marginal accuracy shift (+0.05%), indicating firm-specific alpha dominates broad macro beta. |
| **Phase 3: Lexicon Sentiment Fusion (Exp 3 & 3B)** | Q4 2024 / Q1 2025 | Integrate news headlines via VADER lexicon sentiment. First 10-stock pilot (Exp 3), then all-50 expansion (Exp 3B). | `data/raw/gnews/`, `data/raw/gnews_all50/`, `experiments/experiment_3/`, `experiments/experiment_3b/`. | VADER features yielded 50.99% accuracy on LSTM; failed to beat price-only baseline. |
| **Phase 4: Domain Transformer Concatenation (Exp 4)** | Q2 2025 / Q3 2025 | Replace VADER with ProsusAI FinBERT. Early concatenation of sentiment probabilities with price features. | `data/processed/gnews_all50/daily_gnews_all50_finbert_features.parquet`, `experiments/experiment_4/`. | LSTM-F1_FinBERT reached 50.53%; XGB-F1_FinBERT reached 50.42%. Both trailed price-only LSTM. |
| **Phase 5: Temporal Cross-Attention Fusion (Exp 5)** | Q4 2025 / Q1 2026 | Build end-to-end multimodal network with cross-attention and dual prediction heads (Direction + Volatility). | `src/models/cross_attention_net.py`, `experiments/experiment_5/`. | Cross-Attention achieved 50.16% direction accuracy (loss of alpha). Initial report claimed +44.8% volatility gain. |
| **Phase 6: Independent Statistical Audit & Verification** | Oct 2026 | Perform full statistical audit, cohort parity check, HAC DM tests, and benchmark verification. | `docs/audit/exp1_exp5_statistical_audit/`, `scratch/investigate_inconsistencies.py`. | Uncovered benchmark asymmetry in Exp 5 volatility; confirmed fair Parkinson 20d beats model. Verified parity on 45,136 observations. |
