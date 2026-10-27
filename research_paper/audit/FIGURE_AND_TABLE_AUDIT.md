# FinFusionNet: Figure, Table, and Visual Design Overhaul Audit

**Auditor:** Senior Scientific-Visualization Designer & Academic Research Specialist  
**Date:** October 2026  
**Scope:** Exhaustive audit of all figures, charts, diagrams, tables, and visual styling across the FinFusionNet research paper.

---

## 1. Executive Summary of Visual Audit

The initial visual presentation in the research paper suffered from several notable limitations common to early-stage experimental pipelines:
1. **Demo-Style Block Diagrams**: Figures 1, 2, and 4 utilized colored rounded rectangles resembling presentation slides rather than formal, publication-ready vector schematics with clear tensor dimensions and flow pathways.
2. **Synthetic Data in Visualizations**: Figure 5 (Cross-Attention Weight Heatmap) relied on synthetic, randomly generated values rather than authentic empirical attention weights or real data distributions.
3. **Axis Truncation & Scale Inconsistency**: Figure 6 and 7 displayed truncated vertical/horizontal axes (e.g., 49.5% to 51.5%), which artificially exaggerated negligible accuracy differences (0.92%) and failed to show full probability distributions.
4. **Incompatible Metrics in Shared Plots**: Figures attempted to plot incompatible dimensions (e.g., Brier score alongside percentage accuracy) on dual axes with crowded legends and oversized in-plot titles.
5. **Inconsistent Color Semantics**: Color palettes varied between figures, lacking a unified semantic mapping across model families (Price-only, News sentiment, Multimodal, and Baselines).

This overhaul establishes a **cohesive, journal-grade visual identity** adhering to the design conventions of IEEE Transactions and ACM SIGKDD conferences.

---

## 2. Comprehensive Figure Inventory & Redesign Matrix

| Fig # | Figure Title | Current Purpose | Source Data / Code | Critical Design Problem | Action | Proposed Journal-Quality Redesign |
|---|---|---|---|---|---|---|
| **Fig 1** | Research Methodology & Pipeline | Overview of the end-to-end framework | `scratch/generate_paper_figures.py` | Colorful presentation-style boxes; oversized in-plot title; lacks formal flowchart syntax. | **Redesign** | A minimal, vector-style 3-stage modular pipeline diagram: (1) Causal Ingestion (Price + News), (2) Protocol B Walk-Forward Engine, and (3) Dual-Task Modeling & Multi-Hypothesis Audited Evaluation. Styled in clean monochromatic lines and subtle slate tones. |
| **Fig 2** | News Ingestion & 15:30 IST Cutoff | Illustrating causal news assignment window $\mathcal{W}_t$ | `scripts/process_gnews_all50_finbert.py` | Crude timeline with floating text; fails to clearly show market session boundaries $[09:15, 15:30]$. | **Redesign** | A rigorous dual-timeline schematic comparing the Continuous IST News Stream against Active Market Sessions $t-1, t, t+1$, explicitly highlighting the $(15:30_{t-1}, 15:30_t]$ causal boundary and post-market spillover handling. |
| **Fig 3** | CrossAttentionNet Neural Architecture | Detailed network topology | `src/finfusionnet/models/cross_attention_model.py` | Simple rectangular boxes; missing explicit tensor dimensions, layer operations, and sentinel token routing. | **Redesign** | Publication-grade deep learning architecture diagram with formal tensor shapes ($\mathbb{R}^{B \times 20 \times 24}$, $\mathbb{R}^{B \times 20 \times 8}$, $\mathbb{R}^{B \times 256}$), BiLSTM unfolding, Query/Key/Value projections, Multi-Head Attention, Sentinel-Masked Softmax, and dual prediction heads (BCE + Huber loss). |
| **Fig 4** | Protocol B Walk-Forward Engine | Explaining expanding folds and embargo | `src/finfusionnet/evaluation/walk_forward.py` | Embedded only as ASCII text; no visual representation of the 45 monthly expanding folds. | **New / Redesign** | A high-clarity walk-forward ladder diagram illustrating the 24-month initial training window, monthly step expansion, 1-session embargo gap, and out-of-sample evaluation period (2023–2026). |
| **Fig 5** | Out-of-Sample Direction Classification | Benchmark comparison across 10 models | `audited_direction_performance.csv`, `audited_direction_statistical_significance.csv` | Truncated axes (49.5–51.5%); in-plot titles; lack of confidence intervals. | **Redesign** | Dual-panel publication figure: Panel A displays Date-Level Accuracy with 95% Date-Block Bootstrap Confidence Intervals (color-coded by model family). Panel B contrasts ROC-AUC against Brier Score, visually demonstrating probability calibration shrinkage. |
| **Fig 6** | Volatility Forecasting & Benchmark Asymmetry | Parkinson realized volatility benchmark comparison | `audited_volatility_performance.csv`, `audited_volatility_diebold_mariano.csv` | Two separate horizontal bars with unharmonized scales; fails to explain the $+0.0331$ close-to-close bias. | **Redesign** | Dual-panel diagnostic: Panel A illustrates the Target vs Benchmark distribution means, highlighting the overnight jump bias. Panel B compares RMSE, MAE, and $R^2$ across fair Parkinson benchmarks (`parkinson_20`, `parkinson_10`, `parkinson_5`) vs Cross-Attention Net. |
| **Fig 7** | Statistical Significance & Diebold-Mariano Tests | Forest plot of loss differentials | `audited_direction_statistical_significance.csv` | Missing visual plot (only in tabular text); difficult to understand Holm correction impact. | **New / Redesign** | Forest plot of Date-Level Diebold-Mariano test statistics (Brier loss and 0–1 discrete loss) with raw two-sided p-values and Holm-Bonferroni significance thresholds, proving absence of positive multimodal lift. |
| **Fig 8** | Market Regime & Subgroup Sensitivity | Evaluating fragility in market turmoil | `audited_market_regimes.csv` | Cluttered grouped bars with arbitrary colors and no reference line. | **Redesign** | Clean dot-and-whisker plot displaying Date Accuracy across High Volatility, Normal Volatility, Low Volatility, Bull, and Bear regimes with 50% breakeven baseline, highlighting severe degradation in high volatility. |
| **Fig 9** | Empirical News Arrival & Sentiment Distribution | Empirical news coverage & FinBERT scores | `daily_gnews_all50_finbert_features.parquet`, `experiment_5_common_cohort_predictions.parquet` | Previously used synthetic numbers (`np.random.seed(42)`). | **Redesign** | Authentic empirical distributions from 45,136 cohort observations: Panel A shows News Arrival Coverage (72.5% active days, 27.5% missing-news days). Panel B shows FinBERT sentiment distribution on active news days. |

---

## 3. Standardized Visual Design System

To enforce uniformity across all figures, the following scientific design standards are established:

### A. Professional Color Palette (Semantic Mapping)
- **Price-Only Baseline Models (`LSTM-F0`, `XGB-F0`, `LR-F0`)**: Deep Navy (`#1A365D` / `#2B6CB0`)
- **News Sentiment Augmented Models (`VADER`, `FinBERT`)**: Deep Teal (`#234E52` / `#319795`)
- **Multimodal Cross-Attention Architecture (`CrossAttentionNet`)**: Royal Purple (`#44337A` / `#6B46C1`)
- **Statistical / Naive Baselines (`Always-Up`, `Prev-Day`)**: Neutral Slate Grey (`#718096`)
- **Superior / Positive Benchmark**: Restrained Forest Green (`#276749`)
- **Degraded / Negative Benchmark**: Restrained Crimson (`#9B2C2C`)

### B. Typography & Font Hierarchy
- **Base Font**: Helvetica / Arial / DejaVu Sans (Clean, high-legibility sans-serif)
- **Axis Titles**: 9.5 pt, Medium/Bold, `#2D3748`
- **Tick Labels**: 8.0 pt, Regular, `#4A5568`
- **Annotations & Value Labels**: 7.5–8.0 pt, `#1A202C`
- **In-Plot Titles**: Strictly removed (all titles moved to formal LaTeX/Markdown figure captions).

### C. Technical Export Specifications
- **Raster Resolution**: 300 DPI minimum PNG for print/PDF embedding.
- **Vector Formats**: Scalable Vector Graphics (SVG) generated alongside all PNGs for lossless rendering.
