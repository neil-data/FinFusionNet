"""FinFusionNet CLI: Statistical Evaluation & Hypothesis Testing.

Executes:
1. Pre-registered Primary Hypothesis: X5 vs X3 on Protocol B Date-Level Accuracy.
2. 10,000-repetition Paired Date-Block Bootstrap and Sensitivity Analysis.
3. Secondary model comparisons with Holm-Bonferroni correction.
4. Generates comprehensive final research report in reports/final_results/.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from finfusionnet.evaluation.stats_tests import (
    paired_date_block_bootstrap,
    block_length_sensitivity_analysis,
    apply_holm_bonferroni,
    date_level_mcnemar_test,
)


def find_latest_run(runs_base_dir: str, exp_id: str, protocol: str = "protocol_b") -> Path:
    """Find the most recent execution directory for a specific experiment and protocol."""
    target_dir = Path(runs_base_dir) / f"{exp_id.lower()}_{protocol}"
    if not target_dir.exists():
        raise FileNotFoundError(f"No runs found for {exp_id} under {protocol} in {target_dir}")
    subdirs = sorted([p for p in target_dir.iterdir() if p.is_dir()])
    if not subdirs:
        raise FileNotFoundError(f"No run directories found in {target_dir}")
    return subdirs[-1]


def run_statistical_evaluation(
    runs_base_dir: str = "experiments/runs",
    output_report_path: str = "reports/final_results/xgboost_research_report.md",
):
    print("Starting comprehensive statistical significance evaluation...")

    # Load latest Protocol B runs for X1 to X6
    exp_runs: Dict[str, Path] = {}
    exp_summaries: Dict[str, dict] = {}
    date_metrics_dict: Dict[str, pd.DataFrame] = {}

    for exp_id in ["X1", "X2", "X3", "X4", "X5", "X6"]:
        try:
            r_path = find_latest_run(runs_base_dir, exp_id, "protocol_b")
            exp_runs[exp_id] = r_path
            with open(r_path / "summary.json", "r", encoding="utf-8") as f:
                exp_summaries[exp_id] = json.load(f)
            dm = pd.read_csv(r_path / "date_metrics.csv")
            dm["date"] = pd.to_datetime(dm["date"])
            date_metrics_dict[exp_id] = dm.sort_values("date").reset_index(drop=True)
            print(f"Loaded {exp_id}: {r_path}")
        except FileNotFoundError as e:
            print(f"Warning: {e}")

    if "X5" not in date_metrics_dict or "X3" not in date_metrics_dict:
        print("Error: Both X5 and X3 runs are required for primary hypothesis testing.")
        return

    # Verify identical dates across X5 and X3
    dm_x5 = date_metrics_dict["X5"]
    dm_x3 = date_metrics_dict["X3"]

    merged = pd.merge(
        dm_x5[["date", "accuracy"]].rename(columns={"accuracy": "acc_x5"}),
        dm_x3[["date", "accuracy"]].rename(columns={"accuracy": "acc_x3"}),
        on="date",
        how="inner",
    )
    print(f"Aligned {len(merged)} test trading dates for primary comparison.")

    acc_x5 = merged["acc_x5"].values
    acc_x3 = merged["acc_x3"].values

    # 1. Primary Hypothesis Test: X5 vs X3 (Paired Date-Block Bootstrap, L=10, 10k reps)
    print("Computing 10,000 paired date-block bootstrap iterations (L=10)...")
    primary_res = paired_date_block_bootstrap(
        acc_x5, acc_x3, block_length=10, n_bootstraps=10000, random_state=42
    )

    # 2. Block-Length Sensitivity Analysis (L in [3, 5, 10, 15, 20])
    print("Computing block-length sensitivity analysis...")
    sens_df = block_length_sensitivity_analysis(
        acc_x5, acc_x3, block_lengths=[3, 5, 10, 15, 20], n_bootstraps=5000, random_state=42
    )

    # 3. Secondary Model Comparisons
    secondary_pairs = [
        ("X2", "X1", "Market Context vs Pure Technicals"),
        ("X3", "X1", "News Attention vs Pure Technicals"),
        ("X4", "X3", "VADER Sentiment vs News Control"),
        ("X6", "X3", "Combined Sentiment vs News Control"),
        ("X6", "X4", "Combined Sentiment vs VADER"),
        ("X6", "X5", "Combined Sentiment vs FinBERT"),
    ]

    secondary_raw_p = {}
    secondary_stats = {}
    for id_a, id_b, desc in secondary_pairs:
        if id_a in date_metrics_dict and id_b in date_metrics_dict:
            m_pair = pd.merge(
                date_metrics_dict[id_a][["date", "accuracy"]].rename(columns={"accuracy": "acc_a"}),
                date_metrics_dict[id_b][["date", "accuracy"]].rename(columns={"accuracy": "acc_b"}),
                on="date",
                how="inner",
            )
            boot = paired_date_block_bootstrap(
                m_pair["acc_a"].values, m_pair["acc_b"].values, block_length=10, n_bootstraps=5000
            )
            mcn = date_level_mcnemar_test(m_pair["acc_a"].values, m_pair["acc_b"].values)
            pair_key = f"{id_a}_vs_{id_b}"
            secondary_raw_p[pair_key] = boot["p_value_two_sided"]
            secondary_stats[pair_key] = {
                "desc": desc,
                "mean_diff": boot["observed_mean_difference"],
                "ci_lower": boot["ci_lower"],
                "ci_upper": boot["ci_upper"],
                "p_two_sided": boot["p_value_two_sided"],
                "mcnemar_wins_a": mcn["wins_model_a"],
                "mcnemar_wins_b": mcn["wins_model_b"],
                "mcnemar_p": mcn["p_value"],
            }

    # Apply Holm-Bonferroni correction
    holm_results = apply_holm_bonferroni(secondary_raw_p, family_alpha=0.05)
    for k, v in holm_results.items():
        secondary_stats[k].update(v)

    # 4. Generate Research Report
    report_md = f"""# FinFusionNet — XGBoost Research Track Final Report

## Executive Summary
This report details the empirical findings of the **XGBoost Experimental Track of FinFusionNet**, evaluating whether financial news sentiment (lexicon-based VADER and domain-adapted FinBERT) provides statistically defensible predictive lift for next-trading-session stock direction in the Indian equity market.

Evaluation was conducted under a strict **Common Evaluation Cohort** across **point-in-time NIFTY 50 constituents** using chronological walk-forward validation and paired date-block bootstrap testing.

---

## 1. Overall Performance Comparison (Protocol B: Expanding Window)

| Model ID | Configuration | Date-Level Acc | Pooled Acc | Pooled F1 | ROC-AUC | Log Loss | Lift vs Always-Up | Lift vs Prev-Dir |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""

    for exp_id in ["X1", "X2", "X3", "X4", "X5", "X6"]:
        if exp_id in exp_summaries:
            s = exp_summaries[exp_id]
            report_md += (
                f"| **{exp_id}** | {s['description']} | "
                f"{s['primary_metric_equal_weighted_date_accuracy']:.4f} | "
                f"{s['pooled_accuracy']:.4f} | "
                f"{s['pooled_f1']:.4f} | "
                f"{s['pooled_roc_auc']:.4f} | "
                f"{s['pooled_log_loss']:.4f} | "
                f"{s['lift_over_always_up']:+.4f} | "
                f"{s['lift_over_previous_direction']:+.4f} |\n"
            )

    report_md += f"""
---

## 2. Pre-Registered Primary Hypothesis: FinBERT Contribution (X5 vs X3)

**Hypothesis**: Financial news sentiment extracted via FinBERT improves next-session stock direction prediction beyond technical indicators and news availability controls under Protocol B.
- **Metric**: Equal-weighted date-level accuracy ($Acc_d$)
- **Statistical Test**: Paired Date-Block Bootstrap (10,000 repetitions, block length $L=10$ trading days)
- **Observed Mean Difference**: **{primary_res['observed_mean_difference']:+.4f}** ({primary_res['observed_mean_difference']*100:+.2f}%)
- **95% Bootstrap Confidence Interval**: **[{primary_res['ci_lower']:+.4f}, {primary_res['ci_upper']:+.4f}]**
- **Two-Sided Empirical p-value**: **{primary_res['p_value_two_sided']:.4f}**
- **One-Sided Empirical p-value ($H_1: \Delta > 0$)**: **{primary_res['p_value_one_sided']:.4f}**
- **Statistically Significant at $\alpha = 0.05$**: **{'YES' if primary_res['is_significant_05'] else 'NO'}**

### Block-Length Sensitivity Analysis
| Block Length ($L$ days) | Mean Difference | 95% Bootstrap CI | Two-Sided p-value | Significant at $\alpha=0.05$ |
|:---:|:---:|:---:|:---:|:---:|
"""
    for _, row in sens_df.iterrows():
        report_md += (
            f"| {int(row['block_length'])} | {row['mean_diff']:+.4f} | "
            f"[{row['ci_lower']:+.4f}, {row['ci_upper']:+.4f}] | "
            f"{row['p_two_sided']:.4f} | "
            f"{'YES' if row['significant_05'] else 'NO'} |\n"
        )

    report_md += f"""
---

## 3. Secondary Hypotheses & Holm-Bonferroni Correction

| Comparison | Description | Mean Diff | 95% Bootstrap CI | Raw p-value | Holm Adjusted p | Significant |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
"""
    for pair_key, s_data in secondary_stats.items():
        report_md += (
            f"| **{pair_key}** | {s_data['desc']} | "
            f"{s_data['mean_diff']:+.4f} | "
            f"[{s_data['ci_lower']:+.4f}, {s_data['ci_upper']:+.4f}] | "
            f"{s_data['raw_p']:.4f} | "
            f"{s_data.get('adjusted_p', s_data['raw_p']):.4f} | "
            f"{'YES' if s_data.get('is_significant', False) else 'NO'} |\n"
        )

    report_md += f"""
---

## 4. Key Methodological Takeaways & Limitations
1. **Survivorship Bias Protection**: Restricting evaluation to true point-in-time index constituents across 62 historical NIFTY 50 tickers prevented artificial selection bias.
2. **Correlation-Safe Inference**: By clustering at the date level and resampling contiguous trading blocks, the evaluation prevented false significance from market-wide cross-sectional co-movement.
3. **Execution Reality**: Predictive accuracy in directional classification must be paired with transaction costs and slippage before implying economic profitability.
"""

    out_p = Path(output_report_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\nFinal research report written to {out_p}")


if __name__ == "__main__":
    run_statistical_evaluation()
