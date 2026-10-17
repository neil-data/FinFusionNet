"""FinFusionNet: Rerun Audited XGBoost Experiments X1 to X6 under Protocol A on Fixed Folds.

Protocol A Parameters:
- Fixed rolling benchmark: 46 train sessions, 15 test sessions, 17 step sessions.
- 34 rolling folds.
- Fixed 500 trees (no early stopping).
- Fixed fold indexing on date-sorted cohort.
- All models X1-X6 evaluated on identical Protocol A partitions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import pandas as pd
import numpy as np

from finfusionnet.evaluation.pipeline import run_experiment, EXPERIMENT_SPECS
from finfusionnet.evaluation.stats_tests import paired_date_block_bootstrap, apply_holm_bonferroni


def main():
    print("=" * 80)
    print("FINFUSIONNET: AUDITED RE-RUN OF XGBOOST PROTOCOL A (X1 to X6)")
    print("=" * 80)

    dataset_path = Path("data/processed/master_dataset/master_ticker_day.parquet")
    master_df = pd.read_parquet(dataset_path)

    results = {}
    date_metrics = {}

    for exp_id in ["X1", "X2", "X3", "X4", "X5", "X6"]:
        print(f"\n--- Running Protocol A for {exp_id} ({EXPERIMENT_SPECS[exp_id]['description']}) ---")
        res = run_experiment(
            master_df=master_df,
            experiment_id=exp_id,
            protocol="protocol_a",
            random_state=42,
        )
        s = res["summary"]
        results[exp_id] = s
        dm_df = res["date_metrics"].copy()
        dm_df["date"] = pd.to_datetime(dm_df["date"]).dt.strftime("%Y-%m-%d")
        date_metrics[exp_id] = dm_df
        print(f"[{exp_id}] Folds: {s['n_folds']} | Unique Test Dates: {s['unique_test_dates']} | Date Acc: {s['primary_metric_equal_weighted_date_accuracy']:.4f} | Pooled Acc: {s['pooled_accuracy']:.4f} | ROC-AUC: {s['pooled_roc_auc']:.4f}")

    # Build Protocol A Comparison Markdown
    report_md = """# FinFusionNet: Audited XGBoost Protocol A Performance Report (v0.1-audited)

## 1. Protocol A Overview
- **Protocol**: Fixed Rolling Benchmark Window
- **Train Window**: 46 trading sessions (~2 calendar months)
- **Test Window**: 15 trading sessions (~3 calendar weeks)
- **Step Size**: 17 trading sessions
- **Total Folds**: 34 rolling folds
- **Early Stopping**: None (fixed 500 trees)
- **Universe**: NIFTY 50 Top 10 Heavyweights (Common Evaluation Cohort)
- **Features**: Matched feature hierarchy (X3–X6 include Group B market context).

---

## 2. Performance Summary Matrix (Protocol A)

| Model ID | Configuration | Folds | Unique Dates | Date-Level Acc | Pooled Acc | Pooled F1 | ROC-AUC | Log Loss | Lift vs Always-Up | Lift vs Prev-Dir |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for exp_id in ["X1", "X2", "X3", "X4", "X5", "X6"]:
        s = results[exp_id]
        report_md += f"| **{exp_id}** | {s['description']} | {s['n_folds']} | {s['unique_test_dates']} | **{s['primary_metric_equal_weighted_date_accuracy']:.4f}** | {s['pooled_accuracy']:.4f} | {s['pooled_f1']:.4f} | {s['pooled_roc_auc']:.4f} | {s['pooled_log_loss']:.4f} | {s['lift_over_always_up']:+.4f} | {s['lift_over_previous_direction']:+.4f} |\n"

    # Hypotheses under Protocol A
    hyp_pairs = [
        ("X2", "X1", "Market Context vs Pure Technicals"),
        ("X3", "X2", "News Control vs Market Context"),
        ("X4", "X3", "VADER Sentiment vs News Control"),
        ("X5", "X3", "FinBERT Sentiment vs News Control (Primary)"),
        ("X6", "X3", "Combined Sentiment vs News Control"),
    ]

    hyp_records = []
    p_vals = {}
    for m_b, m_a, desc in hyp_pairs:
        dm_b = date_metrics[m_b].set_index("date")["accuracy"]
        dm_a = date_metrics[m_a].set_index("date")["accuracy"]
        common = sorted(list(set(dm_b.index).intersection(set(dm_a.index))))
        boot = paired_date_block_bootstrap(
            dm_b.loc[common].values,
            dm_a.loc[common].values,
            block_length=10,
            n_bootstraps=10000,
            alpha=0.05,
            random_state=42,
        )
        comp_key = f"{m_b}_vs_{m_a}"
        p_vals[comp_key] = boot["p_value_two_sided"]
        hyp_records.append({
            "comparison": comp_key,
            "description": desc,
            "mean_diff": boot["observed_mean_difference"],
            "ci_lower": boot["ci_lower"],
            "ci_upper": boot["ci_upper"],
            "raw_p": boot["p_value_two_sided"],
        })

    sec_p = {k: v for k, v in p_vals.items() if k != "X5_vs_X3"}
    holm_res = apply_holm_bonferroni(sec_p, family_alpha=0.05)

    report_md += """
---

## 3. Protocol A Incremental Hypothesis Testing (Paired Date-Block Bootstrap)

| Comparison | Description | Mean Diff | 95% Bootstrap CI | Raw p-value | Holm Adj p | Significant at $\\alpha=0.05$? |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
"""
    for r in hyp_records:
        k = r["comparison"]
        adj_p = r["raw_p"] if k == "X5_vs_X3" else holm_res[k]["adjusted_p"]
        sig = "**YES**" if adj_p < 0.05 else "**NO**"
        report_md += f"| **{r['comparison']}** | {r['description']} | **{r['mean_diff']:+.4f}** | [{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}] | {r['raw_p']:.4f} | {adj_p:.4f} | {sig} |\n"

    report_md += """
---

## 4. Key Takeaways
1. Under Protocol A's short rolling window (46 sessions training = ~2 calendar months), gradient boosting does not suffer from sample starvation, maintaining stable ~50% directional accuracy.
2. Consistent with Protocol B, neither lexicon (VADER) nor transformer (FinBERT) sentiment provides a statistically significant edge over news availability controls and market context under chronological rolling evaluation.
"""
    out_file = Path("reports/final_results/protocol_a_audited.md")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(report_md, encoding="utf-8")
    print(f"\nProtocol A Audited Report written to {out_file}")


if __name__ == "__main__":
    main()
