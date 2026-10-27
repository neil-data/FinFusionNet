"""FinFusionNet: Experiment 3B - Comprehensive Tripartite Comparison and Subgroup Analysis.

Computes:
1. Tripartite Performance Comparison:
   - Price-Only Baselines (F0)
   - Experiment 3 (10-Stock News Fusion) (F1_10)
   - Experiment 3B (All-50 News Fusion) (F1_All50)
2. Subgroup Performance Breakdown:
   - Active-News Days vs No-News Days in All-50
   - Original 10 Heavyweight Stocks vs Expanded 40 Stocks
3. Fold-Level Stability Metrics across all 45 Protocol B folds.
4. Sector-Level Performance Breakdown.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, brier_score_loss


EXPERIMENT_DIR = Path("experiments/experiment_3b")
PRED_PATH = EXPERIMENT_DIR / "predictions" / "experiment_3b_common_cohort_predictions.parquet"
UNIVERSE_CSV = Path("data/metadata/universe.csv")


def compute_metrics_dict(y_true, y_prob, dates):
    y_pred = (y_prob >= 0.50).astype(int)
    correct = (y_true == y_pred).astype(float)
    date_acc = pd.Series(correct).groupby(dates).mean().mean()
    pooled_acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.5
    brier = brier_score_loss(y_true, y_prob)

    return {
        "date_level_acc": round(float(date_acc), 4),
        "pooled_acc": round(float(pooled_acc), 4),
        "balanced_acc": round(float(balanced_acc), 4),
        "roc_auc": round(float(auc), 4),
        "brier_score": round(float(brier), 4),
    }


def analyze_experiment_3b():
    if not PRED_PATH.exists():
        raise FileNotFoundError(f"Predictions not found at {PRED_PATH}. Run run_experiment_3b_all50.py first.")

    print(f"Loading Experiment 3B common cohort predictions from {PRED_PATH}...", flush=True)
    df = pd.read_parquet(PRED_PATH)
    print(f"Loaded {len(df):,} observations across {df['date'].nunique()} test dates and {df['ticker'].nunique()} tickers.", flush=True)

    y_true = df["y_true"].values
    dates = df["date"].values

    models = [
        ("XGB-F0 (Price-Only)", "prob_xgb_f0"),
        ("XGB-F1_10 (10-Stock News)", "prob_xgb_f1_10"),
        ("XGB-F1_All50 (All-50 News)", "prob_xgb_f1_all50"),
        ("LR-F0 (Price-Only)", "prob_lr_f0"),
        ("LR-F1_10 (10-Stock News)", "prob_lr_f1_10"),
        ("LR-F1_All50 (All-50 News)", "prob_lr_f1_all50"),
        ("LSTM-F0 (Price-Only, Seed 42)", "prob_lstm_f0"),
        ("LSTM-F1_10 (10-Stock News, Seed 42)", "prob_lstm_f1_10"),
        ("LSTM-F1_All50 (All-50 News, Seed 42)", "prob_lstm_f1_all50"),
    ]

    # -------------------------------------------------------------
    # 1. TRIPARTITE COMPARISON ON FULL COMMON COHORT
    # -------------------------------------------------------------
    print("\n=== 1. TRIPARTITE COMPARISON ON FULL COMMON COHORT (N=45,136) ===", flush=True)
    tripartite_rows = []
    for name, col in models:
        m = compute_metrics_dict(y_true, df[col].values, dates)
        m["model_configuration"] = name
        tripartite_rows.append(m)

    df_tripartite = pd.DataFrame(tripartite_rows)[["model_configuration", "date_level_acc", "pooled_acc", "balanced_acc", "roc_auc", "brier_score"]]
    df_tripartite.to_csv(EXPERIMENT_DIR / "experiment_3b_tripartite_comparison.csv", index=False)
    print(df_tripartite.to_string(index=False), flush=True)

    # -------------------------------------------------------------
    # 2. SUBGROUP ANALYSIS: ACTIVE NEWS VS NO-NEWS DAYS IN ALL-50
    # -------------------------------------------------------------
    print("\n=== 2. SUBGROUP: ACTIVE NEWS VS NO-NEWS DAYS IN ALL-50 ===", flush=True)
    active_mask = (df["has_news_all50"] == 1)
    no_news_mask = (df["has_news_all50"] == 0)

    subgroup_rows = []
    for sub_name, mask in [("All-50 Active News Days", active_mask), ("All-50 No-News Days", no_news_mask)]:
        sub_df = df[mask]
        n_obs = len(sub_df)
        pct = (n_obs / len(df)) * 100
        y_sub = sub_df["y_true"].values
        d_sub = sub_df["date"].values

        for name, col in models:
            m = compute_metrics_dict(y_sub, sub_df[col].values, d_sub)
            m["subgroup"] = sub_name
            m["observations"] = n_obs
            m["pct_cohort"] = round(pct, 2)
            m["model_configuration"] = name
            subgroup_rows.append(m)

    # -------------------------------------------------------------
    # 3. SUBGROUP ANALYSIS: 10 HEAVYWEIGHTS VS REMAINING EXPANDED STOCKS
    # -------------------------------------------------------------
    print("\n=== 3. SUBGROUP: 10 HEAVYWEIGHTS VS REMAINING EXPANDED STOCKS ===", flush=True)
    # Check tickers that had news in 10-stock dataset
    top10_tickers = df[df["has_news_10"] == 1]["ticker"].unique()
    top10_mask = df["ticker"].isin(top10_tickers)
    expanded_mask = ~top10_mask

    for sub_name, mask in [("Original 10 Heavyweights", top10_mask), ("Expanded Constituents", expanded_mask)]:
        sub_df = df[mask]
        n_obs = len(sub_df)
        pct = (n_obs / len(df)) * 100
        y_sub = sub_df["y_true"].values
        d_sub = sub_df["date"].values

        for name, col in models:
            m = compute_metrics_dict(y_sub, sub_df[col].values, d_sub)
            m["subgroup"] = sub_name
            m["observations"] = n_obs
            m["pct_cohort"] = round(pct, 2)
            m["model_configuration"] = name
            subgroup_rows.append(m)

    df_subgroups = pd.DataFrame(subgroup_rows)[["subgroup", "model_configuration", "observations", "pct_cohort", "date_level_acc", "pooled_acc", "balanced_acc", "roc_auc", "brier_score"]]
    df_subgroups.to_csv(EXPERIMENT_DIR / "experiment_3b_subgroups.csv", index=False)
    print(df_subgroups.to_string(index=False), flush=True)

    # -------------------------------------------------------------
    # 4. FOLD-LEVEL STABILITY ACROSS 45 EXPANDING FOLDS
    # -------------------------------------------------------------
    print("\n=== 4. FOLD-LEVEL STABILITY ACROSS 45 PROTOCOL B FOLDS ===", flush=True)
    fold_records = []
    for f_idx, grp in df.groupby("fold_idx"):
        y_f = grp["y_true"].values
        row = {
            "fold_idx": f_idx,
            "test_start": str(grp["date"].min()),
            "test_end": str(grp["date"].max()),
            "n_samples": len(grp),
        }
        for name, col in models:
            pred_f = (grp[col] >= 0.50).astype(int)
            row[f"acc_{col}"] = round(float(np.mean(y_f == pred_f)), 4)
        fold_records.append(row)

    df_folds = pd.DataFrame(fold_records)
    df_folds.to_csv(EXPERIMENT_DIR / "experiment_3b_fold_results.csv", index=False)

    print("\n--- Mean Fold Accuracies (45 Folds) ---")
    for name, col in models:
        mean_acc = df_folds[f"acc_{col}"].mean()
        std_acc = df_folds[f"acc_{col}"].std()
        print(f"  {name:35s}: Mean Acc = {mean_acc:.4f} (std = {std_acc:.4f})")

    # -------------------------------------------------------------
    # 5. SECTOR-LEVEL BREAKDOWN
    # -------------------------------------------------------------
    print("\n=== 5. SECTOR-LEVEL PERFORMANCE BREAKDOWN ===", flush=True)
    if UNIVERSE_CSV.exists():
        df_uni = pd.read_csv(UNIVERSE_CSV)
        ticker_to_sector = dict(zip(df_uni["ticker"], df_uni["sector"]))
        df["sector"] = df["ticker"].map(ticker_to_sector).fillna("Other")

        sector_rows = []
        for sector, grp in df.groupby("sector"):
            n_obs = len(grp)
            y_s = grp["y_true"].values
            d_s = grp["date"].values

            for name, col in [
                ("XGB-F0", "prob_xgb_f0"),
                ("XGB-F1_10", "prob_xgb_f1_10"),
                ("XGB-F1_All50", "prob_xgb_f1_all50"),
                ("LR-F0", "prob_lr_f0"),
                ("LR-F1_10", "prob_lr_f1_10"),
                ("LR-F1_All50", "prob_lr_f1_all50"),
                ("LSTM-F0", "prob_lstm_f0"),
                ("LSTM-F1_10", "prob_lstm_f1_10"),
                ("LSTM-F1_All50", "prob_lstm_f1_all50"),
            ]:
                m = compute_metrics_dict(y_s, grp[col].values, d_s)
                m["sector"] = sector
                m["observations"] = n_obs
                m["model"] = name
                sector_rows.append(m)

        df_sector = pd.DataFrame(sector_rows)[["sector", "model", "observations", "date_level_acc", "pooled_acc", "balanced_acc", "roc_auc", "brier_score"]]
        df_sector.to_csv(EXPERIMENT_DIR / "experiment_3b_sector_breakdown.csv", index=False)
        print(df_sector.to_string(index=False), flush=True)

    print("\nAll Experiment 3B comparisons, subgroups, fold stability, and sector analyses completed successfully.", flush=True)


if __name__ == "__main__":
    analyze_experiment_3b()
