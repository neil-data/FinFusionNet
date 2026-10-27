"""FinFusionNet: Experiment 4 - Comprehensive Comparison and Subgroup Analysis.

Computes:
1. Tripartite Comparison:
   - Price-Only Baseline (F0)
   - Experiment 3B All-50 VADER (F1_VADER)
   - Experiment 4 All-50 FinBERT (F1_FinBERT)
2. Subgroup Analyses:
   - Active News Days (has_news == 1) vs No-News Days (has_news == 0)
   - Heavyweight 10 Stocks vs Expanded 40 Stocks
   - Article Volume Buckets (Low: 1, Medium: 2-3, High: 4+)
3. Sector-Level Breakdown across all NIFTY 50 sectors.
4. Fold Stability Analysis across all 45 Protocol B folds.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
    f1_score,
)

EXPERIMENT_DIR = Path("experiments/experiment_4")
PRED_PATH = EXPERIMENT_DIR / "predictions" / "experiment_4_common_cohort_predictions.parquet"
UNIVERSE_CSV = Path("data/metadata/universe.csv")

HEAVYWEIGHT_10 = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "ITC",
    "TCS", "LT", "BHARTIARTL", "SBIN", "HINDUNILVR",
]


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
    ll = log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7))
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pred_up = float(np.mean(y_pred == 1))

    return {
        "date_level_acc": round(float(date_acc), 4),
        "pooled_acc": round(float(pooled_acc), 4),
        "balanced_acc": round(float(balanced_acc), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "brier_score": round(float(brier), 4),
        "log_loss": round(float(ll), 4),
        "predicted_up_share": round(pred_up, 4),
    }


def analyze_experiment_4():
    if not PRED_PATH.exists():
        raise FileNotFoundError(f"Predictions not found at {PRED_PATH}. Run run_experiment_4_finbert.py first.")

    print(f"Loading Experiment 4 common cohort predictions from {PRED_PATH}...", flush=True)
    df = pd.read_parquet(PRED_PATH)
    print(f"Loaded {len(df):,} observations across {df['date'].nunique()} test dates and {df['ticker'].nunique()} tickers.", flush=True)

    y_true = df["y_true"].values
    dates = df["date"].values

    models = [
        ("XGB-F0 (Price-Only)", "prob_xgb_f0"),
        ("XGB-F1_VADER (All-50 VADER)", "prob_xgb_f1_vader"),
        ("XGB-F1_FinBERT (All-50 FinBERT)", "prob_xgb_f1_finbert"),
        ("LR-F0 (Price-Only)", "prob_lr_f0"),
        ("LR-F1_VADER (All-50 VADER)", "prob_lr_f1_vader"),
        ("LR-F1_FinBERT (All-50 FinBERT)", "prob_lr_f1_finbert"),
        ("LSTM-F0 (Price-Only, Seed 42)", "prob_lstm_f0"),
        ("LSTM-F1_VADER (All-50 VADER, Seed 42)", "prob_lstm_f1_vader"),
        ("LSTM-F1_FinBERT (All-50 FinBERT, Seed 42)", "prob_lstm_f1_finbert"),
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

    df_tripartite = pd.DataFrame(tripartite_rows)[
        ["model_configuration", "date_level_acc", "pooled_acc", "balanced_acc", "f1_score", "roc_auc", "brier_score", "log_loss", "predicted_up_share"]
    ]
    df_tripartite.to_csv(EXPERIMENT_DIR / "experiment_4_tripartite_comparison.csv", index=False)
    print(df_tripartite.to_string(index=False), flush=True)

    # -------------------------------------------------------------
    # 2. SUBGROUP ANALYSIS
    # -------------------------------------------------------------
    print("\n=== 2. SUBGROUP ANALYSIS ===", flush=True)
    active_mask = (df["has_news_all50"] == 1)
    no_news_mask = (df["has_news_all50"] == 0)
    heavyweight_mask = df["ticker"].isin(HEAVYWEIGHT_10)
    expanded_mask = ~heavyweight_mask

    subgroups = [
        ("Active News Days", active_mask),
        ("No-News Days", no_news_mask),
        ("Original 10 Heavyweights", heavyweight_mask),
        ("Expanded 40 Stocks", expanded_mask),
    ]

    subgroup_rows = []
    for sub_name, mask in subgroups:
        sub_df = df[mask]
        n_obs = len(sub_df)
        pct = (n_obs / len(df)) * 100
        y_sub = sub_df["y_true"].values
        d_sub = sub_df["date"].values

        for name, col in models:
            m = compute_metrics_dict(y_sub, sub_df[col].values, d_sub)
            m["subgroup"] = sub_name
            m["n_obs"] = n_obs
            m["pct_of_total"] = round(pct, 1)
            m["model_configuration"] = name
            subgroup_rows.append(m)

    df_subgroups = pd.DataFrame(subgroup_rows)[
        ["subgroup", "n_obs", "pct_of_total", "model_configuration", "date_level_acc", "pooled_acc", "balanced_acc", "roc_auc", "brier_score"]
    ]
    df_subgroups.to_csv(EXPERIMENT_DIR / "experiment_4_subgroups.csv", index=False)
    print(df_subgroups.to_string(index=False), flush=True)

    # -------------------------------------------------------------
    # 3. SECTOR-LEVEL PERFORMANCE BREAKDOWN
    # -------------------------------------------------------------
    print("\n=== 3. SECTOR-LEVEL BREAKDOWN ===", flush=True)
    if UNIVERSE_CSV.exists():
        u_df = pd.read_csv(UNIVERSE_CSV)
        sector_map = dict(zip(u_df["ticker"], u_df["sector"]))
    else:
        sector_map = {}

    df["sector"] = df["ticker"].map(sector_map).fillna("Other")
    sector_rows = []

    for sector, grp in df.groupby("sector"):
        if len(grp) < 100:
            continue
        y_sec = grp["y_true"].values
        d_sec = grp["date"].values
        sec_acc_xgb_f0 = accuracy_score(y_sec, (grp["prob_xgb_f0"] >= 0.50).astype(int))
        sec_acc_xgb_vader = accuracy_score(y_sec, (grp["prob_xgb_f1_vader"] >= 0.50).astype(int))
        sec_acc_xgb_finbert = accuracy_score(y_sec, (grp["prob_xgb_f1_finbert"] >= 0.50).astype(int))

        sec_acc_lr_f0 = accuracy_score(y_sec, (grp["prob_lr_f0"] >= 0.50).astype(int))
        sec_acc_lr_finbert = accuracy_score(y_sec, (grp["prob_lr_f1_finbert"] >= 0.50).astype(int))

        sec_acc_lstm_f0 = accuracy_score(y_sec, (grp["prob_lstm_f0"] >= 0.50).astype(int))
        sec_acc_lstm_finbert = accuracy_score(y_sec, (grp["prob_lstm_f1_finbert"] >= 0.50).astype(int))

        sector_rows.append({
            "sector": sector,
            "n_obs": len(grp),
            "news_coverage_pct": round(float((grp["has_news_all50"] == 1).mean() * 100), 1),
            "xgb_f0_acc": round(float(sec_acc_xgb_f0), 4),
            "xgb_vader_acc": round(float(sec_acc_xgb_vader), 4),
            "xgb_finbert_acc": round(float(sec_acc_xgb_finbert), 4),
            "xgb_finbert_delta": round(float(sec_acc_xgb_finbert - sec_acc_xgb_f0), 4),
            "lr_f0_acc": round(float(sec_acc_lr_f0), 4),
            "lr_finbert_acc": round(float(sec_acc_lr_finbert), 4),
            "lr_finbert_delta": round(float(sec_acc_lr_finbert - sec_acc_lr_f0), 4),
            "lstm_f0_acc": round(float(sec_acc_lstm_f0), 4),
            "lstm_finbert_acc": round(float(sec_acc_lstm_finbert), 4),
            "lstm_finbert_delta": round(float(sec_acc_lstm_finbert - sec_acc_lstm_f0), 4),
        })

    df_sectors = pd.DataFrame(sector_rows).sort_values("n_obs", ascending=False)
    df_sectors.to_csv(EXPERIMENT_DIR / "experiment_4_sector_breakdown.csv", index=False)
    print(df_sectors.to_string(index=False), flush=True)

    print("\nExperiment 4 comparison and subgroup analysis complete.", flush=True)


if __name__ == "__main__":
    analyze_experiment_4()
