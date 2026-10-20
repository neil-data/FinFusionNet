"""Audit B2: Walk-Forward Partitioning Audit on Full NIFTY 50 Universe.

Evaluates Protocol B (24-month initial, monthly expanding, 1-day embargo)
and Protocol A (46/15/17 rolling).
Reports for each:
- folds
- unique test dates
- ticker-days
- mean and min tickers per test date
- distribution of ticker count per test date (documenting why dates deviate from 50).
Outputs experiments/audit/protocol_partitions_summary.csv and per-date constituent audit.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from finfusionnet.evaluation.pipeline import FEATURE_GROUPS
from finfusionnet.evaluation.walk_forward import WalkForwardEngine


def run_walk_forward_universe_audit():
    print("=" * 80, flush=True)
    print("AUDIT B2: WALK-FORWARD PARTITIONING AUDIT (FULL NIFTY 50 UNIVERSE)", flush=True)
    print("=" * 80, flush=True)

    master_path = Path("experiments/audit/master_universe_ticker_day.parquet")
    df = pd.read_parquet(master_path)
    df["date"] = pd.to_datetime(df["date"])

    tech_cols = FEATURE_GROUPS["group_a_technical"]

    # Filter to cohort
    mask_date = (df["date"] >= pd.Timestamp("2021-01-01")) & (df["date"] <= pd.Timestamp("2026-09-30"))
    mask_target = df["target"].notna()
    mask_tech = df[tech_cols].notna().all(axis=1)

    cohort_df = df[mask_date & mask_target & mask_tech].copy()
    cohort_df = cohort_df.sort_values(["date", "ticker"]).reset_index(drop=True)

    print(f"Eligible evaluation cohort rows: {len(cohort_df)}", flush=True)
    print(f"Unique dates: {cohort_df['date'].nunique()}, Unique tickers: {cohort_df['ticker'].nunique()}", flush=True)

    engine = WalkForwardEngine(date_col="date", target_date_col="target_date")

    # 1. Protocol B
    folds_b = engine.generate_protocol_b_folds(cohort_df)
    test_dates_b = set()
    ticker_days_b = 0
    date_counts_b = {}

    for fold in folds_b:
        test_slice = cohort_df.iloc[fold.test_indices]
        ticker_days_b += len(test_slice)
        for dt, grp in test_slice.groupby("date"):
            dt_str = dt.strftime("%Y-%m-%d")
            test_dates_b.add(dt_str)
            date_counts_b[dt_str] = len(grp)

    b_counts_series = pd.Series(date_counts_b)

    # 2. Protocol A
    folds_a = engine.generate_protocol_a_folds(
        cohort_df, train_sessions=46, test_sessions=15, step_sessions=17
    )
    test_dates_a = set()
    ticker_days_a = 0
    date_counts_a = {}

    for fold in folds_a:
        test_slice = cohort_df.iloc[fold.test_indices]
        ticker_days_a += len(test_slice)
        for dt, grp in test_slice.groupby("date"):
            dt_str = dt.strftime("%Y-%m-%d")
            test_dates_a.add(dt_str)
            date_counts_a[dt_str] = len(grp)

    a_counts_series = pd.Series(date_counts_a)

    summary_records = [
        {
            "protocol": "Protocol B (Expanding Monthly)",
            "n_folds": len(folds_b),
            "unique_test_dates": len(test_dates_b),
            "total_ticker_days": ticker_days_b,
            "mean_tickers_per_date": round(float(b_counts_series.mean()), 2),
            "min_tickers_per_date": int(b_counts_series.min()),
            "max_tickers_per_date": int(b_counts_series.max()),
            "pct_dates_exactly_50": round(float((b_counts_series == 50).mean() * 100), 2),
            "dates_below_50_count": int((b_counts_series < 50).sum()),
        },
        {
            "protocol": "Protocol A (Rolling Benchmark)",
            "n_folds": len(folds_a),
            "unique_test_dates": len(test_dates_a),
            "total_ticker_days": ticker_days_a,
            "mean_tickers_per_date": round(float(a_counts_series.mean()), 2),
            "min_tickers_per_date": int(a_counts_series.min()),
            "max_tickers_per_date": int(a_counts_series.max()),
            "pct_dates_exactly_50": round(float((a_counts_series == 50).mean() * 100), 2),
            "dates_below_50_count": int((a_counts_series < 50).sum()),
        },
    ]

    summary_df = pd.DataFrame(summary_records)
    out_csv = Path("experiments/audit/protocol_partitions_summary.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False)

    print("\n--- PROTOCOL PARTITIONS SUMMARY ---", flush=True)
    print(summary_df.to_string(index=False), flush=True)

    # Save per-date ticker count table for Protocol B
    b_date_df = pd.DataFrame({
        "date": b_counts_series.index,
        "n_tickers": b_counts_series.values,
    }).sort_values("date").reset_index(drop=True)
    b_date_df.to_csv("experiments/audit/protocol_b_daily_ticker_counts.csv", index=False)

    print(f"\nProtocol B test dates range: {min(test_dates_b)} to {max(test_dates_b)}", flush=True)
    print(f"Mean tickers per date: {b_counts_series.mean():.2f} (Min: {b_counts_series.min()}, Max: {b_counts_series.max()})", flush=True)
    print(f"Dates with exactly 50 tickers: {(b_counts_series == 50).sum()} / {len(b_counts_series)}", flush=True)
    print(f"Dates with < 50 tickers: {(b_counts_series < 50).sum()} / {len(b_counts_series)}", flush=True)

    return summary_df


if __name__ == "__main__":
    run_walk_forward_universe_audit()
