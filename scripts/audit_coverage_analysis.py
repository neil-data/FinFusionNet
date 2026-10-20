"""Audit B5: 50-Member Daily Coverage & Dropped Ticker-Day Forensics.

Answers explicitly: "Does the pipeline produce predictions for all 50 NIFTY 50 members on every test date?"
Outputs:
1. experiments/audit/daily_constituent_counts.csv (date, active_constituents, evaluated_predictions, missing_count)
2. experiments/audit/per_ticker_prediction_counts.csv (ticker, expected_days, actual_predictions, missing_days, reason)
3. experiments/audit/dropped_ticker_days_audit.csv (date, ticker, reason: warmup, missing price, listing date)
"""

from pathlib import Path
import pandas as pd
import numpy as np

from finfusionnet.data.universe import PointInTimeUniverse
from finfusionnet.data.trading_calendar import get_trading_calendar
from finfusionnet.evaluation.pipeline import FEATURE_GROUPS
from finfusionnet.evaluation.walk_forward import WalkForwardEngine


def run_coverage_analysis():
    print("=" * 80, flush=True)
    print("AUDIT B5: 50-MEMBER DAILY COVERAGE & DROPPED TICKER-DAY FORENSICS", flush=True)
    print("=" * 80, flush=True)

    master_path = Path("experiments/audit/master_universe_ticker_day.parquet")
    df = pd.read_parquet(master_path)
    df["date"] = pd.to_datetime(df["date"])

    tech_cols = FEATURE_GROUPS["group_a_technical"]
    mask_date = (df["date"] >= pd.Timestamp("2021-01-01")) & (df["date"] <= pd.Timestamp("2026-09-30"))
    mask_target = df["target"].notna()
    mask_tech = df[tech_cols].notna().all(axis=1)

    cohort_df = df[mask_date & mask_target & mask_tech].copy()
    cohort_df = cohort_df.sort_values(["date", "ticker"]).reset_index(drop=True)

    engine = WalkForwardEngine(date_col="date", target_date_col="target_date")
    folds = engine.generate_protocol_b_folds(cohort_df)

    # Collect all out-of-sample test predictions
    test_rows = []
    for fold in folds:
        test_slice = cohort_df.iloc[fold.test_indices][["date", "ticker", "target"]].copy()
        test_slice["fold_idx"] = fold.fold_idx
        test_rows.append(test_slice)

    test_df = pd.concat(test_rows, ignore_index=True)
    test_dates = sorted(test_df["date"].unique())
    print(f"Total out-of-sample test dates: {len(test_dates)}", flush=True)

    universe = PointInTimeUniverse()

    # Per-date constituent count accounting
    daily_records = []
    dropped_records = []

    # Map raw data availability per ticker
    available_tickers = set(df["ticker"].unique())

    for dt in test_dates:
        dt_str = dt.strftime("%Y-%m-%d")
        expected_members = universe.get_constituents_on_date(dt)
        actual_members = set(test_df[test_df["date"] == dt]["ticker"].unique())

        missing_members = expected_members - actual_members

        daily_records.append({
            "date": dt_str,
            "expected_constituents": len(expected_members),
            "actual_evaluated_tickers": len(actual_members),
            "missing_tickers_count": len(missing_members),
            "missing_tickers": ", ".join(sorted(missing_members)),
        })

        for m in missing_members:
            # Diagnose reason
            if m not in available_tickers:
                reason = "missing price series (not in data source / delisted)"
            else:
                m_raw = df[(df["ticker"] == m) & (df["date"] == dt)]
                if m_raw.empty:
                    # Check first available date
                    m_all = df[df["ticker"] == m]
                    if not m_all.empty and dt < m_all["date"].min():
                        reason = f"pre-listing date (first price on {m_all['date'].min().strftime('%Y-%m-%d')})"
                    elif not m_all.empty and dt > m_all["date"].max():
                        reason = f"post-delisting date (last price on {m_all['date'].max().strftime('%Y-%m-%d')})"
                    else:
                        reason = "missing trading session price row"
                else:
                    # In raw, but dropped from cohort: check warmup or target
                    row = m_raw.iloc[0]
                    if pd.isna(row["target"]):
                        reason = "dropped forward target (no forward session close / holiday cutoff)"
                    elif not row[tech_cols].notna().all():
                        reason = "incomplete technical indicator warmup (<50 history sessions post-entry)"
                    else:
                        reason = "unclassified filter exclusion"

            dropped_records.append({
                "date": dt_str,
                "ticker": m,
                "reason": reason,
            })

    daily_df = pd.DataFrame(daily_records)
    daily_df.to_csv("experiments/audit/daily_constituent_counts.csv", index=False)

    dropped_df = pd.DataFrame(dropped_records)
    dropped_df.to_csv("experiments/audit/dropped_ticker_days_audit.csv", index=False)

    # Per-ticker prediction counts across out-of-sample test window
    ticker_records = []
    all_test_constituents = set()
    for dt in test_dates:
        all_test_constituents.update(universe.get_constituents_on_date(dt))

    for t in sorted(all_test_constituents):
        # How many test dates was t an active constituent?
        active_days = sum(1 for dt in test_dates if universe.is_constituent(t, dt))
        evaluated_days = int((test_df["ticker"] == t).sum())
        missing_days = active_days - evaluated_days

        # Main reason
        if t not in available_tickers:
            main_reason = "Delisted / Missing data from source"
        elif missing_days > 0:
            reasons = dropped_df[dropped_df["ticker"] == t]["reason"].value_counts()
            main_reason = reasons.index[0] if len(reasons) > 0 else "None"
        else:
            main_reason = "Complete coverage"

        ticker_records.append({
            "ticker": t,
            "active_constituent_days": active_days,
            "evaluated_prediction_days": evaluated_days,
            "missing_days": missing_days,
            "coverage_pct": round(float(evaluated_days / active_days * 100) if active_days > 0 else 0, 1),
            "primary_reason": main_reason,
        })

    ticker_df = pd.DataFrame(ticker_records)
    ticker_df.to_csv("experiments/audit/per_ticker_prediction_counts.csv", index=False)

    print("\n--- FORENSIC COVERAGE RESULTS ---", flush=True)
    dates_with_50 = (daily_df["actual_evaluated_tickers"] == 50).sum()
    print(f"Total Test Dates: {len(daily_df)}", flush=True)
    print(f"Dates with exactly 50 tickers evaluated: {dates_with_50} ({dates_with_50/len(daily_df):.1%})", flush=True)
    print(f"Dates with < 50 tickers evaluated: {len(daily_df) - dates_with_50}", flush=True)
    print(f"Mean tickers evaluated per date: {daily_df['actual_evaluated_tickers'].mean():.2f}", flush=True)
    print(f"Min tickers evaluated: {daily_df['actual_evaluated_tickers'].min()}, Max: {daily_df['actual_evaluated_tickers'].max()}", flush=True)

    print("\nTop Root Causes of Dropped Ticker-Days:")
    print(dropped_df["reason"].value_counts().to_string(), flush=True)

    print("\nSample Per-Ticker Coverage:")
    print(ticker_df.head(15).to_string(index=False), flush=True)

    return daily_df, ticker_df, dropped_df


if __name__ == "__main__":
    run_coverage_analysis()
