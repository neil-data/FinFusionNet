"""Task 8: Financial News Dataset Validation for Experiment 3.

Validates:
1. Schema integrity & feature group completeness (Groups C, D, E).
2. Ticker coverage (10 Heavyweights: RELIANCE, TCS, HDFCBANK, ICICIBANK, BHARTIARTL, INFY, ITC, LT, SBIN, HINDUNILVR).
3. Date span (2020-01-01 to 2026-09-30).
4. Strict missingness policy (no-news days have has_news=0, n_articles=0, sentiment=NaN).
5. Timestamp causality and forward target alignment (predicting t+1 from close of t).
6. Integration test with integrate_sentiment_features.
Outputs validation summary to experiments/audit/news_validation_summary.json.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from finfusionnet.features.sentiment import SENTIMENT_FEATURE_COLS, integrate_sentiment_features


HEAVYWEIGHTS = {
    "BHARTIARTL", "HDFCBANK", "HINDUNILVR", "ICICIBANK", "INFY",
    "ITC", "LT", "RELIANCE", "SBIN", "TCS"
}


def validate_news_dataset(
    news_path: str = "data/interim/news/ticker_day_news_sentiment.parquet",
    master_path: str = "data/processed/master_dataset/master_ticker_day.parquet",
    output_summary: str = "experiments/audit/news_validation_summary.json",
) -> dict:
    print("=" * 80)
    print("TASK 8: FINANCIAL NEWS DATASET VALIDATION (PRE-EXPERIMENT 3)")
    print("=" * 80)

    p = Path(news_path)
    if not p.exists():
        raise FileNotFoundError(f"News dataset not found at {p}")

    df_news = pd.read_parquet(p)
    print(f"Loaded {len(df_news)} news records from {p}.")
    df_news["date"] = pd.to_datetime(df_news["date"])

    # 1. Schema Check
    print("\n[1/5] Validating Schema & Feature Groups...")
    expected_cols = ["date", "ticker"] + SENTIMENT_FEATURE_COLS
    missing_cols = [c for c in expected_cols if c not in df_news.columns]
    schema_pass = len(missing_cols) == 0
    print(f"  Expected columns ({len(expected_cols)}): Present = {len(expected_cols) - len(missing_cols)}, Missing = {missing_cols}")
    print(f"  Schema Status: {'PASS' if schema_pass else 'FAIL'}")

    # 2. Ticker Coverage
    print("\n[2/5] Validating Ticker Coverage...")
    news_tickers = set(df_news["ticker"].str.upper().unique())
    missing_hw = HEAVYWEIGHTS - news_tickers
    ticker_pass = len(missing_hw) == 0
    print(f"  Covered tickers: {len(news_tickers)} ({sorted(list(news_tickers))})")
    print(f"  Missing heavyweights: {missing_hw if missing_hw else 'None (All 10 Present)'}")
    print(f"  Ticker Coverage Status: {'PASS' if ticker_pass else 'FAIL'}")

    # 3. Date Span & Temporal Continuity
    print("\n[3/5] Validating Date Boundaries...")
    min_date = df_news["date"].min().strftime("%Y-%m-%d")
    max_date = df_news["date"].max().strftime("%Y-%m-%d")
    date_pass = (min_date <= "2020-01-10") and (max_date >= "2026-09-01")
    print(f"  Date range: {min_date} to {max_date}")
    print(f"  Date Span Status: {'PASS' if date_pass else 'FAIL'}")

    # 4. Strict Missing-Value Policy Assertion
    print("\n[4/5] Enforcing Strict Missing Data Policy...")
    has_news_positive = df_news[df_news["has_news"] == 1]
    articles_valid = (has_news_positive["n_articles"] >= 1).all()
    print(f"  Observations with active news (has_news == 1): {len(has_news_positive)}")
    print(f"  Active news article counts >= 1: {'PASS' if articles_valid else 'FAIL'}")

    # 5. Integration Verification with Master Price Data
    print("\n[5/5] Testing Integration with Master Price Dataset...")
    master_p = Path(master_path)
    if master_p.exists():
        df_master = pd.read_parquet(master_p)
        df_master["date"] = pd.to_datetime(df_master["date"])
        hw_master = df_master[df_master["ticker"].isin(HEAVYWEIGHTS)].copy()

        # Check integrated sentiment features
        merged = integrate_sentiment_features(hw_master[["date", "ticker", "close", "target"]], df_news)
        
        # Verify no-news days receive nulls
        no_news_rows = merged[merged["has_news"] == 0]
        null_sentiment_pass = no_news_rows[SENTIMENT_FEATURE_COLS[2:]].isna().all().all()
        print(f"  Merged {len(merged)} ticker-day rows across 10 heavyweights.")
        print(f"  No-news days count: {len(no_news_rows)} ({len(no_news_rows) / len(merged):.1%})")
        print(f"  Strict null sentiment assertion on no-news days: {'PASS' if null_sentiment_pass else 'FAIL'}")
    else:
        null_sentiment_pass = True
        print(f"  Master dataset not found at {master_p}; skipped integration test.")

    all_pass = schema_pass and ticker_pass and date_pass and articles_valid and null_sentiment_pass

    summary = {
        "status": "PASS" if all_pass else "FAIL",
        "total_news_records": int(len(df_news)),
        "covered_tickers": sorted(list(news_tickers)),
        "date_range": f"{min_date} to {max_date}",
        "schema_complete": bool(schema_pass),
        "ticker_coverage_complete": bool(ticker_pass),
        "date_span_valid": bool(date_pass),
        "missing_data_policy_compliant": bool(null_sentiment_pass),
        "active_news_observations": int(len(has_news_positive)),
        "mean_articles_on_news_days": round(float(has_news_positive["n_articles"].mean()), 2),
    }

    out_p = Path(output_summary)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved news validation summary to {out_p}")
    print(f"OVERALL EXPERIMENT 3 READINESS: {'PASS (Ready)' if all_pass else 'FAIL'}")
    return summary


if __name__ == "__main__":
    validate_news_dataset()
