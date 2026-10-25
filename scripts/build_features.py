"""FinFusionNet CLI: Build Master Ticker-Day Dataset and Features.

Executes:
1. Ingestion of raw OHLCV price series for NIFTY 50 universe and benchmark (^NSEI).
2. Point-in-time constituent filtering (preventing survivorship bias).
3. Group A Causal Technical Feature Engineering (returns, trend, momentum, volatility, volume, price action).
4. Group B Market Context calculation from NIFTY 50 index.
5. Group C, D, E News Availability & Sentiment Integration (strict null preservation for no-news days).
6. Primary & Secondary Target Formulation (15:30 IST market close cutoff, flat return = 0).
7. Serialization to Parquet: data/processed/master_dataset/master_ticker_day.parquet.
"""

import argparse
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd

from finfusionnet.data.prices import fetch_all_prices
from finfusionnet.data.trading_calendar import get_trading_calendar
from finfusionnet.data.universe import PointInTimeUniverse
from finfusionnet.features.sentiment import integrate_sentiment_features
from finfusionnet.features.targets import compute_targets
from finfusionnet.features.technical import (
    compute_technical_indicators,
    compute_market_context,
)


def build_master_dataset(
    start_date: str = "2020-01-01",
    end_date: str = "2026-10-01",
    tickers: Optional[List[str]] = None,
    output_path: str = "data/processed/master_dataset/master_ticker_day.parquet",
    force_download: bool = False,
) -> pd.DataFrame:
    print("=================================================================")
    print("FinFusionNet: Building Master Ticker-Day Feature Dataset")
    print("=================================================================")

    universe = PointInTimeUniverse()
    if tickers is None:
        # Use complete 62 in-window constituent tickers
        target_tickers = sorted(list(universe.all_tickers))
    else:
        target_tickers = sorted(list(tickers))

    print(f"Target universe: {len(target_tickers)} tickers.")

    # 1. Fetch raw prices
    print("\n[1/6] Ingesting OHLCV market data...")
    stocks_df, nifty_df = fetch_all_prices(
        tickers=target_tickers,
        start_date=start_date,
        end_date=end_date,
        benchmark_symbol="^NSEI",
        force_download=force_download,
    )
    print(f"Ingested {len(stocks_df)} raw stock rows across {stocks_df['ticker'].nunique()} tickers.")
    print(f"Ingested {len(nifty_df)} benchmark rows.")

    if stocks_df.empty:
        raise ValueError("Failed to fetch market data.")

    # 2. Compute Group A Technical Features on raw price series (providing complete warmup)
    print("\n[2/6] Computing causal Group A technical indicators on raw price series...")
    tech_df = compute_technical_indicators(stocks_df)
    print(f"Computed 24 technical indicators.")

    # 3. Compute Group B Market Context
    print("\n[3/6] Computing Group B market context from NIFTY 50...")
    context_df = compute_market_context(nifty_df)
    full_df = pd.merge(tech_df, context_df, on="date", how="left")

    # 4. Integrate News Availability & Sentiment (Groups C, D, E)
    print("\n[4/6] Integrating news availability and sentiment features...")
    news_path = Path("data/interim/news/ticker_day_news_sentiment.parquet")
    if news_path.exists():
        news_sentiment_df = pd.read_parquet(news_path)
        print(f"Loaded {len(news_sentiment_df)} sentiment records from {news_path}.")
    else:
        news_sentiment_df = None
        print("No interim news sentiment parquet found. Preserving null sentiment policy for no-news days.")

    master_df = integrate_sentiment_features(full_df, news_sentiment_df)

    # 5. Target Formulation
    print("\n[5/6] Computing primary and secondary targets...")
    cal = get_trading_calendar(start_date, end_date)
    targets_df = compute_targets(master_df, nifty_df=nifty_df, calendar=cal)

    # 6. Point-in-time filtering (Survivorship Bias Elimination)
    print("\n[6/6] Applying dynamic point-in-time constituent filter...")
    initial_rows = len(targets_df)
    final_df = universe.filter_point_in_time(targets_df)
    print(f"Retained {len(final_df)} eligible observations (dropped {initial_rows - len(final_df)} non-constituent rows).")

    # Save to Parquet
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(out_file, index=False)
    print(f"\nMaster dataset successfully serialized to {out_file}.")
    print(f"Total observations: {len(final_df)} rows | Columns: {len(final_df.columns)}")

    return final_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FinFusionNet Master Feature Dataset")
    parser.add_argument("--start-date", type=str, default="2020-01-01")
    parser.add_argument("--end-date", type=str, default="2026-10-01")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    build_master_dataset(
        start_date=args.start_date,
        end_date=args.end_date,
        force_download=args.force_download,
    )
