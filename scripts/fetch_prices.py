"""FinFusionNet CLI: Fetch Market Price Data for NIFTY 50 Universe.

Ingests daily OHLCV price history from Yahoo Finance across the entire
2020 through September 2026 window and caches Parquet files locally in data/raw/prices/.
"""

import argparse
from finfusionnet.data.prices import fetch_all_prices
from finfusionnet.data.universe import PointInTimeUniverse


def main():
    parser = argparse.ArgumentParser(description="Fetch historical OHLCV data for NIFTY 50 universe")
    parser.add_argument("--start-date", type=str, default="2020-01-01", help="Warm-up start date")
    parser.add_argument("--end-date", type=str, default="2026-10-01", help="Freeze end date")
    parser.add_argument("--force", action="store_true", help="Force redownload ignoring local cache")
    args = parser.parse_args()

    universe = PointInTimeUniverse()
    tickers = sorted(list(universe.all_tickers))

    print(f"Ingesting OHLCV price series for {len(tickers)} universe tickers + benchmark (^NSEI)...")
    print(f"Date range: {args.start_date} to {args.end_date}")

    stocks_df, nifty_df = fetch_all_prices(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        cache_dir="data/raw/prices",
        force_download=args.force,
    )

    print(f"\nFetch complete:")
    print(f"- Stocks records: {len(stocks_df)} rows across {stocks_df['ticker'].nunique()} tickers.")
    print(f"- Benchmark records: {len(nifty_df)} rows.")


if __name__ == "__main__":
    main()
