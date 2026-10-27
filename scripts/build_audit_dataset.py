"""Build Full Point-in-Time NIFTY 50 Universe Feature Dataset for Audit Track.

Ingests all available cached raw OHLCV price series in data/raw/prices/.
Applies point-in-time constituent filtering (PointInTimeUniverse).
Computes Group A technical indicators (returns, SMA ratios, RSI, MACD, ATR, BB, vol, volume, range).
Computes forward next-session return and binary target.
Serializes to experiments/audit/master_universe_ticker_day.parquet without modifying base data folders.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from finfusionnet.data.prices import fetch_all_prices
from finfusionnet.data.trading_calendar import get_trading_calendar
from finfusionnet.data.universe import PointInTimeUniverse
from finfusionnet.features.targets import compute_targets
from finfusionnet.features.technical import compute_technical_indicators, compute_market_context


def build_audit_universe_dataset(
    start_date: str = "2020-01-01",
    end_date: str = "2026-10-01",
    output_path: str = "experiments/audit/master_universe_ticker_day.parquet",
) -> pd.DataFrame:
    print("=" * 80, flush=True)
    print("BUILDING FULL UNIVERSE AUDIT DATASET", flush=True)
    print("=" * 80, flush=True)

    universe = PointInTimeUniverse()
    tickers = sorted(list(universe.all_tickers))
    print(f"Total universe constituent tickers: {len(tickers)}", flush=True)

    # 1. Ingest cached OHLCV
    print("\n[1/5] Loading raw price series...", flush=True)
    stocks_df, nifty_df = fetch_all_prices(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        cache_dir="data/raw/prices",
        benchmark_symbol="^NSEI",
        force_download=False,
    )
    print(f"Loaded {len(stocks_df)} raw records across {stocks_df['ticker'].nunique()} tickers.", flush=True)
    print(f"Loaded {len(nifty_df)} benchmark records.", flush=True)

    # 2. Compute Group A Technical Features on raw price series (providing complete warmup)
    print("\n[2/5] Computing Group A Technical Indicators on raw series...", flush=True)
    tech_df = compute_technical_indicators(stocks_df)
    print(f"Computed technical indicators on full raw series.", flush=True)

    # 3. Market Context (Group B)
    if not nifty_df.empty:
        context_df = compute_market_context(nifty_df)
        full_df = pd.merge(tech_df, context_df, on="date", how="left")
    else:
        full_df = tech_df

    # 4. Compute Targets
    print("\n[3/5] Computing forward next-session targets...", flush=True)
    calendar = get_trading_calendar(start_date, end_date)
    targets_df = compute_targets(full_df, nifty_df=nifty_df, calendar=calendar)

    # 5. Point-in-Time Filtering (Survivorship Bias Elimination)
    print("\n[4/5] Applying Point-in-Time constituent filter...", flush=True)
    final_df = universe.filter_point_in_time(targets_df)
    print(f"Retained {len(final_df)} eligible point-in-time observations.", flush=True)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(out_file, index=False)
    print(f"\n[5/5] Saved full universe dataset to {out_file}", flush=True)
    print(f"Dataset shape: {final_df.shape}", flush=True)
    print(f"Unique tickers: {final_df['ticker'].nunique()}", flush=True)
    print(f"Date range: {final_df['date'].min().strftime('%Y-%m-%d')} to {final_df['date'].max().strftime('%Y-%m-%d')}", flush=True)

    return final_df


if __name__ == "__main__":
    build_audit_universe_dataset()
