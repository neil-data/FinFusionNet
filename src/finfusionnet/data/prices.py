"""FinFusionNet: Market Price Data Ingestion & Preprocessing.

Fetches daily OHLCV prices from Yahoo Finance for NIFTY 50 constituents
and the benchmark index (^NSEI) for 2020 through September 2026.
Caches raw files locally in Parquet format.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import yfinance as yf


def ticker_to_yfinance_symbol(ticker: str) -> str:
    """Map project ticker to Yahoo Finance symbol (e.g. RELIANCE -> RELIANCE.NS, ^NSEI -> ^NSEI)."""
    t = ticker.strip().upper()
    if t.startswith("^") or t.endswith(".NS"):
        return t
    # Handle specific Yahoo ticker quirks if any (e.g., M&M -> M&M.NS)
    return f"{t}.NS"


def yfinance_symbol_to_ticker(symbol: str) -> str:
    """Map Yahoo Finance symbol back to clean project ticker."""
    s = symbol.strip().upper()
    if s.endswith(".NS"):
        return s[:-3]
    return s


def fetch_single_ticker(
    ticker: str,
    start_date: str = "2020-01-01",
    end_date: str = "2026-10-01",
    cache_dir: Optional[Union[str, Path]] = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """Fetch OHLCV data for a single ticker with local caching."""
    clean_ticker = yfinance_symbol_to_ticker(ticker)
    yf_symbol = ticker_to_yfinance_symbol(clean_ticker)

    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{clean_ticker}.parquet"
        if cache_path.exists() and not force_download:
            try:
                df = pd.read_parquet(cache_path)
                return df
            except Exception:
                pass  # Fall back to download if cache is corrupt

    # Download from yfinance
    try:
        raw = yf.download(
            yf_symbol,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
        )
    except Exception as e:
        print(f"Failed to fetch {ticker} ({yf_symbol}): {e}")
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    # Flatten multi-level columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0].lower() for col in raw.columns]
    else:
        raw.columns = [col.lower() for col in raw.columns]

    df = raw.reset_index()
    # Normalize date column
    date_col = "Date" if "Date" in df.columns else "date"
    df["date"] = pd.to_datetime(df[date_col]).dt.normalize()
    df["ticker"] = clean_ticker

    # Ensure required columns
    required = ["date", "ticker", "open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    df = df[required].dropna(subset=["close"]).sort_values("date").reset_index(drop=True)

    if cache_dir is not None and not df.empty:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)

    return df


def fetch_all_prices(
    tickers: List[str],
    start_date: str = "2020-01-01",
    end_date: str = "2026-10-01",
    cache_dir: Optional[Union[str, Path]] = "data/raw/prices",
    benchmark_symbol: str = "^NSEI",
    force_download: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch prices for all tickers in universe plus the benchmark index.

    Returns:
    - stocks_df: Combined long DataFrame [date, ticker, open, high, low, close, volume]
    - nifty_df: Benchmark DataFrame [date, open, high, low, close, volume]
    """
    stocks_dfs = []
    for t in tickers:
        df_t = fetch_single_ticker(
            t,
            start_date=start_date,
            end_date=end_date,
            cache_dir=cache_dir,
            force_download=force_download,
        )
        if not df_t.empty:
            stocks_dfs.append(df_t)

    stocks_df = pd.concat(stocks_dfs, ignore_index=True) if stocks_dfs else pd.DataFrame()

    # Fetch benchmark index
    nifty_df = fetch_single_ticker(
        benchmark_symbol,
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
        force_download=force_download,
    )
    if not nifty_df.empty:
        nifty_df = nifty_df.drop(columns=["ticker"], errors="ignore")

    return stocks_df, nifty_df
