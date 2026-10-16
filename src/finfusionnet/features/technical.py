"""FinFusionNet: Causal Technical Feature Engineering.

Implements Group A technical indicators using strictly backward-looking
rolling windows. Strictly prevents centered windows, future prices, or lookahead leakage.
"""

from typing import List, Optional
import pandas as pd
import numpy as np


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate causal relative strength index (RSI) using exponential moving average (Wilder's)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing via ewm(alpha=1/period, adjust=False)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Calculate causal average true range (ATR)."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr


def compute_technical_indicators(
    df: pd.DataFrame,
    date_col: str = "date",
    ticker_col: str = "ticker",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
) -> pd.DataFrame:
    """Compute Group A technical indicators for each ticker independently.

    Features computed:
    - Returns: return_1d, return_2d, return_5d, return_10d, return_20d
    - Trend: close_sma_5_ratio, close_sma_10_ratio, close_sma_20_ratio, close_sma_50_ratio, sma_5_sma_20_ratio
    - Momentum: rsi_14, macd, macd_signal, macd_histogram
    - Volatility: atr_14_pct, bb_width_20, volatility_10, volatility_20
    - Volume: volume_change_1d, volume_ratio_5, volume_ratio_20
    - Price action: high_low_range_pct, close_open_return, close_position_in_range
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([ticker_col, date_col]).reset_index(drop=True)

    grouped = df.groupby(ticker_col, group_keys=False)

    # 1. Returns (causal backward pct_change)
    df["return_1d"] = grouped[close_col].pct_change(1)
    df["return_2d"] = grouped[close_col].pct_change(2)
    df["return_5d"] = grouped[close_col].pct_change(5)
    df["return_10d"] = grouped[close_col].pct_change(10)
    df["return_20d"] = grouped[close_col].pct_change(20)

    # 2. Trend (SMA ratios)
    sma_5 = grouped[close_col].transform(lambda s: s.rolling(5, min_periods=5).mean())
    sma_10 = grouped[close_col].transform(lambda s: s.rolling(10, min_periods=10).mean())
    sma_20 = grouped[close_col].transform(lambda s: s.rolling(20, min_periods=20).mean())
    sma_50 = grouped[close_col].transform(lambda s: s.rolling(50, min_periods=50).mean())

    df["close_sma_5_ratio"] = df[close_col] / sma_5
    df["close_sma_10_ratio"] = df[close_col] / sma_10
    df["close_sma_20_ratio"] = df[close_col] / sma_20
    df["close_sma_50_ratio"] = df[close_col] / sma_50
    df["sma_5_sma_20_ratio"] = sma_5 / sma_20

    # 3. Momentum
    df["rsi_14"] = grouped[close_col].transform(lambda s: compute_rsi(s, 14))

    # MACD (12, 26, 9)
    ema_12 = grouped[close_col].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    ema_26 = grouped[close_col].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = (
        df.groupby(ticker_col, group_keys=False)["macd"]
        .transform(lambda s: s.ewm(span=9, adjust=False).mean())
    )
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    # 4. Volatility
    # ATR 14 percentage of close
    atr_list = []
    for _, group in df.groupby(ticker_col, sort=False):
        atr_series = compute_atr(group[high_col], group[low_col], group[close_col], 14)
        atr_list.append(atr_series)
    atr_vals = pd.concat(atr_list).sort_index()
    df["atr_14_pct"] = atr_vals / df[close_col]

    # Bollinger Band width 20 (std=2)
    rolling_std_20 = grouped[close_col].transform(lambda s: s.rolling(20, min_periods=20).std())
    upper_bb = sma_20 + (2.0 * rolling_std_20)
    lower_bb = sma_20 - (2.0 * rolling_std_20)
    df["bb_width_20"] = (upper_bb - lower_bb) / sma_20

    # Realized close-to-close volatility (annualized = std * sqrt(252))
    df["volatility_10"] = grouped["return_1d"].transform(
        lambda s: s.rolling(10, min_periods=10).std() * np.sqrt(252)
    )
    df["volatility_20"] = grouped["return_1d"].transform(
        lambda s: s.rolling(20, min_periods=20).std() * np.sqrt(252)
    )

    # 5. Volume dynamics
    df["volume_change_1d"] = grouped[volume_col].pct_change(1)
    vol_sma_5 = grouped[volume_col].transform(lambda s: s.rolling(5, min_periods=5).mean())
    vol_sma_20 = grouped[volume_col].transform(lambda s: s.rolling(20, min_periods=20).mean())
    df["volume_ratio_5"] = df[volume_col] / (vol_sma_5 + 1e-10)
    df["volume_ratio_20"] = df[volume_col] / (vol_sma_20 + 1e-10)

    # 6. Intraday price action
    df["high_low_range_pct"] = (df[high_col] - df[low_col]) / df[close_col]
    df["close_open_return"] = (df[close_col] - df[open_col]) / (df[open_col] + 1e-10)
    hl_spread = df[high_col] - df[low_col]
    df["close_position_in_range"] = np.where(
        hl_spread > 0,
        (df[close_col] - df[low_col]) / (hl_spread + 1e-10),
        0.5,
    )

    # Clean infinite values to null
    num_cols = [
        "return_1d", "return_2d", "return_5d", "return_10d", "return_20d",
        "close_sma_5_ratio", "close_sma_10_ratio", "close_sma_20_ratio", "close_sma_50_ratio",
        "sma_5_sma_20_ratio", "rsi_14", "macd", "macd_signal", "macd_histogram",
        "atr_14_pct", "bb_width_20", "volatility_10", "volatility_20",
        "volume_change_1d", "volume_ratio_5", "volume_ratio_20",
        "high_low_range_pct", "close_open_return", "close_position_in_range",
    ]
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    return df


def compute_market_context(
    nifty_df: pd.DataFrame,
    date_col: str = "date",
    close_col: str = "close",
) -> pd.DataFrame:
    """Compute Group B market context features using NIFTY 50 index prices.

    Features computed:
    - nifty_return_1d, nifty_return_5d, nifty_return_20d
    - nifty_volatility_10, nifty_volatility_20
    """
    nifty = nifty_df.copy()
    nifty[date_col] = pd.to_datetime(nifty[date_col])
    nifty = nifty.sort_values(date_col).reset_index(drop=True)

    nifty["nifty_return_1d"] = nifty[close_col].pct_change(1)
    nifty["nifty_return_5d"] = nifty[close_col].pct_change(5)
    nifty["nifty_return_20d"] = nifty[close_col].pct_change(20)

    nifty["nifty_volatility_10"] = (
        nifty["nifty_return_1d"].rolling(10, min_periods=10).std() * np.sqrt(252)
    )
    nifty["nifty_volatility_20"] = (
        nifty["nifty_return_1d"].rolling(20, min_periods=20).std() * np.sqrt(252)
    )

    context_cols = [
        date_col,
        "nifty_return_1d", "nifty_return_5d", "nifty_return_20d",
        "nifty_volatility_10", "nifty_volatility_20",
    ]
    return nifty[context_cols].replace([np.inf, -np.inf], np.nan)
