"""FinFusionNet: Realized Volatility Target Formulation.

Computes next-trading-session realized volatility using the Parkinson (1980)
high-low estimator, annualized by sqrt(252).

Mathematical formulation:
sigma_{Parkinson, t+1} = sqrt( (ln(H_{t+1} / L_{t+1}))^2 / (4 * ln(2)) ) * sqrt(252)

Guarantees:
1. Target date is strictly determined via the authenticated NSE trading calendar.
2. High >= Low > 0 strictly verified; ratio clipped to >= 1.0 to prevent negative log.
3. Extreme outlier spikes (bad ticks / abnormal splits) winsorized at 99.9th percentile (~2.0)
   to ensure numerical stability in neural network regression.
4. Preserves missing data at universe boundaries (target = NaN when session t+1 is unobserved).
"""

from typing import Optional
import numpy as np
import pandas as pd

from finfusionnet.data.trading_calendar import (
    get_trading_calendar,
    get_next_trading_session,
)


def compute_next_session_parkinson_volatility(
    df: pd.DataFrame,
    date_col: str = "date",
    ticker_col: str = "ticker",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    target_date_col: str = "target_date",
    max_vol_clip: float = 2.0,
    calendar: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """Compute annualized next-session Parkinson realized volatility target."""
    if df.empty:
        return df

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    if target_date_col not in df.columns:
        if calendar is None:
            min_date = df[date_col].min() - pd.Timedelta(days=5)
            max_date = df[date_col].max() + pd.Timedelta(days=30)
            calendar = get_trading_calendar(min_date, max_date)

        unique_dates = pd.DatetimeIndex(sorted(df[date_col].unique()))
        next_session_map = {}
        for d in unique_dates:
            try:
                nxt = get_next_trading_session(d, calendar=calendar)
                next_session_map[d] = nxt
            except ValueError:
                next_session_map[d] = pd.NaT

        df[target_date_col] = df[date_col].map(next_session_map)
    else:
        df[target_date_col] = pd.to_datetime(df[target_date_col])

    # Lookup forward session high and low by matching (ticker, target_date)
    lookup = df[[date_col, ticker_col, high_col, low_col]].rename(
        columns={
            date_col: target_date_col,
            high_col: "_next_high",
            low_col: "_next_low",
        }
    )
    df = df.merge(lookup, on=[ticker_col, target_date_col], how="left")

    # Validate H >= L > 0
    valid_hl = (
        df["_next_high"].notna()
        & df["_next_low"].notna()
        & (df["_next_low"] > 1e-4)
        & (df["_next_high"] >= df["_next_low"])
    )

    hl_ratio = np.where(
        valid_hl,
        df["_next_high"] / np.maximum(df["_next_low"], 1e-6),
        np.nan,
    )
    hl_ratio = np.maximum(hl_ratio, 1.0)

    # Parkinson (1980) annualized realized volatility
    parkinson_vol = np.sqrt((np.log(hl_ratio) ** 2) / (4.0 * np.log(2.0))) * np.sqrt(252.0)

    # Robust winsorization at max_vol_clip (e.g. 200% annualized vol)
    if max_vol_clip is not None:
        parkinson_vol = np.clip(parkinson_vol, 0.0, max_vol_clip)

    df["target_realized_vol"] = parkinson_vol
    df = df.drop(columns=["_next_high", "_next_low"])

    return df
