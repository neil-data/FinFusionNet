"""FinFusionNet: Prediction Target Formulation.

Computes next-trading-session return, primary binary direction label,
and secondary market-relative direction label.
Strictly maps exactly zero return to class 0.
"""

from typing import Optional, Tuple, Union
import pandas as pd
import numpy as np

from finfusionnet.data.trading_calendar import (
    get_trading_calendar,
    get_next_trading_session,
)


def compute_targets(
    df: pd.DataFrame,
    nifty_df: Optional[pd.DataFrame] = None,
    date_col: str = "date",
    ticker_col: str = "ticker",
    close_col: str = "close",
    calendar: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """Compute primary next-session direction target and secondary relative target.

    Mathematical formulation:
    R_{i, t+1} = (Close_{i, t+1} / Close_{i, t}) - 1
    Y_{i, t} = 1 if R_{i, t+1} > 0 else 0 (flat return strictly maps to 0)

    Secondary relative target:
    R_rel = R_{i, t+1} - R_{NIFTY, t+1}
    Y_rel = 1 if R_rel > 0 else 0

    Target date is determined strictly via the NSE trading calendar,
    never by simple calendar-day shifting.
    """
    if df.empty:
        return df

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    if calendar is None:
        min_date = df[date_col].min() - pd.Timedelta(days=5)
        max_date = df[date_col].max() + pd.Timedelta(days=30)
        calendar = get_trading_calendar(min_date, max_date)

    # Sort to guarantee chronological consistency
    df = df.sort_values([ticker_col, date_col]).reset_index(drop=True)

    # Build calendar next-session mapping
    unique_dates = pd.DatetimeIndex(sorted(df[date_col].unique()))
    next_session_map = {}
    for d in unique_dates:
        try:
            nxt = get_next_trading_session(d, calendar=calendar)
            next_session_map[d] = nxt
        except ValueError:
            next_session_map[d] = pd.NaT

    df["target_date"] = df[date_col].map(next_session_map)

    # Compute forward price for each ticker by matching target_date
    lookup = df[[date_col, ticker_col, close_col]].rename(
        columns={date_col: "target_date", close_col: "next_close"}
    )
    df = df.merge(lookup, on=["ticker", "target_date"], how="left")

    # Compute returns
    df["next_session_return"] = (df["next_close"] / df[close_col]) - 1.0

    # Primary target: positive return -> 1, negative or zero -> 0
    # Crucial: exactly zero return maps to 0
    df["target"] = np.where(
        df["next_session_return"].isna(),
        np.nan,
        np.where(df["next_session_return"] > 0.0, 1, 0),
    )

    # Secondary target: market-relative return
    if nifty_df is not None and not nifty_df.empty:
        nifty = nifty_df.copy()
        nifty[date_col] = pd.to_datetime(nifty[date_col])
        nifty = nifty.sort_values(date_col).reset_index(drop=True)

        nifty_lookup = nifty[[date_col, close_col]].rename(
            columns={date_col: "target_date", close_col: "nifty_next_close"}
        )
        nifty["target_date"] = nifty[date_col].map(next_session_map)
        nifty = nifty.merge(nifty_lookup, on="target_date", how="left")
        nifty["nifty_next_session_return"] = (
            nifty["nifty_next_close"] / nifty[close_col]
        ) - 1.0

        nifty_map = dict(zip(nifty[date_col], nifty["nifty_next_session_return"]))
        df["nifty_next_session_return"] = df[date_col].map(nifty_map)

        df["relative_return"] = (
            df["next_session_return"] - df["nifty_next_session_return"]
        )
        df["relative_target"] = np.where(
            df["relative_return"].isna(),
            np.nan,
            np.where(df["relative_return"] > 0.0, 1, 0),
        )
    else:
        df["nifty_next_session_return"] = np.nan
        df["relative_return"] = np.nan
        df["relative_target"] = np.nan

    # Drop interim next_close column
    df = df.drop(columns=["next_close"])

    return df
