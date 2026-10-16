"""FinFusionNet: National Stock Exchange of India (NSE) Trading Calendar.

Provides official NSE trading day resolution, exchange holiday calendar,
next valid trading session lookup, and timestamp alignment (15:30 IST cutoff).
"""

from datetime import datetime, time, date
from typing import List, Optional, Set, Union
import pandas as pd
import numpy as np


# Official NSE Holidays (2020 - 2026) for Capital Market Segment
# Includes gazetted holidays and declared trading closures.
NSE_HOLIDAYS: Set[str] = {
    # 2020
    "2020-02-21", "2020-03-10", "2020-03-25", "2020-04-02", "2020-04-06",
    "2020-04-10", "2020-04-14", "2020-05-01", "2020-05-25", "2020-10-02",
    "2020-11-16", "2020-11-30", "2020-12-25",
    # 2021
    "2021-01-26", "2021-03-11", "2021-03-29", "2021-04-02", "2021-04-14",
    "2021-04-21", "2021-05-13", "2021-07-21", "2021-08-19", "2021-09-10",
    "2021-10-15", "2021-11-05", "2021-11-19",
    # 2022
    "2022-01-26", "2022-03-01", "2022-03-18", "2022-04-14", "2022-04-15",
    "2022-05-03", "2022-08-09", "2022-08-15", "2022-08-31", "2022-10-05",
    "2022-10-24", "2022-10-26", "2022-11-08",
    # 2023
    "2023-01-26", "2023-03-07", "2023-03-30", "2023-04-04", "2023-04-07",
    "2023-04-14", "2023-05-01", "2023-06-29", "2023-08-15", "2023-09-19",
    "2023-10-02", "2023-10-24", "2023-11-14", "2023-11-27", "2023-12-25",
    # 2024
    "2024-01-22", "2024-01-26", "2024-03-08", "2024-03-25", "2024-03-29",
    "2024-04-11", "2024-04-17", "2024-05-01", "2024-05-20", "2024-06-17",
    "2024-07-17", "2024-08-15", "2024-10-02", "2024-11-01", "2024-11-15",
    "2024-11-20", "2024-12-25",
    # 2025
    "2025-02-26", "2025-03-14", "2025-03-31", "2025-04-10", "2025-04-14",
    "2025-04-18", "2025-05-01", "2025-06-07", "2025-08-15", "2025-08-27",
    "2025-10-02", "2025-10-21", "2025-10-22", "2025-11-05", "2025-12-25",
    # 2026 (Projected / Official through September 2026 freeze)
    "2026-01-26", "2026-02-16", "2026-03-04", "2026-03-20", "2026-04-03",
    "2026-04-14", "2026-05-01", "2026-05-27", "2026-08-15", "2026-08-26",
}


def get_trading_calendar(
    start_date: Union[str, pd.Timestamp] = "2020-01-01",
    end_date: Union[str, pd.Timestamp] = "2026-09-30",
) -> pd.DatetimeIndex:
    """Generate the official list of valid NSE trading sessions between start and end date.

    Excludes Saturdays, Sundays, and official NSE holidays.
    """
    all_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    trading_dates = [
        d for d in all_dates
        if d.weekday() < 5 and d.strftime("%Y-%m-%d") not in NSE_HOLIDAYS
    ]
    return pd.DatetimeIndex(trading_dates).sort_values()


def is_trading_session(
    dt: Union[str, date, datetime, pd.Timestamp],
    calendar: Optional[pd.DatetimeIndex] = None,
) -> bool:
    """Check if a given date is an active NSE trading session."""
    ts = pd.Timestamp(dt).normalize()
    if calendar is not None:
        return ts in calendar
    if ts.weekday() >= 5:
        return False
    return ts.strftime("%Y-%m-%d") not in NSE_HOLIDAYS


def get_next_trading_session(
    current_date: Union[str, date, datetime, pd.Timestamp],
    calendar: Optional[pd.DatetimeIndex] = None,
) -> pd.Timestamp:
    """Return the next valid NSE trading session following current_date.

    Never assumes the next calendar day is a trading day.
    """
    ts = pd.Timestamp(current_date).normalize()
    if calendar is None:
        calendar = get_trading_calendar(ts, ts + pd.Timedelta(days=30))

    future_sessions = calendar[calendar > ts]
    if len(future_sessions) == 0:
        # Extend calendar if needed
        calendar = get_trading_calendar(ts, ts + pd.Timedelta(days=60))
        future_sessions = calendar[calendar > ts]
        if len(future_sessions) == 0:
            raise ValueError(f"No future trading sessions found after {ts.strftime('%Y-%m-%d')}")

    return future_sessions[0]


def get_previous_trading_session(
    current_date: Union[str, date, datetime, pd.Timestamp],
    calendar: Optional[pd.DatetimeIndex] = None,
) -> pd.Timestamp:
    """Return the previous valid NSE trading session prior to current_date."""
    ts = pd.Timestamp(current_date).normalize()
    if calendar is None:
        calendar = get_trading_calendar(ts - pd.Timedelta(days=30), ts)

    past_sessions = calendar[calendar < ts]
    if len(past_sessions) == 0:
        calendar = get_trading_calendar(ts - pd.Timedelta(days=60), ts)
        past_sessions = calendar[calendar < ts]
        if len(past_sessions) == 0:
            raise ValueError(f"No past trading sessions found before {ts.strftime('%Y-%m-%d')}")

    return past_sessions[-1]


def align_timestamp_to_session(
    event_timestamp: Union[datetime, pd.Timestamp],
    market_close: time = time(15, 30),
    calendar: Optional[pd.DatetimeIndex] = None,
) -> pd.Timestamp:
    """Align an event/news timestamp (in Asia/Kolkata timezone) to its applicable trading session.

    Protocol:
    1. If event occurs on a trading day at or before 15:30 IST -> session is this trading day.
    2. If event occurs on a trading day after 15:30 IST -> session rolls to next valid trading day.
    3. If event occurs on a weekend or holiday -> session rolls to next valid trading day.
    """
    ts = pd.Timestamp(event_timestamp)
    event_date = ts.normalize()
    event_time = ts.time()

    if calendar is None:
        calendar = get_trading_calendar(
            event_date - pd.Timedelta(days=5),
            event_date + pd.Timedelta(days=30),
        )

    is_today_trading = is_trading_session(event_date, calendar=calendar)

    if is_today_trading:
        if event_time <= market_close:
            return event_date
        else:
            return get_next_trading_session(event_date, calendar=calendar)
    else:
        # Weekend or holiday -> rolls to next session
        future_sessions = calendar[calendar > event_date]
        if len(future_sessions) == 0:
            calendar = get_trading_calendar(event_date, event_date + pd.Timedelta(days=30))
            future_sessions = calendar[calendar > event_date]
        return future_sessions[0]
