"""FinFusionNet: Point-in-Time NIFTY 50 Universe Tracker.

Provides dynamic, survivorship-bias-free constituent membership tracking
for Indian equity research spanning 2020 through September 2026.
Based on historical index reconstitution events documented in Dataset Design.
"""

from typing import Dict, List, Optional, Set, Tuple, Union
import pandas as pd
import numpy as np


# In-window NIFTY 50 Rebalance Events (effective date = first trading day of new status)
# Documented in docs/Dataset Design.docx
REBALANCES: List[Dict[str, Union[str, List[str]]]] = [
    {
        "effective_date": "2021-03-31",
        "in": ["TATACONSUM"],
        "out": ["GAIL"],
    },
    {
        "effective_date": "2022-03-31",
        "in": ["APOLLOHOSP"],
        "out": ["IOC"],
    },
    {
        "effective_date": "2022-09-30",
        "in": ["ADANIENT"],
        "out": ["SHREECEM"],
    },
    {
        "effective_date": "2023-07-13",
        "in": ["LTIM"],
        "out": ["HDFC"],
    },
    {
        "effective_date": "2024-03-28",
        "in": ["SHRIRAMFIN"],
        "out": ["UPL"],
    },
    {
        "effective_date": "2024-09-30",
        "in": ["BEL", "TRENT"],
        "out": ["DIVISLAB", "LTIM"],
    },
    {
        "effective_date": "2025-03-28",
        "in": ["JIOFIN", "ETERNAL"],
        "out": ["BPCL", "BRITANNIA"],
    },
    {
        "effective_date": "2025-09-30",
        "in": ["INDIGO", "MAXHEALTH"],
        "out": ["HEROMOTOCO", "INDUSINDBK"],
    },
    {
        "effective_date": "2026-09-30",
        "in": ["BSE"],
        "out": ["WIPRO"],
    },
]

# Base 50 constituents as of 2021-01-01 (prior to 2021-03-31 rebalance)
BASE_CONSTITUENTS_2021: Set[str] = {
    "ADANIPORTS", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE",
    "BAJAJFINSV", "BPCL", "BHARTIARTL", "BRITANNIA", "CIPLA",
    "COALINDIA", "DIVISLAB", "DRREDDY", "EICHERMOT", "GAIL",
    "GRASIM", "HCLTECH", "HDFC", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "ITC",
    "IOC", "INDUSINDBK", "INFY", "JSWSTEEL", "KOTAKBANK",
    "LT", "M&M", "MARUTI", "NTPC", "NESTLEIND",
    "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SHREECEM",
    "SBIN", "SUNPHARMA", "TCS", "TATAMOTORS", "TATASTEEL",
    "TECHM", "TITAN", "UPL", "ULTRACEMCO", "WIPRO"
}


class PointInTimeUniverse:
    """Manages exact historical NIFTY 50 index constituent membership."""

    def __init__(self):
        self.base_constituents = set(BASE_CONSTITUENTS_2021)
        self.rebalances = sorted(REBALANCES, key=lambda x: str(x["effective_date"]))
        self._all_tickers = self._compute_all_in_window_tickers()

    def _compute_all_in_window_tickers(self) -> Set[str]:
        """Collect all tickers that were in the NIFTY 50 at any point during 2020-2026."""
        all_t = set(self.base_constituents)
        for r in self.rebalances:
            for t_in in r["in"]:
                all_t.add(t_in)
            for t_out in r["out"]:
                all_t.add(t_out)
        return all_t

    @property
    def all_tickers(self) -> Set[str]:
        """Return the complete set of 62 tickers involved across the entire horizon."""
        return set(self._all_tickers)

    def get_constituents_on_date(self, dt: Union[str, pd.Timestamp]) -> Set[str]:
        """Compute the exact set of 50 constituents active on a given date."""
        target_date = pd.Timestamp(dt).strftime("%Y-%m-%d")
        current_members = set(self.base_constituents)

        for reb in self.rebalances:
            reb_date = str(reb["effective_date"])
            if target_date >= reb_date:
                # Apply rebalance
                for t_out in reb["out"]:
                    current_members.discard(t_out)
                for t_in in reb["in"]:
                    current_members.add(t_in)

        return current_members

    def is_constituent(self, ticker: str, dt: Union[str, pd.Timestamp]) -> bool:
        """Check if a specific ticker was an index constituent on a given date."""
        active = self.get_constituents_on_date(dt)
        return ticker.upper() in active

    def filter_point_in_time(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        ticker_col: str = "ticker",
    ) -> pd.DataFrame:
        """Filter a DataFrame so only rows where ticker is an active index member are kept.

        Prevents survivorship bias by removing exited tickers post-exit,
        and new entrants prior to their effective entry date.
        """
        if df.empty:
            return df

        df = df.copy()
        # Vectorized lookup for performance
        dates = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
        tickers = df[ticker_col].str.upper()

        unique_dates = dates.unique()
        membership_map: Dict[str, Set[str]] = {
            d: self.get_constituents_on_date(d) for d in unique_dates
        }

        mask = [
            t in membership_map[d]
            for t, d in zip(tickers, dates)
        ]

        return df[mask].reset_index(drop=True)
