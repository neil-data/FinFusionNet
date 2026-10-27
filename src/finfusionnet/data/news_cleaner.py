"""FinFusionNet: News Ingestion, Cleaning, and Timestamp Alignment.

Implements rigorous timestamp normalization to Indian Standard Time (IST),
market close cutoff assignment (15:30 IST), deduplication, domain filtering,
and causal trading day alignment using the official trading calendar.
"""

import html
import re
from datetime import time, datetime
from typing import List, Optional, Set
import pandas as pd
import numpy as np

from finfusionnet.data.trading_calendar import get_trading_calendar, get_next_trading_session


# Non-editorial / social media domains to exclude from financial news corpus
EXCLUDED_PUBLISHER_PATTERNS = [
    r"facebook",
    r"twitter",
    r"instagram",
    r"linkedin",
    r"youtube",
    r"tiktok",
    r"pinterest",
    r"reddit",
    r"quora",
    r"wikipedia",
]

# Common publisher title suffixes added by Google News RSS (e.g. " - Moneycontrol")
SUFFIX_PATTERN = re.compile(r"\s+[-–|]\s+[A-Za-z0-9\.\s]+$")


def clean_article_title(raw_title: str) -> str:
    """Clean and unescape article headline, removing aggregator publisher suffixes."""
    if not isinstance(raw_title, str) or not raw_title.strip():
        return ""
    
    # 1. HTML unescape (&amp; -> &, &quot; -> ", etc.)
    text = html.unescape(raw_title.strip())

    # 2. Strip publisher suffix (e.g. " - Economic Times", " | Livemint")
    cleaned = SUFFIX_PATTERN.sub("", text).strip()
    if not cleaned:
        cleaned = text  # fallback if regex stripped everything

    # 3. Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def filter_editorial_domains(
    df: pd.DataFrame,
    publisher_title_col: str = "publisher_title",
    publisher_href_col: str = "publisher_href",
) -> pd.DataFrame:
    """Filter out social media, user-generated content, and non-editorial sources."""
    if df.empty:
        return df

    mask = pd.Series(True, index=df.index)
    pattern = "|".join(EXCLUDED_PUBLISHER_PATTERNS)
    
    if publisher_title_col in df.columns:
        is_social_title = df[publisher_title_col].astype(str).str.lower().str.contains(pattern, regex=True)
        mask = mask & (~is_social_title)
        
    if publisher_href_col in df.columns:
        is_social_href = df[publisher_href_col].astype(str).str.lower().str.contains(pattern, regex=True)
        mask = mask & (~is_social_href)

    return df[mask].copy().reset_index(drop=True)


def normalize_timestamps_to_ist(
    df: pd.DataFrame,
    pub_date_col: str = "published_date",
) -> pd.DataFrame:
    """Parse RFC 822 UTC timestamps and convert to tz-aware Asia/Kolkata (IST)."""
    df = df.copy()
    # RFC 822 format typically in GMT / UTC
    dt_utc = pd.to_datetime(df[pub_date_col], utc=True, errors="coerce")
    dt_ist = dt_utc.dt.tz_convert("Asia/Kolkata")

    df["pub_timestamp_ist"] = dt_ist
    df["pub_date_ist"] = dt_ist.dt.date
    df["pub_time_ist"] = dt_ist.dt.time

    # Drop records where timestamp parsing completely failed
    valid_mask = df["pub_timestamp_ist"].notna()
    return df[valid_mask].copy().reset_index(drop=True)


def assign_trading_date_with_cutoff(
    df: pd.DataFrame,
    calendar: List[pd.Timestamp],
    cutoff_time: time = time(15, 30),
) -> pd.DataFrame:
    """Assign each article to the appropriate trading feature date with 3:30 PM IST cutoff.

    Rule:
    1. If article published on Trading Day T at or before 15:30 IST:
       Assigned to Trading Day T (known at close of T to predict T+1).
    2. If article published on Trading Day T after 15:30 IST:
       Assigned to Trading Day T+1 (rolls over to predict T+2).
    3. If article published on non-trading day (weekend / holiday):
       Assigned to the immediate next Trading Day T' (known at close of T' to predict T'+1).
    """
    df = df.copy()
    calendar_dates = set(pd.to_datetime(c).date() for c in calendar)
    sorted_calendar_dates = sorted(list(calendar_dates))
    
    assigned_feature_dates = []

    for idx, row in df.iterrows():
        p_date = row["pub_date_ist"]
        p_time = row["pub_time_ist"]

        if p_date in calendar_dates and p_time <= cutoff_time:
            # Published during trading hours before close: available for Day T close prediction
            assigned_feature_dates.append(pd.Timestamp(p_date))
        else:
            # Published after close or on weekend/holiday: find next trading session
            # Find first trading date strictly greater than p_date
            next_dates = [d for d in sorted_calendar_dates if d > p_date]
            if next_dates:
                assigned_feature_dates.append(pd.Timestamp(next_dates[0]))
            else:
                # Terminal edge case past calendar
                assigned_feature_dates.append(pd.NaT)

    df["feature_date"] = assigned_feature_dates
    # Drop any articles that fell beyond the terminal calendar boundary
    return df[df["feature_date"].notna()].copy().reset_index(drop=True)


def deduplicate_articles(
    df: pd.DataFrame,
    ticker_col: str = "ticker",
    feature_date_col: str = "feature_date",
    title_col: str = "clean_title",
) -> pd.DataFrame:
    """Deduplicate syndicated stories sharing the same normalized title on a ticker-day."""
    df = df.copy()
    # Normalize title for comparison
    df["_norm_title_key"] = (
        df[title_col]
        .astype(str)
        .str.lower()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.strip()
    )
    # Deduplicate keeping first occurrence
    df_deduped = df.drop_duplicates(subset=[ticker_col, feature_date_col, "_norm_title_key"], keep="first")
    df_deduped = df_deduped.drop(columns=["_norm_title_key"]).reset_index(drop=True)
    return df_deduped
