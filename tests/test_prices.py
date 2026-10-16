"""Unit tests for NSE Trading Calendar and Target Formulation."""

from datetime import datetime
import pytest
import pandas as pd
import numpy as np

from finfusionnet.data.trading_calendar import (
    get_trading_calendar,
    is_trading_session,
    get_next_trading_session,
    align_timestamp_to_session,
)
from finfusionnet.features.targets import compute_targets


def test_trading_calendar_excludes_weekends_and_holidays():
    """Verify that weekends and official NSE holidays are excluded."""
    cal = get_trading_calendar("2021-01-20", "2021-01-31")
    date_strs = [d.strftime("%Y-%m-%d") for d in cal]

    # 2021-01-26 is Republic Day (official NSE holiday)
    assert "2021-01-26" not in date_strs

    # Saturdays and Sundays
    assert "2021-01-23" not in date_strs  # Saturday
    assert "2021-01-24" not in date_strs  # Sunday
    assert "2021-01-30" not in date_strs  # Saturday
    assert "2021-01-31" not in date_strs  # Sunday

    # Valid trading days
    assert "2021-01-25" in date_strs
    assert "2021-01-27" in date_strs


def test_next_trading_session_skips_weekends_and_holidays():
    """Verify that next session lookup accurately skips non-trading intervals."""
    cal = get_trading_calendar("2021-01-01", "2021-02-01")

    # Friday to Monday
    friday = pd.Timestamp("2021-01-22")
    next_s = get_next_trading_session(friday, calendar=cal)
    assert next_s == pd.Timestamp("2021-01-25")  # Monday

    # Monday to Wednesday across Republic Day (2021-01-26)
    monday = pd.Timestamp("2021-01-25")
    next_s = get_next_trading_session(monday, calendar=cal)
    assert next_s == pd.Timestamp("2021-01-27")  # Wednesday


def test_news_timestamp_alignment():
    """Verify 15:30 IST cutoff logic for news mapping."""
    cal = get_trading_calendar("2021-01-20", "2021-02-01")

    # Friday at 14:30 (before close) -> mapped to Friday
    ts1 = pd.Timestamp("2021-01-22 14:30:00")
    assert align_timestamp_to_session(ts1, calendar=cal) == pd.Timestamp("2021-01-22")

    # Friday at 15:30 (exactly at close) -> mapped to Friday
    ts2 = pd.Timestamp("2021-01-22 15:30:00")
    assert align_timestamp_to_session(ts2, calendar=cal) == pd.Timestamp("2021-01-22")

    # Friday at 16:00 (after close) -> mapped to Monday
    ts3 = pd.Timestamp("2021-01-22 16:00:00")
    assert align_timestamp_to_session(ts3, calendar=cal) == pd.Timestamp("2021-01-25")

    # Saturday news -> mapped to Monday
    ts4 = pd.Timestamp("2021-01-23 11:00:00")
    assert align_timestamp_to_session(ts4, calendar=cal) == pd.Timestamp("2021-01-25")

    # Monday at 18:00 (after close before Republic Day holiday) -> mapped to Wednesday
    ts5 = pd.Timestamp("2021-01-25 18:00:00")
    assert align_timestamp_to_session(ts5, calendar=cal) == pd.Timestamp("2021-01-27")


def test_compute_targets_binary_and_flat_zero():
    """Verify target calculation, especially the mandatory rule: zero return strictly maps to 0."""
    cal = get_trading_calendar("2021-01-01", "2021-01-10")
    # Sessions: 2021-01-01 (Fri), 2021-01-04 (Mon), 2021-01-05 (Tue), 2021-01-06 (Wed)
    df = pd.DataFrame([
        {"date": "2021-01-01", "ticker": "INFY", "close": 100.0},
        {"date": "2021-01-04", "ticker": "INFY", "close": 105.0},  # +5% -> target=1 for Jan 1
        {"date": "2021-01-05", "ticker": "INFY", "close": 105.0},  # 0.0% -> target=0 for Jan 4 (FLAT)
        {"date": "2021-01-06", "ticker": "INFY", "close": 101.0},  # -3.8% -> target=0 for Jan 5
    ])

    result = compute_targets(df, calendar=cal)

    # Jan 1: next is Jan 4, return = +5%, target = 1
    row_jan1 = result[result["date"] == "2021-01-01"].iloc[0]
    assert row_jan1["target_date"] == pd.Timestamp("2021-01-04")
    assert pytest.approx(row_jan1["next_session_return"], 1e-4) == 0.05
    assert row_jan1["target"] == 1

    # Jan 4: next is Jan 5, return = 0.0, TARGET MUST BE 0 (MANDATORY RULE)
    row_jan4 = result[result["date"] == "2021-01-04"].iloc[0]
    assert row_jan4["target_date"] == pd.Timestamp("2021-01-05")
    assert pytest.approx(row_jan4["next_session_return"], 1e-4) == 0.0
    assert row_jan4["target"] == 0

    # Jan 5: next is Jan 6, return < 0, target = 0
    row_jan5 = result[result["date"] == "2021-01-05"].iloc[0]
    assert row_jan5["target_date"] == pd.Timestamp("2021-01-06")
    assert row_jan5["next_session_return"] < 0
    assert row_jan5["target"] == 0


def test_secondary_relative_target():
    """Verify market-relative direction target calculation."""
    cal = get_trading_calendar("2021-01-01", "2021-01-10")
    df = pd.DataFrame([
        {"date": "2021-01-01", "ticker": "TCS", "close": 100.0},
        {"date": "2021-01-04", "ticker": "TCS", "close": 102.0},  # +2% return
    ])
    # Case 1: Market gained 3% -> relative return is -1% -> relative_target = 0
    nifty_df_1 = pd.DataFrame([
        {"date": "2021-01-01", "close": 1000.0},
        {"date": "2021-01-04", "close": 1030.0},  # +3% return
    ])
    res_1 = compute_targets(df, nifty_df=nifty_df_1, calendar=cal)
    assert res_1.iloc[0]["target"] == 1  # Absolute return is positive (+2%)
    assert res_1.iloc[0]["relative_target"] == 0  # Relative return is negative (-1%)

    # Case 2: Market gained 1% -> relative return is +1% -> relative_target = 1
    nifty_df_2 = pd.DataFrame([
        {"date": "2021-01-01", "close": 1000.0},
        {"date": "2021-01-04", "close": 1010.0},  # +1% return
    ])
    res_2 = compute_targets(df, nifty_df=nifty_df_2, calendar=cal)
    assert res_2.iloc[0]["target"] == 1
    assert res_2.iloc[0]["relative_target"] == 1
