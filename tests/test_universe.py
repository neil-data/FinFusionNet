"""Unit tests for Point-in-Time NIFTY 50 Universe Tracker."""

import pytest
import pandas as pd
from finfusionnet.data.universe import PointInTimeUniverse, BASE_CONSTITUENTS_2021


@pytest.fixture
def universe():
    return PointInTimeUniverse()


def test_base_constituents_count(universe):
    """Verify that base constituent set contains exactly 50 stocks."""
    assert len(universe.base_constituents) == 50
    assert "RELIANCE" in universe.base_constituents
    assert "TCS" in universe.base_constituents
    assert "INFY" in universe.base_constituents
    assert "GAIL" in universe.base_constituents
    assert "TATACONSUM" not in universe.base_constituents


def test_total_in_window_tickers(universe):
    """Verify that the 2020-2026 universe comprises exactly 62 unique tickers."""
    assert len(universe.all_tickers) == 62


def test_rebalance_2021_03_31(universe):
    """Verify March 2021 rebalance: GAIL out, TATACONSUM in."""
    before = universe.get_constituents_on_date("2021-03-30")
    assert "GAIL" in before
    assert "TATACONSUM" not in before
    assert len(before) == 50

    after = universe.get_constituents_on_date("2021-03-31")
    assert "GAIL" not in after
    assert "TATACONSUM" in after
    assert len(after) == 50


def test_rebalance_2023_07_13(universe):
    """Verify July 2023 HDFC/HDFCBANK merger: HDFC out, LTIM in."""
    before = universe.get_constituents_on_date("2023-07-12")
    assert "HDFC" in before
    assert "LTIM" not in before
    assert len(before) == 50

    after = universe.get_constituents_on_date("2023-07-13")
    assert "HDFC" not in after
    assert "LTIM" in after
    assert len(after) == 50


def test_rebalance_2026_09_30(universe):
    """Verify September 2026 rebalance: WIPRO out, BSE in."""
    before = universe.get_constituents_on_date("2026-09-29")
    assert "WIPRO" in before
    assert "BSE" not in before
    assert len(before) == 50

    after = universe.get_constituents_on_date("2026-09-30")
    assert "WIPRO" not in after
    assert "BSE" in after
    assert len(after) == 50


def test_filter_point_in_time(universe):
    """Verify that filter_point_in_time drops exited stocks and prevents early entrants."""
    records = [
        # GAIL was constituent on 2021-03-30, but not on 2021-04-01
        {"date": "2021-03-30", "ticker": "GAIL", "close": 100.0},
        {"date": "2021-04-01", "ticker": "GAIL", "close": 102.0},
        # TATACONSUM was not constituent on 2021-03-30, but entered on 2021-03-31
        {"date": "2021-03-30", "ticker": "TATACONSUM", "close": 600.0},
        {"date": "2021-04-01", "ticker": "TATACONSUM", "close": 610.0},
        # RELIANCE was constituent throughout
        {"date": "2021-03-30", "ticker": "RELIANCE", "close": 2000.0},
        {"date": "2021-04-01", "ticker": "RELIANCE", "close": 2020.0},
    ]
    df = pd.DataFrame(records)
    filtered = universe.filter_point_in_time(df)

    assert len(filtered) == 4
    # GAIL only on 2021-03-30
    gail_dates = filtered[filtered["ticker"] == "GAIL"]["date"].tolist()
    assert gail_dates == ["2021-03-30"]

    # TATACONSUM only on 2021-04-01
    tata_dates = filtered[filtered["ticker"] == "TATACONSUM"]["date"].tolist()
    assert tata_dates == ["2021-04-01"]

    # RELIANCE on both dates
    rel_dates = filtered[filtered["ticker"] == "RELIANCE"]["date"].tolist()
    assert len(rel_dates) == 2
