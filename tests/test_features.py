"""Unit tests for Feature Engineering Pipelines (Technical & Sentiment)."""

import pytest
import pandas as pd
import numpy as np

from finfusionnet.features.technical import (
    compute_technical_indicators,
    compute_market_context,
)
from finfusionnet.features.sentiment import (
    integrate_sentiment_features,
    SENTIMENT_FEATURE_COLS,
)


@pytest.fixture
def sample_ohlcv():
    dates = pd.date_range("2021-01-01", periods=60, freq="B")
    np.random.seed(42)
    rows = []
    for ticker in ["RELIANCE", "TCS"]:
        base_price = 1000.0 if ticker == "RELIANCE" else 2000.0
        returns = np.random.normal(0.0005, 0.015, len(dates))
        close = base_price * np.cumprod(1 + returns)
        high = close * (1 + np.random.uniform(0.002, 0.015, len(dates)))
        low = close * (1 - np.random.uniform(0.002, 0.015, len(dates)))
        open_p = low + np.random.uniform(0.2, 0.8, len(dates)) * (high - low)
        volume = np.random.uniform(100000, 500000, len(dates))

        for d, o, h, l, c, v in zip(dates, open_p, high, low, close, volume):
            rows.append({
                "date": d,
                "ticker": ticker,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
            })
    return pd.DataFrame(rows)


def test_technical_indicators_shape_and_ranges(sample_ohlcv):
    """Verify technical indicator computation produces valid ranges and expected columns."""
    df_feat = compute_technical_indicators(sample_ohlcv)

    # Check RSI range
    valid_rsi = df_feat["rsi_14"].dropna()
    assert len(valid_rsi) > 0
    assert (valid_rsi >= 0.0).all()
    assert (valid_rsi <= 100.0).all()

    # Check ATR positive
    valid_atr = df_feat["atr_14_pct"].dropna()
    assert len(valid_atr) > 0
    assert (valid_atr > 0.0).all()

    # Check Bollinger width positive
    valid_bb = df_feat["bb_width_20"].dropna()
    assert len(valid_bb) > 0
    assert (valid_bb > 0.0).all()

    # Check close position in range [0, 1]
    valid_pos = df_feat["close_position_in_range"].dropna()
    assert (valid_pos >= 0.0).all()
    assert (valid_pos <= 1.0).all()


def test_market_context_computation():
    """Verify Group B market context features calculated on index prices."""
    dates = pd.date_range("2021-01-01", periods=30, freq="B")
    nifty_prices = 14000.0 + np.cumsum(np.random.normal(5, 50, len(dates)))
    nifty_df = pd.DataFrame({"date": dates, "close": nifty_prices})

    context = compute_market_context(nifty_df)
    assert "nifty_return_1d" in context.columns
    assert "nifty_volatility_10" in context.columns
    assert len(context) == 30


def test_sentiment_integration_missing_data_policy():
    """Verify mandatory policy: no-news days receive null sentiment, never 0.0."""
    market_df = pd.DataFrame([
        {"date": "2021-01-04", "ticker": "INFY"},
        {"date": "2021-01-05", "ticker": "INFY"},  # will have news
        {"date": "2021-01-06", "ticker": "INFY"},  # no news
    ])

    news_df = pd.DataFrame([
        {
            "date": "2021-01-05",
            "ticker": "INFY",
            "n_articles": 3,
            "vader_mean": 0.45,
            "finbert_positive_mean": 0.82,
            "finbert_negative_mean": 0.08,
            "finbert_neutral_mean": 0.10,
        }
    ])

    merged = integrate_sentiment_features(market_df, news_df)

    # Jan 05: has news
    jan05 = merged[merged["date"] == "2021-01-05"].iloc[0]
    assert jan05["has_news"] == 1
    assert jan05["n_articles"] == 3
    assert jan05["vader_mean"] == 0.45
    assert jan05["finbert_positive_mean"] == 0.82

    # Jan 04 and Jan 06: no news
    for d in ["2021-01-04", "2021-01-06"]:
        row = merged[merged["date"] == d].iloc[0]
        assert row["has_news"] == 0
        assert row["n_articles"] == 0
        # MUST BE NULL, NOT 0.0
        assert np.isnan(row["vader_mean"])
        assert np.isnan(row["finbert_positive_mean"])
        assert np.isnan(row["finbert_negative_mean"])
