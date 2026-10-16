"""Unit tests to strictly verify NO LOOKAHEAD LEAKAGE in features."""

import pytest
import pandas as pd
import numpy as np

from finfusionnet.features.technical import compute_technical_indicators


def test_no_future_information_in_technical_indicators():
    """Verify causal rolling calculations: changing future prices does NOT alter past indicators."""
    dates = pd.date_range("2021-01-01", periods=60, freq="B")
    np.random.seed(123)
    base_close = 100.0 + np.cumsum(np.random.normal(0.1, 1.0, len(dates)))
    high = base_close + 2.0
    low = base_close - 2.0
    open_p = base_close - 0.5
    volume = np.random.uniform(1000, 5000, len(dates))

    df_original = pd.DataFrame({
        "date": dates,
        "ticker": "TEST",
        "open": open_p,
        "high": high,
        "low": low,
        "close": base_close,
        "volume": volume,
    })

    # Compute baseline indicators
    feat_orig = compute_technical_indicators(df_original)

    # Create modified dataset where future observations (sessions 40 to 59) are drastically changed
    df_modified = df_original.copy()
    cutoff_idx = 35
    df_modified.loc[cutoff_idx:, "close"] = df_modified.loc[cutoff_idx:, "close"] * 10.0
    df_modified.loc[cutoff_idx:, "high"] = df_modified.loc[cutoff_idx:, "high"] * 10.0
    df_modified.loc[cutoff_idx:, "low"] = df_modified.loc[cutoff_idx:, "low"] * 10.0
    df_modified.loc[cutoff_idx:, "open"] = df_modified.loc[cutoff_idx:, "open"] * 10.0
    df_modified.loc[cutoff_idx:, "volume"] = df_modified.loc[cutoff_idx:, "volume"] * 50.0

    feat_mod = compute_technical_indicators(df_modified)

    # For all dates strictly before cutoff_idx (i.e. dates 0 to 34),
    # ALL technical indicator values must be EXACTLY IDENTICAL.
    cols_to_check = [
        "return_1d", "return_2d", "return_5d", "return_10d", "return_20d",
        "close_sma_5_ratio", "close_sma_10_ratio", "close_sma_20_ratio", "close_sma_50_ratio",
        "sma_5_sma_20_ratio", "rsi_14", "macd", "macd_signal", "macd_histogram",
        "atr_14_pct", "bb_width_20", "volatility_10", "volatility_20",
        "volume_change_1d", "volume_ratio_5", "volume_ratio_20",
        "high_low_range_pct", "close_open_return", "close_position_in_range",
    ]

    for col in cols_to_check:
        orig_past = feat_orig.iloc[:cutoff_idx][col].values
        mod_past = feat_mod.iloc[:cutoff_idx][col].values

        # Handle NaNs from initial warmup
        nan_mask = np.isnan(orig_past)
        assert (nan_mask == np.isnan(mod_past)).all(), f"NaN mask mismatch in {col}"
        np.testing.assert_allclose(
            orig_past[~nan_mask],
            mod_past[~nan_mask],
            rtol=1e-7,
            atol=1e-7,
            err_msg=f"Future price leak detected in {col}!",
        )
