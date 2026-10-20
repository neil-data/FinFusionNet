"""Audit A5 Unit Tests: Timing, Calendar Alignment & Causal Proof."""

import pytest
import pandas as pd
import numpy as np

from finfusionnet.data.trading_calendar import get_trading_calendar, NSE_HOLIDAYS
from finfusionnet.features.targets import compute_targets
from finfusionnet.features.technical import compute_technical_indicators
from finfusionnet.evaluation.pipeline import build_common_evaluation_cohort


@pytest.fixture
def cohort_data():
    master_path = "data/processed/master_dataset/master_ticker_day.parquet"
    df = pd.read_parquet(master_path)
    cohort_df, _ = build_common_evaluation_cohort(df)
    return cohort_df.sort_values(["date", "ticker"]).reset_index(drop=True)


def test_calendar_next_session_weekend_and_holidays():
    """Verify that next session lookup correctly bridges weekends and exchange holidays."""
    cal = get_trading_calendar("2023-01-01", "2023-04-01")

    # Friday before weekend: 2023-01-06 -> next trading date should be Monday 2023-01-09
    dt_fri = pd.Timestamp("2023-01-06")
    next_dt = cal[cal > dt_fri][0]
    assert next_dt == pd.Timestamp("2023-01-09"), f"Weekend bridge failed: got {next_dt}"

    # Day before Republic Day (2023-01-26 is holiday): 2023-01-25 -> next should be 2023-01-27
    dt_rep = pd.Timestamp("2023-01-25")
    next_dt_rep = cal[cal > dt_rep][0]
    assert next_dt_rep == pd.Timestamp("2023-01-27"), f"Holiday bridge failed: got {next_dt_rep}"


def test_target_calculation_and_flat_return_zero_mapping(cohort_data):
    """Verify target == 1[close_{t+1}/close_t - 1 > 0] and that flat returns strictly map to 0."""
    df = cohort_data.copy()
    tickers = ["RELIANCE", "TCS", "INFY"]

    for t in tickers:
        df_t = df[df["ticker"] == t].sort_values("date").reset_index(drop=True)
        # Check consecutive session target consistency
        for i in range(len(df_t) - 1):
            c_t = df_t.iloc[i]["close"]
            c_next = df_t.iloc[i + 1]["close"]
            t_date = df_t.iloc[i]["target_date"]
            next_date = df_t.iloc[i + 1]["date"]

            if t_date == next_date:
                expected_ret = (c_next / c_t) - 1.0
                expected_target = 1 if expected_ret > 0.0 else 0
                actual_target = int(df_t.iloc[i]["target"])
                assert actual_target == expected_target, (
                    f"Target mismatch for {t} at date {df_t.iloc[i]['date']}: got {actual_target}, expected {expected_target}"
                )

    # Test synthetic flat return mapping
    synthetic_flat = pd.DataFrame({
        "date": [pd.Timestamp("2023-01-02"), pd.Timestamp("2023-01-03")],
        "ticker": ["FLAT_TEST", "FLAT_TEST"],
        "close": [100.0, 100.0],
    })
    cal = get_trading_calendar("2023-01-01", "2023-01-10")
    res_flat = compute_targets(synthetic_flat, calendar=cal)
    # Return is 0.0 -> target must be 0, not 1
    assert res_flat.iloc[0]["target"] == 0, "Flat return (0.0) must map strictly to class 0"


def test_no_future_data_leakage_in_features(cohort_data):
    """Verify that changing future prices does not alter technical features at t."""
    df_raw = cohort_data[cohort_data["ticker"] == "RELIANCE"][
        ["date", "ticker", "open", "high", "low", "close", "volume"]
    ].sort_values("date").reset_index(drop=True)

    feats_orig = compute_technical_indicators(df_raw)

    cutoff = 50
    df_mod = df_raw.copy()
    df_mod.loc[cutoff:, "close"] *= 5.0
    df_mod.loc[cutoff:, "volume"] *= 10.0

    feats_mod = compute_technical_indicators(df_mod)

    tech_cols = [
        "return_1d", "return_2d", "return_5d", "return_10d", "return_20d",
        "close_sma_5_ratio", "close_sma_10_ratio", "close_sma_20_ratio", "close_sma_50_ratio",
        "sma_5_sma_20_ratio", "rsi_14", "macd", "macd_signal", "macd_histogram",
        "atr_14_pct", "bb_width_20", "volatility_10", "volatility_20",
        "volume_change_1d", "volume_ratio_5", "volume_ratio_20",
        "high_low_range_pct", "close_open_return", "close_position_in_range",
    ]

    for col in tech_cols:
        orig_past = feats_orig.iloc[:cutoff][col].values
        mod_past = feats_mod.iloc[:cutoff][col].values
        nan_mask = np.isnan(orig_past)
        assert (nan_mask == np.isnan(mod_past)).all()
        np.testing.assert_allclose(orig_past[~nan_mask], mod_past[~nan_mask], rtol=1e-7, atol=1e-7)
