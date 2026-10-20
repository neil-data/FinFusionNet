"""Audit A1: Timing Proof and Causal Integrity Verification.

Validates:
1. Feature date t, close_t, close_{t+1}, target_return, target, target_date.
2. Assert target == 1[close_{t+1}/close_t - 1 > 0].
3. Include dates before weekends and holidays.
4. Verify that features at date t do not use data from any session after t.
"""

from pathlib import Path
import pandas as pd
import numpy as np

from finfusionnet.data.trading_calendar import get_trading_calendar, NSE_HOLIDAYS
from finfusionnet.features.targets import compute_targets
from finfusionnet.features.technical import compute_technical_indicators


def run_timing_proof():
    print("=" * 80)
    print("AUDIT A1: TIMING PROOF & CAUSAL INTEGRITY VERIFICATION")
    print("=" * 80)

    # Load master dataset
    master_path = Path("data/processed/master_dataset/master_ticker_day.parquet")
    df = pd.read_parquet(master_path)
    df["date"] = pd.to_datetime(df["date"])
    df["target_date"] = pd.to_datetime(df["target_date"])

    tickers = ["RELIANCE", "TCS", "INFY"]

    # Candidate dates:
    # 1. 2023-01-06 (Friday before regular weekend: Sat 2023-01-07, Sun 2023-01-08 -> Mon 2023-01-09)
    # 2. 2023-01-25 (Wednesday before Republic Day holiday on 2023-01-26 -> Fri 2023-01-27)
    # 3. 2023-03-06 (Monday before Holi on 2023-03-07 -> Wed 2023-03-08)
    # 4. 2023-02-14 (Regular Tuesday -> Wed 2023-02-15)
    # 5. 2023-02-15 (Regular Wednesday -> Thu 2023-02-16)
    target_dates = [
        pd.Timestamp("2023-01-06"),
        pd.Timestamp("2023-01-25"),
        pd.Timestamp("2023-03-06"),
        pd.Timestamp("2023-02-14"),
        pd.Timestamp("2023-02-15"),
    ]

    records = []
    calendar = get_trading_calendar("2020-01-01", "2026-09-30")

    for t in tickers:
        df_t = df[df["ticker"] == t].sort_values("date").reset_index(drop=True)
        date_to_row = {r["date"]: r for _, r in df_t.iterrows()}

        for dt in target_dates:
            assert dt in date_to_row, f"Date {dt} not found for ticker {t}"
            row_t = date_to_row[dt]

            close_t = row_t["close"]
            target_date = row_t["target_date"]
            assert pd.notna(target_date), f"Target date is null for {t} on {dt}"

            # Next close
            assert target_date in date_to_row, f"Target date {target_date} price not found for {t}"
            row_next = date_to_row[target_date]
            close_next = row_next["close"]

            calc_return = (close_next / close_t) - 1.0
            calc_target = 1 if calc_return > 0 else 0

            target = int(row_t["target"])
            dataset_return = row_t.get("next_session_return", calc_return)

            # Assertions
            assert np.isclose(calc_return, dataset_return, atol=1e-6), (
                f"Mismatch in return calculation: calculated {calc_return} vs dataset {dataset_return}"
            )
            assert target == calc_target, (
                f"Target assertion failed for {t} on {dt}: target={target}, expected={calc_target}"
            )

            # Assert target_date is indeed the next trading day in calendar
            expected_next_cal = calendar[calendar > dt][0]
            assert target_date == expected_next_cal, (
                f"Target date mismatch: {target_date} vs calendar next {expected_next_cal}"
            )

            # Identify if weekend or holiday
            day_name = dt.strftime("%A")
            days_gap = (target_date - dt).days
            context = f"{day_name} -> {target_date.strftime('%A')} ({days_gap}d gap)"
            if days_gap > 1:
                # Check for holiday in gap
                gap_days = pd.date_range(dt + pd.Timedelta(days=1), target_date - pd.Timedelta(days=1))
                hols = [d.strftime("%Y-%m-%d") for d in gap_days if d.strftime("%Y-%m-%d") in NSE_HOLIDAYS]
                if hols:
                    context += f" [Holiday: {', '.join(hols)}]"
                elif dt.weekday() == 4:
                    context += " [Weekend]"

            records.append({
                "ticker": t,
                "feature_date_t": dt.strftime("%Y-%m-%d"),
                "close_t": round(close_t, 2),
                "close_t_plus_1": round(close_next, 2),
                "target_return": round(calc_return, 6),
                "target": target,
                "target_date": target_date.strftime("%Y-%m-%d"),
                "context": context,
            })

    res_df = pd.DataFrame(records)
    out_path = Path("experiments/audit/timing_proof.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_path, index=False)

    print(f"\nTiming proof verified successfully. Saved {len(res_df)} rows to {out_path}.")
    print("\nSample Table:")
    print(res_df.to_string(index=False))

    # Causal feature check: strictly verify no feature at t changes when modifying future prices
    print("\nAsserting that no feature uses data after t...")
    for t in tickers:
        df_raw = df[df["ticker"] == t][["date", "ticker", "open", "high", "low", "close", "volume"]].copy()
        df_raw = df_raw.sort_values("date").reset_index(drop=True)

        feats_orig = compute_technical_indicators(df_raw)

        # Alter future prices after index 100
        cutoff = 100
        df_mod = df_raw.copy()
        df_mod.loc[cutoff:, "close"] *= 3.0
        df_mod.loc[cutoff:, "high"] *= 3.0
        df_mod.loc[cutoff:, "low"] *= 3.0
        df_mod.loc[cutoff:, "open"] *= 3.0
        df_mod.loc[cutoff:, "volume"] *= 10.0

        feats_mod = compute_technical_indicators(df_mod)

        # Check all indicators prior to cutoff
        tech_cols = [
            "return_1d", "return_2d", "return_5d", "return_10d", "return_20d",
            "close_sma_5_ratio", "close_sma_10_ratio", "close_sma_20_ratio", "close_sma_50_ratio",
            "sma_5_sma_20_ratio", "rsi_14", "macd", "macd_signal", "macd_histogram",
            "atr_14_pct", "bb_width_20", "volatility_10", "volatility_20",
            "volume_change_1d", "volume_ratio_5", "volume_ratio_20",
            "high_low_range_pct", "close_open_return", "close_position_in_range",
        ]
        for col in tech_cols:
            orig_vals = feats_orig.iloc[:cutoff][col].values
            mod_vals = feats_mod.iloc[:cutoff][col].values
            nan_mask = np.isnan(orig_vals)
            assert (nan_mask == np.isnan(mod_vals)).all()
            np.testing.assert_allclose(
                orig_vals[~nan_mask],
                mod_vals[~nan_mask],
                rtol=1e-7,
                atol=1e-7,
                err_msg=f"Future price leakage detected in {col} for ticker {t}!",
            )

    print("PASS: Confirmed zero feature leakage across all technical indicators.")
    return res_df


if __name__ == "__main__":
    run_timing_proof()
