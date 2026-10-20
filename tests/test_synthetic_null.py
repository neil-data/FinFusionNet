"""Audit A5 Unit Tests: Synthetic Random Walk Null Predictability."""

import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from finfusionnet.data.trading_calendar import get_trading_calendar
from finfusionnet.features.technical import compute_technical_indicators
from finfusionnet.features.targets import compute_targets
from finfusionnet.evaluation.walk_forward import WalkForwardEngine


def test_synthetic_random_walk_null():
    """Verify that models trained on unforecastable random walk returns achieve ~50% accuracy."""
    calendar = get_trading_calendar("2021-01-01", "2023-06-30")
    rng = np.random.RandomState(123)

    # 10 synthetic tickers
    records = []
    for t in [f"SYN_{i}" for i in range(10)]:
        rets = rng.normal(0.0002, 0.015, size=len(calendar))
        price = 500.0 * np.cumprod(1.0 + rets)
        for i, dt in enumerate(calendar):
            c = price[i]
            prev_c = price[i - 1] if i > 0 else c
            o = prev_c * (1.0 + rng.normal(0, 0.002))
            h = max(o, c) * 1.005
            l = min(o, c) * 0.995
            records.append({
                "date": dt,
                "ticker": t,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": 100000,
            })

    raw_df = pd.DataFrame(records)
    feat_df = compute_technical_indicators(raw_df)
    df = compute_targets(feat_df, calendar=calendar)

    tech_cols = ["return_1d", "return_2d", "return_5d", "rsi_14", "macd", "volatility_10"]
    clean_df = df.dropna(subset=tech_cols + ["target"]).reset_index(drop=True)

    engine = WalkForwardEngine(date_col="date", target_date_col="target_date")
    folds = engine.generate_protocol_b_folds(clean_df, initial_train_months=12, test_period_months=1, step_months=1)

    preds_list = []
    for fold in folds[-5:]:
        train_df = clean_df.iloc[fold.train_indices]
        test_df = clean_df.iloc[fold.test_indices]

        X_tr = train_df[tech_cols].values
        y_tr = train_df["target"].values.astype(int)
        X_te = test_df[tech_cols].values
        y_te = test_df["target"].values.astype(int)

        mu = X_tr.mean(axis=0)
        sig = X_tr.std(axis=0)
        sig[sig == 0] = 1.0

        clf = LogisticRegression(C=0.1)
        clf.fit((X_tr - mu) / sig, y_tr)
        preds = clf.predict((X_te - mu) / sig)

        preds_list.append(pd.DataFrame({
            "date": test_df["date"],
            "y_true": y_te,
            "y_pred": preds,
        }))

    all_preds = pd.concat(preds_list, ignore_index=True)
    date_acc = all_preds.groupby("date").apply(lambda g: accuracy_score(g["y_true"], g["y_pred"]), include_groups=False).mean()

    # Accuracy must be within statistical margin of 50%
    assert 0.44 <= date_acc <= 0.56, f"Synthetic null accuracy {date_acc} unexpectedly drifted from 50%"
