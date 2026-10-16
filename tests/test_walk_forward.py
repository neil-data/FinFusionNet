"""Unit tests for Walk-Forward Validation Engine and Metrics."""

import pytest
import pandas as pd
import numpy as np

from finfusionnet.evaluation.walk_forward import WalkForwardEngine
from finfusionnet.evaluation.metrics import (
    compute_classification_metrics,
    compute_date_level_metrics,
    compute_naive_baselines,
)
from finfusionnet.data.trading_calendar import get_trading_calendar, get_next_trading_session


@pytest.fixture
def mock_dataset():
    """Create synthetic multi-stock ticker-day dataset across 70 trading days."""
    cal = get_trading_calendar("2021-01-01", "2021-05-01")[:70]
    dates = list(cal)
    tickers = ["RELIANCE", "TCS", "INFY"]

    rows = []
    for i, d in enumerate(dates):
        nxt = cal[i + 1] if i + 1 < len(cal) else pd.NaT
        for t in tickers:
            ret_1d = np.random.normal(0.001, 0.02)
            next_ret = np.random.normal(0.001, 0.02)
            rows.append({
                "date": d,
                "target_date": nxt,
                "ticker": t,
                "return_1d": ret_1d,
                "next_session_return": next_ret,
                "target": 1 if next_ret > 0 else 0,
            })
    return pd.DataFrame(rows)


def test_protocol_a_folds_structure_and_purging(mock_dataset):
    """Verify Protocol A generates valid rolling folds and respects target-date purging."""
    engine = WalkForwardEngine(date_col="date", target_date_col="target_date")
    # Using 30 train, 10 test, 10 step on 70 dates
    folds = engine.generate_protocol_a_folds(
        mock_dataset, train_sessions=30, test_sessions=10, step_sessions=10
    )

    assert len(folds) >= 3

    for fold in folds:
        assert len(fold.train_dates) == 30
        assert len(fold.test_dates) == 10

        train_sub = mock_dataset.iloc[fold.train_indices]
        test_sub = mock_dataset.iloc[fold.test_indices]

        # Target-date purging check: NO train observation's target_date >= first test date
        first_test_date = test_sub["date"].min()
        assert (train_sub["target_date"] < first_test_date).all(), "Purging violation in Protocol A"

        # Date atomic check: all tickers for a date are in the same partition
        train_dates = set(train_sub["date"])
        test_dates = set(test_sub["date"])
        assert len(train_dates.intersection(test_dates)) == 0


def test_protocol_b_expanding_window_and_purging(mock_dataset):
    """Verify Protocol B expanding folds, validation partition, and boundary purging."""
    engine = WalkForwardEngine(date_col="date", target_date_col="target_date")
    # Test on monthly stepping with custom parameters to fit 70-day mock
    folds = engine.generate_protocol_b_folds(
        mock_dataset,
        initial_train_months=1,
        test_period_months=1,
        step_months=1,
        validation_fraction=0.20,
        min_val_sessions=5,
    )

    assert len(folds) >= 1

    for fold in folds:
        assert fold.val_indices is not None
        assert len(fold.train_indices) > 0
        assert len(fold.val_indices) > 0
        assert len(fold.test_indices) > 0

        train_sub = mock_dataset.iloc[fold.train_indices]
        val_sub = mock_dataset.iloc[fold.val_indices]
        test_sub = mock_dataset.iloc[fold.test_indices]

        first_val_date = val_sub["date"].min()
        first_test_date = test_sub["date"].min()

        # Purging: train target_date < first_val_date
        assert (train_sub["target_date"] < first_val_date).all(), "Train/Val purging violation"

        # Purging: val target_date < first_test_date
        assert (val_sub["target_date"] < first_test_date).all(), "Val/Test purging violation"


def test_metrics_and_naive_baselines():
    """Verify metric computation and naive baselines."""
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0])
    y_prob = np.array([0.8, 0.2, 0.4, 0.9, 0.3])

    m = compute_classification_metrics(y_true, y_pred, y_prob)
    assert m["accuracy"] == 0.8
    assert m["tp"] == 2
    assert m["tn"] == 2
    assert m["fp"] == 0
    assert m["fn"] == 1

    # Naive baselines
    df_eval = pd.DataFrame({
        "return_1d": [0.02, -0.01, 0.00, 0.05, -0.03],
    })
    always_up, prev_dir = compute_naive_baselines(df_eval)
    assert (always_up == 1).all()
    assert list(prev_dir) == [1, 0, 0, 1, 0]
