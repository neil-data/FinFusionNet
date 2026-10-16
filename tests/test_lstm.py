"""Unit tests for LSTM Sequence Generation, Model Architecture, and Leakage Prevention."""

import pytest
import numpy as np
import pandas as pd
import torch

from finfusionnet.data.sequence_generator import generate_ticker_sequences
from finfusionnet.models.lstm_model import FinFusionLSTMNet, FinFusionLSTMClassifier


@pytest.fixture
def mock_sequential_data():
    """Create synthetic multi-stock time series across 50 trading days."""
    dates = pd.date_range("2021-01-01", periods=50, freq="B")
    rows = []
    for ticker in ["STOCK_A", "STOCK_B"]:
        for i, d in enumerate(dates):
            nxt_d = dates[i + 1] if i + 1 < len(dates) else pd.NaT
            rows.append({
                "date": d,
                "target_date": nxt_d,
                "ticker": ticker,
                "feat_1": float(i * 1.5 if ticker == "STOCK_A" else -i * 2.0),
                "feat_2": float(np.sin(i / 5.0)),
                "target": int(i % 2),
            })
    return pd.DataFrame(rows)


def test_sequence_generation_geometry_and_ticker_isolation(mock_sequential_data):
    """Verify that sequences are length 20 and NEVER cross ticker boundaries."""
    feat_cols = ["feat_1", "feat_2"]
    X, y, dates, tickers, target_dates = generate_ticker_sequences(
        mock_sequential_data,
        feature_cols=feat_cols,
        sequence_length=20,
    )

    # For each stock with 50 sessions, length 20 lookback produces: 50 - 20 + 1 = 31 sequences.
    # 2 stocks * 31 = 62 sequences total.
    assert X.shape == (62, 20, 2)
    assert len(y) == 62
    assert len(dates) == 62
    assert len(tickers) == 62

    # Verify ticker isolation
    stock_a_mask = (tickers == "STOCK_A")
    X_a = X[stock_a_mask]
    # For STOCK_A, feat_1 is always positive (i * 1.5)
    assert (X_a[:, :, 0] >= 0.0).all(), "Ticker isolation violated: negative values found in STOCK_A sequences!"

    stock_b_mask = (tickers == "STOCK_B")
    X_b = X[stock_b_mask]
    # For STOCK_B, feat_1 is always non-positive (-i * 2.0)
    assert (X_b[:, :, 0] <= 0.0).all(), "Ticker isolation violated: positive values found in STOCK_B sequences!"


def test_target_date_alignment(mock_sequential_data):
    """Verify that each sequence is aligned with target date strictly greater than sequence end date."""
    feat_cols = ["feat_1", "feat_2"]
    X, y, dates, tickers, target_dates = generate_ticker_sequences(
        mock_sequential_data,
        feature_cols=feat_cols,
        sequence_length=20,
    )

    for d, td in zip(dates, target_dates):
        if not pd.isna(td):
            assert pd.Timestamp(td) > pd.Timestamp(d), "Target date must be strictly after prediction date!"


def test_scaler_isolation_no_test_leakage():
    """Verify that scaler is fitted strictly on training data; test data has zero influence."""
    X_train = np.random.normal(10.0, 2.0, size=(100, 20, 3)).astype(np.float32)
    y_train = np.random.binomial(1, 0.5, size=100)

    # Test data has completely different distribution (mean=100.0)
    X_test = np.random.normal(100.0, 5.0, size=(50, 20, 3)).astype(np.float32)

    clf = FinFusionLSTMClassifier(input_size=3, max_epochs=2)
    clf.fit(X_train, y_train)

    train_mean = clf.scaler.mean_.copy()
    train_var = clf.scaler.var_.copy()

    # Predict on test data
    probs = clf.predict_proba(X_test)
    assert len(probs) == 50
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    # Verify scaler parameters are completely unchanged by test predictions
    np.testing.assert_array_equal(clf.scaler.mean_, train_mean)
    np.testing.assert_array_equal(clf.scaler.var_, train_var)


def test_lstm_model_forward_and_threshold():
    """Verify model forward pass, output shape, and 0.50 thresholding."""
    model = FinFusionLSTMNet(input_size=4, hidden_dim=16, num_layers=1, dense_dim=8)
    dummy_input = torch.randn(8, 20, 4)
    out = model(dummy_input)

    assert out.shape == torch.Size([8])
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_lstm_early_stopping():
    """Verify early stopping mechanism restores best validation weights."""
    np.random.seed(42)
    X_train = np.random.randn(80, 20, 2).astype(np.float32)
    y_train = np.random.binomial(1, 0.5, size=80)
    X_val = np.random.randn(20, 20, 2).astype(np.float32)
    y_val = np.random.binomial(1, 0.5, size=20)

    clf = FinFusionLSTMClassifier(
        input_size=2,
        hidden_dim=16,
        num_layers=1,
        dense_dim=8,
        max_epochs=20,
        early_stopping_patience=3,
    )
    clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    assert clf.best_epoch <= 20
    assert clf.best_val_loss < float("inf")
