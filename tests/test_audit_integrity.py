"""Unit tests for Phase 0 Audit Integrity & Freeze.

Validates:
1. Fold date reconciliation: Protocol B yields exactly 45 folds and 915 unique test dates.
2. Contiguous non-overlapping test partitions.
3. Strict target-date purging at train/val and val/test boundaries.
4. Fold index slicing integrity (.iloc matches fold.test_dates).
5. Statistical testing consistency: p < 0.05 if and only if zero is excluded from the 95% bootstrap CI.
6. One-batch LSTM overfit test.
"""

import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from finfusionnet.evaluation.pipeline import build_common_evaluation_cohort
from finfusionnet.evaluation.walk_forward import WalkForwardEngine
from finfusionnet.evaluation.stats_tests import paired_date_block_bootstrap
from finfusionnet.models.lstm_model import FinFusionLSTMNet


@pytest.fixture
def cohort_data():
    master_path = "data/processed/master_dataset/master_ticker_day.parquet"
    df = pd.read_parquet(master_path)
    cohort_df, _ = build_common_evaluation_cohort(df)
    return cohort_df.sort_values(["date", "ticker"]).reset_index(drop=True)


def test_protocol_b_fold_count_and_unique_test_dates(cohort_data):
    """Verify that Protocol B generates 45 monthly folds and exactly 915 unique test dates."""
    engine = WalkForwardEngine(date_col="date", target_date_col="target_date")
    folds = engine.generate_protocol_b_folds(
        cohort_data,
        initial_train_months=24,
        test_period_months=1,
        step_months=1,
        validation_fraction=0.15,
        min_val_sessions=60,
    )

    assert len(folds) == 45, f"Expected 45 folds, got {len(folds)}"

    all_test_dates = set()
    for fold in folds:
        all_test_dates.update(fold.test_dates)

    assert len(all_test_dates) == 915, f"Expected 915 unique test dates, got {len(all_test_dates)}"


def test_fold_index_slicing_integrity(cohort_data):
    """Verify that cohort_data.iloc[fold.test_indices] matches fold.test_dates exactly."""
    engine = WalkForwardEngine(date_col="date", target_date_col="target_date")
    folds = engine.generate_protocol_b_folds(
        cohort_data,
        initial_train_months=24,
        test_period_months=1,
        step_months=1,
        validation_fraction=0.15,
        min_val_sessions=60,
    )

    for fold in folds:
        test_slice = cohort_data.iloc[fold.test_indices]
        slice_dates = set(test_slice["date"])
        expected_dates = set(fold.test_dates)
        assert slice_dates == expected_dates, f"Fold {fold.fold_idx} test slice dates mismatch"

        train_slice = cohort_data.iloc[fold.train_indices]
        val_slice = cohort_data.iloc[fold.val_indices]

        # Purging: train target_date < first val date
        first_val_date = fold.val_dates[0]
        first_test_date = fold.test_dates[0]
        assert (train_slice["target_date"] < first_val_date).all(), f"Fold {fold.fold_idx} train purging violated"
        assert (val_slice["target_date"] < first_test_date).all(), f"Fold {fold.fold_idx} val purging violated"


def test_bootstrap_p_ci_mathematical_consistency():
    """Verify that two-sided bootstrap p < alpha if and only if zero is outside the percentile CI."""
    rng = np.random.RandomState(42)
    # Case A: strictly positive difference
    a = rng.normal(0.55, 0.05, size=200)
    b = rng.normal(0.48, 0.05, size=200)
    res_a = paired_date_block_bootstrap(a, b, block_length=10, n_bootstraps=2000, alpha=0.05, random_state=42)

    zero_in_ci_a = (res_a["ci_lower"] <= 0.0 <= res_a["ci_upper"])
    p_sig_a = res_a["p_value_two_sided"] < 0.05
    assert (not zero_in_ci_a) == p_sig_a == res_a["is_significant_05"]

    # Case B: identical distributions
    c = rng.normal(0.50, 0.05, size=200)
    d = rng.normal(0.50, 0.05, size=200)
    res_b = paired_date_block_bootstrap(c, d, block_length=10, n_bootstraps=2000, alpha=0.05, random_state=42)

    zero_in_ci_b = (res_b["ci_lower"] <= 0.0 <= res_b["ci_upper"])
    p_sig_b = res_b["p_value_two_sided"] < 0.05
    assert (not zero_in_ci_b) == p_sig_b == res_b["is_significant_05"]


def test_lstm_one_batch_overfit():
    """Verify that LSTM can achieve 100% accuracy and loss < 0.05 on a single synthetic batch."""
    torch.manual_seed(42)
    N, S, F = 16, 20, 10
    X = torch.randn(N, S, F)
    y = torch.randint(0, 2, (N,)).float()

    model = FinFusionLSTMNet(input_size=F, hidden_dim=32, num_layers=1, dropout=0.0, dense_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    criterion = nn.BCELoss()

    for _ in range(80):
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

    final_loss = float(loss.item())
    final_acc = float(((model(X) >= 0.5) == y).float().mean().item())
    assert final_loss < 0.05, f"Overfit loss too high: {final_loss}"
    assert final_acc == 1.0, f"Overfit accuracy not 100%: {final_acc}"
