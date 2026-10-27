"""Audit A5 Unit Tests: Fold Integrity, Embargo & Purging Assertions."""

import pytest
import pandas as pd
import numpy as np

from finfusionnet.evaluation.pipeline import build_common_evaluation_cohort
from finfusionnet.evaluation.walk_forward import WalkForwardEngine


@pytest.fixture
def cohort_data():
    master_path = "data/processed/master_dataset/master_ticker_day.parquet"
    df = pd.read_parquet(master_path)
    cohort_df, _ = build_common_evaluation_cohort(df)
    return cohort_df.sort_values(["date", "ticker"]).reset_index(drop=True)


def test_protocol_b_fold_chronology_and_embargo(cohort_data):
    """Assert for Protocol B: train < valid < test, embargo purging, and partition uniqueness."""
    engine = WalkForwardEngine(date_col="date", target_date_col="target_date")
    folds = engine.generate_protocol_b_folds(cohort_data)

    seen_test_dates = set()

    for fold in folds:
        train_dates = fold.train_dates
        val_dates = fold.val_dates
        test_dates = fold.test_dates

        max_train = max(train_dates)
        min_val = min(val_dates)
        max_val = max(val_dates)
        min_test = min(test_dates)

        # 1. Chronology: train dates < valid dates < test dates
        assert max_train < min_val, f"Fold {fold.fold_idx}: max train {max_train} >= min val {min_val}"
        assert max_val < min_test, f"Fold {fold.fold_idx}: max val {max_val} >= min test {min_test}"

        # 2. Embargo / Purging
        train_slice = cohort_data.iloc[fold.train_indices]
        val_slice = cohort_data.iloc[fold.val_indices]
        test_slice = cohort_data.iloc[fold.test_indices]

        # All target_dates in train must be strictly before min_val
        assert (train_slice["target_date"] < min_val).all(), f"Fold {fold.fold_idx}: train target_date leak into val"
        # All target_dates in val must be strictly before min_test
        assert (val_slice["target_date"] < min_test).all(), f"Fold {fold.fold_idx}: val target_date leak into test"

        # 3. Test dates are unique across all folds
        for d in test_dates:
            assert d not in seen_test_dates, f"Test date {d} evaluated in multiple folds!"
            seen_test_dates.add(d)

        # 4. Zero (ticker, date) overlap between partitions
        train_keys = set(zip(train_slice["ticker"], train_slice["date"]))
        val_keys = set(zip(val_slice["ticker"], val_slice["date"]))
        test_keys = set(zip(test_slice["ticker"], test_slice["date"]))

        assert len(train_keys.intersection(val_keys)) == 0, f"Fold {fold.fold_idx}: train/val overlap"
        assert len(train_keys.intersection(test_keys)) == 0, f"Fold {fold.fold_idx}: train/test overlap"
        assert len(val_keys.intersection(test_keys)) == 0, f"Fold {fold.fold_idx}: val/test overlap"


def test_protocol_a_fold_chronology_and_purging(cohort_data):
    """Assert for Protocol A: train < test, target_date purging, and zero overlap."""
    engine = WalkForwardEngine(date_col="date", target_date_col="target_date")
    folds = engine.generate_protocol_a_folds(
        cohort_data, train_sessions=46, test_sessions=15, step_sessions=17
    )

    for fold in folds:
        train_dates = fold.train_dates
        test_dates = fold.test_dates

        max_train = max(train_dates)
        min_test = min(test_dates)

        # 1. Chronology: train dates < test dates
        assert max_train < min_test, f"Fold {fold.fold_idx}: max train {max_train} >= min test {min_test}"

        # 2. Embargo / Purging
        train_slice = cohort_data.iloc[fold.train_indices]
        test_slice = cohort_data.iloc[fold.test_indices]

        assert (train_slice["target_date"] < min_test).all(), (
            f"Fold {fold.fold_idx}: Protocol A train target_date leak into test"
        )

        # 3. Zero overlap
        train_keys = set(zip(train_slice["ticker"], train_slice["date"]))
        test_keys = set(zip(test_slice["ticker"], test_slice["date"]))
        assert len(train_keys.intersection(test_keys)) == 0, f"Fold {fold.fold_idx}: train/test overlap"
