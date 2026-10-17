"""FinFusionNet: Walk-Forward Validation Engine.

Implements:
1. Protocol A: Fixed rolling window benchmark (46 train, 15 test, 17 step).
2. Protocol B: Monthly expanding window with target-date purging and chronological validation.
Enforces date-level atomic partitioning (all stocks on date d stay in the same split).
"""

from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class FoldData:
    """Holds data partitions and metadata for a single evaluation fold."""
    fold_idx: int
    train_dates: List[pd.Timestamp]
    val_dates: Optional[List[pd.Timestamp]]
    test_dates: List[pd.Timestamp]
    train_indices: np.ndarray
    val_indices: Optional[np.ndarray]
    test_indices: np.ndarray


class WalkForwardEngine:
    """Generates leakage-safe chronological train/validation/test splits."""

    def __init__(self, date_col: str = "date", target_date_col: str = "target_date"):
        self.date_col = date_col
        self.target_date_col = target_date_col

    def generate_protocol_a_folds(
        self,
        df: pd.DataFrame,
        train_sessions: int = 46,
        test_sessions: int = 15,
        step_sessions: int = 17,
        expected_folds: int = 34,
    ) -> List[FoldData]:
        """Generate Protocol A rolling folds.

        Window sizing:
        Total sessions for N folds = train_sessions + test_sessions + (N - 1) * step_sessions.
        For 34 folds: 46 + 15 + 33 * 17 = 622 sessions.
        """
        unique_dates = pd.DatetimeIndex(sorted(df[self.date_col].unique()))
        n_dates = len(unique_dates)

        folds: List[FoldData] = []
        fold_idx = 0
        start_idx = 0

        while True:
            train_end_idx = start_idx + train_sessions
            test_start_idx = train_end_idx
            test_end_idx = test_start_idx + test_sessions

            if test_end_idx > n_dates:
                break

            train_dates = unique_dates[start_idx:train_end_idx]
            test_dates = unique_dates[test_start_idx:test_end_idx]

            # Purge train boundary: last train observation's target_date must precede first test date
            first_test_date = test_dates[0]
            train_mask = (df[self.date_col].isin(train_dates)) & (
                df[self.target_date_col] < first_test_date
            )
            test_mask = df[self.date_col].isin(test_dates)

            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]

            folds.append(
                FoldData(
                    fold_idx=fold_idx,
                    train_dates=list(train_dates),
                    val_dates=None,
                    test_dates=list(test_dates),
                    train_indices=train_indices,
                    val_indices=None,
                    test_indices=test_indices,
                )
            )

            fold_idx += 1
            start_idx += step_sessions
            if expected_folds and fold_idx >= expected_folds:
                break

        return folds

    def generate_protocol_b_folds(
        self,
        df: pd.DataFrame,
        initial_train_months: int = 24,
        test_period_months: int = 1,
        step_months: int = 1,
        validation_fraction: float = 0.15,
        min_val_sessions: int = 60,
    ) -> List[FoldData]:
        """Generate Protocol B expanding monthly folds with target-date purging.

        1. Train window expands monthly.
        2. Validation partition is the latest chronological fraction (15%) of train dates.
        3. Target-date purging applied at train/val boundary and val/test boundary.
        """
        unique_dates = pd.DatetimeIndex(sorted(df[self.date_col].unique()))
        
        start_date = unique_dates.min()
        end_date = unique_dates.max()

        folds: List[FoldData] = []
        fold_idx = 0

        current_train_end = start_date + pd.DateOffset(months=initial_train_months)

        while True:
            current_test_end = current_train_end + pd.DateOffset(months=test_period_months)
            
            # Identify dates in current window
            train_pool_dates = unique_dates[
                (unique_dates >= start_date) & (unique_dates < current_train_end)
            ]
            test_dates = unique_dates[
                (unique_dates >= current_train_end) & (unique_dates < current_test_end)
            ]

            if len(test_dates) == 0 or len(train_pool_dates) < min_val_sessions:
                break

            # Chronological validation split from the end of train_pool_dates
            n_pool = len(train_pool_dates)
            n_val = max(int(n_pool * validation_fraction), min_val_sessions)
            if n_val >= n_pool:
                # Fallback to 15% if pool is too small to support 60 sessions
                n_val = max(1, int(n_pool * validation_fraction))

            train_dates = train_pool_dates[:-n_val]
            val_dates = train_pool_dates[-n_val:]

            first_val_date = val_dates[0]
            first_test_date = test_dates[0]

            # Target-date purging
            # 1. Train target_date must be strictly before first_val_date
            train_mask = (df[self.date_col].isin(train_dates)) & (
                df[self.target_date_col] < first_val_date
            )
            # 2. Val target_date must be strictly before first_test_date
            val_mask = (df[self.date_col].isin(val_dates)) & (
                df[self.target_date_col] < first_test_date
            )
            # 3. Test rows
            test_mask = df[self.date_col].isin(test_dates)

            train_indices = np.where(train_mask)[0]
            val_indices = np.where(val_mask)[0]
            test_indices = np.where(test_mask)[0]

            folds.append(
                FoldData(
                    fold_idx=fold_idx,
                    train_dates=list(train_dates),
                    val_dates=list(val_dates),
                    test_dates=list(test_dates),
                    train_indices=train_indices,
                    val_indices=val_indices,
                    test_indices=test_indices,
                )
            )

            fold_idx += 1
            current_train_end = current_train_end + pd.DateOffset(months=step_months)
            if current_train_end >= end_date:
                break

        return folds
