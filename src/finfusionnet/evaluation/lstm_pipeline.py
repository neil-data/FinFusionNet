"""FinFusionNet: Master LSTM Experiment Pipeline Orchestrator.

Manages end-to-end execution of LSTM research track experiments (L1 to L6).
Enforces:
1. Exact common evaluation cohort parity with XGBoost track.
2. Independent 20-session rolling sequence generation per ticker.
3. Preceding training/validation context preserved for test boundaries (zero lost test dates).
4. In-fold feature scaling strictly using training data parameters.
5. Chronological validation loss early stopping (Protocol B).
6. Structured run artifacts and Parquet prediction serialization.
"""

from datetime import datetime
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import yaml

import numpy as np
import pandas as pd
import torch

from finfusionnet.data.sequence_generator import generate_ticker_sequences
from finfusionnet.evaluation.metrics import (
    compute_classification_metrics,
    compute_date_level_metrics,
    compute_naive_baselines,
)
from finfusionnet.evaluation.pipeline import FEATURE_GROUPS, build_common_evaluation_cohort
from finfusionnet.evaluation.walk_forward import FoldData, WalkForwardEngine
from finfusionnet.models.lstm_model import FinFusionLSTMClassifier


LSTM_EXPERIMENT_SPECS: Dict[str, Dict[str, Union[str, List[str]]]] = {
    "L1": {
        "description": "Technical-only sequential baseline",
        "groups": ["group_a_technical"],
    },
    "L2": {
        "description": "Technical + market context",
        "groups": ["group_a_technical", "group_b_market_context"],
    },
    "L3": {
        "description": "Technical + market context + news availability control",
        "groups": ["group_a_technical", "group_b_market_context", "group_c_news_availability"],
    },
    "L4": {
        "description": "Technical + market context + news availability + VADER",
        "groups": [
            "group_a_technical",
            "group_b_market_context",
            "group_c_news_availability",
            "group_d_vader",
        ],
    },
    "L5": {
        "description": "Technical + market context + news availability + FinBERT",
        "groups": [
            "group_a_technical",
            "group_b_market_context",
            "group_c_news_availability",
            "group_e_finbert",
        ],
    },
    "L6": {
        "description": "Technical + market context + news availability + VADER + FinBERT",
        "groups": [
            "group_a_technical",
            "group_b_market_context",
            "group_c_news_availability",
            "group_d_vader",
            "group_e_finbert",
        ],
    },
}


def get_lstm_experiment_features(experiment_id: str) -> List[str]:
    """Retrieve the exact list of feature columns for an LSTM experiment."""
    if experiment_id not in LSTM_EXPERIMENT_SPECS:
        raise ValueError(f"Unknown experiment {experiment_id}. Must be one of {list(LSTM_EXPERIMENT_SPECS.keys())}")
    groups = LSTM_EXPERIMENT_SPECS[experiment_id]["groups"]
    features: List[str] = []
    for g in groups:
        features.extend(FEATURE_GROUPS[g])
    return features


def extract_boundary_context_sequences(
    full_df: pd.DataFrame,
    context_dates: List[pd.Timestamp],
    eval_dates: List[pd.Timestamp],
    feature_cols: List[str],
    sequence_length: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Extract rolling sequences for eval_dates using up to 19 sessions of preceding context.

    Guarantees:
    - Zero test dates lost at partition boundaries.
    - Context sessions are strictly input features and never evaluated as targets.
    - Sequences never cross ticker boundaries.
    """
    all_dates = list(context_dates) + list(eval_dates)
    sub_df = full_df[full_df["date"].isin(all_dates)].copy()

    X_seq, y, dates, tickers, target_dates = generate_ticker_sequences(
        sub_df,
        feature_cols=feature_cols,
        sequence_length=sequence_length,
    )

    # Filter sequences to keep strictly those where prediction date is in eval_dates
    eval_dates_set = set(pd.to_datetime(eval_dates))
    keep_mask = np.array([pd.Timestamp(d) in eval_dates_set for d in dates])

    X_eval = X_seq[keep_mask]
    y_eval = y[keep_mask]
    dates_eval = dates[keep_mask]
    tickers_eval = tickers[keep_mask]
    target_dates_eval = target_dates[keep_mask]

    # Meta dataframe for prediction tracking
    meta_df = pd.DataFrame({
        "date": dates_eval,
        "ticker": tickers_eval,
        "target_date": target_dates_eval,
        "target": y_eval,
    })

    return X_eval, y_eval, dates_eval, tickers_eval, target_dates_eval, meta_df


def run_lstm_experiment(
    master_df: pd.DataFrame,
    experiment_id: str,
    protocol: str = "protocol_b",
    runs_dir: str = "experiments/runs",
    sequence_length: int = 20,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.20,
    dense_dim: int = 32,
    learning_rate: float = 0.001,
    batch_size: int = 64,
    max_epochs: int = 25,
    early_stopping_patience: int = 5,
    random_state: int = 42,
    news_days_only: bool = False,
) -> Dict[str, Union[str, float, pd.DataFrame]]:
    """Execute walk-forward evaluation for an LSTM deep learning experiment."""
    feature_cols = get_lstm_experiment_features(experiment_id)
    features_desc = LSTM_EXPERIMENT_SPECS[experiment_id]["description"]

    # Filter to identical Common Evaluation Cohort as XGBoost track
    cohort_df, cohort_stats = build_common_evaluation_cohort(master_df)

    if news_days_only:
        cohort_df = cohort_df[cohort_df["has_news"] == 1].reset_index(drop=True)

    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    exp_dir_name = f"{experiment_id.lower()}_{protocol}"
    if news_days_only:
        exp_dir_name += "_news_only"
    run_path = Path(runs_dir) / exp_dir_name / timestamp_str
    run_path.mkdir(parents=True, exist_ok=True)

    engine = WalkForwardEngine(date_col="date", target_date_col="target_date")

    if protocol == "protocol_a":
        folds = engine.generate_protocol_a_folds(
            cohort_df, train_sessions=46, test_sessions=15, step_sessions=17
        )
        patience = 999  # Protocol A uses fixed epochs (no early stopping)
        epochs = 15
    elif protocol == "protocol_b":
        folds = engine.generate_protocol_b_folds(
            cohort_df,
            initial_train_months=24,
            test_period_months=1,
            step_months=1,
            validation_fraction=0.15,
            min_val_sessions=60,
        )
        patience = early_stopping_patience
        epochs = max_epochs
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    predictions_list = []
    fold_metrics_list = []

    print(f"Executing {len(folds)} walk-forward folds for {experiment_id} ({protocol})...")

    for fold in folds:
        train_dates = fold.train_dates
        val_dates = fold.val_dates
        test_dates = fold.test_dates

        # 1. Generate Training Sequences strictly within train_dates
        train_df = cohort_df[cohort_df["date"].isin(train_dates)].copy()
        X_train, y_train, _, _, _ = generate_ticker_sequences(
            train_df, feature_cols=feature_cols, sequence_length=sequence_length
        )

        if len(X_train) == 0:
            continue

        # 2. Generate Validation Sequences with boundary context if present
        if val_dates is not None and len(val_dates) > 0:
            X_val, y_val, _, _, _, _ = extract_boundary_context_sequences(
                cohort_df,
                context_dates=train_dates[-sequence_length + 1:],
                eval_dates=val_dates,
                feature_cols=feature_cols,
                sequence_length=sequence_length,
            )
        else:
            X_val, y_val = None, None

        # 3. Generate Test Sequences with boundary context
        prior_dates = val_dates if (val_dates is not None and len(val_dates) > 0) else train_dates
        X_test, y_test, dates_test, tickers_test, target_dates_test, meta_test = extract_boundary_context_sequences(
            cohort_df,
            context_dates=prior_dates[-sequence_length + 1:],
            eval_dates=test_dates,
            feature_cols=feature_cols,
            sequence_length=sequence_length,
        )

        if len(X_test) == 0:
            continue

        # 4. Initialize and fit LSTM Classifier
        clf = FinFusionLSTMClassifier(
            input_size=len(feature_cols),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            dense_dim=dense_dim,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=epochs,
            early_stopping_patience=patience,
            random_state=random_state,
            threshold=0.50,
        )

        clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        # 5. Predict on Test Sequences
        y_prob = clf.predict_proba(X_test)
        y_pred = clf.predict(X_test)

        # Baselines
        test_rows = cohort_df[cohort_df["date"].isin(test_dates)].copy()
        test_merged = meta_test.merge(test_rows[["date", "ticker", "return_1d"]], on=["date", "ticker"], how="left")
        always_up, prev_dir = compute_naive_baselines(test_merged)

        meta_test["y_true"] = y_test
        meta_test["y_pred"] = y_pred
        meta_test["y_prob"] = y_prob
        meta_test["baseline_always_up"] = always_up
        meta_test["baseline_previous_direction"] = prev_dir
        meta_test["fold_idx"] = fold.fold_idx
        predictions_list.append(meta_test)

        # Fold metrics
        m = compute_classification_metrics(y_test, y_pred, y_prob)
        m["fold_idx"] = fold.fold_idx
        m["best_epoch"] = clf.best_epoch
        m["n_train_seq"] = len(X_train)
        m["n_val_seq"] = len(X_val) if X_val is not None else 0
        m["n_test_seq"] = len(X_test)
        m["test_start_date"] = pd.Timestamp(dates_test.min()).strftime("%Y-%m-%d")
        m["test_end_date"] = pd.Timestamp(dates_test.max()).strftime("%Y-%m-%d")
        fold_metrics_list.append(m)

    all_preds = pd.concat(predictions_list, ignore_index=True)
    fold_metrics_df = pd.DataFrame(fold_metrics_list)

    # Date-level equal-weighted metrics
    date_metrics_df = compute_date_level_metrics(all_preds)
    mean_date_acc = float(date_metrics_df["accuracy"].mean())

    # Overall pooled metrics
    pooled_metrics = compute_classification_metrics(
        all_preds["y_true"].values,
        all_preds["y_pred"].values,
        all_preds["y_prob"].values,
    )

    acc_always_up = float(np.mean(all_preds["y_true"].values == all_preds["baseline_always_up"].values))
    acc_prev_dir = float(np.mean(all_preds["y_true"].values == all_preds["baseline_previous_direction"].values))
    lift_always_up = pooled_metrics["accuracy"] - acc_always_up
    lift_prev_dir = pooled_metrics["accuracy"] - acc_prev_dir

    # Per-stock metrics
    stock_records = []
    for ticker, grp in all_preds.groupby("ticker"):
        stock_m = compute_classification_metrics(grp["y_true"].values, grp["y_pred"].values, grp["y_prob"].values)
        stock_m["ticker"] = ticker
        stock_m["n_obs"] = len(grp)
        stock_records.append(stock_m)
    stock_metrics_df = pd.DataFrame(stock_records)

    # Save artifacts
    all_preds.to_parquet(run_path / "predictions.parquet", index=False)
    fold_metrics_df.to_csv(run_path / "fold_metrics.csv", index=False)
    date_metrics_df.to_csv(run_path / "date_metrics.csv", index=False)
    stock_metrics_df.to_csv(run_path / "stock_metrics.csv", index=False)

    summary = {
        "experiment_id": experiment_id,
        "model_type": "LSTM",
        "description": features_desc,
        "protocol": protocol,
        "news_days_only": news_days_only,
        "sequence_length": sequence_length,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "execution_timestamp": timestamp_str,
        "n_folds": len(fold_metrics_df),
        "total_test_predictions": len(all_preds),
        "unique_test_dates": int(all_preds["date"].nunique()),
        "primary_metric_equal_weighted_date_accuracy": mean_date_acc,
        "pooled_accuracy": pooled_metrics["accuracy"],
        "pooled_f1": pooled_metrics["f1"],
        "pooled_roc_auc": pooled_metrics["roc_auc"],
        "pooled_log_loss": pooled_metrics["log_loss"],
        "baseline_always_up_accuracy": acc_always_up,
        "baseline_previous_direction_accuracy": acc_prev_dir,
        "lift_over_always_up": lift_always_up,
        "lift_over_previous_direction": lift_prev_dir,
        "cohort_stats": cohort_stats,
    }

    with open(run_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "run_path": str(run_path),
        "summary": summary,
        "predictions": all_preds,
        "date_metrics": date_metrics_df,
    }
