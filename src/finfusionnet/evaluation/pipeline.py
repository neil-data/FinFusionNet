"""FinFusionNet: Master Experiment Pipeline Orchestrator.

Manages end-to-end execution of XGBoost research track experiments (X1 to X6).
Enforces:
1. Strict common evaluation cohort parity across X1 to X6.
2. In-fold ticker one-hot encoding.
3. Target-date purging and chronological early stopping (Protocol B).
4. Equal-weighted date-level metric computation.
5. Structured run artifacts and Parquet prediction serialization.
"""

from datetime import datetime
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import yaml

import numpy as np
import pandas as pd

from finfusionnet.evaluation.metrics import (
    compute_classification_metrics,
    compute_date_level_metrics,
    compute_naive_baselines,
)
from finfusionnet.evaluation.walk_forward import FoldData, WalkForwardEngine
from finfusionnet.models.xgboost_model import FinFusionXGBoost


FEATURE_GROUPS: Dict[str, List[str]] = {
    "group_a_technical": [
        "return_1d", "return_2d", "return_5d", "return_10d", "return_20d",
        "close_sma_5_ratio", "close_sma_10_ratio", "close_sma_20_ratio", "close_sma_50_ratio",
        "sma_5_sma_20_ratio", "rsi_14", "macd", "macd_signal", "macd_histogram",
        "atr_14_pct", "bb_width_20", "volatility_10", "volatility_20",
        "volume_change_1d", "volume_ratio_5", "volume_ratio_20",
        "high_low_range_pct", "close_open_return", "close_position_in_range",
    ],
    "group_b_market_context": [
        "nifty_return_1d", "nifty_return_5d", "nifty_return_20d",
        "nifty_volatility_10", "nifty_volatility_20",
    ],
    "group_c_news_availability": [
        "n_articles", "has_news",
    ],
    "group_d_vader": [
        "vader_mean", "vader_positive_mean", "vader_negative_mean",
        "vader_neutral_mean", "vader_dispersion",
    ],
    "group_e_finbert": [
        "finbert_positive_mean", "finbert_negative_mean",
        "finbert_neutral_mean", "finbert_dispersion", "finbert_confidence",
    ],
}

EXPERIMENT_SPECS: Dict[str, Dict[str, Union[str, List[str]]]] = {
    "X1": {
        "description": "Technical-only baseline",
        "groups": ["group_a_technical"],
    },
    "X2": {
        "description": "Technical + market context",
        "groups": ["group_a_technical", "group_b_market_context"],
    },
    "X3": {
        "description": "Technical + market context + news availability control",
        "groups": ["group_a_technical", "group_b_market_context", "group_c_news_availability"],
    },
    "X4": {
        "description": "Technical + market context + news availability + VADER",
        "groups": [
            "group_a_technical",
            "group_b_market_context",
            "group_c_news_availability",
            "group_d_vader",
        ],
    },
    "X5": {
        "description": "Technical + market context + news availability + FinBERT",
        "groups": [
            "group_a_technical",
            "group_b_market_context",
            "group_c_news_availability",
            "group_e_finbert",
        ],
    },
    "X6": {
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


def get_experiment_features(experiment_id: str) -> List[str]:
    """Retrieve the exact list of feature columns for an experiment."""
    if experiment_id not in EXPERIMENT_SPECS:
        raise ValueError(f"Unknown experiment {experiment_id}. Must be one of {list(EXPERIMENT_SPECS.keys())}")
    groups = EXPERIMENT_SPECS[experiment_id]["groups"]
    features: List[str] = []
    for g in groups:
        features.extend(FEATURE_GROUPS[g])
    return features


def build_common_evaluation_cohort(
    master_df: pd.DataFrame,
    start_date: str = "2021-01-01",
    end_date: str = "2026-09-30",
    manifest_save_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Filter master dataset to the strict Common Evaluation Cohort.

    Guarantees that X1 through X6 are evaluated on the exact same ticker-date observations.
    Filters:
    - Modeling date window [start_date, end_date]
    - Target is not null
    - All Group A technical features are non-null
    - Group B market context features are non-null
    - No-news observations are retained (with null sentiment and has_news=0)
    """
    df = master_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    total_rows = len(df)
    stats: Dict[str, int] = {"total_input_rows": total_rows}

    # 1. Date window
    mask_date = (df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))
    stats["excluded_outside_modeling_window"] = int((~mask_date).sum())
    df = df[mask_date].copy()

    # 2. Target validity
    mask_target = df["target"].notna()
    stats["excluded_missing_target"] = int((~mask_target).sum())
    df = df[mask_target].copy()

    # 3. Technical warmup completeness (Group A)
    tech_cols = FEATURE_GROUPS["group_a_technical"]
    mask_tech = df[tech_cols].notna().all(axis=1)
    stats["excluded_incomplete_technical_warmup"] = int((~mask_tech).sum())
    df = df[mask_tech].copy()

    # 4. Market context completeness (Group B)
    if all(col in df.columns for col in FEATURE_GROUPS["group_b_market_context"]):
        context_cols = FEATURE_GROUPS["group_b_market_context"]
        mask_context = df[context_cols].notna().all(axis=1)
        stats["excluded_missing_market_context"] = int((~mask_context).sum())
        df = df[mask_context].copy()

    stats["final_eligible_cohort_rows"] = len(df)
    stats["unique_stocks"] = int(df["ticker"].nunique())
    stats["unique_dates"] = int(df["date"].nunique())

    if manifest_save_path is not None:
        Path(manifest_save_path).parent.mkdir(parents=True, exist_ok=True)
        manifest_df = df[["date", "ticker", "target_date", "target"]].copy()
        manifest_df.to_parquet(manifest_save_path, index=False)

    return df.reset_index(drop=True), stats


def run_experiment(
    master_df: pd.DataFrame,
    experiment_id: str,
    protocol: str = "protocol_b",
    runs_dir: str = "experiments/runs",
    random_state: int = 42,
    news_days_only: bool = False,
) -> Dict[str, Union[str, float, pd.DataFrame]]:
    """Execute walk-forward evaluation for a single XGBoost experiment.

    Protocol: 'protocol_a' (historical rolling) or 'protocol_b' (expanding window).
    """
    feature_cols = get_experiment_features(experiment_id)
    features_desc = EXPERIMENT_SPECS[experiment_id]["description"]

    # Filter to cohort
    cohort_df, cohort_stats = build_common_evaluation_cohort(master_df)

    # Ensure chronological date-level ordering for consistent indexing
    cohort_df = cohort_df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Optional secondary news-days-only filter
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
        early_stopping_rounds = None
    elif protocol == "protocol_b":
        folds = engine.generate_protocol_b_folds(
            cohort_df,
            initial_train_months=24,
            test_period_months=1,
            step_months=1,
            validation_fraction=0.15,
            min_val_sessions=60,
        )
        early_stopping_rounds = 50
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    predictions_list = []
    fold_metrics_list = []
    importance_list = []

    for fold in folds:
        train_df = cohort_df.iloc[fold.train_indices].copy()
        val_df = cohort_df.iloc[fold.val_indices].copy() if fold.val_indices is not None else None
        test_df = cohort_df.iloc[fold.test_indices].copy()

        model = FinFusionXGBoost(
            random_state=random_state,
            n_estimators=500,
            threshold=0.50,
            tree_method="hist",
            n_jobs=4,
        )

        model.fit(
            train_df=train_df,
            feature_cols=feature_cols,
            target_col="target",
            ticker_col="ticker",
            val_df=val_df,
            early_stopping_rounds=early_stopping_rounds,
        )

        # Predict on test partition
        y_prob = model.predict_proba(test_df, feature_cols=feature_cols, ticker_col="ticker")
        y_pred = model.predict(test_df, feature_cols=feature_cols, ticker_col="ticker")
        y_true = test_df["target"].values.astype(int)

        # Baseline predictions
        always_up, prev_dir = compute_naive_baselines(test_df)

        test_pred_df = test_df[["date", "ticker", "target_date"]].copy()
        test_pred_df["y_true"] = y_true
        test_pred_df["y_pred"] = y_pred
        test_pred_df["y_prob"] = y_prob
        test_pred_df["baseline_always_up"] = always_up
        test_pred_df["baseline_previous_direction"] = prev_dir
        test_pred_df["fold_idx"] = fold.fold_idx
        predictions_list.append(test_pred_df)

        # Fold metrics
        m = compute_classification_metrics(y_true, y_pred, y_prob)
        m["fold_idx"] = fold.fold_idx
        m["n_train"] = len(train_df)
        m["n_val"] = len(val_df) if val_df is not None else 0
        m["n_test"] = len(test_df)
        m["test_start_date"] = test_df["date"].min().strftime("%Y-%m-%d")
        m["test_end_date"] = test_df["date"].max().strftime("%Y-%m-%d")
        fold_metrics_list.append(m)

        # Feature importance
        imp = model.get_feature_importance()
        imp["fold_idx"] = fold.fold_idx
        importance_list.append(imp)

    all_preds = pd.concat(predictions_list, ignore_index=True)
    fold_metrics_df = pd.DataFrame(fold_metrics_list)
    feature_imp_df = pd.concat(importance_list, ignore_index=True)

    # Date-level equal-weighted metrics
    date_metrics_df = compute_date_level_metrics(all_preds)
    mean_date_acc = float(date_metrics_df["accuracy"].mean())

    # Overall pooled metrics
    pooled_metrics = compute_classification_metrics(
        all_preds["y_true"].values,
        all_preds["y_pred"].values,
        all_preds["y_prob"].values,
    )

    # Lift over naive baselines
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

    # Aggregate feature importance across folds
    mean_imp = (
        feature_imp_df.groupby("feature")[["gain", "weight", "cover"]]
        .mean()
        .sort_values("gain", ascending=False)
        .reset_index()
    )
    mean_imp.to_csv(run_path / "feature_importance.csv", index=False)

    # Metadata & summary
    summary = {
        "experiment_id": experiment_id,
        "description": features_desc,
        "protocol": protocol,
        "news_days_only": news_days_only,
        "execution_timestamp": timestamp_str,
        "n_folds": len(folds),
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
