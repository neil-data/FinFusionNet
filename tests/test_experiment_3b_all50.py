"""Tests for FinFusionNet Experiment 3B: All-50 Company News Coverage Expansion."""

from datetime import time
from pathlib import Path
import pandas as pd
import pytest


def test_processed_articles_leakage_and_cutoff():
    clean_path = Path("data/processed/gnews_all50/gnews_all50_articles_clean.parquet")
    if not clean_path.exists():
        pytest.skip("Processed articles dataset not yet generated.")

    df_clean = pd.read_parquet(clean_path)
    df_clean["pub_date_ist"] = pd.to_datetime(df_clean["pub_date_ist"])
    df_clean["feature_date"] = pd.to_datetime(df_clean["feature_date"])

    # 1. No future publication leaks (pub_date > feature_date)
    future_leaks = df_clean[df_clean["pub_date_ist"].dt.normalize() > df_clean["feature_date"]]
    assert len(future_leaks) == 0, f"Found {len(future_leaks)} future publication leaks!"

    # 2. No post-15:30 IST same-day leaks
    same_day = df_clean["pub_date_ist"].dt.normalize() == df_clean["feature_date"]
    after_cutoff = df_clean["pub_date_ist"].dt.time > time(15, 30)
    cutoff_leaks = df_clean[same_day & after_cutoff]
    assert len(cutoff_leaks) == 0, f"Found {len(cutoff_leaks)} post-15:30 IST same-day leaks!"


def test_daily_vader_features_missing_policy():
    daily_path = Path("data/processed/gnews_all50/daily_gnews_all50_vader_features.parquet")
    if not daily_path.exists():
        pytest.skip("Daily VADER features dataset not yet generated.")

    df_daily = pd.read_parquet(daily_path)

    # Active news days have has_news == 1 and n_articles >= 1
    assert (df_daily["has_news"] == 1).all()
    assert (df_daily["n_articles"] >= 1).all()
    assert df_daily["vader_mean"].notna().all()


def test_experiment_3b_common_cohort_integrity():
    pred_path = Path("experiments/experiment_3b/predictions/experiment_3b_common_cohort_predictions.parquet")
    if not pred_path.exists():
        pytest.skip("Experiment 3B predictions not yet generated.")

    df_pred = pd.read_parquet(pred_path)

    # Must match the common evaluation cohort
    assert len(df_pred) == 45136, f"Expected 45,136 cohort rows, found {len(df_pred)}"
    assert df_pred["date"].nunique() == 916, f"Expected 916 dates, found {df_pred['date'].nunique()}"

    # Required probability columns
    required_cols = [
        "prob_xgb_f0", "prob_xgb_f1_10", "prob_xgb_f1_all50",
        "prob_lr_f0", "prob_lr_f1_10", "prob_lr_f1_all50",
        "prob_lstm_f0", "prob_lstm_f1_10", "prob_lstm_f1_all50",
        "has_news_all50", "n_articles_all50", "has_news_10", "n_articles_10",
        "y_true"
    ]
    for col in required_cols:
        assert col in df_pred.columns, f"Missing required column {col} in predictions"
        assert df_pred[col].notna().all(), f"Found NaNs in required column {col}"


def test_experiment_3b_statistical_significance_artifacts():
    stats_path = Path("experiments/experiment_3b/experiment_3b_statistical_significance.csv")
    metrics_path = Path("experiments/experiment_3b/experiment_3b_results.csv")
    if not stats_path.exists() or not metrics_path.exists():
        pytest.skip("Experiment 3B results not yet generated.")

    df_stats = pd.read_csv(stats_path)
    df_metrics = pd.read_csv(metrics_path)

    assert len(df_stats) == 6, f"Expected 6 statistical comparisons, found {len(df_stats)}"
    assert "holm_bonferroni_p_value" in df_stats.columns
    assert "bootstrap_p_value" in df_stats.columns
    assert len(df_metrics) >= 11, f"Expected at least 11 model configurations, found {len(df_metrics)}"
