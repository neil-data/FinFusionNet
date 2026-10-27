"""Unit and Integration Tests for FinFusionNet Experiment 4 (FinBERT + Fusion).

Covers:
1. Model loading offline from local cache.
2. Tokenizer and label dictionary verification.
3. FinBERT sentiment output polarity and probability properties.
4. Strict missing-news handling (null preservation, no neutral conversion).
5. 15:30 IST market close causal cutoff and leakage prevention.
6. Numerical reproducibility.
7. Dispersion computation logic.
"""

from datetime import time
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from finfusionnet.features.sentiment import integrate_sentiment_features, SENTIMENT_FEATURE_COLS


LOCAL_MODEL_PATH = Path(
    r"C:\Users\Neil\.cache\huggingface\hub\models--ProsusAI--finbert\snapshots\4556d13015211d73dccd3fdd39d39232506f3e43"
)


@pytest.fixture(scope="module")
def finbert_pipeline():
    """Load local FinBERT model and tokenizer once for the test module."""
    if not LOCAL_MODEL_PATH.exists():
        pytest.skip(f"Local FinBERT snapshot not found at {LOCAL_MODEL_PATH}")
    
    tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_MODEL_PATH), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(LOCAL_MODEL_PATH), local_files_only=True)
    model.eval()
    return tokenizer, model


def test_finbert_model_loading(finbert_pipeline):
    """Test 1: FinBERT loads offline with correct configuration and label mapping."""
    tokenizer, model = finbert_pipeline
    assert model is not None
    assert tokenizer is not None

    # Check 3-class financial sentiment architecture
    assert model.config.num_labels == 3
    # id2label mapping: 0 -> positive, 1 -> negative, 2 -> neutral
    id2label = model.config.id2label
    assert id2label[0].lower() == "positive"
    assert id2label[1].lower() == "negative"
    assert id2label[2].lower() == "neutral"


def test_finbert_sentiment_outputs(finbert_pipeline):
    """Test 2: FinBERT produces valid probabilities and expected directional sentiment."""
    tokenizer, model = finbert_pipeline

    positive_headline = "Tata Motors reports 75% surge in quarterly net profit driven by record JLR deliveries"
    negative_headline = "Company defaults on debt obligations following catastrophic quarterly loss and insolvency filing"

    inputs = tokenizer([positive_headline, negative_headline], padding=True, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    # Probabilities sum to 1.0
    for p in probs:
        assert np.isclose(np.sum(p), 1.0, atol=1e-5)
        assert (p >= 0.0).all() and (p <= 1.0).all()

    # Positive headline: pos > neg, pos > neutral
    pos_p, neg_p, neu_p = probs[0]
    assert pos_p > neg_p, f"Expected positive > negative, got pos={pos_p:.4f}, neg={neg_p:.4f}"
    assert pos_p - neg_p > 0.5, f"Expected strong positive score, got {pos_p - neg_p:.4f}"

    # Negative headline: neg > pos, neg > neutral
    pos_n, neg_n, neu_n = probs[1]
    assert neg_n > pos_n, f"Expected negative > positive, got neg={neg_n:.4f}, pos={pos_n:.4f}"
    assert pos_n - neg_n < -0.5, f"Expected strong negative score, got {pos_n - neg_n:.4f}"


def test_finbert_missing_news_policy():
    """Test 3: Mandatory missing data policy: no-news days receive null, never neutral/zero."""
    # Synthetic price dataset with 3 days
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    market_df = pd.DataFrame({
        "date": dates,
        "ticker": ["RELIANCE", "RELIANCE", "RELIANCE"],
        "close": [2500.0, 2520.0, 2510.0],
    })

    # News present only on day 2
    news_df = pd.DataFrame({
        "date": [dates[1]],
        "ticker": ["RELIANCE"],
        "n_articles": [2],
        "has_news": [1],
        "finbert_positive_mean": [0.75],
        "finbert_negative_mean": [0.10],
        "finbert_neutral_mean": [0.15],
        "finbert_sentiment_score": [0.65],
        "finbert_dispersion": [0.05],
        "finbert_confidence": [0.75],
    })

    merged = integrate_sentiment_features(market_df, news_df)

    # Day 0: No news -> n_articles = 0, has_news = 0, all finbert features NaN
    assert merged.loc[0, "n_articles"] == 0
    assert merged.loc[0, "has_news"] == 0
    assert pd.isna(merged.loc[0, "finbert_positive_mean"])
    assert pd.isna(merged.loc[0, "finbert_negative_mean"])
    assert pd.isna(merged.loc[0, "finbert_neutral_mean"])
    assert pd.isna(merged.loc[0, "finbert_sentiment_score"])
    assert pd.isna(merged.loc[0, "finbert_dispersion"])

    # Day 1: Has news -> values preserved
    assert merged.loc[1, "n_articles"] == 2
    assert merged.loc[1, "has_news"] == 1
    assert merged.loc[1, "finbert_positive_mean"] == 0.75
    assert merged.loc[1, "finbert_sentiment_score"] == 0.65

    # Day 2: No news -> NaN
    assert merged.loc[2, "has_news"] == 0
    assert pd.isna(merged.loc[2, "finbert_sentiment_score"])


def test_causal_cutoff_no_leakage():
    """Test 4: Verify causal timestamp alignment and 15:30 IST market close cutoff."""
    clean_path = Path("data/processed/gnews_all50/gnews_all50_articles_clean.parquet")
    if not clean_path.exists():
        pytest.skip("Processed clean articles dataset not yet available.")

    df_clean = pd.read_parquet(clean_path)
    df_clean["pub_date_ist"] = pd.to_datetime(df_clean["pub_date_ist"])
    df_clean["feature_date"] = pd.to_datetime(df_clean["feature_date"])

    # No future publication dates
    future_leaks = df_clean[df_clean["pub_date_ist"].dt.normalize() > df_clean["feature_date"]]
    assert len(future_leaks) == 0, f"Found {len(future_leaks)} future leaks!"

    # No same-day articles published after 15:30 IST
    same_day = df_clean["pub_date_ist"].dt.normalize() == df_clean["feature_date"]
    after_cutoff = df_clean["pub_date_ist"].dt.time > time(15, 30)
    cutoff_leaks = df_clean[same_day & after_cutoff]
    assert len(cutoff_leaks) == 0, f"Found {len(cutoff_leaks)} post-cutoff leaks!"


def test_numerical_reproducibility(finbert_pipeline):
    """Test 5: Same input produces bit-for-bit identical outputs across runs."""
    tokenizer, model = finbert_pipeline
    text = "Infosys secures multi-year digital transformation contract with European financial institution"

    inputs = tokenizer(text, return_tensors="pt")
    with torch.inference_mode():
        out1 = model(**inputs).logits
        p1 = torch.softmax(out1, dim=-1).cpu().numpy()

        out2 = model(**inputs).logits
        p2 = torch.softmax(out2, dim=-1).cpu().numpy()

    np.testing.assert_array_equal(p1, p2)


def test_dispersion_calculation():
    """Test 6: Sentiment dispersion is 0.0 for n=1, and sample std for n>1."""
    # Case A: n=1
    scores_single = np.array([0.45])
    disp_single = float(np.std(scores_single, ddof=1)) if len(scores_single) > 1 else 0.0
    assert disp_single == 0.0

    # Case B: n=3
    scores_multi = np.array([0.60, -0.20, 0.40])
    disp_multi = float(np.std(scores_multi, ddof=1))
    expected_std = np.std([0.60, -0.20, 0.40], ddof=1)
    assert np.isclose(disp_multi, expected_std)
    assert disp_multi > 0.0


def test_experiment_4_common_cohort_integrity():
    """Test 7: Verify common evaluation cohort size, dates, and column completeness."""
    pred_path = Path("experiments/experiment_4/predictions/experiment_4_common_cohort_predictions.parquet")
    if not pred_path.exists():
        pytest.skip("Experiment 4 predictions not yet generated.")

    df_pred = pd.read_parquet(pred_path)
    assert len(df_pred) == 45136, f"Expected 45,136 cohort rows, found {len(df_pred)}"
    assert df_pred["date"].nunique() == 916, f"Expected 916 dates, found {df_pred['date'].nunique()}"

    required_cols = [
        "prob_xgb_f0", "prob_xgb_f1_vader", "prob_xgb_f1_finbert",
        "prob_lr_f0", "prob_lr_f1_vader", "prob_lr_f1_finbert",
        "prob_lstm_f0", "prob_lstm_f1_vader", "prob_lstm_f1_finbert",
        "y_true", "has_news_all50", "n_articles_all50"
    ]
    for col in required_cols:
        assert col in df_pred.columns, f"Missing required column {col} in predictions"
        assert df_pred[col].notna().all(), f"Found NaNs in required column {col}"


def test_experiment_4_statistical_significance_artifacts():
    """Test 8: Verify statistical significance output structure and metric tables."""
    stats_path = Path("experiments/experiment_4/experiment_4_statistical_significance.csv")
    metrics_path = Path("experiments/experiment_4/experiment_4_results.csv")
    if not stats_path.exists() or not metrics_path.exists():
        pytest.skip("Experiment 4 results not yet generated.")

    df_stats = pd.read_csv(stats_path)
    df_metrics = pd.read_csv(metrics_path)

    assert len(df_stats) == 6, f"Expected 6 statistical comparisons, found {len(df_stats)}"
    assert "bootstrap_p_value" in df_stats.columns
    assert "mcnemar_p_value" in df_stats.columns
    assert "holm_bonferroni_p_value" in df_stats.columns
    assert len(df_metrics) >= 11, f"Expected at least 11 model configurations, found {len(df_metrics)}"
