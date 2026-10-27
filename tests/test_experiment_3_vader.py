"""Unit Tests for FinFusionNet Experiment 3 (News Cleaning, VADER Sentiment, and Cutoff Timing)."""

from datetime import time
import pandas as pd
import numpy as np
import pytest
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from finfusionnet.data.news_cleaner import (
    clean_article_title,
    filter_editorial_domains,
    normalize_timestamps_to_ist,
    assign_trading_date_with_cutoff,
    deduplicate_articles,
)
from finfusionnet.features.sentiment import integrate_sentiment_features


def test_clean_article_title():
    # Test HTML unescaping
    raw1 = "Reliance &amp; Disney Announce &quot;Mega Merger&quot; - Moneycontrol"
    clean1 = clean_article_title(raw1)
    assert "&amp;" not in clean1
    assert "&quot;" not in clean1
    assert "Reliance & Disney Announce \"Mega Merger\"" in clean1
    assert not clean1.endswith("- Moneycontrol")

    # Test publisher suffix stripping with different dashes
    raw2 = "TCS Q3 Results: Net Profit Rises 8% – Economic Times"
    clean2 = clean_article_title(raw2)
    assert clean2 == "TCS Q3 Results: Net Profit Rises 8%"


def test_filter_editorial_domains():
    df = pd.DataFrame({
        "title": ["Article 1", "Social Post 1", "Article 2", "Video 1"],
        "publisher_title": ["Moneycontrol", "Facebook", "Livemint", "YouTube"],
        "publisher_href": ["https://moneycontrol.com", "https://facebook.com", "https://livemint.com", "https://youtube.com"]
    })
    df_filtered = filter_editorial_domains(df)
    assert len(df_filtered) == 2
    assert set(df_filtered["publisher_title"]) == {"Moneycontrol", "Livemint"}


def test_normalize_timestamps_to_ist():
    df = pd.DataFrame({
        "published_date": [
            "Wed, 22 Jan 2020 08:00:00 GMT",  # 08:00 UTC = 13:30 IST
            "Fri, 15 Sep 2023 11:30:00 GMT",  # 11:30 UTC = 17:00 IST
        ]
    })
    df_ist = normalize_timestamps_to_ist(df)
    assert len(df_ist) == 2
    assert "pub_timestamp_ist" in df_ist.columns
    # Check 08:00 UTC -> 13:30 IST
    t1 = df_ist.loc[0, "pub_time_ist"]
    assert t1.hour == 13 and t1.minute == 30
    # Check 11:30 UTC -> 17:00 IST
    t2 = df_ist.loc[1, "pub_time_ist"]
    assert t2.hour == 17 and t2.minute == 0


def test_assign_trading_date_with_cutoff():
    # Trading calendar: Monday 2024-09-02, Tuesday 2024-09-03, Wednesday 2024-09-04
    calendar = [
        pd.Timestamp("2024-09-02"),
        pd.Timestamp("2024-09-03"),
        pd.Timestamp("2024-09-04"),
    ]

    df = pd.DataFrame({
        "ticker": ["RELIANCE", "RELIANCE", "RELIANCE", "RELIANCE"],
        "clean_title": ["Headline 1", "Headline 2", "Headline 3", "Headline 4"],
        "pub_date_ist": [
            pd.to_datetime("2024-09-02").date(),  # Monday 10:00 (before cutoff)
            pd.to_datetime("2024-09-02").date(),  # Monday 16:00 (after cutoff)
            pd.to_datetime("2024-09-01").date(),  # Sunday (weekend)
            pd.to_datetime("2024-09-03").date(),  # Tuesday 14:00 (before cutoff)
        ],
        "pub_time_ist": [
            time(10, 0),
            time(16, 0),
            time(12, 0),
            time(14, 0),
        ]
    })

    df_aligned = assign_trading_date_with_cutoff(df, calendar, cutoff_time=time(15, 30))

    # Headline 1 (Monday 10:00) -> Monday 2024-09-02
    assert df_aligned.loc[0, "feature_date"] == pd.Timestamp("2024-09-02")
    # Headline 2 (Monday 16:00) -> Tuesday 2024-09-03 (rolled forward)
    assert df_aligned.loc[1, "feature_date"] == pd.Timestamp("2024-09-03")
    # Headline 3 (Sunday 12:00) -> Monday 2024-09-02 (rolled forward to next trading session)
    assert df_aligned.loc[2, "feature_date"] == pd.Timestamp("2024-09-02")
    # Headline 4 (Tuesday 14:00) -> Tuesday 2024-09-03
    assert df_aligned.loc[3, "feature_date"] == pd.Timestamp("2024-09-03")


def test_deduplicate_articles():
    df = pd.DataFrame({
        "ticker": ["TCS", "TCS", "TCS"],
        "feature_date": [pd.Timestamp("2024-09-02"), pd.Timestamp("2024-09-02"), pd.Timestamp("2024-09-03")],
        "clean_title": ["TCS Q2 Results Beat Estimates", "tcs q2 results beat estimates", "TCS Q2 Results Beat Estimates"],
    })
    df_deduped = deduplicate_articles(df)
    # The two articles on 2024-09-02 should deduplicate to 1; article on 2024-09-03 retained
    assert len(df_deduped) == 2


def test_vader_polarity_bounds():
    analyzer = SentimentIntensityAnalyzer()
    positive_title = "Reliance Industries Reports Record Quarterly Profit and 25% Revenue Growth"
    negative_title = "Tech Giant Faces Severe Regulatory Penalty and Massive Margin Decline"

    pos_score = analyzer.polarity_scores(positive_title)["compound"]
    neg_score = analyzer.polarity_scores(negative_title)["compound"]

    assert pos_score > 0.3
    assert neg_score < -0.3
    assert -1.0 <= pos_score <= 1.0
    assert -1.0 <= neg_score <= 1.0


def test_strict_missing_news_is_null():
    market_df = pd.DataFrame({
        "date": [pd.Timestamp("2024-09-02"), pd.Timestamp("2024-09-02")],
        "ticker": ["RELIANCE", "TCS"],
        "close": [2500.0, 3500.0],
    })
    news_df = pd.DataFrame({
        "date": [pd.Timestamp("2024-09-02")],
        "ticker": ["RELIANCE"],
        "n_articles": [3],
        "has_news": [1],
        "vader_mean": [0.45],
        "vader_positive_mean": [0.30],
        "vader_negative_mean": [0.05],
        "vader_neutral_mean": [0.65],
        "vader_dispersion": [0.10],
    })

    merged = integrate_sentiment_features(market_df, news_df)
    assert len(merged) == 2

    # RELIANCE has news
    rel = merged[merged["ticker"] == "RELIANCE"].iloc[0]
    assert rel["has_news"] == 1
    assert rel["n_articles"] == 3
    assert rel["vader_mean"] == 0.45

    # TCS has NO news -> must be strictly NaN, never 0.0 or neutral
    tcs = merged[merged["ticker"] == "TCS"].iloc[0]
    assert tcs["has_news"] == 0
    assert tcs["n_articles"] == 0
    assert pd.isna(tcs["vader_mean"])
    assert pd.isna(tcs["vader_positive_mean"])
    assert pd.isna(tcs["vader_negative_mean"])
    assert pd.isna(tcs["vader_neutral_mean"])
    assert pd.isna(tcs["vader_dispersion"])
