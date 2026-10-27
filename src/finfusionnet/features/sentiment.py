"""FinFusionNet: News Availability and Sentiment Feature Integration.

Implements Group C (News availability controls), Group D (VADER),
and Group E (FinBERT) sentiment feature integration into ticker-day rows.
Enforces strict missing-value handling: no-news days receive null sentiment,
never 0.0 or neutral.
"""

from typing import List, Optional
import pandas as pd
import numpy as np


SENTIMENT_FEATURE_COLS = [
    # Group C: News Availability Controls
    "n_articles",
    "has_news",
    # Group D: VADER Sentiment
    "vader_mean",
    "vader_positive_mean",
    "vader_negative_mean",
    "vader_neutral_mean",
    "vader_dispersion",
    # Group E: FinBERT Sentiment
    "finbert_positive_mean",
    "finbert_negative_mean",
    "finbert_neutral_mean",
    "finbert_sentiment_score",
    "finbert_mean",
    "finbert_dispersion",
    "finbert_confidence",
]


def integrate_sentiment_features(
    market_df: pd.DataFrame,
    news_sentiment_df: Optional[pd.DataFrame] = None,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """Merge ticker-day news sentiment features with market price data.

    Enforces mandatory missing data policy:
    1. If a ticker-day observation has no news:
       - n_articles = 0
       - has_news = 0
       - All sentiment features (VADER & FinBERT) = np.nan (null)
    2. Missing sentiment is NEVER replaced with 0.0 or neutral probability.
    3. Legitimate missingness is preserved for XGBoost native handling.
    """
    df = market_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    if news_sentiment_df is None or news_sentiment_df.empty:
        # No news at all for entire dataset: populate standard nulls
        df["n_articles"] = 0
        df["has_news"] = 0
        for col in SENTIMENT_FEATURE_COLS[2:]:
            df[col] = np.nan
        return df

    news = news_sentiment_df.copy()
    news[date_col] = pd.to_datetime(news[date_col])
    news[ticker_col] = news[ticker_col].str.upper()

    # Left join to preserve all eligible market ticker-day observations
    df = df.merge(news, on=[date_col, ticker_col], how="left")

    # Enforce no-news rules
    df["n_articles"] = df["n_articles"].fillna(0).astype(int)
    df["has_news"] = np.where(df["n_articles"] > 0, 1, 0)

    # For any rows where has_news == 0, guarantee sentiment features are null
    no_news_mask = (df["has_news"] == 0)
    for col in SENTIMENT_FEATURE_COLS[2:]:
        if col in df.columns:
            df.loc[no_news_mask, col] = np.nan
        else:
            df[col] = np.nan

    return df
