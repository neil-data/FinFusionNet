"""FinFusionNet: Clean GNews Raw Articles and Compute Daily VADER Sentiment Features.

Enforces:
1. Strict IST timezone normalization.
2. Market close cutoff (15:30 IST) trading calendar alignment.
3. Deduplication of syndicated titles.
4. Non-editorial domain filtering.
5. VADER sentiment scoring and aggregation to ticker-day features.
"""

import os
from datetime import time
from pathlib import Path
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from finfusionnet.data.trading_calendar import get_trading_calendar
from finfusionnet.data.news_cleaner import (
    clean_article_title,
    filter_editorial_domains,
    normalize_timestamps_to_ist,
    assign_trading_date_with_cutoff,
    deduplicate_articles,
)

RAW_PARQUET = Path("data/raw/gnews/raw_articles.parquet")
PROCESSED_DIR = Path("data/processed/gnews")
CLEAN_ARTICLES_PARQUET = PROCESSED_DIR / "gnews_articles_clean.parquet"
DAILY_FEATURES_PARQUET = PROCESSED_DIR / "daily_gnews_vader_features.parquet"


def process_gnews_and_extract_vader():
    if not RAW_PARQUET.exists():
        raise FileNotFoundError(f"Raw articles not found at {RAW_PARQUET}. Please run historical collection first.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[1/6] Loading raw articles from {RAW_PARQUET}...", flush=True)
    df_raw = pd.read_parquet(RAW_PARQUET)
    print(f"Loaded {len(df_raw):,} raw article records across {df_raw['ticker'].nunique()} tickers.", flush=True)

    # 1. Title Cleaning
    print("[2/6] Cleaning headlines and unescaping HTML...", flush=True)
    df_raw["clean_title"] = df_raw["title"].apply(clean_article_title)
    # Drop records with empty cleaned title
    df_clean = df_raw[df_raw["clean_title"].str.strip() != ""].copy()
    print(f"Retained {len(df_clean):,} articles with non-empty cleaned titles.", flush=True)

    # 2. Filter Editorial Domains
    print("[3/6] Filtering non-editorial / social media domains...", flush=True)
    df_editorial = filter_editorial_domains(df_clean)
    dropped_social = len(df_clean) - len(df_editorial)
    print(f"Dropped {dropped_social:,} social/non-editorial items; retained {len(df_editorial):,} editorial records.", flush=True)

    # 3. Normalize Timestamps to IST
    print("[4/6] Parsing RFC 822 timestamps and converting to Asia/Kolkata (IST)...", flush=True)
    df_ist = normalize_timestamps_to_ist(df_editorial)
    print(f"Successfully converted {len(df_ist):,} timestamps to IST.", flush=True)

    # 4. Trading Day Alignment with 3:30 PM IST Cutoff
    print("[5/6] Aligning articles to causal trading dates using 15:30 IST market close cutoff...", flush=True)
    calendar = get_trading_calendar("2020-01-01", "2026-09-30")
    df_aligned = assign_trading_date_with_cutoff(df_ist, calendar, cutoff_time=time(15, 30))
    print(f"Aligned {len(df_aligned):,} articles to trading dates in calendar.", flush=True)

    # 5. Deduplicate Articles on (ticker, feature_date, normalized_title)
    df_deduped = deduplicate_articles(df_aligned)
    dropped_dupes = len(df_aligned) - len(df_deduped)
    print(f"Deduplication complete: dropped {dropped_dupes:,} duplicate syndicated titles; retained {len(df_deduped):,} unique articles.", flush=True)

    # 6. Compute Article-Level VADER Sentiment
    print("[6/6] Computing VADER polarity scores (compound, pos, neg, neu)...", flush=True)
    analyzer = SentimentIntensityAnalyzer()
    
    compounds = []
    pos_scores = []
    neg_scores = []
    neu_scores = []

    for title in df_deduped["clean_title"]:
        scores = analyzer.polarity_scores(title)
        compounds.append(scores["compound"])
        pos_scores.append(scores["pos"])
        neg_scores.append(scores["neg"])
        neu_scores.append(scores["neu"])

    df_deduped["vader_compound"] = compounds
    df_deduped["vader_pos"] = pos_scores
    df_deduped["vader_neg"] = neg_scores
    df_deduped["vader_neu"] = neu_scores

    # Save cleaned article-level dataset
    df_deduped.to_parquet(CLEAN_ARTICLES_PARQUET, index=False)
    print(f"Saved cleaned article dataset to {CLEAN_ARTICLES_PARQUET} ({len(df_deduped):,} records).", flush=True)

    # 7. Aggregate to Ticker-Day Level
    print("\nAggregating to Daily Ticker-Day Sentiment Features...", flush=True)
    daily_groups = df_deduped.groupby(["ticker", "feature_date"])

    daily_records = []
    for (ticker, f_date), grp in daily_groups:
        n_art = len(grp)
        v_mean = float(grp["vader_compound"].mean())
        v_pos = float(grp["vader_pos"].mean())
        v_neg = float(grp["vader_neg"].mean())
        v_neu = float(grp["vader_neu"].mean())
        v_disp = float(grp["vader_compound"].std()) if n_art > 1 else 0.0

        daily_records.append({
            "ticker": ticker,
            "date": pd.Timestamp(f_date),
            "n_articles": n_art,
            "has_news": 1,
            "vader_mean": v_mean,
            "vader_positive_mean": v_pos,
            "vader_negative_mean": v_neg,
            "vader_neutral_mean": v_neu,
            "vader_dispersion": v_disp,
        })

    df_daily = pd.DataFrame(daily_records)
    df_daily = df_daily.sort_values(["date", "ticker"]).reset_index(drop=True)
    df_daily.to_parquet(DAILY_FEATURES_PARQUET, index=False)
    print(f"Saved daily VADER features to {DAILY_FEATURES_PARQUET} ({len(df_daily):,} ticker-day records).", flush=True)

    # Validation Checks
    print("\n--- Daily VADER Dataset Summary ---", flush=True)
    print(f"Date Range: {df_daily['date'].min().date()} to {df_daily['date'].max().date()}", flush=True)
    print(f"Unique Tickers Covered: {df_daily['ticker'].nunique()} ({list(df_daily['ticker'].unique())})", flush=True)
    print(f"Total Ticker-Day Observations with News: {len(df_daily):,}", flush=True)
    print(f"Mean Articles per Active News Day: {df_daily['n_articles'].mean():.2f}", flush=True)
    print(f"VADER Compound Sentiment Mean: {df_daily['vader_mean'].mean():.4f} (std: {df_daily['vader_mean'].std():.4f})", flush=True)


if __name__ == "__main__":
    process_gnews_and_extract_vader()
