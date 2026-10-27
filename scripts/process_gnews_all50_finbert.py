"""FinFusionNet: Clean All-50 GNews Corpus and Compute Daily FinBERT Sentiment Features.

Reuses the audited 15:30 IST causally aligned cleaned articles:
    data/processed/gnews_all50/gnews_all50_articles_clean.parquet

Computes FinBERT sentiment for unique headlines:
    - Positive probability
    - Negative probability
    - Neutral probability
    - Net sentiment score (pos - neg)
    - Sentiment dispersion (std across day's articles, 0.0 if n=1)
    - Confidence (mean maximum class probability)

Maintains checkpointing in data/processed/gnews_all50/finbert_title_cache.parquet.
Saves daily ticker-day features to data/processed/gnews_all50/daily_gnews_all50_finbert_features.parquet.
"""

from datetime import time
from pathlib import Path
import time as time_lib
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

RAW_PARQUET = Path("data/raw/gnews_all50/raw_articles_all50.parquet")
PROCESSED_DIR = Path("data/processed/gnews_all50")
CLEAN_ARTICLES_PARQUET = PROCESSED_DIR / "gnews_all50_articles_clean.parquet"
CACHE_PARQUET = PROCESSED_DIR / "finbert_title_cache.parquet"
DAILY_FEATURES_PARQUET = PROCESSED_DIR / "daily_gnews_all50_finbert_features.parquet"

MODEL_IDENTIFIER = "ProsusAI/finbert"
LOCAL_SNAPSHOT_PATH = Path(
    r"C:\Users\Neil\.cache\huggingface\hub\models--ProsusAI--finbert\snapshots\4556d13015211d73dccd3fdd39d39232506f3e43"
)


def load_finbert():
    """Load local FinBERT tokenizer and sequence classification model."""
    load_path = str(LOCAL_SNAPSHOT_PATH) if LOCAL_SNAPSHOT_PATH.exists() else MODEL_IDENTIFIER
    print(f"Loading FinBERT model and tokenizer from {load_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(load_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(load_path, local_files_only=True)
    model.eval()
    torch.set_num_threads(8)
    # FinBERT label mapping: 0 -> positive, 1 -> negative, 2 -> neutral
    print(f"Loaded FinBERT. id2label: {model.config.id2label}", flush=True)
    return tokenizer, model


def extract_finbert_unique_titles(unique_titles, tokenizer, model, batch_size=64):
    """Run batch FinBERT inference on unique headlines with checkpointing."""
    CACHE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    
    cached_df = pd.DataFrame()
    if CACHE_PARQUET.exists():
        try:
            cached_df = pd.read_parquet(CACHE_PARQUET)
            print(f"Found existing FinBERT title cache with {len(cached_df):,} scored titles.", flush=True)
        except Exception as e:
            print(f"Warning: could not read existing cache ({e}). Starting fresh.", flush=True)

    cached_titles = set(cached_df["clean_title"]) if not cached_df.empty else set()
    remaining_titles = [t for t in unique_titles if t not in cached_titles]
    print(f"Total unique titles: {len(unique_titles):,} | Cached: {len(cached_titles):,} | Remaining: {len(remaining_titles):,}", flush=True)

    if not remaining_titles:
        return cached_df

    # Sort remaining by length to minimize dynamic padding overhead
    remaining_titles = sorted(remaining_titles, key=len)

    records = cached_df.to_dict("records") if not cached_df.empty else []
    total = len(remaining_titles)
    t0 = time_lib.time()

    for idx in range(0, total, batch_size):
        batch = remaining_titles[idx : idx + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")

        with torch.inference_mode():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        for title, prob in zip(batch, probs):
            p_pos = float(prob[0])
            p_neg = float(prob[1])
            p_neu = float(prob[2])
            score = p_pos - p_neg
            conf = float(max(p_pos, p_neg, p_neu))
            records.append({
                "clean_title": title,
                "finbert_pos": p_pos,
                "finbert_neg": p_neg,
                "finbert_neu": p_neu,
                "finbert_score": score,
                "finbert_conf": conf,
            })

        processed_so_far = idx + len(batch)
        if (processed_so_far % 5000 < batch_size) or (processed_so_far == total):
            elapsed = time_lib.time() - t0
            rate = processed_so_far / elapsed if elapsed > 0 else 0
            eta_min = (total - processed_so_far) / rate / 60 if rate > 0 else 0
            print(
                f"  Scored {processed_so_far:,}/{total:,} headlines ({rate:.1f} items/s, ETA: {eta_min:.1f}m)... Checkpointing...",
                flush=True,
            )
            df_checkpoint = pd.DataFrame(records)
            df_checkpoint.to_parquet(CACHE_PARQUET, index=False)

    df_scored = pd.DataFrame(records)
    df_scored.to_parquet(CACHE_PARQUET, index=False)
    print(f"Title scoring complete. Total cached titles: {len(df_scored):,}.", flush=True)
    return df_scored


def process_all50_gnews_and_extract_finbert():
    if not CLEAN_ARTICLES_PARQUET.exists():
        raise FileNotFoundError(f"Cleaned articles not found at {CLEAN_ARTICLES_PARQUET}.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[1/4] Loading cleaned articles from {CLEAN_ARTICLES_PARQUET}...", flush=True)
    df_clean = pd.read_parquet(CLEAN_ARTICLES_PARQUET)
    print(f"Loaded {len(df_clean):,} cleaned articles across {df_clean['ticker'].nunique()} tickers.", flush=True)

    # 2. Extract and Score Unique Titles
    print("[2/4] Extracting and scoring unique headlines with FinBERT...", flush=True)
    unique_titles = df_clean["clean_title"].dropna().unique().tolist()
    tokenizer, model = load_finbert()
    df_titles_scored = extract_finbert_unique_titles(unique_titles, tokenizer, model, batch_size=64)

    # 3. Merge Back to Article Level
    print("[3/4] Merging FinBERT scores back to article-level records...", flush=True)
    df_merged = df_clean.merge(df_titles_scored, on="clean_title", how="left")

    # 4. Aggregate to Ticker-Day Level
    print("[4/4] Aggregating to Daily Ticker-Day FinBERT Features...", flush=True)
    daily_groups = df_merged.groupby(["ticker", "feature_date"])

    daily_records = []
    for (ticker, f_date), grp in daily_groups:
        n_art = len(grp)
        p_pos = float(grp["finbert_pos"].mean())
        p_neg = float(grp["finbert_neg"].mean())
        p_neu = float(grp["finbert_neu"].mean())
        s_score = float(grp["finbert_score"].mean())
        s_disp = float(grp["finbert_score"].std()) if n_art > 1 else 0.0
        s_conf = float(grp["finbert_conf"].mean())

        daily_records.append({
            "ticker": ticker,
            "date": pd.Timestamp(f_date),
            "n_articles": n_art,
            "has_news": 1,
            "finbert_positive_mean": p_pos,
            "finbert_negative_mean": p_neg,
            "finbert_neutral_mean": p_neu,
            "finbert_sentiment_score": s_score,
            "finbert_mean": s_score,
            "finbert_dispersion": s_disp,
            "finbert_confidence": s_conf,
        })

    df_daily = pd.DataFrame(daily_records)
    df_daily = df_daily.sort_values(["date", "ticker"]).reset_index(drop=True)
    df_daily.to_parquet(DAILY_FEATURES_PARQUET, index=False)
    print(f"Saved daily FinBERT features to {DAILY_FEATURES_PARQUET} ({len(df_daily):,} ticker-day records across {df_daily['ticker'].nunique()} tickers).", flush=True)

    # Validation Summary
    print("\n--- Daily FinBERT Dataset Summary ---", flush=True)
    print(f"Date Range: {df_daily['date'].min().date()} to {df_daily['date'].max().date()}", flush=True)
    print(f"Unique Tickers Covered: {df_daily['ticker'].nunique()}", flush=True)
    print(f"Total Ticker-Day Observations with News: {len(df_daily):,}", flush=True)
    print(f"Mean Articles per Active News Day: {df_daily['n_articles'].mean():.2f}", flush=True)
    print(f"FinBERT Net Sentiment Mean: {df_daily['finbert_sentiment_score'].mean():.4f} (std: {df_daily['finbert_sentiment_score'].std():.4f})", flush=True)
    print(f"FinBERT Positive Prob Mean: {df_daily['finbert_positive_mean'].mean():.4f}", flush=True)
    print(f"FinBERT Negative Prob Mean: {df_daily['finbert_negative_mean'].mean():.4f}", flush=True)
    print(f"FinBERT Neutral Prob Mean: {df_daily['finbert_neutral_mean'].mean():.4f}", flush=True)
    print(f"FinBERT Dispersion Mean: {df_daily['finbert_dispersion'].mean():.4f}", flush=True)
    print(f"FinBERT Confidence Mean: {df_daily['finbert_confidence'].mean():.4f}", flush=True)


if __name__ == "__main__":
    process_all50_gnews_and_extract_finbert()
