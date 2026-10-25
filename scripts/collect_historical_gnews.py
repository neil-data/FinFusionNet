"""FinFusionNet: Fast Resumable Historical GNews Collector.

Collects historical news headlines across 10 NIFTY 50 heavyweights from 2020-01-01 to 2026-09-30
via direct Google News RSS feed parsing, monthly chunking, file-based checkpointing,
polite rate limiting, and manifest logging.
"""

import os
import json
import time
import calendar
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import feedparser


TICKER_QUERIES = {
    "BHARTIARTL": "Bharti Airtel",
    "HDFCBANK": "HDFC Bank",
    "HINDUNILVR": "Hindustan Unilever",
    "ICICIBANK": "ICICI Bank",
    "INFY": "Infosys",
    "ITC": "ITC Limited",
    "LT": "Larsen & Toubro",
    "RELIANCE": "Reliance Industries",
    "SBIN": "State Bank of India",
    "TCS": "Tata Consultancy Services",
}

RAW_CHUNKS_DIR = Path("data/raw/gnews/chunks")
RAW_OUTPUT_PARQUET = Path("data/raw/gnews/raw_articles.parquet")
MANIFEST_PATH = Path("data/raw/gnews/manifest.jsonl")


def generate_month_ranges(start_year=2020, end_year=2026, end_month=9):
    months = []
    for y in range(start_year, end_year + 1):
        max_m = end_month if y == end_year else 12
        for m in range(1, max_m + 1):
            months.append((y, m))
    return months


def fetch_monthly_chunk(ticker: str, query: str, year: int, month: int, max_retries: int = 3) -> List[Dict[str, Any]]:
    # Format start and end date strings (YYYY-MM-DD)
    start_str = f"{year}-{month:02d}-01"
    _, last_day = calendar.monthrange(year, month)
    end_str = f"{year}-{month:02d}-{last_day:02d}"

    # Google News search query with date range
    encoded_query = urllib.parse.quote(f"{query} after:{start_str} before:{end_str}")
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

    for attempt in range(1, max_retries + 1):
        try:
            feed = feedparser.parse(url)
            
            # Check for rate limit or empty feed
            if hasattr(feed, "status") and feed.status == 429:
                delay = attempt * 3.0
                time.sleep(delay)
                continue

            clean_articles = []
            for entry in feed.entries:
                source = getattr(entry, "source", {})
                pub_title = ""
                pub_href = ""
                if isinstance(source, dict):
                    pub_title = source.get("title", "")
                    pub_href = source.get("href", "")
                elif hasattr(source, "title"):
                    pub_title = getattr(source, "title", "")
                    pub_href = getattr(source, "href", "")

                clean_articles.append({
                    "ticker": ticker,
                    "query": query,
                    "year": year,
                    "month": month,
                    "title": getattr(entry, "title", ""),
                    "description": getattr(entry, "summary", ""),
                    "published_date": getattr(entry, "published", ""),
                    "url": getattr(entry, "link", ""),
                    "publisher_title": pub_title,
                    "publisher_href": pub_href,
                })
            return clean_articles
        except Exception as e:
            if attempt < max_retries:
                time.sleep(attempt * 2.0)
            else:
                print(f"Failed [{ticker} {year}-{month:02d}] after {max_retries} attempts: {e}", flush=True)
                return []


def collect_historical_gnews():
    RAW_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    months = generate_month_ranges(2020, 2026, 9)
    total_tasks = len(TICKER_QUERIES) * len(months)
    print(f"Starting Fast GNews Historical Collection across {len(TICKER_QUERIES)} tickers and {len(months)} monthly slices ({total_tasks} chunks total)...", flush=True)

    completed_count = 0
    newly_fetched = 0
    all_articles = []

    for ticker, query in TICKER_QUERIES.items():
        ticker_articles_count = 0
        t0 = time.time()
        for year, month in months:
            chunk_file = RAW_CHUNKS_DIR / f"{ticker}_{year}_{month:02d}.json"
            
            if chunk_file.exists():
                try:
                    with open(chunk_file, "r", encoding="utf-8") as f:
                        records = json.load(f)
                    all_articles.extend(records)
                    ticker_articles_count += len(records)
                    completed_count += 1
                    continue
                except Exception:
                    pass  # If corrupt, re-fetch

            # Fetch fresh chunk
            records = fetch_monthly_chunk(ticker, query, year, month)
            with open(chunk_file, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)

            all_articles.extend(records)
            ticker_articles_count += len(records)
            completed_count += 1
            newly_fetched += 1

            # Log to manifest
            manifest_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "ticker": ticker,
                "query": query,
                "year": year,
                "month": month,
                "articles_count": len(records),
                "status": "SUCCESS" if len(records) > 0 else "EMPTY"
            }
            with open(MANIFEST_PATH, "a", encoding="utf-8") as mf:
                mf.write(json.dumps(manifest_entry) + "\n")

            if completed_count % 25 == 0:
                print(f"Progress: [{completed_count}/{total_tasks}] chunks processed ({newly_fetched} new, {len(all_articles)} total articles)...", flush=True)

            # Polite pacing delay
            time.sleep(0.3)

        elapsed = time.time() - t0
        print(f"Completed {ticker}: {ticker_articles_count} total articles across {len(months)} months in {elapsed:.1f}s.", flush=True)

    print(f"\nCollection Complete! Processed: {completed_count}/{total_tasks} chunks. Total raw articles collected: {len(all_articles)}.", flush=True)

    # Save consolidated raw dataset to parquet
    df_raw = pd.DataFrame(all_articles)
    df_raw.to_parquet(RAW_OUTPUT_PARQUET, index=False)
    print(f"Saved raw articles dataset to {RAW_OUTPUT_PARQUET} ({len(df_raw)} records, {df_raw['ticker'].nunique()} tickers).", flush=True)


if __name__ == "__main__":
    collect_historical_gnews()
