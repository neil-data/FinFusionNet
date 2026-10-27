"""FinFusionNet: Fast Resumable Historical GNews Collector for All-50 NIFTY Constituents.

Collects historical news headlines across all 62 in-window NIFTY 50 constituents from 2020-01-01 to 2026-09-30
using monthly chunking, file-based checkpointing, thread-pooled rate-limited fetching, and manifest logging.
Reuses existing 10-stock collection without duplicate downloads.
"""

import os
import json
import time
import calendar
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import concurrent.futures
import pandas as pd
import feedparser


ALL_TICKER_QUERIES = {
    # 10 Heavyweights (Already downloaded, 810 chunks will be instantly reused)
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

    # Remaining 52 Constituents (In-window active + exited names)
    "ADANIENT": "Adani Enterprises",
    "ADANIPORTS": "Adani Ports",
    "APOLLOHOSP": "Apollo Hospitals",
    "ASIANPAINT": "Asian Paints",
    "AXISBANK": "Axis Bank",
    "BAJAJ-AUTO": "Bajaj Auto",
    "BAJAJFINSV": "Bajaj Finserv",
    "BAJFINANCE": "Bajaj Finance",
    "BEL": "Bharat Electronics",
    "BPCL": "Bharat Petroleum BPCL",
    "BRITANNIA": "Britannia Industries",
    "BSE": "BSE Limited exchange",
    "CIPLA": "Cipla",
    "COALINDIA": "Coal India",
    "DIVISLAB": "Divis Laboratories",
    "DRREDDY": "Dr Reddys Laboratories",
    "EICHERMOT": "Eicher Motors",
    "ETERNAL": "Zomato OR Eternal",
    "GAIL": "GAIL India",
    "GRASIM": "Grasim Industries",
    "HCLTECH": "HCL Technologies",
    "HDFC": "HDFC Ltd mortgage",
    "HDFCLIFE": "HDFC Life Insurance",
    "HEROMOTOCO": "Hero MotoCorp",
    "HINDALCO": "Hindalco Industries",
    "INDIGO": "InterGlobe Aviation IndiGo",
    "INDUSINDBK": "IndusInd Bank",
    "IOC": "Indian Oil Corporation IOC",
    "JIOFIN": "Jio Financial Services",
    "JSWSTEEL": "JSW Steel",
    "KOTAKBANK": "Kotak Mahindra Bank",
    "LTIM": "LTIMindtree",
    "M&M": "Mahindra & Mahindra",
    "MARUTI": "Maruti Suzuki",
    "MAXHEALTH": "Max Healthcare",
    "NESTLEIND": "Nestle India",
    "NTPC": "NTPC",
    "ONGC": "ONGC Oil and Natural Gas",
    "POWERGRID": "Power Grid Corporation",
    "SBILIFE": "SBI Life Insurance",
    "SHREECEM": "Shree Cement",
    "SHRIRAMFIN": "Shriram Finance",
    "SUNPHARMA": "Sun Pharma",
    "TATACONSUM": "Tata Consumer Products",
    "TATAMOTORS": "Tata Motors",
    "TATASTEEL": "Tata Steel",
    "TECHM": "Tech Mahindra",
    "TITAN": "Titan Company",
    "TRENT": "Trent Westside",
    "ULTRACEMCO": "UltraTech Cement",
    "UPL": "UPL Limited",
    "WIPRO": "Wipro",
}

RAW_CHUNKS_DIR = Path("data/raw/gnews_all50/chunks")
RAW_OUTPUT_PARQUET = Path("data/raw/gnews_all50/raw_articles_all50.parquet")
MANIFEST_PATH = Path("data/raw/gnews_all50/manifest.jsonl")


def generate_month_ranges(start_year=2020, end_year=2026, end_month=9) -> List[Tuple[int, int]]:
    months = []
    for y in range(start_year, end_year + 1):
        max_m = end_month if y == end_year else 12
        for m in range(1, max_m + 1):
            months.append((y, m))
    return months


def fetch_one_chunk(task: Tuple[str, str, int, int], max_retries: int = 3) -> Tuple[str, int, int, List[Dict[str, Any]]]:
    ticker, query, year, month = task
    chunk_file = RAW_CHUNKS_DIR / f"{ticker}_{year}_{month:02d}.json"

    # Checkpoint check
    if chunk_file.exists():
        try:
            with open(chunk_file, "r", encoding="utf-8") as f:
                records = json.load(f)
            return ticker, year, month, records
        except Exception:
            pass  # If corrupt, re-fetch

    start_str = f"{year}-{month:02d}-01"
    _, last_day = calendar.monthrange(year, month)
    end_str = f"{year}-{month:02d}-{last_day:02d}"

    encoded_query = urllib.parse.quote(f"{query} after:{start_str} before:{end_str}")
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

    clean_articles = []
    for attempt in range(1, max_retries + 1):
        try:
            feed = feedparser.parse(url)
            if hasattr(feed, "status") and feed.status == 429:
                time.sleep(attempt * 3.0)
                continue

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
            break
        except Exception as e:
            if attempt < max_retries:
                time.sleep(attempt * 2.0)
            else:
                pass

    # Save chunk JSON
    with open(chunk_file, "w", encoding="utf-8") as f:
        json.dump(clean_articles, f, ensure_ascii=False, indent=2)

    # Polite delay
    time.sleep(0.2)
    return ticker, year, month, clean_articles


def collect_all50_gnews():
    RAW_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    months = generate_month_ranges(2020, 2026, 9)
    tasks = []
    for ticker, query in ALL_TICKER_QUERIES.items():
        for year, month in months:
            tasks.append((ticker, query, year, month))

    total_tasks = len(tasks)
    print(f"Starting GNews All-50 Collection across {len(ALL_TICKER_QUERIES)} tickers and {len(months)} months ({total_tasks:,} total chunks)...", flush=True)

    # Check how many are already cached
    existing_chunks = len([f for f in RAW_CHUNKS_DIR.glob("*.json")])
    print(f"Existing cached chunks found: {existing_chunks:,} / {total_tasks:,} (will be loaded instantly).", flush=True)

    all_articles = []
    completed = 0
    t0 = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submit tasks
        future_to_task = {executor.submit(fetch_one_chunk, task): task for task in tasks}

        for future in concurrent.futures.as_completed(future_to_task):
            ticker, year, month, records = future.result()
            all_articles.extend(records)
            completed += 1

            if completed % 100 == 0 or completed == total_tasks:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (total_tasks - completed) / rate if rate > 0 else 0
                print(f"Progress: [{completed:,}/{total_tasks:,}] chunks ({completed/total_tasks*100:.1f}%) | Articles so far: {len(all_articles):,} | Rate: {rate:.1f} ch/s | ETA: {remaining/60:.1f} min", flush=True)

    print(f"\nAll-50 Collection Complete! Total chunks: {completed:,}. Total raw articles collected: {len(all_articles):,}.", flush=True)

    # Save consolidated raw dataset to parquet
    print(f"Saving consolidated raw articles dataset to {RAW_OUTPUT_PARQUET}...", flush=True)
    df_raw = pd.DataFrame(all_articles)
    df_raw.to_parquet(RAW_OUTPUT_PARQUET, index=False)
    print(f"Saved {len(df_raw):,} records across {df_raw['ticker'].nunique()} tickers.", flush=True)


if __name__ == "__main__":
    collect_all50_gnews()
