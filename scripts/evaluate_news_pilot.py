"""Evaluate News Provider Pilot Data."""
import os
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def main():
    csv_path = 'experiments/audit/pilot/gnews_pilot_30d.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    analyzer = SentimentIntensityAnalyzer()

    print("=== GNEWS 30-DAY PILOT ANALYSIS (2024-09-01 to 2024-09-30) ===")
    print(f"Total Articles Collected: {len(df)}")
    print("\n--- Articles by Ticker ---")
    print(df['ticker'].value_counts())

    print("\n--- Publisher Diversity (Top 10 Publishers) ---")
    print(df['publisher'].value_counts().head(10))
    print(f"Unique Publishers: {df['publisher'].nunique()}")

    # Timestamps & Days Covered
    df['pub_dt'] = pd.to_datetime(df['published_date'], utc=True, errors='coerce')
    df['date_utc'] = df['pub_dt'].dt.date
    # Convert to IST (UTC + 5:30)
    df['pub_ist'] = df['pub_dt'].dt.tz_convert('Asia/Kolkata')
    df['date_ist'] = df['pub_ist'].dt.date
    df['time_ist'] = df['pub_ist'].dt.time

    print("\n--- Daily Coverage (IST Dates) ---")
    for ticker, grp in df.groupby('ticker'):
        unique_days = grp['date_ist'].nunique()
        print(f"{ticker}: {len(grp)} articles across {unique_days} unique calendar days in September 2024")

    # Duplicates Analysis
    exact_dupes = df.duplicated(subset=['ticker', 'title']).sum()
    print(f"\nExact Duplicate Titles (within ticker): {exact_dupes} ({exact_dupes / len(df) * 100:.1f}%)")

    # Relevance Check
    relevance = []
    for idx, row in df.iterrows():
        text = (str(row['title']) + " " + str(row['description'])).lower()
        q_terms = str(row['query']).lower().split()
        rel = any(term in text for term in q_terms) or str(row['ticker']).lower() in text
        relevance.append(rel)
    df['is_relevant'] = relevance
    rel_rate = df['is_relevant'].mean() * 100
    print(f"Keyword Relevance Rate: {rel_rate:.1f}% ({df['is_relevant'].sum()}/{len(df)})")

    # VADER Scoring Preview
    vader_compounds = [analyzer.polarity_scores(str(t))['compound'] for t in df['title']]
    df['vader_compound'] = vader_compounds
    print("\n--- VADER Compound Sentiment Summary ---")
    print(df.groupby('ticker')['vader_compound'].describe())

    # Check timing cutoff (3:30 PM IST)
    cutoff_time = pd.to_datetime("15:30:00").time()
    df['is_pre_cutoff'] = df['time_ist'] <= cutoff_time
    print(f"\nArticles published before 3:30 PM IST cutoff: {df['is_pre_cutoff'].sum()} ({df['is_pre_cutoff'].mean() * 100:.1f}%)")
    print(f"Articles published after 3:30 PM IST (to roll to next session): {(~df['is_pre_cutoff']).sum()} ({(~df['is_pre_cutoff']).mean() * 100:.1f}%)")

    # Save enriched pilot analysis
    df.to_csv('experiments/audit/pilot/pilot_evaluated_metrics.csv', index=False)
    print("\nSaved enriched pilot evaluation to experiments/audit/pilot/pilot_evaluated_metrics.csv")

if __name__ == "__main__":
    main()
