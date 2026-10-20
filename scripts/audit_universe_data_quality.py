"""Audit B1: Universe & Data Quality Forensic Audit.

Ingests OHLCV price series for all 62 NIFTY 50 universe tickers (2020 - 2026).
Applies point-in-time constituent filtering.
Generates comprehensive data quality audit table:
- first/last price date
- total rows
- missing days relative to trading calendar
- eligible point-in-time ticker-days
- flags |daily return| > 20%
- flags +-35-45% jumps (e.g., TMPV / Tata Motors around 2025-10-14)
- documents splits, bonus issues, delisted HDFC, late listings (JIOFIN, ETERNAL).
Populates data/metadata/universe.csv and outputs experiments/audit/data_quality_table.csv.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from finfusionnet.data.prices import fetch_all_prices, fetch_single_ticker
from finfusionnet.data.trading_calendar import get_trading_calendar
from finfusionnet.data.universe import PointInTimeUniverse, REBALANCES, BASE_CONSTITUENTS_2021


SECTOR_MAP = {
    "RELIANCE": ("Oil & Gas", "Reliance Industries Ltd."),
    "TCS": ("IT", "Tata Consultancy Services Ltd."),
    "HDFCBANK": ("Financials", "HDFC Bank Ltd."),
    "ICICIBANK": ("Financials", "ICICI Bank Ltd."),
    "BHARTIARTL": ("Telecom", "Bharti Airtel Ltd."),
    "INFY": ("IT", "Infosys Ltd."),
    "ITC": ("FMCG", "ITC Ltd."),
    "LT": ("Construction", "Larsen & Toubro Ltd."),
    "SBIN": ("Financials", "State Bank of India"),
    "HINDUNILVR": ("FMCG", "Hindustan Unilever Ltd."),
    "ADANIENT": ("Metals & Mining", "Adani Enterprises Ltd."),
    "ADANIPORTS": ("Services", "Adani Ports & SEZ Ltd."),
    "APOLLOHOSP": ("Healthcare", "Apollo Hospitals Enterprise Ltd."),
    "ASIANPAINT": ("Consumer Durables", "Asian Paints Ltd."),
    "AXISBANK": ("Financials", "Axis Bank Ltd."),
    "BAJAJ-AUTO": ("Automobile", "Bajaj Auto Ltd."),
    "BAJAJFINSV": ("Financials", "Bajaj Finserv Ltd."),
    "BAJFINANCE": ("Financials", "Bajaj Finance Ltd."),
    "BEL": ("Capital Goods", "Bharat Electronics Ltd."),
    "BPCL": ("Oil & Gas", "Bharat Petroleum Corporation Ltd."),
    "BRITANNIA": ("FMCG", "Britannia Industries Ltd."),
    "BSE": ("Financials", "BSE Ltd."),
    "CIPLA": ("Healthcare", "Cipla Ltd."),
    "COALINDIA": ("Metals & Mining", "Coal India Ltd."),
    "DIVISLAB": ("Healthcare", "Divi's Laboratories Ltd."),
    "DRREDDY": ("Healthcare", "Dr. Reddy's Laboratories Ltd."),
    "EICHERMOT": ("Automobile", "Eicher Motors Ltd."),
    "ETERNAL": ("Consumer Services", "Eternal Ltd. / Zomato"),
    "GAIL": ("Oil & Gas", "GAIL (India) Ltd."),
    "GRASIM": ("Materials", "Grasim Industries Ltd."),
    "HCLTECH": ("IT", "HCL Technologies Ltd."),
    "HDFC": ("Financials", "Housing Development Finance Corp (Merged)"),
    "HDFCLIFE": ("Financials", "HDFC Life Insurance Co Ltd."),
    "HEROMOTOCO": ("Automobile", "Hero MotoCorp Ltd."),
    "HINDALCO": ("Metals & Mining", "Hindalco Industries Ltd."),
    "INDIGO": ("Consumer Services", "InterGlobe Aviation Ltd."),
    "INDUSINDBK": ("Financials", "IndusInd Bank Ltd."),
    "IOC": ("Oil & Gas", "Indian Oil Corporation Ltd."),
    "JIOFIN": ("Financials", "Jio Financial Services Ltd."),
    "JSWSTEEL": ("Metals & Mining", "JSW Steel Ltd."),
    "KOTAKBANK": ("Financials", "Kotak Mahindra Bank Ltd."),
    "LTIM": ("IT", "LTIMindtree Ltd."),
    "M&M": ("Automobile", "Mahindra & Mahindra Ltd."),
    "MARUTI": ("Automobile", "Maruti Suzuki India Ltd."),
    "MAXHEALTH": ("Healthcare", "Max Healthcare Institute Ltd."),
    "NESTLEIND": ("FMCG", "Nestle India Ltd."),
    "NTPC": ("Power", "NTPC Ltd."),
    "ONGC": ("Oil & Gas", "Oil & Natural Gas Corp Ltd."),
    "POWERGRID": ("Power", "Power Grid Corporation of India Ltd."),
    "SBILIFE": ("Financials", "SBI Life Insurance Co Ltd."),
    "SHREECEM": ("Materials", "Shree Cement Ltd."),
    "SHRIRAMFIN": ("Financials", "Shriram Finance Ltd."),
    "SUNPHARMA": ("Healthcare", "Sun Pharmaceutical Industries Ltd."),
    "TATACONSUM": ("FMCG", "Tata Consumer Products Ltd."),
    "TATAMOTORS": ("Automobile", "Tata Motors Ltd. (TMPV)"),
    "TATASTEEL": ("Metals & Mining", "Tata Steel Ltd."),
    "TECHM": ("IT", "Tech Mahindra Ltd."),
    "TITAN": ("Consumer Durables", "Titan Company Ltd."),
    "TRENT": ("Consumer Services", "Trent Ltd."),
    "ULTRACEMCO": ("Materials", "UltraTech Cement Ltd."),
    "UPL": ("Chemicals", "UPL Ltd."),
    "WIPRO": ("IT", "Wipro Ltd."),
}


def populate_universe_metadata(tickers):
    records = []
    pit = PointInTimeUniverse()
    latest_constituents = pit.get_constituents_on_date("2026-09-30")

    for t in sorted(tickers):
        sector, name = SECTOR_MAP.get(t, ("Other", t))
        is_active = 1 if t in latest_constituents else 0
        records.append({
            "ticker": t,
            "name": name,
            "sector": sector,
            "market_cap": "Large",
            "is_active": is_active,
        })
    df = pd.DataFrame(records)
    out_path = Path("data/metadata/universe.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Populated {len(df)} tickers to {out_path}", flush=True)
    return df


def run_universe_data_quality_audit():
    print("=" * 80, flush=True)
    print("AUDIT B1: UNIVERSE & DATA QUALITY FORENSIC AUDIT", flush=True)
    print("=" * 80, flush=True)

    pit = PointInTimeUniverse()
    tickers = sorted(list(pit.all_tickers))
    print(f"Total unique tickers in point-in-time universe: {len(tickers)}", flush=True)

    # Populate universe metadata
    populate_universe_metadata(tickers)

    # Ingest OHLCV prices
    print("\nIngesting / validating OHLCV prices for all tickers (2020-01-01 to 2026-10-01)...", flush=True)
    stocks_df, nifty_df = fetch_all_prices(
        tickers=tickers,
        start_date="2020-01-01",
        end_date="2026-10-01",
        cache_dir="data/raw/prices",
        benchmark_symbol="^NSEI",
        force_download=False,
    )
    print(f"Total ingested price rows: {len(stocks_df)} across {stocks_df['ticker'].nunique()} tickers.", flush=True)

    calendar = get_trading_calendar("2020-01-01", "2026-09-30")
    print(f"Official trading calendar sessions: {len(calendar)}", flush=True)

    # Apply Point-in-Time filter
    pit_df = pit.filter_point_in_time(stocks_df)
    print(f"Total eligible point-in-time ticker-day rows: {len(pit_df)}", flush=True)

    audit_records = []
    anomalies_list = []

    for ticker in tickers:
        df_t = stocks_df[stocks_df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        df_pit_t = pit_df[pit_df["ticker"] == ticker].sort_values("date").reset_index(drop=True)

        if df_t.empty:
            audit_records.append({
                "ticker": ticker,
                "first_date": "N/A",
                "last_date": "N/A",
                "raw_rows": 0,
                "missing_days": len(calendar),
                "eligible_pit_rows": 0,
                "flag_gt_20pct_ret": 0,
                "flag_jump_35_45pct": 0,
                "notes": "Missing all price data from source",
            })
            continue

        first_dt = df_t["date"].min()
        last_dt = df_t["date"].max()
        raw_rows = len(df_t)

        # Expected calendar dates within [first_dt, last_dt]
        sub_cal = calendar[(calendar >= first_dt) & (calendar <= last_dt)]
        missing_days = len(sub_cal) - raw_rows

        eligible_pit_rows = len(df_pit_t)

        # Compute daily returns
        df_t["daily_ret"] = df_t["close"].pct_change()
        extreme_20 = df_t[df_t["daily_ret"].abs() > 0.20]
        n_extreme_20 = len(extreme_20)

        # Specific 35-45% jumps
        jumps_35_45 = df_t[(df_t["daily_ret"].abs() >= 0.35) & (df_t["daily_ret"].abs() <= 0.45)]
        n_jumps = len(jumps_35_45)

        notes = []
        if ticker == "HDFC":
            notes.append("Delisted 2023-07-13 (merged into HDFCBANK)")
        if ticker in ["JIOFIN", "ETERNAL"]:
            notes.append(f"Late listing: first date {first_dt.strftime('%Y-%m-%d')}")
        if ticker == "TATAMOTORS":
            notes.append("Tata Motors / TMPV demerger scrutiny")
        if n_extreme_20 > 0:
            dates_20 = [f"{d.strftime('%Y-%m-%d')} ({r:+.1%})" for d, r in zip(extreme_20["date"], extreme_20["daily_ret"])]
            notes.append(f"Extreme returns: {'; '.join(dates_20[:3])}")
            for d, r in zip(extreme_20["date"], extreme_20["daily_ret"]):
                anomalies_list.append({
                    "ticker": ticker,
                    "date": d.strftime("%Y-%m-%d"),
                    "daily_return": round(float(r), 4),
                    "anomaly_type": ">20% price move",
                })

        audit_records.append({
            "ticker": ticker,
            "first_date": first_dt.strftime("%Y-%m-%d"),
            "last_date": last_dt.strftime("%Y-%m-%d"),
            "raw_rows": raw_rows,
            "missing_days": max(0, missing_days),
            "eligible_pit_rows": eligible_pit_rows,
            "flag_gt_20pct_ret": n_extreme_20,
            "flag_jump_35_45pct": n_jumps,
            "notes": " | ".join(notes) if notes else "Normal",
        })

    audit_df = pd.DataFrame(audit_records)
    out_table = Path("experiments/audit/data_quality_table.csv")
    out_table.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(out_table, index=False)
    print(f"\nSaved data quality table to {out_table}", flush=True)

    if anomalies_list:
        anom_df = pd.DataFrame(anomalies_list)
        anom_out = Path("experiments/audit/extreme_returns_anomalies.csv")
        anom_df.to_csv(anom_out, index=False)
        print(f"Saved {len(anom_df)} flagged return anomalies to {anom_out}", flush=True)

    print("\n--- DATA QUALITY SUMMARY ---", flush=True)
    print(f"Total Tickers Audited: {len(audit_df)}", flush=True)
    print(f"Tickers with |return| > 20%: {(audit_df['flag_gt_20pct_ret'] > 0).sum()}", flush=True)
    print(f"Total Eligible PIT Observations: {audit_df['eligible_pit_rows'].sum()}", flush=True)
    print("\nSample Ticker Audit Rows:")
    print(audit_df.head(15).to_string(index=False), flush=True)

    return audit_df


if __name__ == "__main__":
    run_universe_data_quality_audit()
