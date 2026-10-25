import pandas as pd

df = pd.read_csv('data/metadata/universe.csv')
print(f"Total Tickers: {len(df)}")
for idx, r in df.iterrows():
    print(f"[{r['ticker']:<12}] Name: {r['name']:<35} | Sector: {r['sector']:<25} | Active: {r['is_active']}")
