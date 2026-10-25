import pandas as pd

df = pd.read_csv('experiments/audit/pilot/pilot_evaluated_metrics.csv')
irrel = df[~df['is_relevant']]
print(f"Total non-matching rows: {len(irrel)}")
for idx, r in irrel.head(10).iterrows():
    title_clean = r['title'].encode('ascii', errors='replace').decode()
    desc_clean = str(r['description']).encode('ascii', errors='replace').decode()
    print(f"[{r['ticker']}] Title: {title_clean}")
    print(f"  Publisher: {r['publisher']}")
    print(f"  Desc: {desc_clean[:100]}...")
