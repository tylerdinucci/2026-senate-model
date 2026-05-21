"""
Compares catalist baselines to actual 2024 statewide House popular vote.
Run from project root: python data/check_baselines.py
"""
import sys
sys.path.insert(0, 'data')
from fetch_catalist import _get_signed_url, _clean_text
import requests, io, pandas as pd

print('Fetching MIT raw data...')
url = _get_signed_url()
r = requests.get(url, timeout=60)
df = pd.read_csv(io.StringIO(_clean_text(r.text)), low_memory=False)

df24 = df[(df['year'] == 2024) & (df['stage'] == 'GEN') & (df['writein'] == False)].copy()
df24['candidatevotes'] = pd.to_numeric(df24['candidatevotes'], errors='coerce').fillna(0)
df24['totalvotes']     = pd.to_numeric(df24['totalvotes'], errors='coerce').fillna(0)

results = []
for abbr, grp in df24.groupby('state_po'):
    d_votes = grp[grp['party'] == 'DEMOCRAT']['candidatevotes'].sum()
    r_votes = grp[grp['party'] == 'REPUBLICAN']['candidatevotes'].sum()
    total   = grp.groupby('district')['totalvotes'].max().sum()
    if total > 0:
        d_margin = (d_votes - r_votes) / total * 100
        results.append({'abbr': abbr, 'act': round(d_margin, 1)})

actual   = pd.DataFrame(results).set_index('abbr')
catalist = pd.read_csv('data/catalist_2024.csv').set_index('abbr')

print(f"{'St':<5} {'Catalist':>9} {'Actual':>8} {'Diff':>7} {'Uncont%':>8}  Note")
print('-'*60)
flagged = []
for abbr in sorted(actual.index):
    if abbr not in catalist.index:
        continue
    cat   = catalist.loc[abbr, 'd_margin_2024']
    act   = actual.loc[abbr, 'act']
    n_unc = catalist.loc[abbr, 'n_uncontested']
    n_tot = catalist.loc[abbr, 'n_districts']
    pct   = n_unc / n_tot * 100 if n_tot > 0 else 0

    if pd.isna(cat):
        print(f"{abbr:<5} {'n/a':>9} {act:>+8.1f} {'':>7} {pct:>7.0f}%")
        continue

    diff = cat - act
    flag = '***' if abs(diff) > 4 else ''
    print(f"{abbr:<5} {cat:>+9.1f} {act:>+8.1f} {diff:>+7.1f} {pct:>7.0f}%  {flag}")
    if abs(diff) > 4:
        flagged.append((abbr, cat, act, diff, pct))

print()
print(f'{len(flagged)} states with >4pp divergence:')
for abbr, cat, act, diff, pct in sorted(flagged, key=lambda x: -abs(x[3])):
    print(f'  {abbr}: catalist {cat:+.1f}  actual {act:+.1f}  diff {diff:+.1f}pp  ({pct:.0f}% uncontested)')
