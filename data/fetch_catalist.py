"""
Fetches 2024 US House results from MIT Election Lab and computes
state-level D vs R margin (D% - R% of two-party vote).

Handles:
- Uncontested races: flag them, exclude from margin calculation
  (uncontested D races inflate D margin, uncontested R inflate R)
- Third party candidates: use two-party vote share only
- States with all-uncontested delegations: note as unreliable

Output: data/catalist_2024.csv
"""

import pandas as pd
import requests
import io

# MIT Election Lab — U.S. House 1976-2024 (district-level, tab-separated)
# Dataset DOI: https://doi.org/10.7910/DVN/IG0UN2
# File ID confirmed via API: datafile 13592823 = 1976-2024-house.tab
MIT_URL = "https://dataverse.harvard.edu/api/access/datafile/13592823"

def _get_signed_url() -> str:
    """Submit guestbook POST to receive a signed download URL."""
    r = requests.post(MIT_URL, json={}, timeout=15)
    r.raise_for_status()
    return r.json()["data"]["signedUrl"]


def _clean_text(text: str) -> str:
    """
    The Dataverse-served file wraps every data row in outer double quotes:
      header: year,state,state_po,...           ← plain CSV, no outer quotes
      row:    "1976,ALABAMA,AL,..."             ← entire row outer-quoted
    Some party names contain commas and are inner-escaped: \"JOBS, PEACE\"
    Fix:
      1. Strip outer " from data rows
      2. Unescape \" → " so inner quoted fields are standard CSV
    """
    lines = text.split('\n')
    cleaned = []
    for i, line in enumerate(lines):
        line = line.rstrip('\r')
        if i == 0:
            cleaned.append(line)               # header — keep as-is
        elif line.startswith('"') and line.endswith('"'):
            inner = line[1:-1]                 # strip outer quotes
            inner = inner.replace('\\"', '"')  # unescape inner quotes
            cleaned.append(inner)
        elif line.strip():
            cleaned.append(line)
    return '\n'.join(cleaned)


def fetch_mit_house_2024():
    """Download MIT Election Lab House returns and filter to 2024."""
    print("  Obtaining signed download URL (guestbook POST)...")
    signed_url = _get_signed_url()
    print("  Downloading 1976-2024-house (~4 MB)...")
    resp = requests.get(signed_url, timeout=120)
    resp.raise_for_status()
    clean = _clean_text(resp.text)
    df = pd.read_csv(io.StringIO(clean), low_memory=False)
    return df[df['year'] == 2024].copy()

def compute_state_margins(df):
    """
    Aggregate district-level results to state level.

    Rules:
    1. Filter to Democrat and Republican candidates only
    2. Flag districts where one party ran unopposed (no D or no R)
    3. Exclude uncontested districts from the margin calculation
    4. For each state: sum D votes, sum R votes across contested districts
    5. margin = (D_total - R_total) / (D_total + R_total) * 100
    6. If ALL districts in a state are uncontested: flag state as unreliable
    """
    results = []

    for state_abbr, state_df in df.groupby('state_po'):
        state_name = state_df['state'].iloc[0]

        # Get D and R votes per district
        districts = {}
        for district, dist_df in state_df.groupby('district'):
            d_votes = dist_df[dist_df['party'] == 'DEMOCRAT']['candidatevotes'].sum()
            r_votes = dist_df[dist_df['party'] == 'REPUBLICAN']['candidatevotes'].sum()
            total = dist_df['totalvotes'].max()
            uncontested = (d_votes == 0 or r_votes == 0)
            districts[district] = {
                'd_votes': d_votes, 'r_votes': r_votes,
                'total': total, 'uncontested': uncontested
            }

        contested = {k: v for k, v in districts.items() if not v['uncontested']}
        n_total = len(districts)
        n_contested = len(contested)
        n_uncontested = n_total - n_contested

        if n_contested == 0:
            # All uncontested — can't compute reliable margin
            results.append({
                'state': state_name, 'abbr': state_abbr,
                'd_margin_2024': None,
                'n_districts': n_total,
                'n_contested': 0,
                'n_uncontested': n_uncontested,
                'reliable': False,
                'notes': 'All districts uncontested'
            })
            continue

        d_total = sum(v['d_votes'] for v in contested.values())
        r_total = sum(v['r_votes'] for v in contested.values())
        two_party = d_total + r_total

        margin = round((d_total - r_total) / two_party * 100, 2) if two_party > 0 else None
        reliable = n_uncontested == 0  # Only fully reliable if no uncontested races

        results.append({
            'state': state_name, 'abbr': state_abbr,
            'd_margin_2024': margin,
            'n_districts': n_total,
            'n_contested': n_contested,
            'n_uncontested': n_uncontested,
            'reliable': reliable,
            'notes': f'{n_uncontested} uncontested districts excluded' if n_uncontested > 0 else ''
        })

    return pd.DataFrame(results).sort_values('abbr')

if __name__ == '__main__':
    print("Fetching MIT Election Lab 2024 House data...")
    df = fetch_mit_house_2024()
    print(f"  {len(df)} district-candidate rows loaded")

    margins = compute_state_margins(df)

    # Save
    margins.to_csv('data/catalist_2024.csv', index=False)
    print(f"\nSaved data/catalist_2024.csv ({len(margins)} states)")
    print(f"  Reliable (fully contested): {margins['reliable'].sum()}")
    print(f"  Some uncontested districts: {(margins['n_uncontested'] > 0).sum()}")
    print(f"  All uncontested (no margin): {margins['d_margin_2024'].isna().sum()}")
    print("\nKey competitive states:")
    key = ['NC', 'GA', 'MI', 'TX', 'AK', 'OH', 'IA', 'ME', 'NH', 'NE', 'FL', 'MN']
    print(margins[margins['abbr'].isin(key)][
        ['abbr', 'd_margin_2024', 'n_contested', 'n_uncontested', 'reliable']
    ].to_string(index=False))
