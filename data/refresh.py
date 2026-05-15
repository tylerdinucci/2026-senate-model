"""
Pulls latest data from two public sources:
1. 538 Senate polls CSV (public, no auth)
2. Silver Bulletin GCB CSV (public Google Sheets export)

Run: python data/refresh.py
Flags: --gcb-only, --polls-only
"""

import pandas as pd
import requests
import io
from datetime import date
import argparse

# ── SOURCES ────────────────────────────────────────────────────────────────────
SENATE_POLLS_URL = "https://projects.fivethirtyeight.com/polls/data/senate_polls.csv"

# Silver Bulletin GCB — public Google Sheets export confirmed live May 15 2026
GCB_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vRsvXNCZ0ubJr8D_yNcU5q6C0_HBa35K7oDK03KpO7Ca43UwdXaIdvVLWoXEmHHph0EREz5430Hm5yZ"
    "/pub?output=csv"
)

# ── 538 SENATE POLLS ───────────────────────────────────────────────────────────
def refresh_senate_polls():
    print("Fetching 538 Senate polls...")
    resp = requests.get(SENATE_POLLS_URL, timeout=30)
    resp.raise_for_status()

    # 538 moved to ABC News; the old URL now returns HTML instead of CSV.
    # Detect this and fail cleanly rather than crashing on parse.
    if resp.text.lstrip().startswith('<'):
        print(
            "  WARNING: 538 polls URL returned HTML (not CSV).\n"
            "  The projects.fivethirtyeight.com data feeds were retired when 538\n"
            "  merged into ABC News. Update SENATE_POLLS_URL in refresh.py when\n"
            "  a new public CSV endpoint is identified.\n"
            "  Polls unchanged — use python data/entry.py to add polls manually."
        )
        return

    raw = pd.read_csv(io.StringIO(resp.text), low_memory=False)

    # Filter to 2026 Senate general election polls
    df = raw[
        (raw['cycle'] == 2026) &
        (raw['office_type'] == 'U.S. Senate') &
        (raw['stage'] == 'general')
    ].copy()

    if len(df) == 0:
        print("  No 2026 Senate general election polls found yet")
        return

    # Map to our schema
    # 538 has one row per candidate — we need D vs R per poll question
    # Group by poll_id + question_id to get matchup pairs
    existing = pd.read_csv('data/state_polls.csv')
    existing_ids = set(existing['poll_id'].astype(str))

    new_rows = []
    for (poll_id, q_id), grp in df.groupby(['poll_id', 'question_id']):
        if str(poll_id) in existing_ids:
            continue  # Already have this poll

        d_row = grp[grp['party'] == 'DEM']
        r_row = grp[grp['party'] == 'REP']

        if len(d_row) == 0 or len(r_row) == 0:
            continue  # Need both parties for the model

        d = d_row.iloc[0]
        r = r_row.iloc[0]

        new_rows.append({
            'poll_id': f"538_{poll_id}_{q_id}",
            'state': d.get('state', ''),
            'abbr': d.get('state', ''),  # 538 uses state name; we'll map to abbr
            'd_candidate': d.get('candidate_name', 'D Candidate'),
            'r_candidate': r.get('candidate_name', 'R Candidate'),
            'd_pct': d.get('pct', None),
            'r_pct': r.get('pct', None),
            'undi_pct': None,
            'n': d.get('sample_size', None),
            'pollster': d.get('pollster', ''),
            'grade': d.get('numeric_grade', None),
            'partisan': str(d.get('partisan', '')).strip() != '',
            'date_start': d.get('start_date', ''),
            'date_end': d.get('end_date', ''),
            'methodology': d.get('methodology', ''),
            'note': ''
        })

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        updated = pd.concat([existing, new_df], ignore_index=True)
        updated.to_csv('data/state_polls.csv', index=False)
        print(f"  Added {len(new_rows)} new Senate polls (total: {len(updated)})")
    else:
        print(f"  No new polls (existing: {len(existing)})")

# ── SILVER BULLETIN GCB ────────────────────────────────────────────────────────
def refresh_gcb():
    print("Fetching Silver Bulletin GCB data...")
    resp = requests.get(GCB_CSV_URL, timeout=30)
    resp.raise_for_status()

    raw = pd.read_csv(io.StringIO(resp.text), low_memory=False)

    # Get the most recent poll date and the influence-weighted average
    # Columns: subgroup, pollster, startdate, enddate, samplesize,
    #          dem, rep, net, adjusted_net, influence, modeldate

    # Filter to "All polls" subgroup only
    df = raw[raw['subgroup'] == 'All polls'].copy()
    df['enddate'] = pd.to_datetime(df['enddate'], errors='coerce')
    df = df.dropna(subset=['enddate', 'net'])

    # Get model date (Silver's current average date)
    model_date = df['modeldate'].iloc[0] if len(df) > 0 else str(date.today())

    # Compute influence-weighted average (this is Silver's tracker number)
    total_influence = df['influence'].sum()
    if total_influence > 0:
        weighted_net = (df['net'] * df['influence']).sum() / total_influence
    else:
        weighted_net = df['net'].mean()

    # Load existing and check if we already have today's reading
    existing = pd.read_csv('data/gcb_national.csv')
    today_str = str(date.today())

    if today_str in existing['date'].values:
        print(f"  Already have GCB reading for {today_str}: D{existing[existing['date']==today_str]['gcb_d_net'].iloc[0]:+.2f}")
        return

    new_row = pd.DataFrame([{
        'date': today_str,
        'gcb_d_net': round(weighted_net, 2),
        'source': 'Silver Bulletin GCB tracker (influence-weighted)',
        'note': f'Model date: {model_date} | {len(df)} polls in database'
    }])

    updated = pd.concat([existing, new_row], ignore_index=True)
    updated.to_csv('data/gcb_national.csv', index=False)
    print(f"  GCB updated: D{weighted_net:+.2f} ({today_str})")
    print(f"  Total polls in Silver database: {len(df)}")

# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gcb-only',   action='store_true')
    parser.add_argument('--polls-only', action='store_true')
    args = parser.parse_args()

    if not args.polls_only:
        refresh_gcb()
    if not args.gcb_only:
        refresh_senate_polls()

    print("\nDone.")
