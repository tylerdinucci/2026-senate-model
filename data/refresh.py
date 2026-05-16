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

# Pollster grade map for GCB weighting — mirrors our crosstab grade scheme (0–3 scale).
# Unknown pollsters default to 1.5.
GCB_POLLSTER_GRADES = {
    'The New York Times/Siena College': 3.0,
    'Marist University':                2.9,
    'Marist College':                   2.9,
    'Quinnipiac University':            2.8,
    'CNN/SSRS':                         2.8,
    'YouGov':                           2.7,
    'Public Religion Research Institute': 2.6,
    'Ipsos':                            2.5,
    'Emerson College':                  2.5,
    'AtlasIntel':                       2.4,
    'Focaldata':                        2.4,
    'Morning Consult':                  2.3,
    'Public Opinion Strategies':        2.2,
    'HarrisX':                          2.2,
    'Cygnal':                           2.2,
    'Echelon Insights':                 2.0,
    'RMG Research':                     2.0,
    'Rasmussen Reports':                1.8,
    'McLaughlin & Associates':          1.5,
}

GCB_RECENCY_HALF_LIFE = 30   # days
GCB_CUTOFF_DAYS       = 60   # ignore polls older than this


# ── SILVER BULLETIN GCB ────────────────────────────────────────────────────────
def refresh_gcb():
    print("Fetching Silver Bulletin GCB data...")
    resp = requests.get(GCB_CSV_URL, timeout=30)
    resp.raise_for_status()

    import numpy as np

    raw = pd.read_csv(io.StringIO(resp.text), low_memory=False)
    df = raw[raw['subgroup'] == 'All polls'].copy()
    df['enddate'] = pd.to_datetime(df['enddate'], errors='coerce')
    df = df.dropna(subset=['enddate', 'adjusted_net'])

    today = date.today()
    df['days_ago'] = (pd.Timestamp(today) - df['enddate']).dt.days
    df = df[df['days_ago'] <= GCB_CUTOFF_DAYS]

    if len(df) == 0:
        print("  No polls within cutoff window — GCB unchanged")
        return

    df['recency_w'] = np.exp(-df['days_ago'] * np.log(2) / GCB_RECENCY_HALF_LIFE)
    df['grade_w']   = df['pollster'].map(GCB_POLLSTER_GRADES).fillna(1.5)
    df['combined_w'] = df['recency_w'] * df['grade_w']

    weighted_net = (df['adjusted_net'] * df['combined_w']).sum() / df['combined_w'].sum()
    model_date   = raw['modeldate'].iloc[0] if len(raw) > 0 else str(today)

    # Load existing and check if we already have today's reading
    existing  = pd.read_csv('data/gcb_national.csv')
    today_str = str(today)

    if today_str in existing['date'].values:
        print(f"  Already have GCB reading for {today_str}: "
              f"D{existing[existing['date']==today_str]['gcb_d_net'].iloc[0]:+.2f}")
        return

    new_row = pd.DataFrame([{
        'date':      today_str,
        'gcb_d_net': round(weighted_net, 2),
        'source':    'Silver Bulletin GCB CSV — adjusted_net × grade × recency (30d half-life, 60d window)',
        'note':      f'Model date: {model_date} | n={len(df)} polls in window',
    }])

    updated = pd.concat([existing, new_row], ignore_index=True)
    updated.to_csv('data/gcb_national.csv', index=False)
    print(f"  GCB updated: D{weighted_net:+.2f} ({today_str})  "
          f"[n={len(df)}, window={GCB_CUTOFF_DAYS}d, half-life={GCB_RECENCY_HALF_LIFE}d]")

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
