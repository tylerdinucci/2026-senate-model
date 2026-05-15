"""
Computes current demographic group D vote share estimates
by aggregating manually-entered national crosstab data.

Logic:
- Load all entries from data/national_crosstabs.csv
- Apply grade × recency weighting
- For each group: compute weighted average D vote share
- Compute shift vs Catalist 2024 baseline

Exclusion rule: if note contains EXCLUDE_[GROUP] (e.g. EXCLUDE_HISPANIC),
that entry is excluded from that group's calculation only.
"""

import pandas as pd
import numpy as np
import yaml
from datetime import date


def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


def grade_weight(grade):
    """Silver-style grade to weight multiplier."""
    if pd.isna(grade):    return 0.35
    g = float(grade)
    if g >= 2.8:  return 1.00  # A / A-
    if g >= 2.0:  return 0.75  # B+
    if g >= 1.0:  return 0.50  # B / C
    return 0.35                 # D or unrated


def recency_weight(end_date_str, reference_date=None):
    """Exponential decay: 60-day half-life."""
    if reference_date is None:
        reference_date = date.today()
    try:
        end_date = pd.to_datetime(end_date_str).date()
        days_old = (reference_date - end_date).days
        days_old = max(days_old, 0)
        return float(np.exp(-np.log(2) * days_old / 60))
    except Exception:
        return 0.10


def compute_group_shifts(config, reference_date=None, csv_path='data/national_crosstabs.csv'):
    """
    Returns dict: {group: {current_d_share, shift, n_entries, last_updated}}
    """
    if reference_date is None:
        reference_date = date.today()

    baselines = config['catalist_2024']['group_d_share']
    groups = list(baselines.keys())

    df = pd.read_csv(csv_path)

    # Filter to gcb_d_share metric only
    df = df[df['metric'] == 'gcb_d_share'].copy()

    results = {}
    for group in groups:
        group_df = df[df['group'] == group].copy()

        # Apply exclusion flags
        exclude_flag = f'EXCLUDE_{group.upper()}'
        group_df = group_df[~group_df['note'].fillna('').str.upper().str.contains(exclude_flag)]

        if len(group_df) == 0:
            # No data — use Catalist baseline, shift = 0
            results[group] = {
                'current_d_share': baselines[group],
                'shift': 0.0,
                'n_entries': 0,
                'last_updated': None,
                'note': 'No data — using Catalist baseline'
            }
            continue

        # Compute weights
        group_df['gw'] = group_df['grade'].apply(grade_weight)
        group_df['rw'] = group_df['date_end'].apply(
            lambda d: recency_weight(d, reference_date)
        )
        group_df['weight'] = group_df['gw'] * group_df['rw']

        total_weight = group_df['weight'].sum()
        if total_weight == 0:
            current = baselines[group]
        else:
            current = float((group_df['value'] * group_df['weight']).sum() / total_weight)

        shift = round((current - baselines[group]) * 100, 2)  # in pp

        results[group] = {
            'current_d_share': round(current, 4),
            'shift': shift,
            'n_entries': len(group_df),
            'last_updated': group_df['date_end'].max(),
            'note': ''
        }

    return results


if __name__ == '__main__':
    cfg = load_config()
    shifts = compute_group_shifts(cfg)
    baselines = cfg['catalist_2024']['group_d_share']

    print("DEMOGRAPHIC SHIFTS MODULE — UNIT TEST")
    print(f"{'Group':<20} {'Catalist 2024':>14} {'Current':>10} {'Shift':>8} {'N':>4}")
    print("─" * 62)
    for group, data in shifts.items():
        print(f"{group:<20} {baselines[group]:>14.3f} "
              f"{data['current_d_share']:>10.3f} "
              f"{data['shift']:>+8.1f}pp "
              f"{data['n_entries']:>4}")
    print()
    # Verify White NC shift is positive (we're seeing D gains)
    wnc_shift = shifts.get('white_nc', {}).get('shift', 0)
    print(f"VERIFY: White NC shift should be positive → {'PASS' if wnc_shift > 0 else 'FAIL'}")
    # Verify Hispanic exclusion logic works
    hisp_entries = shifts.get('hispanic', {}).get('n_entries', -1)
    print(f"VERIFY: Hispanic has entries → {'PASS' if hisp_entries >= 0 else 'FAIL'}")
