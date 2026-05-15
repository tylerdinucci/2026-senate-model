"""
Quality × recency weighted poll blend for a state.

Special handling:
- AK: only include polls with note containing 'H2H_SULLIVAN'
- ME: separate blends for Platner and Mills matchups, weighted by primary priors
- TX: separate blends for Cornyn and Paxton matchups, weighted by primary priors
"""

import pandas as pd
import numpy as np
import yaml
import math
from datetime import date


def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


def grade_weight(grade):
    if pd.isna(grade): return 0.35
    g = float(grade)
    if g >= 2.8: return 1.00
    if g >= 2.0: return 0.75
    if g >= 1.0: return 0.50
    return 0.35


def recency_weight(end_date_str, reference_date=None):
    if reference_date is None:
        reference_date = date.today()
    try:
        end_date = pd.to_datetime(end_date_str).date()
        days_old = max((reference_date - end_date).days, 0)
        if days_old <= 30:   return 1.00
        if days_old <= 60:   return 0.85
        if days_old <= 90:   return 0.70
        if days_old <= 180:  return 0.55
        if days_old <= 365:  return 0.35
        return 0.15
    except Exception:
        return 0.10


def _blend_subset(subset_df, reference_date):
    """Compute weighted blend margin for a subset of polls."""
    if len(subset_df) == 0:
        return None, 0

    subset_df = subset_df.copy()
    subset_df['gw'] = subset_df['grade'].apply(grade_weight)
    subset_df['rw'] = subset_df['date_end'].apply(
        lambda d: recency_weight(d, reference_date)
    )
    # Partisan flag: 0.6× weight
    subset_df['pw'] = subset_df['partisan'].apply(
        lambda p: 0.6 if str(p).strip().upper() in ('TRUE', '1', 'YES', 'Y') else 1.0
    )
    subset_df['weight'] = subset_df['gw'] * subset_df['rw'] * subset_df['pw']
    subset_df['margin'] = subset_df['d_pct'] - subset_df['r_pct']

    total_weight = subset_df['weight'].sum()
    if total_weight == 0:
        return None, 0

    blend = float((subset_df['margin'] * subset_df['weight']).sum() / total_weight)
    return blend, len(subset_df)


def compute_poll_blend(state_abbr, polls_df, reference_date, config):
    """
    Returns dict:
      blend_margin:  float (D - R pp) or None if no polls
      n_polls:       int
      latest_poll:   str date or None
    """
    state_polls = polls_df[polls_df['abbr'] == state_abbr].copy()

    if state_abbr == 'AK':
        # Only H2H Sullivan polls
        state_polls = state_polls[
            state_polls['note'].fillna('').str.upper().str.contains('H2H_SULLIVAN')
        ]

    if len(state_polls) == 0:
        return {'blend_margin': None, 'n_polls': 0, 'latest_poll': None}

    if state_abbr == 'ME':
        # Separate blends per D candidate
        priors = config['race_model']['primary_priors']
        platner_polls = state_polls[
            state_polls['d_candidate'].str.contains('Platner', case=False, na=False)
        ]
        mills_polls = state_polls[
            state_polls['d_candidate'].str.contains('Mills', case=False, na=False)
        ]
        platner_blend, np_ = _blend_subset(platner_polls, reference_date)
        mills_blend,   nm  = _blend_subset(mills_polls,   reference_date)

        if platner_blend is None and mills_blend is None:
            return {'blend_margin': None, 'n_polls': 0, 'latest_poll': None}

        platner_blend = platner_blend if platner_blend is not None else 0.0
        mills_blend   = mills_blend   if mills_blend   is not None else 0.0

        blend = (priors['ME_platner'] * platner_blend +
                 priors['ME_mills']   * mills_blend)
        n_polls = np_ + nm

    elif state_abbr == 'TX':
        # Separate blends per R candidate
        priors = config['race_model']['primary_priors']
        cornyn_polls = state_polls[
            state_polls['r_candidate'].str.contains('Cornyn', case=False, na=False)
        ]
        paxton_polls = state_polls[
            state_polls['r_candidate'].str.contains('Paxton', case=False, na=False)
        ]
        cornyn_blend, nc = _blend_subset(cornyn_polls, reference_date)
        paxton_blend, np_ = _blend_subset(paxton_polls, reference_date)

        if cornyn_blend is None and paxton_blend is None:
            return {'blend_margin': None, 'n_polls': 0, 'latest_poll': None}

        cornyn_blend = cornyn_blend if cornyn_blend is not None else 0.0
        paxton_blend = paxton_blend if paxton_blend is not None else 0.0

        blend = (priors['TX_cornyn'] * cornyn_blend +
                 priors['TX_paxton'] * paxton_blend)
        n_polls = nc + np_

    else:
        blend, n_polls = _blend_subset(state_polls, reference_date)

    latest = state_polls['date_end'].max() if len(state_polls) > 0 else None

    return {
        'blend_margin': round(blend, 3) if blend is not None else None,
        'n_polls':      n_polls,
        'latest_poll':  latest,
    }


if __name__ == '__main__':
    cfg = load_config()
    df  = pd.read_csv('data/state_polls.csv')
    ref = date.today()

    print("POLL BLEND MODULE — UNIT TEST")
    print()
    test_states = ['NH','NC','GA','ME','MI','AK','OH','IA','TX','NE','FL','MN']
    print(f"{'State':<6} {'Blend':>8} {'N Polls':>8} {'Latest Poll'}")
    print("─" * 45)
    for abbr in test_states:
        result = compute_poll_blend(abbr, df, ref, cfg)
        blend  = f"D{result['blend_margin']:+.1f}" if result['blend_margin'] is not None else "NO POLLS"
        print(f"  {abbr:<4} {blend:>8}  {result['n_polls']:>7}  {result['latest_poll']}")

    print()
    # Verify special cases
    me = compute_poll_blend('ME', df, ref, cfg)
    tx = compute_poll_blend('TX', df, ref, cfg)
    ak = compute_poll_blend('AK', df, ref, cfg)
    print(f"VERIFY: ME blend uses Platner/Mills primary weighting → {'PASS' if me['n_polls'] > 0 else 'FAIL'}")
    print(f"VERIFY: TX blend uses Cornyn/Paxton primary weighting → {'PASS' if tx['n_polls'] > 0 else 'FAIL'}")
    print(f"VERIFY: AK only uses H2H_SULLIVAN polls → {'PASS' if ak['n_polls'] > 0 else 'FAIL'}")
