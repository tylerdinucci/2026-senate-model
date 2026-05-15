"""
Synthetic MRP: applies demographic group shifts × state electorate composition
to produce state-specific GCB election day estimates.

Formula per state:
  demo_shift_state = Σ (acs_share[group] × national_shift[group])
  gcb_eday = catalist_2024_state_margin + demo_shift_state + oop_shift

Note: 'other' group uses the 'total' shift as a proxy.
"""

import pandas as pd
import numpy as np
import yaml


def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


def compute_state_environments(oop_result, demo_shifts, config,
                               catalist_csv='data/catalist_2024.csv',
                               acs_csv='data/acs_composition.csv'):
    """
    Returns dict: {abbr: {catalist_baseline, demo_shift, oop_shift,
                           gcb_eday, gcb_eday_lo, gcb_eday_hi, reliable}}
    """
    catalist = pd.read_csv(catalist_csv)
    acs      = pd.read_csv(acs_csv)

    gcb_sigma = oop_result['gcb_sigma']
    oop_shift = oop_result['oop_shift']

    # Map group shifts to a simple dict (in pp)
    # ACS composition groups: white_nc, white_college, hispanic, black, aapi, other
    # Demographic model groups: same + youth_18_29, seniors_65p, total
    # For 'other' use 'total' shift as proxy
    group_shifts_pp = {g: d['shift'] for g, d in demo_shifts.items()}
    other_shift = group_shifts_pp.get('total', 0.0)

    acs_groups = ['white_nc', 'white_college', 'hispanic', 'black', 'aapi', 'other']

    # State-level overrides for unrepresentative MIT House baselines
    catalist_overrides = config.get('catalist_state_overrides', {})

    results = {}

    for _, cat_row in catalist.iterrows():
        abbr = cat_row['abbr']
        baseline = cat_row['d_margin_2024']
        reliable = cat_row.get('reliable', True)

        # Apply override if configured (replaces MIT House margin entirely)
        if abbr in catalist_overrides:
            baseline = catalist_overrides[abbr]
            reliable = True

        if pd.isna(baseline):
            # State had all-uncontested races — can't use
            results[abbr] = {
                'catalist_baseline': None,
                'demo_shift': None,
                'oop_shift': oop_shift,
                'gcb_eday': None,
                'gcb_eday_lo': None,
                'gcb_eday_hi': None,
                'reliable': False,
                'note': 'No catalist baseline (all uncontested 2024)'
            }
            continue

        # Get ACS composition for this state
        acs_row = acs[acs['abbr'] == abbr]
        if len(acs_row) == 0:
            results[abbr] = {
                'catalist_baseline': float(baseline),
                'demo_shift': 0.0,  # no composition data
                'oop_shift': oop_shift,
                'gcb_eday': round(float(baseline) + oop_shift, 3),
                'gcb_eday_lo': round(float(baseline) + oop_shift - gcb_sigma, 3),
                'gcb_eday_hi': round(float(baseline) + oop_shift + gcb_sigma, 3),
                'reliable': False,
                'note': 'No ACS composition data — demo_shift set to 0'
            }
            continue

        acs_row = acs_row.iloc[0]

        # Compute demographic shift for this state
        demo_shift = 0.0
        for group in acs_groups:
            share = float(acs_row.get(group, 0.0))
            if group == 'other':
                shift_pp = other_shift
            else:
                shift_pp = group_shifts_pp.get(group, 0.0)
            demo_shift += share * shift_pp

        gcb_eday = round(float(baseline) + demo_shift + oop_shift, 3)

        results[abbr] = {
            'catalist_baseline': round(float(baseline), 3),
            'demo_shift':        round(demo_shift, 3),
            'oop_shift':         oop_shift,
            'gcb_eday':          gcb_eday,
            'gcb_eday_lo':       round(gcb_eday - gcb_sigma, 3),
            'gcb_eday_hi':       round(gcb_eday + gcb_sigma, 3),
            'reliable':          bool(reliable),
            'note':              str(cat_row.get('note', ''))
        }

    # Sanity check: implied national GCB
    # Weight by state population (approximate using n_districts as proxy)
    eday_vals = [v['gcb_eday'] for v in results.values() if v['gcb_eday'] is not None]
    implied_national = round(np.mean(eday_vals), 2)
    gap = implied_national - oop_result['gcb_eday_central']
    if abs(gap) > 1.5:
        print(f"  WARNING: Implied national GCB ({implied_national:+.2f}) "
              f"differs from projected ({oop_result['gcb_eday_central']:+.2f}) by {gap:+.2f}pp")

    return results, implied_national


if __name__ == '__main__':
    from model.oop_shift import compute_oop_shift
    from model.demographic_shifts import compute_group_shifts

    cfg = load_config()
    oop   = compute_oop_shift(cfg)
    demo  = compute_group_shifts(cfg)
    envs, implied_national = compute_state_environments(oop, demo, cfg)

    print("STATE ENVIRONMENT MODULE — UNIT TEST")
    print(f"OOP shift: {oop['oop_shift']:+.2f}pp | GCB eday: D{oop['gcb_eday_central']:+.2f}")
    print(f"Implied national GCB from state rollup: D{implied_national:+.2f}")
    print(f"Gap vs projected: {implied_national - oop['gcb_eday_central']:+.2f}pp")
    print()

    key_states = ['NC','GA','MI','TX','AK','OH','IA','ME','NH','NE','FL','MN','CA','WY']
    print(f"{'St':<4} {'Baseline':>10} {'DemoShift':>10} {'OOP':>6} {'GCB Eday':>10} {'Lo':>8} {'Hi':>8}")
    print("─" * 70)
    for abbr in key_states:
        if abbr not in envs: continue
        e = envs[abbr]
        if e['gcb_eday'] is None:
            print(f"{abbr:<4} {'NO DATA':>10}")
            continue
        print(f"{abbr:<4} {e['catalist_baseline']:>+10.2f} {e['demo_shift']:>+10.2f} "
              f"{e['oop_shift']:>+6.2f} {e['gcb_eday']:>+10.2f} "
              f"{e['gcb_eday_lo']:>+8.2f} {e['gcb_eday_hi']:>+8.2f}")

    # Verify competitive state directions
    print()
    nc_eday = envs.get('NC', {}).get('gcb_eday')
    oh_eday = envs.get('OH', {}).get('gcb_eday')
    wy_eday = envs.get('WY', {}).get('gcb_eday')
    print(f"VERIFY: NC GCB eday positive (D-favorable) → {'PASS' if nc_eday and nc_eday > 0 else 'FAIL'} ({nc_eday:+.2f})")
    print(f"VERIFY: OH GCB eday negative (R-favorable) → {'PASS' if oh_eday and oh_eday < 0 else 'FAIL'} ({oh_eday:+.2f})")
    print(f"VERIFY: WY GCB eday very negative (deep R)  → {'PASS' if wy_eday and wy_eday < -30 else 'FAIL'} ({wy_eday:+.2f})")
