"""
Synthetic MRP: GCB tracker sets the national level; demographics distribute it.

Formula per state:
  national_shift = gcb_eday_projected − catalist_national_gcb
  demo_tilt      = state_demo_shift − national_demo_shift (n_districts weighted)
  gcb_eday       = catalist_state_baseline + national_shift + demo_tilt

The GCB tracker (today's polling + OOP shift) anchors the national level.
Demographic group shifts from crosstabs determine each state's deviation:
a Latino surge matters more in TX than in ME.
Uncertainty about whether the projected GCB holds lives in the Monte Carlo
national shock (gcb_national_sigma in config), not in the central estimate.
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
    Returns dict: {abbr: {catalist_baseline, demo_shift, demo_tilt,
                           national_shift, gcb_eday, gcb_eday_lo,
                           gcb_eday_hi, reliable}}
    and implied_national (float).
    """
    catalist = pd.read_csv(catalist_csv)
    acs      = pd.read_csv(acs_csv)

    gcb_sigma         = oop_result['gcb_sigma']
    gcb_eday_national = oop_result['gcb_eday_central']
    catalist_nat_gcb  = config['catalist_2024']['national_gcb']   # -3.06 House
    national_pres     = config['catalist_2024'].get('national_pres', -1.50)  # -1.50 pres

    group_shifts_pp = {g: d['shift'] for g, d in demo_shifts.items()}
    other_shift     = group_shifts_pp.get('total', 0.0)
    acs_groups      = ['white_nc', 'white_college', 'hispanic', 'black', 'aapi', 'other']
    catalist_overrides = config.get('catalist_state_overrides', {})

    # ── DOWNBALLOT LAG AUTO-OVERRIDES ─────────────────────────────────────────
    # States where |house_2024 - pres_2024| > threshold get baseline replaced
    # with 2-cycle presidential average. Manual overrides take precedence.
    lag_cfg   = config.get('downballot_lag', {})
    lag_enabled   = lag_cfg.get('enabled', False)
    lag_threshold = float(lag_cfg.get('threshold_pp', 2.5))
    auto_overrides = {}

    if lag_enabled:
        pres_path = catalist_csv.replace('catalist_2024.csv', 'presidential_margins.csv')
        # Handle backtest paths (catalist_2020.csv etc.)
        import os
        pres_candidate = os.path.join(os.path.dirname(catalist_csv), 'presidential_margins.csv')
        if os.path.exists(pres_candidate):
            pres_df = pd.read_csv(pres_candidate)
            pres_map = {row['abbr']: (float(row['d_margin_pres_2024']),
                                      float(row['d_margin_pres_2020']))
                        for _, row in pres_df.iterrows()}

            flagged = []
            for _, row in catalist.iterrows():
                abbr  = row['abbr']
                house = row['d_margin_2024']
                if pd.isna(house) or abbr in catalist_overrides or abbr not in pres_map:
                    continue
                p24, p20  = pres_map[abbr]
                gap       = float(house) - p24
                if abs(gap) > lag_threshold:
                    two_cycle_avg = round((p24 + p20) / 2, 2)
                    auto_overrides[abbr] = two_cycle_avg
                    flagged.append((abbr, float(house), p24, gap, two_cycle_avg))

            if flagged:
                print(f"  Downballot lag overrides (|house−pres| > {lag_threshold}pp):")
                for abbr, house, p24, gap, avg in sorted(flagged, key=lambda x: x[3]):
                    tag = 'H>R' if gap < 0 else 'H>D'
                    print(f"    {abbr}: house {house:+.1f} → pres avg {avg:+.1f}  "
                          f"(gap {gap:+.1f}pp, {tag})")

    # Merge: auto_overrides first, then manual overrides take precedence
    effective_overrides = {**auto_overrides, **catalist_overrides}

    # ── PASS 1: compute per-state demo shifts ──────────────────────────────────
    state_demo_shifts = {}   # abbr → float
    state_weights     = {}   # abbr → n_districts (population proxy)

    for _, cat_row in catalist.iterrows():
        abbr     = cat_row['abbr']
        baseline = effective_overrides.get(abbr, cat_row['d_margin_2024'])
        if pd.isna(baseline):
            continue
        acs_row = acs[acs['abbr'] == abbr]
        if len(acs_row) == 0:
            continue
        acs_row = acs_row.iloc[0]

        demo_shift = sum(
            float(acs_row.get(g, 0.0)) * (other_shift if g == 'other' else group_shifts_pp.get(g, 0.0))
            for g in acs_groups
        )
        state_demo_shifts[abbr] = demo_shift
        state_weights[abbr]     = float(cat_row.get('n_districts', 1))

    # National average demo shift — weighted by n_districts
    w = np.array(list(state_weights.values()))
    d = np.array([state_demo_shifts[a] for a in state_weights])
    national_demo_shift = float((d * w).sum() / w.sum())

    # National shifts: two anchors depending on baseline type.
    # House-baseline states use the 2024 House GCB (-3.06) as anchor.
    # Presidential-override states use the 2024 presidential margin (-1.50) as anchor,
    # since their baselines are presidential data — mixing scales otherwise.
    national_shift_house = gcb_eday_national - catalist_nat_gcb   # e.g. +10.82pp
    national_shift_pres  = gcb_eday_national - national_pres       # e.g. +9.26pp

    print(f"  GCB anchor: D{gcb_eday_national:+.2f} projected election day  "
          f"(today D{oop_result['gcb_today']:+.2f} + OOP {oop_result['oop_shift']:+.2f}pp)  "
          f"→ house shift {national_shift_house:+.2f}pp / pres shift {national_shift_pres:+.2f}pp")

    # ── PASS 2: build state estimates anchored to GCB tracker ─────────────────
    results = {}

    for _, cat_row in catalist.iterrows():
        abbr     = cat_row['abbr']
        baseline = cat_row['d_margin_2024']
        reliable = cat_row.get('reliable', True)

        if abbr in effective_overrides:
            baseline = effective_overrides[abbr]
            reliable = True

        # Presidential-override states use pres anchor; House-baseline states use House anchor.
        national_shift = national_shift_pres if abbr in effective_overrides else national_shift_house

        if pd.isna(baseline):
            results[abbr] = {
                'catalist_baseline': None,
                'demo_shift':        None,
                'demo_tilt':         None,
                'national_shift':    round(national_shift_house, 3),
                'gcb_eday':          None,
                'gcb_eday_lo':       None,
                'gcb_eday_hi':       None,
                'reliable':          False,
                'note': 'No catalist baseline (all uncontested 2024)'
            }
            continue

        if abbr not in state_demo_shifts:
            # No ACS data — apply national shift with no tilt
            gcb_eday = round(float(baseline) + national_shift, 3)
            results[abbr] = {
                'catalist_baseline': round(float(baseline), 3),
                'demo_shift':        0.0,
                'demo_tilt':         0.0,
                'national_shift':    round(national_shift, 3),
                'gcb_eday':          gcb_eday,
                'gcb_eday_lo':       round(gcb_eday - gcb_sigma, 3),
                'gcb_eday_hi':       round(gcb_eday + gcb_sigma, 3),
                'reliable':          False,
                'note': 'No ACS composition data — demo_tilt set to 0'
            }
            continue

        demo_shift = state_demo_shifts[abbr]
        demo_tilt  = demo_shift - national_demo_shift   # state deviation from avg

        gcb_eday = round(float(baseline) + national_shift + demo_tilt, 3)

        results[abbr] = {
            'catalist_baseline': round(float(baseline), 3),
            'demo_shift':        round(demo_shift, 3),
            'demo_tilt':         round(demo_tilt, 3),
            'national_shift':    round(national_shift, 3),
            'gcb_eday':          gcb_eday,
            'gcb_eday_lo':       round(gcb_eday - gcb_sigma, 3),
            'gcb_eday_hi':       round(gcb_eday + gcb_sigma, 3),
            'reliable':          bool(reliable),
            'note':              str(cat_row.get('note', ''))
        }

    # Sanity check — should now be close to gcb_eday_national
    eday_vals = [v['gcb_eday'] for v in results.values() if v['gcb_eday'] is not None]
    implied_national = round(np.mean(eday_vals), 2)
    gap = implied_national - gcb_eday_national
    if abs(gap) > 1.0:
        print(f"  NOTE: Implied national GCB ({implied_national:+.2f}) vs "
              f"projected ({gcb_eday_national:+.2f}): gap {gap:+.2f}pp "
              f"(from unequal state population weighting)")
    else:
        print(f"  GCB anchor check: implied={implied_national:+.2f}  "
              f"projected={gcb_eday_national:+.2f}  gap={gap:+.2f}pp ✓")

    return results, implied_national


if __name__ == '__main__':
    from model.oop_shift import compute_oop_shift
    from model.demographic_shifts import compute_group_shifts

    cfg  = load_config()
    oop  = compute_oop_shift(cfg)
    demo = compute_group_shifts(cfg)
    envs, implied_national = compute_state_environments(oop, demo, cfg)

    print("STATE ENVIRONMENT MODULE — UNIT TEST")
    print(f"GCB today: D{oop['gcb_today']:+.2f} | OOP shift: +{oop['oop_shift']:.2f}pp | "
          f"GCB eday projected: D{oop['gcb_eday_central']:+.2f}")
    print(f"National shift applied:                 {oop['gcb_eday_central'] - cfg['catalist_2024']['national_gcb']:+.2f}pp")
    print(f"Implied national from state rollup:     D{implied_national:+.2f}")
    print()

    key_states = ['NC', 'GA', 'MI', 'TX', 'AK', 'OH', 'IA', 'ME', 'NH', 'NE', 'FL', 'MN', 'WY']
    print(f"{'St':<4} {'Baseline':>10} {'DemoTilt':>10} {'NatShift':>10} {'GCB Eday':>10} {'Lo':>8} {'Hi':>8}")
    print("─" * 72)
    for abbr in key_states:
        if abbr not in envs:
            continue
        e = envs[abbr]
        if e['gcb_eday'] is None:
            print(f"{abbr:<4} {'NO DATA':>10}")
            continue
        print(f"{abbr:<4} {e['catalist_baseline']:>+10.2f} {e['demo_tilt']:>+10.2f} "
              f"{e['national_shift']:>+10.2f} {e['gcb_eday']:>+10.2f} "
              f"{e['gcb_eday_lo']:>+8.2f} {e['gcb_eday_hi']:>+8.2f}")

    print()
    nc_eday = envs.get('NC', {}).get('gcb_eday')
    oh_eday = envs.get('OH', {}).get('gcb_eday')
    wy_eday = envs.get('WY', {}).get('gcb_eday')
    print(f"VERIFY NC positive:  {'PASS' if nc_eday and nc_eday > 0 else 'FAIL'} ({nc_eday:+.2f})")
    print(f"VERIFY OH negative:  {'PASS' if oh_eday and oh_eday < 0 else 'FAIL'} ({oh_eday:+.2f})")
    print(f"VERIFY WY very neg:  {'PASS' if wy_eday and wy_eday < -20 else 'FAIL'} ({wy_eday:+.2f})")
