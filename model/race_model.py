"""
Computes central estimate, sigma, and D win probability for each Senate race.
Combines state environment (MRP), direct poll blend, WAR, and fundraising.
"""

import math
import yaml
import pandas as pd
import numpy as np
from scipy import stats
from datetime import date

from model.oop_shift import compute_oop_shift
from model.demographic_shifts import compute_group_shifts
from model.state_environment import compute_state_environments
from model.poll_blend import compute_poll_blend
from model.blending import compute_env_weight, compute_time_sigma


def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


# ── ALL 37 SENATE RACES 2026 ──────────────────────────────────────────────────
# (abbr, incumbent_party, d_candidate, r_candidate, competitive)
SENATE_RACES_2026 = [
    ('AL', 'R', 'TBD',               'TBD',                  False),
    ('AK', 'R', 'Mary Peltola',      'Dan Sullivan',         True),
    ('AR', 'R', 'TBD',               'John Boozman',         False),
    ('CO', 'D', 'Michael Bennet',    'TBD',                  False),
    ('DE', 'D', 'Lisa Blunt Roch.',  'TBD',                  False),
    ('FL', 'R', 'Alex Vindman',      'Ashley Moody',         True),
    ('GA', 'D', 'Jon Ossoff',        'TBD',                  True),
    ('ID', 'R', 'TBD',               'Mike Crapo',           False),
    ('IL', 'D', 'TBD',               'TBD',                  False),
    ('IA', 'R', 'Josh Turek',        'Ashley Hinson',        True),
    ('KS', 'R', 'TBD',               'Roger Marshall',       False),
    ('KY', 'R', 'TBD',               'TBD',                  False),
    ('LA', 'R', 'TBD',               'Bill Cassidy',         False),
    ('MA', 'D', 'Ed Markey',         'TBD',                  False),
    ('MD', 'D', 'Chris Van Hollen',  'TBD',                  False),
    ('ME', 'R', 'Graham Platner',    'Susan Collins',        True),
    ('MI', 'D', 'Stevens/McMorrow',  'Mike Rogers',          True),
    ('MN', 'D', 'Craig/Flanagan',    'Michele Tafoya',       True),
    ('MS', 'R', 'TBD',               'Roger Wicker',         False),
    ('MT', 'R', 'TBD',               'Kurt Alme',            False),
    ('NE', 'R', 'Dan Osborn (I)',    'Pete Ricketts',        True),
    ('NH', 'D', 'Chris Pappas',      'John Sununu',          True),
    ('OH', 'R', 'Sherrod Brown',     'Jon Husted',           True),   # special — Vance term remainder
    ('NJ', 'D', 'Cory Booker',       'TBD',                  False),
    ('NM', 'D', 'Ben Ray Lujan',     'TBD',                  False),
    ('NC', 'R', 'Roy Cooper',        'Michael Whatley',      True),
    ('OK', 'R', 'TBD',               'James Lankford',       False),
    ('OR', 'D', 'Jeff Merkley',      'TBD',                  False),
    ('RI', 'D', 'Sheldon Whitehouse','TBD',                  False),
    ('SC', 'R', 'TBD',               'Tim Scott',            False),
    ('SD', 'R', 'TBD',               'Mike Rounds',          False),
    ('TN', 'R', 'TBD',               'Marsha Blackburn',     False),
    ('TX', 'R', 'James Talarico',    'Cornyn/Paxton',        True),
    ('VA', 'D', 'Mark Warner',       'TBD',                  False),
    ('WA', 'D', 'Maria Cantwell',    'TBD',                  False),
    ('WV', 'R', 'TBD',               'Shelley Moore Capito', False),
    ('WY', 'R', 'TBD',               'TBD',                  False),
]


def get_rating(d_prob):
    if d_prob > 0.95: return 'Safe D'
    if d_prob > 0.80: return 'Likely D'
    if d_prob > 0.60: return 'Lean D'
    if d_prob > 0.40: return 'Tossup'
    if d_prob > 0.20: return 'Lean R'
    if d_prob > 0.08: return 'Likely R'
    return 'Safe R'


def compute_fundraising_signal(abbr, fec_csv='data/fec_fundraising.csv'):
    """
    log(d_raised / r_raised) / log(5) * 2.0, clipped to [-2.0, +2.0]
    Returns 0.0 if no data.
    """
    try:
        df = pd.read_csv(fec_csv)
        state_df = df[df['abbr'] == abbr]
        d_raised = state_df[state_df['party'] == 'D']['raised_k'].sum()
        r_raised = state_df[state_df['party'] == 'R']['raised_k'].sum()
        if d_raised <= 0 or r_raised <= 0:
            return 0.0
        signal = math.log(d_raised / r_raised) / math.log(5) * 2.0
        return max(-2.0, min(2.0, signal))
    except Exception:
        return 0.0


def compute_race(abbr, state_env, poll_blend_result,
                 days_remaining, config):
    """
    Returns full race result dict with all intermediate values.

    Weight logic:
    1. Get base weights from config for n_polls bucket
    2. Compute dynamic env/poll split from blending.py
    3. Scale env and poll weights proportionally
    4. WAR and fund weights stay at config values
    5. Renormalize all four to sum to 1
    """
    n_polls    = poll_blend_result['n_polls']
    poll_blend = poll_blend_result['blend_margin']  # may be None

    # ── Base weights from config ──────────────────────────────────────────────
    rm = config['race_model']
    w_cfg = rm['weights']

    # State override check
    state_overrides = w_cfg.get('state_weight_overrides', {})
    if abbr in state_overrides:
        base_w = state_overrides[abbr]
    elif n_polls >= 5:
        base_w = w_cfg['n_gte_5']
    elif n_polls >= 1:
        base_w = w_cfg['n_1_to_4']
    else:
        base_w = w_cfg['n_0']

    # ── Dynamic env/poll split ────────────────────────────────────────────────
    env_weight, poll_weight = compute_env_weight(abbr, n_polls, days_remaining, config)

    # Scale env and poll weights from base, keeping WAR/fund fixed
    w_war  = base_w['war']
    w_fund = base_w['fund']
    remaining = 1.0 - w_war - w_fund
    w_env  = env_weight  * remaining
    w_poll = poll_weight * remaining

    # Renormalize
    total = w_env + w_poll + w_war + w_fund
    w_env  /= total
    w_poll /= total
    w_war  /= total
    w_fund /= total

    # ── Central estimate ──────────────────────────────────────────────────────
    gcb_eday = state_env.get('gcb_eday')
    if gcb_eday is None:
        gcb_eday = 0.0  # fallback for states with no catalist data

    war_cfg  = rm.get('war', {})
    war_net  = 0.0
    if abbr in war_cfg:
        war_net = war_cfg[abbr].get('d', 0.0) - war_cfg[abbr].get('r', 0.0)

    fund_signal = compute_fundraising_signal(abbr)

    if poll_blend is not None and n_polls > 0:
        central = (w_env  * gcb_eday +
                   w_poll * poll_blend +
                   w_war  * war_net +
                   w_fund * fund_signal)
    else:
        # No polls — env + war + fund only, renormalized
        denom = w_env + w_war + w_fund
        central = ((w_env / denom)  * gcb_eday +
                   (w_war / denom)  * war_net +
                   (w_fund / denom) * fund_signal)

    # ── Sigma ──────────────────────────────────────────────────────────────────
    gcb_sigma = state_env.get('gcb_sigma', config['oop']['gcb_tracker_uncertainty'])

    # Poll sigma: scales with poll count
    if n_polls >= 10:  poll_sigma = gcb_sigma * 0.85
    elif n_polls >= 5: poll_sigma = gcb_sigma * 0.92
    elif n_polls >= 1: poll_sigma = gcb_sigma * 1.00
    else:              poll_sigma = gcb_sigma * 1.20

    time_sigma = compute_time_sigma(days_remaining, config)

    prim_sigma = rm.get('primary_extra_sigma', {}).get(abbr, 0.0)

    total_sigma = math.sqrt(poll_sigma**2 + time_sigma**2 + prim_sigma**2)

    # ── D win probability ─────────────────────────────────────────────────────
    d_prob = float(stats.norm.cdf(central / total_sigma)) if total_sigma > 0 else 0.5

    return {
        'abbr':         abbr,
        'central':      round(central, 3),
        'total_sigma':  round(total_sigma, 3),
        'poll_sigma':   round(poll_sigma, 3),
        'time_sigma':   round(time_sigma, 3),
        'prim_sigma':   round(prim_sigma, 3),
        'gcb_eday':     round(gcb_eday, 3),
        'poll_blend':   round(poll_blend, 3) if poll_blend is not None else None,
        'n_polls':      n_polls,
        'war_net':      round(war_net, 3),
        'fund_signal':  round(fund_signal, 3),
        'w_env':        round(w_env, 3),
        'w_poll':       round(w_poll, 3),
        'w_war':        round(w_war, 3),
        'w_fund':       round(w_fund, 3),
        'd_prob':       round(d_prob, 4),
        'rating':       get_rating(d_prob),
    }


if __name__ == '__main__':
    cfg  = load_config()
    oop  = compute_oop_shift(cfg)
    demo = compute_group_shifts(cfg)
    envs, implied_national = compute_state_environments(oop, demo, cfg)
    polls_df = pd.read_csv('data/state_polls.csv')
    ref_date = date.today()
    election_date = date(2026, 11, 3)
    days_remaining = (election_date - ref_date).days

    print(f"RACE MODEL — UNIT TEST  |  {days_remaining} days to election")
    print(f"GCB eday: D{oop['gcb_eday_central']:+.2f}  |  σ=±{oop['gcb_sigma']:.2f}pp")
    print()

    results = {}
    for abbr, inc, dc, rc, competitive in SENATE_RACES_2026:
        if abbr not in envs:
            continue
        blend  = compute_poll_blend(abbr, polls_df, ref_date, cfg)
        result = compute_race(abbr, envs[abbr], blend, days_remaining, cfg)
        results[abbr] = result

    # Print competitive races sorted by D probability
    comp_abbrs = [r[0] for r in SENATE_RACES_2026 if r[4]]
    comp = sorted([results[a] for a in comp_abbrs if a in results],
                  key=lambda x: -x['d_prob'])

    print(f"{'St':<4} {'D Win%':>7} {'Central':>8} {'σ':>6} {'Polls':>6}  {'Rating':<12} {'GCB':>7} {'Poll':>7} {'WAR':>5}")
    print("─" * 80)
    for r in comp:
        pb = f"D{r['poll_blend']:+.1f}" if r['poll_blend'] is not None else "  n/a"
        print(f"{r['abbr']:<4} {r['d_prob']:>7.1%} {r['central']:>+8.2f} "
              f"{r['total_sigma']:>6.2f} {r['n_polls']:>6}  "
              f"{r['rating']:<12} {r['gcb_eday']:>+7.2f} {pb:>7} {r['war_net']:>+5.1f}")

    print()
    # Verify key races
    nc = results.get('NC', {}); oh = results.get('OH', {})
    me = results.get('ME', {}); ak = results.get('AK', {})
    print(f"VERIFY: NC D prob > 80%  → {'PASS' if nc.get('d_prob',0) > 0.80 else 'FAIL'} ({nc.get('d_prob',0):.1%})")
    print(f"VERIFY: OH D prob < 60%  → {'PASS' if oh.get('d_prob',1) < 0.60 else 'FAIL'} ({oh.get('d_prob',1):.1%})")
    print(f"VERIFY: ME D prob > 70%  → {'PASS' if me.get('d_prob',0) > 0.70 else 'FAIL'} ({me.get('d_prob',0):.1%})")
    print(f"VERIFY: AK D prob 40-70% → {'PASS' if 0.40 < ak.get('d_prob',0) < 0.70 else 'FAIL'} ({ak.get('d_prob',0):.1%})")
