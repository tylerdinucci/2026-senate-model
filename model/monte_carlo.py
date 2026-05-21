"""
Correlated Monte Carlo simulation.
100,000 runs → D majority probability, seat distribution.
"""

import numpy as np
import yaml
from scipy import stats


def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


def build_correlation_matrix(competitive_states, config):
    """
    Build correlation matrix from config.correlations.
    Unspecified pairs get national_baseline_correlation.
    Enforce PSD via eigenvalue floor.
    """
    n = len(competitive_states)
    idx = {s: i for i, s in enumerate(competitive_states)}
    baseline = config['simulation']['national_baseline_correlation']

    corr = np.full((n, n), baseline)
    np.fill_diagonal(corr, 1.0)

    for pair in config['simulation'].get('correlations', []):
        states = pair['states']
        value  = pair['value']
        if states[0] in idx and states[1] in idx:
            i, j = idx[states[0]], idx[states[1]]
            corr[i, j] = value
            corr[j, i] = value

    # Enforce PSD
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 1e-8)
    corr_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return corr_psd


def run_simulation(race_results, oop_result, config, seed=None):
    """
    race_results: dict {abbr: compute_race() output}
    oop_result:   dict from compute_oop_shift()

    Returns full simulation result dict.
    """
    sim_cfg = config['simulation']
    n_sims  = sim_cfg['n_sims']
    rng_seed = seed if seed is not None else sim_cfg.get('seed', 42)
    np.random.seed(rng_seed)

    d_seats_not_up   = sim_cfg['d_seats_not_up']
    majority_threshold = sim_cfg['majority_threshold']
    gcb_sigma          = oop_result['gcb_sigma']

    # Split races into competitive (correlated) and non-competitive (independent)
    # Nebraska: Osborn (I) modeled as D-equivalent
    comp_abbrs = sorted([
        abbr for abbr, r in race_results.items()
        if 0.05 < r['d_prob'] < 0.95  # only races with real uncertainty
    ])
    noncomp_abbrs = [a for a in race_results if a not in comp_abbrs]

    # Build correlation matrix for competitive states
    corr_matrix = build_correlation_matrix(comp_abbrs, config)

    # Centrals and sigmas for competitive states
    centrals = np.array([race_results[a]['central'] for a in comp_abbrs])
    sigmas   = np.array([race_results[a]['total_sigma'] for a in comp_abbrs])

    # Covariance matrix
    cov = corr_matrix * np.outer(sigmas, sigmas)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-8)
    cov_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Draw correlated margins: shape (n_sims, n_comp)
    draws = np.random.multivariate_normal(centrals, cov_psd, n_sims)

    # National GCB environment shock: represents uncertainty about whether the
    # projected election-day GCB (e.g. D+7.73) actually materializes.
    # Applied identically to all states — a fully correlated national tide shift.
    # gcb_national_sigma in config controls the width; empirically the GCB at
    # 6 months out has ~3-4pp std dev vs. the final result.
    gcb_national_sigma = float(sim_cfg.get('gcb_national_sigma', 3.5))
    national_shock = np.random.normal(0, gcb_national_sigma, n_sims)
    draws = draws + national_shock[:, np.newaxis]

    comp_wins = (draws > 0).astype(int)  # shape (n_sims, n_comp)

    # Non-competitive states: independent draws
    noncomp_wins = np.zeros(n_sims, dtype=int)
    for abbr in noncomp_abbrs:
        r = race_results[abbr]
        nc_draws = np.random.normal(r['central'], r['total_sigma'], n_sims)
        noncomp_wins += (nc_draws > 0).astype(int)

    total_d = d_seats_not_up + comp_wins.sum(axis=1) + noncomp_wins

    # Majority probability
    d_majority_prob = float((total_d >= majority_threshold).mean())
    exp_d_seats     = float(total_d.mean())
    tie_50_prob     = float((total_d == 50).mean())

    # Scenario sims: shift all competitive margins by ±gcb_sigma
    lo_draws = draws - gcb_sigma
    hi_draws = draws + gcb_sigma
    lo_wins  = (lo_draws > 0).astype(int).sum(axis=1) + noncomp_wins
    hi_wins  = (hi_draws > 0).astype(int).sum(axis=1) + noncomp_wins
    maj_lo   = float(((d_seats_not_up + lo_wins) >= majority_threshold).mean())
    maj_hi   = float(((d_seats_not_up + hi_wins) >= majority_threshold).mean())

    # Seat distribution
    seat_dist = {}
    for seats in range(int(total_d.min()), int(total_d.max()) + 1):
        p = float((total_d == seats).mean())
        if p >= 0.001:
            seat_dist[seats] = round(p, 4)

    # Per-race win probabilities from simulation
    sim_race_probs = {}
    for i, abbr in enumerate(comp_abbrs):
        sim_race_probs[abbr] = round(float(comp_wins[:, i].mean()), 4)
    for abbr in noncomp_abbrs:
        sim_race_probs[abbr] = race_results[abbr]['d_prob']

    return {
        'd_majority_prob':  round(d_majority_prob, 4),
        'exp_d_seats':      round(exp_d_seats, 2),
        'tie_50_prob':      round(tie_50_prob, 4),
        'maj_lo':           round(maj_lo, 4),
        'maj_hi':           round(maj_hi, 4),
        'seat_distribution': seat_dist,
        'sim_race_probs':   sim_race_probs,
        'comp_states':      comp_abbrs,
        'n_sims':           n_sims,
    }


if __name__ == '__main__':
    from datetime import date
    import pandas as pd
    from model.oop_shift import compute_oop_shift
    from model.demographic_shifts import compute_group_shifts
    from model.state_environment import compute_state_environments
    from model.poll_blend import compute_poll_blend
    from model.race_model import compute_race, SENATE_RACES_2026

    cfg  = load_config()
    oop  = compute_oop_shift(cfg)
    demo = compute_group_shifts(cfg)
    envs, _ = compute_state_environments(oop, demo, cfg)
    polls_df = pd.read_csv('data/state_polls.csv')
    ref_date = date.today()
    election_date = date(2026, 11, 3)
    days_remaining = (election_date - ref_date).days

    race_results = {}
    for abbr, inc, dc, rc, competitive in SENATE_RACES_2026:
        if abbr not in envs:
            continue
        blend  = compute_poll_blend(abbr, polls_df, ref_date, cfg)
        result = compute_race(abbr, envs[abbr], blend, days_remaining, cfg)
        race_results[abbr] = result

    print("MONTE CARLO — UNIT TEST  (100,000 simulations)")
    print("Running...")
    sim = run_simulation(race_results, oop, cfg)

    print(f"\n{'='*55}")
    print(f"  D MAJORITY PROBABILITY:  {sim['d_majority_prob']:.1%}")
    print(f"  Expected D seats:        {sim['exp_d_seats']:.1f}")
    print(f"  50-seat tie:             {sim['tie_50_prob']:.1%}")
    print(f"  Low scenario:            {sim['maj_lo']:.1%}")
    print(f"  High scenario:           {sim['maj_hi']:.1%}")
    print(f"  Competitive states:      {', '.join(sim['comp_states'])}")
    print(f"{'='*55}")

    print(f"\nSeat distribution (top outcomes):")
    sorted_dist = sorted(sim['seat_distribution'].items(),
                         key=lambda x: -x[1])
    for seats, prob in sorted_dist[:8]:
        bar = '█' * int(prob * 200)
        marker = ' ← majority' if seats == 51 else (' ← tie' if seats == 50 else '')
        print(f"  {seats} seats: {prob:>6.1%}  {bar}{marker}")

    print(f"\nSim race probs (competitive):")
    comp_probs = [(a, p) for a, p in sim['sim_race_probs'].items()
                  if a in sim['comp_states']]
    for abbr, prob in sorted(comp_probs, key=lambda x: -x[1]):
        print(f"  {abbr}: {prob:.1%}")

    print()
    print(f"VERIFY: D majority prob 50-90%  → {'PASS' if 0.50 < sim['d_majority_prob'] < 0.90 else 'FAIL'} ({sim['d_majority_prob']:.1%})")
    print(f"VERIFY: Exp D seats 49-56       → {'PASS' if 49 < sim['exp_d_seats'] < 56 else 'FAIL'} ({sim['exp_d_seats']:.1f})")
    print(f"VERIFY: Low < central < high    → {'PASS' if sim['maj_lo'] < sim['d_majority_prob'] < sim['maj_hi'] else 'FAIL'}")
    print(f"VERIFY: 50-seat tie > 0%        → {'PASS' if sim['tie_50_prob'] > 0 else 'FAIL'} ({sim['tie_50_prob']:.1%})")
