"""
Computes the recency-weighted OOP shift and GCB election day projection.

Key outputs:
  oop_shift:        weighted mean shift (pp) toward OOP party
  gcb_sigma:        combined uncertainty (pp)
  gcb_eday_central: GCB today + oop_shift
  gcb_eday_lo/hi:   central ± gcb_sigma

Verify: with current data should produce oop_shift ≈ +1.60pp, gcb_sigma ≈ ±2.15pp
"""

import numpy as np
import pandas as pd
import yaml
from datetime import date


def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


def compute_oop_shift(config, gcb_csv='data/gcb_national.csv'):
    """
    Steps:
    1. Load OOP cycles from config
    2. Compute exponential decay weights (half-life from config)
    3. Weighted mean shift + weighted std dev
    4. Combined sigma = sqrt(weighted_std^2 + tracker_uncertainty^2)
    5. Load latest GCB from gcb_national.csv
    6. Project election day GCB
    """
    oop_cfg = config['oop']
    half_life = oop_cfg['half_life_years']
    tracker_unc = oop_cfg['gcb_tracker_uncertainty']
    cycles = oop_cfg['cycles']  # list of [year, shift]

    years  = np.array([c[0] for c in cycles], dtype=float)
    shifts = np.array([c[1] for c in cycles], dtype=float)

    # Recency weights: exponential decay from 2026
    raw_weights = np.exp(-np.log(2) * (2026 - years) / half_life)
    weights = raw_weights / raw_weights.sum()

    oop_shift    = float(np.sum(weights * shifts))
    weighted_var = float(np.sum(weights * (shifts - oop_shift) ** 2))
    oop_std      = float(np.sqrt(weighted_var))
    gcb_sigma    = float(np.sqrt(oop_std ** 2 + tracker_unc ** 2))

    # Load latest GCB reading
    gcb_df = pd.read_csv(gcb_csv, parse_dates=['date'])
    gcb_df = gcb_df.sort_values('date')
    latest = gcb_df.iloc[-1]
    gcb_today = float(latest['gcb_d_net'])
    gcb_date  = str(latest['date'].date())

    gcb_eday_central = round(gcb_today + oop_shift, 3)
    gcb_eday_lo      = round(gcb_eday_central - gcb_sigma, 3)
    gcb_eday_hi      = round(gcb_eday_central + gcb_sigma, 3)

    return {
        'gcb_today':        gcb_today,
        'gcb_date':         gcb_date,
        'oop_shift':        round(oop_shift, 3),
        'oop_std':          round(oop_std, 3),
        'gcb_sigma':        round(gcb_sigma, 3),
        'gcb_eday_central': gcb_eday_central,
        'gcb_eday_lo':      gcb_eday_lo,
        'gcb_eday_hi':      gcb_eday_hi,
        'cycle_weights':    [(int(y), round(w, 4), round(s, 2))
                             for y, w, s in zip(years, weights, shifts)],
    }


if __name__ == '__main__':
    cfg = load_config()
    result = compute_oop_shift(cfg)
    print("OOP SHIFT MODULE — UNIT TEST")
    print(f"  GCB today:          D{result['gcb_today']:+.2f} ({result['gcb_date']})")
    print(f"  OOP shift:          {result['oop_shift']:+.2f}pp")
    print(f"  OOP std:            ±{result['oop_std']:.2f}pp")
    print(f"  GCB sigma:          ±{result['gcb_sigma']:.2f}pp")
    print(f"  GCB eday central:   D{result['gcb_eday_central']:+.2f}")
    print(f"  GCB eday range:     D{result['gcb_eday_lo']:+.2f} to D{result['gcb_eday_hi']:+.2f}")
    print(f"\n  Cycle weights:")
    for yr, wt, sh in result['cycle_weights']:
        print(f"    {yr}: weight={wt:.4f}  shift={sh:+.1f}pp")
    print(f"\n  VERIFY: oop_shift should be ≈ +1.60pp  → {'PASS' if abs(result['oop_shift'] - 1.60) < 0.05 else 'FAIL'}")
    print(f"  VERIFY: gcb_sigma should be ≈ ±2.15pp  → {'PASS' if abs(result['gcb_sigma'] - 2.15) < 0.05 else 'FAIL'}")
