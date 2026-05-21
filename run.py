"""
2026 Senate Model — Main Runner

Usage:
  python run.py                  # Full rebuild: model + Excel + charts
  python run.py --quick          # Console output only, no files
  python run.py --no-deck        # Skip PPTX
  python run.py --refresh        # Pull latest 538 + GCB data first
  python run.py --add-poll       # Interactive state poll entry
  python run.py --add-crosstab   # Interactive demographic crosstab entry
"""

import argparse
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yaml


def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


def compute_days_remaining(config):
    election_date = datetime.strptime(config['election_date'], '%Y-%m-%d').date()
    return (election_date - date.today()).days


def print_summary(oop, demo, state_envs, race_results, sim, days_remaining, config):
    from model.race_model import SENATE_RACES_2026

    line = '═' * 63
    print(f"\n{line}")
    print(f"  2026 SENATE MODEL  |  {date.today()}  |  {days_remaining} days to election")
    print(line)

    # Layer 1
    print(f"\nLAYER 1: NATIONAL OOP SHIFT")
    print(f"  GCB today ({oop['gcb_date']}):    D{oop['gcb_today']:+.2f}")
    print(f"  OOP shift (recency-wtd):  {oop['oop_shift']:+.2f}pp")
    print(f"  GCB election day:         D{oop['gcb_eday_central']:+.2f}  "
          f"(σ=±{oop['gcb_sigma']:.2f}pp)")
    print(f"  Range:                    D{oop['gcb_eday_lo']:+.2f} to D{oop['gcb_eday_hi']:+.2f}")
    print(f"  Cycle weights (top 3):    ", end='')
    top3 = sorted(oop['cycle_weights'], key=lambda x: -x[1])[:3]
    print('  '.join(f"{yr}: {wt:.1%}" for yr, wt, _ in top3))

    # Layer 2
    print(f"\nLAYER 2: DEMOGRAPHIC SHIFTS vs CATALIST 2024")
    baselines = config['catalist_2024']['group_d_share']
    for group, data in demo.items():
        if group == 'total':
            continue
        shift = data['shift']
        arrow = '↑' if shift > 1 else ('↓' if shift < -1 else '→')
        n = data['n_entries']
        print(f"  {group:<20} {arrow} {shift:>+5.1f}pp  "
              f"(current: {data['current_d_share']:.3f}  "
              f"baseline: {baselines[group]:.3f}  n={n})")

    # Layer 3 — competitive races
    print(f"\nLAYER 3: RACE RATINGS")
    comp_abbrs = [r[0] for r in SENATE_RACES_2026 if r[4]]
    comp = sorted(
        [race_results[a] for a in comp_abbrs if a in race_results],
        key=lambda x: -x['d_prob']
    )
    print(f"  {'St':<4} {'D Win%':>7} {'Central':>8} {'σ':>6} "
          f"{'Polls':>6}  {'Rating':<12}")
    print(f"  {'─'*55}")
    for r in comp:
        print(f"  {r['abbr']:<4} {r['d_prob']:>7.1%} {r['central']:>+8.2f} "
              f"{r['total_sigma']:>6.2f} {r['n_polls']:>6}  {r['rating']:<12}")

    # Simulation
    print(f"\nSIMULATION ({sim['n_sims']:,} runs)")
    print(f"  D majority (≥51 seats):  {sim['d_majority_prob']:.1%}")
    print(f"  Expected D seats:        {sim['exp_d_seats']:.1f}")
    print(f"  50-seat tie:             {sim['tie_50_prob']:.1%}")
    print(f"  Low scenario:            {sim['maj_lo']:.1%}")
    print(f"  High scenario:           {sim['maj_hi']:.1%}")

    # Top seat outcomes
    print(f"\n  Top seat outcomes:")
    top_seats = sorted(sim['seat_distribution'].items(),
                       key=lambda x: -x[1])[:6]
    for seats, prob in top_seats:
        marker = ' ← majority' if seats == 51 else (' ← tie→R' if seats == 50 else '')
        bar = '█' * int(prob * 150)
        print(f"    {seats} seats: {prob:>5.1%}  {bar}{marker}")

    print(f"\n{line}\n")


def run_model(config, days_remaining, ref_date=None):
    """Run all model layers and return results."""
    if ref_date is None:
        ref_date = date.today()

    from model.oop_shift import compute_oop_shift
    from model.demographic_shifts import compute_group_shifts
    from model.state_environment import compute_state_environments
    from model.poll_blend import compute_poll_blend
    from model.race_model import compute_race, SENATE_RACES_2026
    from model.monte_carlo import run_simulation

    print("  Computing OOP shift...")
    oop = compute_oop_shift(config)

    print("  Computing demographic shifts...")
    demo = compute_group_shifts(config, ref_date)

    print("  Computing state environments...")
    state_envs, implied_national = compute_state_environments(oop, demo, config)

    print("  Loading state polls...")
    polls_df = pd.read_csv('data/state_polls.csv')

    print("  Computing race probabilities...")
    race_results = {}
    for abbr, inc, dc, rc, competitive in SENATE_RACES_2026:
        if abbr not in state_envs:
            continue
        blend  = compute_poll_blend(abbr, polls_df, ref_date, config)
        result = compute_race(abbr, state_envs[abbr], blend, days_remaining, config)
        race_results[abbr] = result

    print(f"  Running Monte Carlo ({config['simulation']['n_sims']:,} sims)...")
    sim = run_simulation(race_results, oop, config)

    return oop, demo, state_envs, race_results, sim


def save_results(oop, demo, state_envs, race_results, sim, config, args):
    """Save datestamped outputs."""
    ds = date.today().strftime('%Y-%m-%d')
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)

    # Save JSON snapshot
    import json
    from model.race_model import SENATE_RACES_2026

    snapshot = {
        'run_date': str(date.today()),
        'model_version': config.get('model_version', 'v1'),
        'oop': oop,
        'simulation': {
            'd_majority_prob': sim['d_majority_prob'],
            'exp_d_seats': sim['exp_d_seats'],
            'tie_50_prob': sim['tie_50_prob'],
            'maj_lo': sim['maj_lo'],
            'maj_hi': sim['maj_hi'],
            'seat_distribution': sim['seat_distribution'],
        },
        'races': {
            abbr: {
                'd_prob': r['d_prob'],
                'central': r['central'],
                'sigma': r['total_sigma'],
                'rating': r['rating'],
                'n_polls': r['n_polls'],
            }
            for abbr, r in race_results.items()
        },
        'demographic_shifts': {
            g: {'shift': d['shift'], 'current': d['current_d_share'], 'n_entries': d['n_entries']}
            for g, d in demo.items()
        }
    }

    json_path = output_dir / f'model_snapshot_{ds}.json'
    with open(json_path, 'w') as f:
        json.dump(snapshot, f, indent=2)
    print(f"  Snapshot: {json_path}")

    # Excel
    try:
        from outputs.build_excel import build_excel
        excel_path = output_dir / f'senate_model_{ds}.xlsx'
        build_excel(oop, demo, state_envs, race_results, sim, config, str(excel_path))
        print(f"  Excel:    {excel_path}")
    except ImportError:
        print("  Excel:    skipped (build_excel.py not yet built)")
    except Exception as e:
        print(f"  Excel:    ERROR — {e}")

    # Charts
    try:
        from outputs.build_charts import build_charts
        charts_dir = output_dir / f'charts_{ds}'
        charts_dir.mkdir(exist_ok=True)
        build_charts(state_envs, race_results, sim, oop, str(charts_dir))
        print(f"  Charts:   {charts_dir}/")
    except ImportError:
        print("  Charts:   skipped (build_charts.py not yet built)")
    except Exception as e:
        print(f"  Charts:   ERROR — {e}")


def main():
    parser = argparse.ArgumentParser(description='2026 Senate Model')
    parser.add_argument('--quick',        action='store_true',
                        help='Console output only — no files generated')
    parser.add_argument('--no-deck',      action='store_true',
                        help='Skip PPTX generation')
    parser.add_argument('--refresh',      action='store_true',
                        help='Pull latest 538 and GCB data before running')
    parser.add_argument('--add-poll',     action='store_true',
                        help='Interactive state poll entry')
    parser.add_argument('--add-crosstab', action='store_true',
                        help='Interactive demographic crosstab entry')
    args = parser.parse_args()

    # Load config
    config = load_config()
    days_remaining = compute_days_remaining(config)
    ref_date = date.today()

    # Data entry modes — run and exit
    if args.add_poll:
        from data.entry import add_state_poll
        add_state_poll(config)
        return

    if args.add_crosstab:
        from data.entry import add_crosstab
        add_crosstab(config)
        return

    # Refresh data sources
    if args.refresh:
        print("Refreshing data sources...")
        result = subprocess.run(
            [sys.executable, 'data/refresh.py'],
            capture_output=False
        )
        if result.returncode != 0:
            print("WARNING: refresh.py returned an error — continuing with existing data")
        print()

    # Run model
    print(f"Running model ({days_remaining} days to election)...")
    oop, demo, state_envs, race_results, sim = run_model(config, days_remaining, ref_date)

    # Print summary
    print_summary(oop, demo, state_envs, race_results, sim, days_remaining, config)

    # Save outputs
    if not args.quick:
        print("Saving outputs...")
        save_results(oop, demo, state_envs, race_results, sim, config, args)
        print("Done.\n")
    else:
        print("(--quick mode: no files saved)\n")


if __name__ == '__main__':
    main()
