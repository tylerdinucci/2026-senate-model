"""
2022 Senate Backtest Runner

Usage:
  python backtest/run_backtest.py           # runs 2022 backtest
  python backtest/run_backtest.py --year 2022
"""

import argparse
import sys
import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.state_environment import compute_state_environments
from model.poll_blend import compute_poll_blend
from model.race_model import compute_race
from model.monte_carlo import run_simulation


def load_backtest_config(year):
    path = Path(__file__).parent / str(year) / 'config.yaml'
    with open(path) as f:
        return yaml.safe_load(f)


def load_actual_results(year):
    path = Path(__file__).parent / str(year) / 'actual_results.yaml'
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get('results', {})


def build_oop_result(config):
    """Construct fixed OOP result from backtest config — bypasses compute_oop_shift."""
    bt = config['backtest']
    sigma = float(bt.get('gcb_sigma', 1.5))
    central = float(bt['gcb_eday_central'])
    return {
        'gcb_today':        float(bt['gcb_today']),
        'gcb_date':         bt['ref_date'],
        'oop_shift':        0.0,
        'oop_std':          0.0,
        'gcb_sigma':        sigma,
        'gcb_eday_central': central,
        'gcb_eday_lo':      round(central - sigma, 3),
        'gcb_eday_hi':      round(central + sigma, 3),
        'cycle_weights':    [],
    }


def build_zero_demo_shifts(config):
    """Zero demographic shifts — bypasses compute_group_shifts."""
    baselines = config['catalist_2024']['group_d_share']
    return {
        g: {'current_d_share': v, 'shift': 0.0, 'n_entries': 0, 'last_updated': None}
        for g, v in baselines.items()
    }


def load_races(year):
    """Import SENATE_RACES_{YEAR} from backtest/{year}/races.py."""
    import importlib.util
    races_path = Path(__file__).parent / str(year) / 'races.py'
    spec = importlib.util.spec_from_file_location(f'races_{year}', races_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, f'SENATE_RACES_{year}')


def run_backtest(year=2022):
    base_dir = Path(__file__).parent / str(year)
    config   = load_backtest_config(year)
    actuals  = load_actual_results(year)

    bt_cfg   = config['backtest']
    ref_date = datetime.strptime(bt_cfg['ref_date'], '%Y-%m-%d').date()
    eday     = datetime.strptime(config['election_date'], '%Y-%m-%d').date()
    days_remaining = (eday - ref_date).days

    catalist_csv = str(base_dir / 'data' / 'catalist_2020.csv')
    acs_csv      = str(Path(__file__).parent.parent / 'data' / 'acs_composition.csv')
    polls_csv    = str(base_dir / 'data' / 'state_polls.csv')

    print(f"\n{'═'*65}")
    print(f"  2022 SENATE BACKTEST  |  Model date: {ref_date}  |  {days_remaining}d out")
    print(f"{'═'*65}")

    print(f"\n  GCB election-eve:  D{bt_cfg['gcb_today']:+.1f}")
    print(f"  National shift:    {bt_cfg['gcb_eday_central'] - config['catalist_2024']['national_gcb']:+.1f}pp from 2020 baseline")

    oop       = build_oop_result(config)
    demo      = build_zero_demo_shifts(config)
    state_envs, implied = compute_state_environments(oop, demo, config,
                                                      catalist_csv=catalist_csv,
                                                      acs_csv=acs_csv)

    polls_df  = pd.read_csv(polls_csv)
    races     = load_races(year)

    race_results = {}
    for abbr, inc, dc, rc, competitive in races:
        if abbr not in state_envs:
            continue
        blend  = compute_poll_blend(abbr, polls_df, ref_date, config)
        result = compute_race(abbr, state_envs[abbr], blend, days_remaining, config)
        race_results[abbr] = result

    sim = run_simulation(race_results, oop, config)

    # Print ALL races sorted by D probability
    all_results = sorted(race_results.values(), key=lambda x: -x['d_prob'])
    comp_abbrs  = {r[0] for r in races if r[4]}

    print(f"\n{'─'*70}")
    print(f"  {'St':<4} {'Rating':<12} {'D Win%':>7} {'Model':>8}  {'Actual':>8}  {'Error':>7}  {'Poll?'}")
    print(f"  {'─'*65}")
    n_correct = 0
    n_total   = 0
    for r in all_results:
        abbr   = r['abbr']
        actual = actuals.get(abbr)
        actual_str = f"D{actual:+.1f}" if actual is not None else "    —"
        polled = '●' if r['n_polls'] > 0 else ''

        if actual is not None:
            error     = actual - r['central']
            error_str = f"{error:+.1f}pp"
            predicted_d = r['d_prob'] > 0.5
            actual_d    = actual > 0
            correct     = (predicted_d == actual_d)
            if abbr in comp_abbrs:
                n_total  += 1
                n_correct += int(correct)
            miss = '' if correct else '  ✗'
        else:
            error_str = '    —'
            miss      = ''

        print(f"  {abbr:<4} {r['rating']:<12} {r['d_prob']:>7.1%} {r['central']:>+8.2f}"
              f"  {actual_str:>8}  {error_str:>7}  {polled}{miss}")

    print(f"\n  Competitive direction: {n_correct}/{n_total} correct")

    print(f"\n  Simulation ({sim['n_sims']:,} runs)")
    print(f"  D majority (≥51):  {sim['d_majority_prob']:.1%}")
    print(f"  Expected D seats:  {sim['exp_d_seats']:.1f}")
    print(f"  (Actual 2022:      48D + 3I = 51 D caucus)")
    print()

    top = sorted(sim['seat_distribution'].items(), key=lambda x: -x[1])[:6]
    for seats, prob in top:
        marker = ' ← majority' if seats == 51 else (' ← tie' if seats == 50 else '')
        bar = '█' * int(prob * 120)
        print(f"    {seats} seats: {prob:>5.1%}  {bar}{marker}")

    print(f"\n{'═'*65}\n")
    return sim, race_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Senate backtest runner')
    parser.add_argument('--year', type=int, default=2022, help='Backtest year')
    args = parser.parse_args()
    run_backtest(args.year)
