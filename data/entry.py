"""
Interactive CLI for manual data entry.
Called via: python run.py --add-poll
            python run.py --add-crosstab
"""

import pandas as pd
import uuid
from datetime import datetime, date
import yaml

def load_config():
    with open('config.yaml') as f:
        return yaml.safe_load(f)

# ── ADD STATE POLL ─────────────────────────────────────────────────────────────
def add_state_poll(config):
    print("\n── ADD STATE POLL ──────────────────────────────────────────────────")

    abbr        = input("STATE ABBREVIATION (e.g. TX): ").strip().upper()
    state_name  = input("STATE NAME (e.g. Texas): ").strip()
    d_candidate = input("D CANDIDATE NAME: ").strip()
    r_candidate = input("R CANDIDATE NAME: ").strip()
    d_pct       = float(input("D PCT (e.g. 47.0): ").strip())
    r_pct       = float(input("R PCT (e.g. 44.0): ").strip())
    undi_raw    = input("UNDECIDED PCT (Enter to skip): ").strip()
    undi_pct    = float(undi_raw) if undi_raw else None
    n           = int(input("SAMPLE SIZE: ").strip())
    pollster    = input("POLLSTER NAME: ").strip()
    grade       = float(input("SILVER GRADE (0.0-3.0, e.g. 2.8 for A-): ").strip())
    partisan    = input("PARTISAN POLL? (y/n): ").strip().lower() == 'y'
    date_start  = input("DATE START (YYYY-MM-DD): ").strip()
    date_end    = input("DATE END (YYYY-MM-DD): ").strip()
    methodology = input("METHODOLOGY (LV/RV/A, Enter to skip): ").strip()
    note        = input("NOTE (H2H_SULLIVAN for AK, or Enter to skip): ").strip()

    poll_id = f"MANUAL_{abbr}_{date_end.replace('-','')}_{str(uuid.uuid4())[:6].upper()}"

    new_row = {
        'poll_id': poll_id, 'state': state_name, 'abbr': abbr,
        'd_candidate': d_candidate, 'r_candidate': r_candidate,
        'd_pct': d_pct, 'r_pct': r_pct, 'undi_pct': undi_pct,
        'n': n, 'pollster': pollster, 'grade': grade,
        'partisan': partisan, 'date_start': date_start,
        'date_end': date_end, 'methodology': methodology, 'note': note
    }

    # Preview
    print(f"\nAdding: {d_candidate} vs {r_candidate} in {state_name}")
    print(f"  Result: D{d_pct-r_pct:+.1f}  (D={d_pct}% R={r_pct}%)  n={n}  grade={grade}")
    print(f"  Pollster: {pollster}  |  Dates: {date_start} to {date_end}")
    if note: print(f"  Note: {note}")

    confirm = input("\nAdd this poll? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return

    df = pd.read_csv('data/state_polls.csv')
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv('data/state_polls.csv', index=False)

    # Show updated blend for this state
    from model.poll_blend import compute_poll_blend
    blend = compute_poll_blend(abbr, df, date.today(), config)
    print(f"\nPoll added. {abbr} updated blend: D{blend['blend_margin']:+.1f} ({blend['n_polls']} polls)")

    run = input("Run full model now? (y/n): ").strip().lower()
    if run == 'y':
        import subprocess
        subprocess.run(['python', 'run.py', '--quick'])

# ── ADD DEMOGRAPHIC CROSSTAB ───────────────────────────────────────────────────
def add_crosstab(config):
    print("\n── ADD DEMOGRAPHIC CROSSTAB ────────────────────────────────────────")
    print("Enter national poll demographic data.")
    print("Values are D vote share as decimal (e.g. 0.441 = 44.1%)")
    print("Press Enter to skip a group.\n")

    pollster   = input("POLLSTER NAME: ").strip()
    grade      = float(input("SILVER GRADE (0.0-3.0): ").strip())
    date_start = input("DATE START (YYYY-MM-DD): ").strip()
    date_end   = input("DATE END (YYYY-MM-DD): ").strip()
    n          = int(input("TOTAL SAMPLE SIZE: ").strip())
    note       = input("NOTE (e.g. EXCLUDE_HISPANIC, or Enter to skip): ").strip()

    baselines = config['catalist_2024']['group_d_share']
    groups = [
        ('white_nc',      'White Non-College'),
        ('white_college', 'White College'),
        ('hispanic',      'Hispanic'),
        ('black',         'Black'),
        ('aapi',          'AAPI'),
        ('youth_18_29',   '18-29'),
        ('seniors_65p',   '65+'),
        ('total',         'Total'),
    ]

    new_rows = []
    entry_id_base = str(uuid.uuid4())[:8].upper()

    print()
    for i, (group_key, group_label) in enumerate(groups):
        baseline = baselines[group_key]
        raw = input(f"  {group_label:<20} [Catalist 2024: {baseline:.3f}]: ").strip()
        if not raw:
            continue

        value = float(raw)
        shift = value - baseline
        print(f"    shift: {shift:+.1f}pp")

        # Check for exclusion flag
        entry_note = note
        if group_key == 'hispanic' and 'EXCLUDE_HISPANIC' in note.upper():
            entry_note = 'EXCLUDE_HISPANIC'

        new_rows.append({
            'entry_id': f"{entry_id_base}_{i:02d}",
            'pollster': pollster,
            'grade': grade,
            'date_start': date_start,
            'date_end': date_end,
            'n': n,
            'group': group_key,
            'metric': 'gcb_d_share',
            'value': value,
            'note': entry_note
        })

    if not new_rows:
        print("No data entered.")
        return

    print(f"\n{len(new_rows)} group readings to add.")
    confirm = input("Add these? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return

    df = pd.read_csv('data/national_crosstabs.csv')
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv('data/national_crosstabs.csv', index=False)

    print(f"\n{len(new_rows)} crosstab entries added.")

    # Show updated demographic shifts
    from model.demographic_shifts import compute_group_shifts
    import yaml
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)
    shifts = compute_group_shifts(cfg, date.today())
    print("\nUpdated demographic shifts vs Catalist 2024:")
    for g, data in shifts.items():
        print(f"  {g:<20} {data['shift']:+.1f}pp  (current: {data['current_d_share']:.3f}, n_entries: {data['n_entries']})")

    run = input("\nRun full model now? (y/n): ").strip().lower()
    if run == 'y':
        import subprocess
        subprocess.run(['python', 'run.py', '--quick'])
