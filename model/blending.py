"""
Dynamic blending weights for environment vs direct state polls.
Also computes time-decay sigma.
"""

import yaml
import math


def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


def compute_env_weight(state_abbr, n_polls, days_remaining, config):
    """
    env_weight = base × time_factor  (time-only — poll count does NOT affect weight)
    poll_weight = 1 - env_weight

    All races use the same env/poll split at the same point in time, regardless
    of how many polls have been conducted. The transition from environment-dominant
    to poll-dominant is purely a function of how close we are to election day.

    Returns (env_weight, poll_weight) both in [0, 1]
    """
    blend_cfg = config['race_model']['blending']

    # State override or global base
    state_overrides = blend_cfg.get('state_overrides', {})
    if state_abbr in state_overrides:
        base = state_overrides[state_abbr].get('base_env_weight',
               blend_cfg['base_env_weight'])
    else:
        base = blend_cfg['base_env_weight']

    time_decay_power = blend_cfg['time_decay_power']

    # Time decay: (days_remaining / 193)^power
    # Clamp days_remaining to [0, 365]
    days_remaining = max(0, min(days_remaining, 365))
    time_factor = (days_remaining / 193.0) ** time_decay_power

    env_weight  = base * time_factor
    env_weight  = max(0.0, min(1.0, env_weight))
    poll_weight = 1.0 - env_weight

    return env_weight, poll_weight


def compute_time_sigma(days_remaining, config):
    """
    sigma_time = base × sqrt(days_remaining / 193)
    Decays smoothly from base at day 193 to 0 at day 0.
    """
    base = config['race_model']['time_sigma_base']
    days_remaining = max(0, days_remaining)
    return base * math.sqrt(days_remaining / 193.0)


if __name__ == '__main__':
    cfg = load_config()
    print("BLENDING MODULE — UNIT TEST")
    print()
    print(f"{'Scenario':<40} {'EnvWt':>8} {'PollWt':>8}")
    print("─" * 58)
    scenarios = [
        ("Day 193, 0 polls (NH, baseline)",     'NH',  0,  193),
        ("Day 193, 10 polls (AK, override)",    'AK', 10,  193),
        ("Day 193, 10 polls (TX, override)",    'TX', 10,  193),
        ("Day 193, 10 polls (NC, baseline)",    'NC', 10,  193),
        ("Day 90,  10 polls (NC)",              'NC', 10,   90),
        ("Day 30,  15 polls (NC)",              'NC', 15,   30),
        ("Day 7,   17 polls (NC)",              'NC', 17,    7),
        ("Day 0,   17 polls (NC, election day)",'NC', 17,    0),
    ]
    for label, state, n, days in scenarios:
        ew, pw = compute_env_weight(state, n, days, cfg)
        print(f"  {label:<40} {ew:>8.3f} {pw:>8.3f}")

    print()
    print("Time sigma at different days:")
    for days in [193, 150, 90, 60, 30, 14, 7, 0]:
        sigma = compute_time_sigma(days, cfg)
        print(f"  Day {days:>3}: σ = {sigma:.3f}pp")

    print()
    # Verify
    ew_193_0, _ = compute_env_weight('NC', 0, 193, cfg)
    ew_0_17,  _ = compute_env_weight('NC', 17, 0, cfg)
    ts_193 = compute_time_sigma(193, cfg)
    ts_0   = compute_time_sigma(0, cfg)
    print(f"VERIFY: Env weight at day 193 / 0 polls ≈ 0.80 → {'PASS' if 0.75 < ew_193_0 < 0.85 else 'FAIL'} ({ew_193_0:.3f})")
    print(f"VERIFY: Env weight at day 0 / 17 polls ≈ 0.00 → {'PASS' if ew_0_17 < 0.05 else 'FAIL'} ({ew_0_17:.3f})")
    print(f"VERIFY: Time sigma at day 193 = 3.50pp        → {'PASS' if abs(ts_193 - 3.50) < 0.01 else 'FAIL'} ({ts_193:.3f})")
    print(f"VERIFY: Time sigma at day 0   = 0.00pp        → {'PASS' if ts_0 < 0.01 else 'FAIL'} ({ts_0:.3f})")
