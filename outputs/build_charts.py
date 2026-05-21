"""
Generates all charts and maps from model results.
Called by run.py or directly.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import matplotlib.colors as mc
from pathlib import Path


# ── TILE MAP GRID ──────────────────────────────────────────────────────────────
GRID = {
    'ME':(11,0),'VT':(10,0),'NH':(11,1),
    'WA':(1,0),'MT':(3,0),'ND':(4,0),'MN':(5,0),'WI':(6,0),'MI':(7,1),
    'NY':(9,1),'MA':(10,1),
    'OR':(1,1),'ID':(2,1),'WY':(3,1),'SD':(4,1),'IA':(5,1),'IL':(6,1),
    'IN':(7,2),'OH':(8,1),'PA':(9,0),'CT':(11,2),'RI':(11,3),
    'CA':(1,2),'NV':(2,2),'UT':(2,3),'CO':(3,2),'NE':(4,2),'MO':(5,2),
    'KY':(7,3),'WV':(8,2),'VA':(9,2),'MD':(9,3),'DE':(10,3),'NJ':(10,2),
    'AZ':(2,4),'NM':(2,5),'KS':(4,3),'OK':(4,4),'AR':(5,4),'TN':(6,3),
    'NC':(8,3),'SC':(8,4),
    'TX':(4,5),'LA':(5,5),'MS':(6,4),'AL':(7,4),'GA':(7,5),'FL':(8,5),
    'AK':(0,7),'HI':(1,7),
}

RATING_COLS = {
    'Safe D':   '#1B4F8A',
    'Likely D': '#4A90D9',
    'Lean D':   '#7EC8E3',
    'Tossup':   '#7C3AED',
    'Lean R':   '#F4A261',
    'Likely R': '#D94A4A',
    'Safe R':   '#8B1A1A',
}

BG = '#F8F9FC'


def rdbu_cmap():
    return LinearSegmentedColormap.from_list('rdbu', [
        (0.50,0.08,0.08),(0.78,0.22,0.22),(0.95,0.60,0.60),
        (0.97,0.94,0.94),(0.94,0.97,0.97),
        (0.58,0.74,0.90),(0.24,0.50,0.78),(0.08,0.28,0.54)
    ])


def gcb_color(v, vmin=-45, vmax=45):
    cmap = rdbu_cmap()
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    return cmap(norm(np.clip(v, vmin, vmax)))


def setup_ax(ax):
    ax.set_xlim(-0.2, 12.2)
    ax.set_ylim(-0.3, 8.2)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.axis('off')


def _luminance(color_rgba):
    """Relative luminance of an RGBA tuple (0–1 each)."""
    r, g, b = color_rgba[:3]
    def lin(c):
        return c/12.92 if c <= 0.04045 else ((c+0.055)/1.055)**2.4
    return 0.2126*lin(r) + 0.7152*lin(g) + 0.0722*lin(b)


def _text_color(bg_rgba):
    """Return white or near-black for max contrast against bg."""
    return 'white' if _luminance(bg_rgba) < 0.35 else '#1A1A1A'


def draw_tile(ax, state, color, label_color, abbr, val_str='',
              border_color='white', border_w=0.8):
    if state not in GRID:
        return
    ci, ri = GRID[state]
    rect = mpatches.FancyBboxPatch(
        (ci+0.04, ri+0.04), 0.90, 0.90,
        boxstyle='round,pad=0.05',
        facecolor=color, edgecolor=border_color,
        linewidth=border_w, zorder=2
    )
    ax.add_patch(rect)
    y_abbr = ri + (0.40 if val_str else 0.49)
    ax.text(ci+0.49, y_abbr, abbr,
            ha='center', va='center', fontsize=9.5,
            fontweight='bold', color=label_color, zorder=3)
    if val_str:
        ax.text(ci+0.49, ri+0.68, val_str,
                ha='center', va='center',
                fontsize=7.5, fontweight='bold', color=label_color, zorder=3)


# ── CHART 1: SENATE RATINGS MAP ────────────────────────────────────────────────
def chart_senate_ratings(race_results, sim, oop_result, out_dir):
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    setup_ax(ax)

    senate_states = set(race_results.keys())

    for state in GRID:
        if state in race_results:
            r = race_results[state]
            hex_col = RATING_COLS.get(r['rating'], '#D4D4D4')
            color = hex_col
            lc = _text_color(mc.to_rgba(hex_col))
            prob = r['d_prob']
            val = f"{prob*100:.0f}%" if 0.02 < prob < 0.98 else ''
            draw_tile(ax, state, color, lc, state, val, 'white', 0.7)
        else:
            draw_tile(ax, state, '#D4D4D4', '#666666', state, '', 'white', 0.5)

    # Legend
    leg_items = [
        ('Safe D (≥95%)',    '#1B4F8A'),
        ('Likely D (80-95%)', '#4A90D9'),
        ('Lean D (60-80%)',   '#7EC8E3'),
        ('Tossup (40-60%)',   '#7C3AED'),
        ('Lean R (20-40%)',   '#F4A261'),
        ('Likely R (8-20%)',  '#D94A4A'),
        ('Safe R (<8%)',      '#8B1A1A'),
        ('Not up 2026',       '#D4D4D4'),
    ]
    lx = np.linspace(0.03, 0.97, len(leg_items)+1)[:-1]
    for (label, c), x in zip(leg_items, lx):
        rect = mpatches.Rectangle(
            (x, 0.055), 0.095, 0.03,
            transform=fig.transFigure,
            facecolor=c, edgecolor='white', linewidth=0.5
        )
        fig.add_artist(rect)
        fig.text(x+0.048, 0.092, label,
                 ha='center', fontsize=6.5, color='#334155')

    fig.text(0.5, 0.97,
             f"2026 Senate Ratings — D Win Probability  |  "
             f"D majority: {sim['d_majority_prob']:.1%}  |  "
             f"Expected D seats: {sim['exp_d_seats']:.1f}",
             ha='center', fontsize=12, fontweight='bold', color='#0D1B3E')
    fig.text(0.5, 0.93,
             f"Model v1 — bottom-up MRP  |  GCB eday: D{oop_result['gcb_eday_central']:+.2f}  |  "
             f"σ=±{oop_result['gcb_sigma']:.2f}pp",
             ha='center', fontsize=8.5, color='#64748B', style='italic')
    fig.tight_layout(rect=[0, 0.11, 1, 0.92])

    path = Path(out_dir) / 'map_senate_ratings.png'
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()
    return str(path)


# ── CHART 2: SEAT DISTRIBUTION ─────────────────────────────────────────────────
def chart_seat_distribution(sim, oop_result, out_dir):
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    seat_dist = sim['seat_distribution']
    seats = sorted(seat_dist.keys())
    probs = [seat_dist[s] * 100 for s in seats]

    colors = []
    for s in seats:
        if s >= 52:   colors.append('#1B4F8A')
        elif s == 51: colors.append('#2E8B57')
        elif s == 50: colors.append('#DAA520')
        else:         colors.append('#8B1A1A')

    ax.bar(seats, probs, color=colors, edgecolor='white', linewidth=0.8, zorder=3)

    for s, p in zip(seats, probs):
        if p > 1.5:
            ax.text(s, p+0.3, f'{p:.1f}%',
                    ha='center', va='bottom', fontsize=8,
                    color='#0D1B3E',
                    fontweight='bold' if s in (50, 51) else 'normal')

    ax.axvline(50.5, color='#334155', linewidth=2.0, linestyle='--', zorder=5)
    ax.text(50.5, max(probs)*0.88, '  MAJORITY\n  THRESHOLD',
            fontsize=8.5, color='#334155', fontweight='bold', va='top')

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor='#1B4F8A', label='52+ seats (comfortable D majority)'),
        Patch(facecolor='#2E8B57', label='51 seats (bare D majority)'),
        Patch(facecolor='#DAA520', label='50 seats (tie → R VP breaks tie)'),
        Patch(facecolor='#8B1A1A', label='≤49 seats (R majority)'),
    ], fontsize=8, loc='upper left',
       framealpha=0.9, edgecolor='#C8D9EE', facecolor=BG)

    ax.text(0.98, 0.97,
            f"D majority (≥51): {sim['d_majority_prob']:.1%}\n"
            f"Expected D seats: {sim['exp_d_seats']:.1f}\n"
            f"50-seat tie: {sim['tie_50_prob']:.1%}",
            transform=ax.transAxes, fontsize=9, color='#0D1B3E',
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#EBF3FB',
                      edgecolor='#4A90D9', linewidth=1.5))

    ax.set_xlabel('Total Democratic Senate Seats', fontsize=10, color='#64748B')
    ax.set_ylabel('Probability (%)', fontsize=10, color='#64748B')
    ax.set_xticks(seats)
    ax.tick_params(labelsize=9, colors='#64748B')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.text(0.5, 0.97, 'Monte Carlo Seat Distribution — 100,000 Simulations',
             ha='center', fontsize=12, fontweight='bold', color='#0D1B3E')
    fig.text(0.5, 0.93,
             f'Correlated 12-state simulation  |  '
             f'GCB eday D{oop_result["gcb_eday_central"]:+.2f}  |  '
             f'σ_time decays to 0 by election day',
             ha='center', fontsize=8, color='#64748B', style='italic')
    plt.tight_layout(rect=[0, 0.02, 1, 0.92])

    path = Path(out_dir) / 'chart_seat_distribution.png'
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()
    return str(path)


# ── CHART 3: DEMOGRAPHIC SHIFTS ────────────────────────────────────────────────
def chart_demo_shifts(demo_shifts, catalist_baselines, out_dir):
    groups = [g for g in demo_shifts if g != 'total']
    labels = {
        'white_nc': 'White NC',
        'white_college': 'White College',
        'hispanic': 'Hispanic',
        'black': 'Black',
        'aapi': 'AAPI',
        'youth_18_29': '18–29',
        'seniors_65p': '65+',
    }

    shifts = [demo_shifts[g]['shift'] for g in groups]
    group_labels = [labels.get(g, g) for g in groups]
    y_pos = range(len(groups))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    colors = ['#1B4F8A' if v >= 0 else '#D94A4A' for v in shifts]
    ax.barh(list(y_pos), shifts, color=colors, height=0.6,
            edgecolor='white', linewidth=0.5)
    ax.axvline(0, color='#334155', linewidth=1.0)

    for i, (v, label) in enumerate(zip(shifts, group_labels)):
        if v >= 0:
            ax.text(v + 0.15, i, f'{v:+.1f}pp',
                    va='center', ha='left', fontsize=9, color='#334155')
        else:
            # Place label inside the bar (from left edge rightward) so it
            # never collides with the y-axis tick labels
            ax.text(v + 0.15, i, f'{v:+.1f}pp',
                    va='center', ha='left', fontsize=9, color='white',
                    fontweight='bold')

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(group_labels, fontsize=10.5, color='#0D1B3E')
    ax.set_xlabel('Shift vs Catalist 2024 Baseline (pp)', fontsize=9, color='#64748B')
    ax.tick_params(axis='x', labelsize=8, colors='#64748B')
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)

    fig.text(0.5, 0.97,
             'Demographic GCB Shifts vs Catalist 2024 Baseline',
             ha='center', fontsize=12, fontweight='bold', color='#0D1B3E')
    fig.text(0.5, 0.93,
             'Recency × grade weighted average of manually entered national poll crosstabs',
             ha='center', fontsize=8, color='#64748B', style='italic')
    plt.tight_layout(rect=[0, 0.04, 1, 0.92])

    path = Path(out_dir) / 'chart_demo_shifts.png'
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()
    return str(path)


# ── CHART 4: STATE ENVIRONMENT MAP ─────────────────────────────────────────────
def chart_state_environment(state_envs, race_results, oop_result, out_dir):
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    setup_ax(ax)

    senate_states = set(race_results.keys())
    cmap = rdbu_cmap()

    for state in GRID:
        env = state_envs.get(state, {})
        gcb = env.get('gcb_eday')
        if gcb is None:
            draw_tile(ax, state, '#D4D4D4', '#999999', state, '')
            continue
        color = gcb_color(gcb, -45, 45)
        lc = _text_color(mc.to_rgba(color))
        bc = '#FFD700' if state in senate_states else 'white'
        bw = 2.2 if state in senate_states else 0.7
        draw_tile(ax, state, color, lc, state, f'{gcb:+.0f}', bc, bw)

    # Colorbar
    ax_cb = fig.add_axes([0.12, 0.04, 0.76, 0.025])
    norm = TwoSlopeNorm(vmin=-45, vcenter=0, vmax=45)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cb, orientation='horizontal')
    cb.set_ticks([-45, -22, 0, 22, 45])
    cb.set_ticklabels(['R+45', 'R+22', '0', 'D+22', 'D+45'])
    cb.ax.tick_params(labelsize=7, colors='#334155', length=3)
    cb.ax.set_xlabel('Projected GCB on Election Day',
                     fontsize=7.5, color='#0D1B3E', fontweight='bold', labelpad=4)
    cb.outline.set_linewidth(0.5)

    fig.text(0.5, 0.97,
             f'State GCB Election Day Projection — D{oop_result["gcb_eday_central"]:+.2f} National',
             ha='center', fontsize=12, fontweight='bold', color='#0D1B3E')
    fig.text(0.5, 0.93,
             f'Bottom-up MRP: Catalist 2024 + demographic shifts + OOP {oop_result["oop_shift"]:+.2f}pp  |  Gold outline = 2026 Senate race',
             ha='center', fontsize=8.5, color='#64748B', style='italic')
    fig.tight_layout(rect=[0, 0.10, 1, 0.92])

    path = Path(out_dir) / 'map_state_environment.png'
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()
    return str(path)


# ── MAIN ───────────────────────────────────────────────────────────────────────
def build_charts(state_envs, race_results, sim, oop_result, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = []

    import yaml
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    baselines = config['catalist_2024']['group_d_share']

    from model.demographic_shifts import compute_group_shifts
    from datetime import date
    demo = compute_group_shifts(config, date.today())

    print('  Generating charts...')
    paths.append(chart_senate_ratings(race_results, sim, oop_result, out_dir))
    paths.append(chart_seat_distribution(sim, oop_result, out_dir))
    paths.append(chart_demo_shifts(demo, baselines, out_dir))
    paths.append(chart_state_environment(state_envs, race_results, oop_result, out_dir))

    print(f'  {len(paths)} charts saved to {out_dir}/')
    return paths


if __name__ == '__main__':
    from datetime import date
    import pandas as pd
    from model.oop_shift import compute_oop_shift
    from model.demographic_shifts import compute_group_shifts
    from model.state_environment import compute_state_environments
    from model.poll_blend import compute_poll_blend
    from model.race_model import compute_race, SENATE_RACES_2026
    from model.monte_carlo import run_simulation
    import yaml

    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)

    oop  = compute_oop_shift(cfg)
    demo = compute_group_shifts(cfg)
    envs, _ = compute_state_environments(oop, demo, cfg)
    polls_df = pd.read_csv('data/state_polls.csv')
    ref = date.today()
    days = (date(2026, 11, 3) - ref).days

    race_results = {}
    for abbr, inc, dc, rc, comp in SENATE_RACES_2026:
        if abbr not in envs: continue
        blend = compute_poll_blend(abbr, polls_df, ref, cfg)
        race_results[abbr] = compute_race(abbr, envs[abbr], blend, days, cfg)

    sim = run_simulation(race_results, oop, cfg)
    build_charts(envs, race_results, sim, oop, 'outputs/charts_test')
    print('Done — check outputs/charts_test/')
