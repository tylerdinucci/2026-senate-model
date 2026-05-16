"""
2026 Senate Model Dashboard
Run with: streamlit run outputs/dashboard.py
"""

import streamlit as st
import pandas as pd
import json
import yaml
import glob
from pathlib import Path
from datetime import date
import subprocess
import sys

st.set_page_config(
    page_title="2026 Senate Model",
    page_icon="🗳️",
    layout="wide"
)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_latest_snapshot():
    snapshots = sorted(glob.glob('outputs/model_snapshot_*.json'), reverse=True)
    if not snapshots:
        return None, None
    with open(snapshots[0]) as f:
        return json.load(f), snapshots[0]

@st.cache_data
def load_config():
    with open('config.yaml') as f:
        return yaml.safe_load(f)

snap, snap_path = load_latest_snapshot()
if snap is None:
    st.error("No model snapshot found. Run `python run.py` first.")
    st.stop()

cfg = load_config()
sim = snap['simulation']
oop = snap['oop']
races = snap.get('races', {})
demo  = snap.get('demographic_shifts', {})
baselines = cfg['catalist_2024']['group_d_share']

run_date = snap.get('run_date', 'Unknown')
days_remaining = (date(2026, 11, 3) - date.today()).days

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🗳️ 2026 Senate Model")
    st.caption(f"Last run: {run_date}")
    st.caption(f"Days to election: **{days_remaining}**")
    st.divider()

    if st.button("🔄 Refresh & Rerun", use_container_width=True, type="primary"):
        with st.spinner("Pulling latest data and rerunning model..."):
            subprocess.run([sys.executable, 'data/refresh.py'])
            subprocess.run([sys.executable, 'run.py', '--quick'])
            st.cache_data.clear()
        st.success("Done!")
        st.rerun()

    st.divider()
    st.caption("**Add data manually:**")
    st.code("python run.py --add-poll", language="bash")
    st.code("python run.py --add-crosstab", language="bash")
    st.divider()
    st.caption(f"Snapshot: `{Path(snap_path).name}`")
    st.caption(f"σ_time base: {cfg['race_model']['time_sigma_base']}pp")
    st.caption(f"OOP half-life: {cfg['oop']['half_life_years']}yr")

# ── HEADLINE KPIs ─────────────────────────────────────────────────────────────
st.markdown("## 2026 Senate Forecast")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("D Majority Probability",
          f"{sim['d_majority_prob']:.1%}",
          help="P(Democrats win ≥51 seats) across 100k simulations")
k2.metric("Expected D Seats",
          f"{sim['exp_d_seats']:.1f}")
k3.metric("GCB Today",
          f"D{oop['gcb_today']:+.2f}",
          help=f"As of {oop['gcb_date']}")
k4.metric("GCB Election Day",
          f"D{oop['gcb_eday_central']:+.2f}",
          help=f"±{oop['gcb_sigma']:.2f}pp | OOP: +{oop['oop_shift']:.2f}pp")
k5.metric("50-Seat Tie Risk",
          f"{sim['tie_50_prob']:.1%}",
          help="R VP breaks tie → R control")

st.divider()

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ State Environment",
    "📊 Race Ratings",
    "👥 Demographics",
    "🎲 Seat Distribution"
])

# ── TAB 1: STATE ENVIRONMENT ──────────────────────────────────────────────────
with tab1:
    st.subheader("State-Level GCB Environment on Election Day")
    st.caption(
        f"Bottom-up MRP: Catalist 2024 baseline + demographic group shifts "
        f"+ OOP shift {oop['oop_shift']:+.2f}pp. "
        f"No external model outputs used."
    )

    # Map image
    maps = sorted(glob.glob('outputs/charts_*/map_state_environment.png'), reverse=True)
    if maps:
        st.image(maps[0], use_container_width=True)
    else:
        st.info("Run `python run.py` to generate the state environment map.")

    st.subheader("State Detail Table")

    # Build state table
    state_rows = []
    for abbr, r in sorted(races.items(), key=lambda x: -x[1].get('d_prob', 0)):
        state_rows.append({
            'St':      abbr,
            'D Win%':  r['d_prob'],
            'Central': f"{r['central']:+.1f}pp",
            'σ':       f"{r['sigma']:.2f}pp",
            'Polls':   r['n_polls'],
            'Rating':  r['rating'],
        })

    df = pd.DataFrame(state_rows)

    def highlight_rating(val):
        colors = {
            'Safe D':   '#1B4F8A', 'Likely D': '#4A90D9',
            'Lean D':   '#7EC8E3', 'Tossup':   '#9F7AEA',
            'Lean R':   '#F6AD55', 'Likely R': '#D94A4A',
            'Safe R':   '#8B1A1A',
        }
        bg = colors.get(val, '#FFFFFF')
        fg = 'white' if val in ('Safe D','Likely D','Safe R','Likely R') else '#1A1A1A'
        return f'background-color: {bg}; color: {fg}; font-weight: bold'

    def highlight_prob(val):
        if val >= 0.80: return 'color: #1B4F8A; font-weight: bold'
        if val >= 0.60: return 'color: #4A90D9'
        if val <= 0.20: return 'color: #8B1A1A; font-weight: bold'
        if val <= 0.40: return 'color: #D94A4A'
        return 'color: #7C3AED; font-weight: bold'

    styled = (df.style
              .applymap(highlight_rating, subset=['Rating'])
              .applymap(highlight_prob, subset=['D Win%'])
              .format({'D Win%': '{:.1%}'}))

    st.dataframe(styled, use_container_width=True, hide_index=True,
                 height=min(600, len(df)*36 + 40))

# ── TAB 2: RACE RATINGS ───────────────────────────────────────────────────────
with tab2:
    st.subheader("Senate Race Ratings")

    rating_map = sorted(
        glob.glob('outputs/charts_*/map_senate_ratings.png'), reverse=True)
    if rating_map:
        st.image(rating_map[0], use_container_width=True)
    else:
        st.info("Run `python run.py` to generate the ratings map.")

    st.subheader("Competitive Races")

    comp = sorted(
        [(a, r) for a, r in races.items() if 0.05 < r['d_prob'] < 0.95],
        key=lambda x: -x[1]['d_prob']
    )

    for abbr, r in comp:
        rating = r['rating']
        prob   = r['d_prob']
        polls  = r['n_polls']
        central = r['central']

        color_map = {
            'Safe D': '#1B4F8A', 'Likely D': '#4A90D9', 'Lean D': '#7EC8E3',
            'Tossup': '#9F7AEA', 'Lean R': '#F6AD55',
            'Likely R': '#D94A4A', 'Safe R': '#8B1A1A',
        }
        bar_color = color_map.get(rating, '#888888')

        col_a, col_b, col_c = st.columns([1, 5, 2])
        with col_a:
            st.markdown(f"**{abbr}**")
            st.caption(rating)
        with col_b:
            st.progress(prob, text=f"D Win: {prob:.1%}  |  Central: {central:+.1f}pp  |  Polls: {polls}")
        with col_c:
            st.caption(f"σ = {r['sigma']:.2f}pp")

    st.divider()
    st.caption(
        "⚠️ Polls in model: "
        + str(sum(r['n_polls'] for _, r in comp))
        + " total across competitive races. "
        "Add more via `python run.py --add-poll` or `python data/refresh.py`."
    )

# ── TAB 3: DEMOGRAPHIC SHIFTS ─────────────────────────────────────────────────
with tab3:
    st.subheader("Demographic Shifts vs Catalist 2024")
    st.caption(
        "These shifts are computed from manually entered national poll crosstabs. "
        "Each new crosstab you add updates the weighted estimates automatically. "
        "Catalist 2024 is the fixed anchor — these are shifts from 2024 actuals."
    )

    chart = sorted(glob.glob('outputs/charts_*/chart_demo_shifts.png'), reverse=True)
    if chart:
        st.image(chart[0], use_container_width=True)

    group_labels = {
        'white_nc': 'White Non-College',
        'white_college': 'White College',
        'hispanic': 'Hispanic',
        'black': 'Black',
        'aapi': 'AAPI',
        'youth_18_29': '18–29',
        'seniors_65p': '65+',
        'total': 'Total',
    }

    rows = []
    for group, data in demo.items():
        shift = data['shift']
        rows.append({
            'Group':        group_labels.get(group, group),
            'Catalist 2024': baselines.get(group, ''),
            'Current':      data['current'],
            'Shift (pp)':   shift,
            'N Entries':    data.get('n_entries', 0),
            'Signal':       '↑ D gaining' if shift > 1.5
                            else ('↓ R gaining' if shift < -1.5
                            else '→ Stable'),
        })

    df_d = pd.DataFrame(rows)

    def highlight_shift(val):
        if isinstance(val, float):
            if val >= 3:   return 'background-color: #C8E6C9; color: #1B5E20; font-weight: bold'
            if val >= 1:   return 'background-color: #E8F5E9; color: #2E7D32'
            if val <= -3:  return 'background-color: #FFCDD2; color: #B71C1C; font-weight: bold'
            if val <= -1:  return 'background-color: #FFEBEE; color: #C62828'
        return ''

    styled_d = (df_d.style
                .applymap(highlight_shift, subset=['Shift (pp)'])
                .format({
                    'Catalist 2024': '{:.3f}',
                    'Current': '{:.3f}',
                    'Shift (pp)': '{:+.1f}',
                }))
    st.dataframe(styled_d, use_container_width=True, hide_index=True)

    st.info(
        "📊 To add a new national poll crosstab:\n\n"
        "`python run.py --add-crosstab`\n\n"
        "You'll be prompted to enter D vote share by demographic group. "
        "The model will automatically update state environments."
    )

# ── TAB 4: SEAT DISTRIBUTION ──────────────────────────────────────────────────
with tab4:
    st.subheader("Monte Carlo Seat Distribution")
    st.caption(f"100,000 correlated simulations | Time σ = {cfg['race_model']['time_sigma_base']}pp base")

    chart2 = sorted(glob.glob('outputs/charts_*/chart_seat_distribution.png'), reverse=True)
    if chart2:
        st.image(chart2[0], use_container_width=True)

    # Scenario summary
    st.subheader("Scenarios")
    s1, s2, s3 = st.columns(3)
    s1.metric(
        f"Low (D{oop['gcb_eday_lo']:+.2f})",
        f"{sim['maj_lo']:.1%}",
        help="−1σ national GCB scenario"
    )
    s2.metric(
        f"Central (D{oop['gcb_eday_central']:+.2f})",
        f"{sim['d_majority_prob']:.1%}",
        delta="Base case",
        help="Central OOP shift applied"
    )
    s3.metric(
        f"High (D{oop['gcb_eday_hi']:+.2f})",
        f"{sim['maj_hi']:.1%}",
        help="+1σ national GCB scenario"
    )

    st.divider()
    st.subheader("Most Likely Outcomes")
    seat_dist = sim['seat_distribution']
    top = sorted(seat_dist.items(), key=lambda x: -x[1])[:10]
    for seats, prob in top:
        if seats >= 51:   label = f"**{seats} seats** ✅ D majority"
        elif seats == 50: label = f"**{seats} seats** ⚠️ Tie → R control"
        else:             label = f"**{seats} seats** ❌ R majority"
        st.progress(float(prob), text=f"{label} — {prob:.1%}")
