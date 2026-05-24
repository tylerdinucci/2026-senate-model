"""
2026 Senate Model Dashboard
Run with: streamlit run outputs/dashboard.py
"""

import streamlit as st
import pandas as pd
import json
import yaml
import glob
import requests
import io
import numpy as np
from scipy import stats
import altair as alt
from pathlib import Path
from datetime import date, datetime
import subprocess
import sys

_GCB_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vRsvXNCZ0ubJr8D_yNcU5q6C0_HBa35K7oDK03KpO7Ca43UwdXaIdvVLWoXEmHHph0EREz5430Hm5yZ"
    "/pub?output=csv"
)


@st.cache_data(ttl=3600)
def load_gcb_polls():
    """Fetch raw Silver Bulletin GCB CSV — returns (all_polls, scatter_polls).
    all_polls: full history for trend computation (needs deep window at start).
    scatter_polls: 2026-cycle only for display dots.
    """
    try:
        resp = requests.get(_GCB_CSV_URL, timeout=15)
        resp.raise_for_status()
        raw = pd.read_csv(io.StringIO(resp.text), low_memory=False)
        df = raw[raw['subgroup'] == 'All polls'].copy()
        df['enddate'] = pd.to_datetime(df['enddate'], errors='coerce')
        df['adjusted_net'] = pd.to_numeric(df['adjusted_net'], errors='coerce')
        df = df.dropna(subset=['enddate', 'adjusted_net'])
        df = df.sort_values('enddate')
        # All polls for trend (full history so early window has enough polls)
        all_polls = df.copy()
        # Scatter dots: 2026 cycle only (Jan 1 2026 onwards — avoids pre-cycle clutter)
        scatter_polls = df[df['enddate'] >= pd.Timestamp('2026-01-01')].copy()
        return all_polls, scatter_polls
    except Exception:
        return None, None


_GCB_GRADES = {
    'The New York Times/Siena College': 3.0, 'Marist University': 2.9,
    'Marist College': 2.9, 'Quinnipiac University': 2.8, 'CNN/SSRS': 2.8,
    'YouGov': 2.7, 'Public Religion Research Institute': 2.6, 'Ipsos': 2.5,
    'Emerson College': 2.5, 'AtlasIntel': 2.4, 'Focaldata': 2.4,
    'Morning Consult': 2.3, 'Public Opinion Strategies': 2.2,
    'HarrisX': 2.2, 'Cygnal': 2.2, 'Echelon Insights': 2.0,
    'RMG Research': 2.0, 'Rasmussen Reports': 1.8, 'McLaughlin & Associates': 1.5,
}

def compute_gcb_trend(polls_df, half_life=30, window=60, step_days=3):
    """
    Compute rolling grade×recency weighted GCB average at every `step_days` interval.
    Returns DataFrame with columns: date, gcb_trend.
    """
    if polls_df is None or len(polls_df) == 0:
        return pd.DataFrame(columns=['date', 'gcb_trend'])

    polls_df = polls_df.copy()
    polls_df['grade_w'] = polls_df['pollster'].map(_GCB_GRADES).fillna(1.5)

    start = polls_df['enddate'].min().to_pydatetime()
    end   = datetime.combine(date.today(), datetime.min.time())
    dates = pd.date_range(start, end, freq=f'{step_days}D')

    rows = []
    for dt in dates:
        df_w = polls_df.copy()
        df_w['days_ago'] = (dt - df_w['enddate']).dt.days
        df_w = df_w[(df_w['days_ago'] >= 0) & (df_w['days_ago'] <= window)]
        if len(df_w) < 8:
            continue
        df_w['recency_w'] = np.exp(-df_w['days_ago'] * np.log(2) / half_life)
        df_w['w'] = df_w['recency_w'] * df_w['grade_w']
        wavg = (df_w['adjusted_net'] * df_w['w']).sum() / df_w['w'].sum()
        rows.append({'date': dt, 'gcb_trend': round(float(wavg), 3)})

    return pd.DataFrame(rows)


def build_gcb_sensitivity(races, d_seats_not_up, gcb_base):
    """Expected D seats across a range of election-day GCB values."""
    rows = []
    for gcb_val in np.arange(-5.0, 16.5, 0.5):
        delta = gcb_val - gcb_base
        exp_seats = float(d_seats_not_up)
        for r in races.values():
            sigma = r.get('sigma', 4.5)
            new_central = r['central'] + delta
            prob = float(stats.norm.cdf(new_central / sigma)) if sigma > 0 else (1.0 if new_central > 0 else 0.0)
            exp_seats += prob
        rows.append({'gcb_eday': round(float(gcb_val), 1), 'exp_d_seats': round(exp_seats, 2)})
    return pd.DataFrame(rows)


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
            subprocess.run([sys.executable, 'run.py'])
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

# ── GCB TRACKER: ALL POLLS ────────────────────────────────────────────────────
st.subheader("Generic Congressional Ballot — All Polls")

gcb_polls_all, gcb_polls_scatter = load_gcb_polls()
tracker_df = pd.read_csv('data/gcb_national.csv', parse_dates=['date'])

election_day = datetime(2026, 11, 3)

if gcb_polls_scatter is not None and len(gcb_polls_scatter) > 0:
    # Individual polls scatter — 2026 cycle only
    poll_chart = (
        alt.Chart(gcb_polls_scatter)
        .mark_circle(size=45, opacity=0.55)
        .encode(
            x=alt.X('enddate:T', title='Poll End Date',
                    scale=alt.Scale(domain=['2026-01-01', '2026-11-03'])),
            y=alt.Y('adjusted_net:Q', title='D Net Margin (pp)',
                    axis=alt.Axis(format='+d')),
            color=alt.condition(
                alt.datum.adjusted_net > 0,
                alt.value('#3A7DC9'),
                alt.value('#C0392B')
            ),
            tooltip=(
                [alt.Tooltip('pollster:N', title='Pollster'),
                 alt.Tooltip('enddate:T', title='Date', format='%b %d %Y'),
                 alt.Tooltip('adjusted_net:Q', title='Adj. Net', format='+.1f')]
                + ([alt.Tooltip('samplesize:Q', title='n=')] if 'samplesize' in gcb_polls_scatter.columns else [])
            ),
        )
    )
else:
    poll_chart = alt.Chart(pd.DataFrame({'enddate': [], 'adjusted_net': []})).mark_circle()

# Continuous trend line — uses full history for computation (stable window),
# then clips display to 2026-01-01 so the pre-cycle rise doesn't show
trend_df = compute_gcb_trend(gcb_polls_all)
if len(trend_df) > 0:
    trend_df = trend_df[trend_df['date'] >= pd.Timestamp('2026-01-01')]
if len(trend_df) > 0:
    trend_line = (
        alt.Chart(trend_df)
        .mark_line(color='#F59E0B', strokeWidth=2.5)
        .encode(
            x=alt.X('date:T'),
            y=alt.Y('gcb_trend:Q'),
            tooltip=[alt.Tooltip('date:T', title='Date', format='%b %d %Y'),
                     alt.Tooltip('gcb_trend:Q', title='Trend', format='+.2f')],
        )
    )
else:
    trend_line = alt.Chart(pd.DataFrame({'date': [], 'gcb_trend': []})).mark_line()

# Our manual snapshot readings (dots only — trend line replaces the connecting line)
tracker_pts = (
    alt.Chart(tracker_df)
    .mark_circle(color='#F59E0B', size=80, opacity=1.0, stroke='white', strokeWidth=1)
    .encode(
        x='date:T',
        y='gcb_d_net:Q',
        tooltip=[alt.Tooltip('date:T', title='Date', format='%b %d %Y'),
                 alt.Tooltip('gcb_d_net:Q', title='Model reading', format='+.2f')],
    )
)

# Election day projected range: vertical rule + error band
eday_df = pd.DataFrame({
    'x':  [election_day],
    'y':  [oop['gcb_eday_central']],
    'lo': [oop['gcb_eday_lo']],
    'hi': [oop['gcb_eday_hi']],
})
eday_rule = (
    alt.Chart(eday_df)
    .mark_rule(color='#6B7280', strokeDash=[5, 3], strokeWidth=1.5)
    .encode(x='x:T')
)
eday_point = (
    alt.Chart(eday_df)
    .mark_point(shape='diamond', size=120, color='#7C3AED', filled=True)
    .encode(
        x='x:T',
        y='y:Q',
        tooltip=[alt.Tooltip('y:Q', title='Projected eday GCB', format='+.2f'),
                 alt.Tooltip('lo:Q', title='Low', format='+.2f'),
                 alt.Tooltip('hi:Q', title='High', format='+.2f')],
    )
)
eday_bar = (
    alt.Chart(eday_df)
    .mark_errorbar(color='#7C3AED', ticks=True)
    .encode(x='x:T', y='lo:Q', y2='hi:Q')
)

# Zero reference line
zero_rule = (
    alt.Chart(pd.DataFrame({'y': [0]}))
    .mark_rule(color='#9CA3AF', strokeDash=[3, 3], strokeWidth=1.0)
    .encode(y='y:Q')
)

gcb_chart = (
    (zero_rule + poll_chart + trend_line + tracker_pts + eday_rule + eday_bar + eday_point)
    .properties(height=320)
    .configure_axis(labelFontSize=11, titleFontSize=12)
    .configure_view(strokeWidth=0)
)
st.altair_chart(gcb_chart, use_container_width=True)

col_leg1, col_leg2, col_leg3 = st.columns(3)
col_leg1.caption("🔵 Individual D-favored polls  🔴 Individual R-favored polls")
col_leg2.caption("🟡 Continuous trend line (grade × recency weighted)  ·  dots = model snapshot readings")
col_leg3.caption("🟣 Election day projection ± σ")

st.divider()

# ── GCB SENSITIVITY ───────────────────────────────────────────────────────────
st.subheader("GCB Sensitivity — Expected D Seats vs. Election Day GCB")
st.caption(
    "If the election-day GCB comes in higher or lower than projected, "
    "how many seats do Democrats expect to win? "
    f"Purple band = ±1σ (±{oop['gcb_sigma']:.1f}pp) around the current projection."
)

d_seats_not_up = cfg['simulation']['d_seats_not_up']
gcb_base       = oop['gcb_eday_central']
gcb_sig        = oop['gcb_sigma']

sens_df = build_gcb_sensitivity(races, d_seats_not_up, gcb_base)

# Pin y domain across every layer so Altair doesn't default to 0
_y_min = max(40, int(sens_df['exp_d_seats'].min()) - 1)
_y_max = min(60, int(sens_df['exp_d_seats'].max()) + 2)
_y_scale = alt.Scale(domain=[_y_min, _y_max])
_x_axis  = alt.Axis(format='+d', values=list(range(-6, 17, 2)))

# Main expected-seats line
sens_line = (
    alt.Chart(sens_df)
    .mark_line(color='#1B4F8A', strokeWidth=2.5)
    .encode(
        x=alt.X('gcb_eday:Q', title='Election Day GCB (D margin, pp)', axis=_x_axis),
        y=alt.Y('exp_d_seats:Q', title='Expected D Seats', scale=_y_scale),
        tooltip=[alt.Tooltip('gcb_eday:Q', title='GCB eday', format='+.1f'),
                 alt.Tooltip('exp_d_seats:Q', title='Exp. D seats', format='.1f')],
    )
)

# Majority (51) and tie (50) threshold lines — explicit x range so scale stays pinned
thresh_data = pd.DataFrame([
    {'x': -5.0, 'y': 51.0, 'label': '51 — D majority'},
    {'x': 16.0, 'y': 51.0, 'label': '51 — D majority'},
    {'x': -5.0, 'y': 50.0, 'label': '50 — Tie (R VP)'},
    {'x': 16.0, 'y': 50.0, 'label': '50 — Tie (R VP)'},
])
thresh_lines = (
    alt.Chart(thresh_data)
    .mark_line(strokeDash=[5, 3], strokeWidth=1.5)
    .encode(
        x=alt.X('x:Q', axis=_x_axis),
        y=alt.Y('y:Q', scale=_y_scale),
        color=alt.Color('label:N',
                        scale=alt.Scale(domain=['51 — D majority', '50 — Tie (R VP)'],
                                        range=['#16A34A', '#D97706']),
                        legend=alt.Legend(title='', orient='bottom-right')),
        detail='label:N',
    )
)

# Current projection vertical line
curr_rule = (
    alt.Chart(pd.DataFrame({'x': [gcb_base]}))
    .mark_rule(color='#7C3AED', strokeWidth=2)
    .encode(x=alt.X('x:Q', axis=_x_axis))
)

# ±1σ shaded band
band_df = pd.DataFrame([{
    'x1': gcb_base - gcb_sig, 'x2': gcb_base + gcb_sig,
    'y1': float(_y_min),      'y2': float(_y_max),
}])
sigma_band = (
    alt.Chart(band_df)
    .mark_rect(opacity=0.08, color='#7C3AED')
    .encode(
        x=alt.X('x1:Q'), x2='x2:Q',
        y=alt.Y('y1:Q', scale=_y_scale), y2='y2:Q',
    )
)

# Mark the current scenario point
curr_exp = sens_df.loc[(sens_df['gcb_eday'] - gcb_base).abs().idxmin(), 'exp_d_seats']
curr_pt_df = pd.DataFrame([{'x': gcb_base, 'y': curr_exp}])
curr_point = (
    alt.Chart(curr_pt_df)
    .mark_point(shape='diamond', size=130, color='#7C3AED', filled=True)
    .encode(
        x=alt.X('x:Q', axis=_x_axis),
        y=alt.Y('y:Q', scale=_y_scale),
        tooltip=[alt.Tooltip('x:Q', title='Current projection', format='+.2f'),
                 alt.Tooltip('y:Q', title='Exp. D seats', format='.1f')],
    )
)

sens_chart = (
    (sigma_band + thresh_lines + curr_rule + sens_line + curr_point)
    .properties(height=300)
    .configure_axis(labelFontSize=11, titleFontSize=12)
    .configure_view(strokeWidth=0)
)
st.altair_chart(sens_chart, use_container_width=True)

# Quick scenario callouts
sa, sb, sc, sd = st.columns(4)
def seats_at(gcb_val):
    row = sens_df.loc[(sens_df['gcb_eday'] - gcb_val).abs().idxmin()]
    return row['exp_d_seats']

sa.metric(f"D{oop['gcb_eday_lo']:+.1f}  (−1σ)",  f"{seats_at(oop['gcb_eday_lo']):.1f} seats")
sb.metric(f"D{gcb_base:+.1f}  (central)",         f"{seats_at(gcb_base):.1f} seats", delta="projected")
sc.metric(f"D{oop['gcb_eday_hi']:+.1f}  (+1σ)",   f"{seats_at(oop['gcb_eday_hi']):.1f} seats")
sd.metric("50-seat tie (R control)",               f"GCB ≈ D{sens_df.loc[(sens_df['exp_d_seats'] - 50).abs().idxmin(), 'gcb_eday']:+.1f}")

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
    maps = sorted(glob.glob('outputs/charts_20[0-9][0-9]-[0-9][0-9]-[0-9][0-9]/map_state_environment.png'), reverse=True)
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
              .map(highlight_rating, subset=['Rating'])
              .map(highlight_prob, subset=['D Win%'])
              .format({'D Win%': '{:.1%}'}))

    st.dataframe(styled, use_container_width=True, hide_index=True,
                 height=min(600, len(df)*36 + 40))

# ── TAB 2: RACE RATINGS ───────────────────────────────────────────────────────
with tab2:
    st.subheader("Senate Race Ratings")

    rating_map = sorted(
        glob.glob('outputs/charts_20[0-9][0-9]-[0-9][0-9]-[0-9][0-9]/map_senate_ratings.png'), reverse=True)
    if rating_map:
        st.image(rating_map[0], use_container_width=True)
    else:
        st.info("Run `python run.py` to generate the ratings map.")

    st.subheader("Competitive Races")

    comp = sorted(
        [(a, r) for a, r in races.items() if 0.01 < r["d_prob"] < 0.99],
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

    chart = sorted(glob.glob('outputs/charts_20[0-9][0-9]-[0-9][0-9]-[0-9][0-9]/chart_demo_shifts.png'), reverse=True)
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
            'Group':           group_labels.get(group, group),
            '2024 Baseline':   baselines.get(group, 0.0),
            'Current (2026)':  data['current'],
            'Shift':           shift,
            'N Polls':         data.get('n_entries', 0),
            'Trend':           '↑ D gaining' if shift > 1.5
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
                .map(highlight_shift, subset=['Shift'])
                .format({
                    '2024 Baseline':  '{:.1%}',
                    'Current (2026)': '{:.1%}',
                    'Shift':          '{:+.1f}pp',
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

    chart2 = sorted(glob.glob('outputs/charts_20[0-9][0-9]-[0-9][0-9]-[0-9][0-9]/chart_seat_distribution.png'), reverse=True)
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
        seats = int(seats)
        if seats >= 51:   label = f"**{seats} seats** ✅ D majority"
        elif seats == 50: label = f"**{seats} seats** ⚠️ Tie → R control"
        else:             label = f"**{seats} seats** ❌ R majority"
        st.progress(float(prob), text=f"{label} — {prob:.1%}")
