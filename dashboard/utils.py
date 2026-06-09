"""
Shared utilities, design system, and data loaders for the WC2026 dashboard.
All heavy data is loaded once and cached with st.cache_data / st.cache_resource.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st

# ── Path helpers ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Design tokens ─────────────────────────────────────────────────────────────
COLORS = {
    "bg_deep":    "#080C14",
    "bg_card":    "#0D1526",
    "bg_card2":   "#111827",
    "border":     "#1C2E4A",
    "border_hi":  "#2A4470",
    "accent":     "#38BDF8",   # sky-400
    "accent_dim": "#0EA5E9",   # sky-500
    "red":        "#F87171",   # red-400
    "gold":       "#FCD34D",   # amber-300
    "green":      "#34D399",   # emerald-400
    "purple":     "#A78BFA",   # violet-400
    "text1":      "#F1F5F9",
    "text2":      "#94A3B8",
    "text3":      "#475569",
    # legacy aliases used in pages
    "primary":    "#0EA5E9",
    "secondary":  "#F87171",
}

STAGE_COLORS = {
    "champion":      "#FCD34D",
    "runner_up":     "#94A3B8",
    "semifinal":     "#A78BFA",
    "quarterfinal":  "#38BDF8",
    "round_of_16":   "#34D399",
    "round_of_32":   "#6EE7B7",
    "group_stage":   "#1C2E4A",
}

CONF_COLORS = {
    "UEFA":     "#38BDF8",
    "CONMEBOL": "#34D399",
    "CONCACAF": "#FCD34D",
    "CAF":      "#F87171",
    "AFC":      "#A78BFA",
    "OFC":      "#FB923C",
}

# ── Plotly dark chart template ─────────────────────────────────────────────────
CHART = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,21,38,0.5)",
    font=dict(family="Inter, system-ui, sans-serif", color="#94A3B8", size=11),
    margin=dict(l=4, r=4, t=36, b=4),
    showlegend=True,
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(255,255,255,0.06)",
        borderwidth=1,
        font=dict(color="#94A3B8", size=11),
    ),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.08)",
        tickfont=dict(color="#64748B"),
        title_font=dict(color="#64748B"),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.08)",
        tickfont=dict(color="#64748B"),
        title_font=dict(color="#64748B"),
    ),
)

def chart_layout(**overrides) -> dict:
    """Return a merged chart layout dict."""
    base = {**CHART}
    base.update(overrides)
    return base


# ── Data loaders (cached) ─────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data/processed/model_dataset.csv", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_elo() -> dict[str, float]:
    with open(ROOT / "outputs/final_elo_ratings.json") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_monte_carlo() -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs/monte_carlo_results.csv")


@st.cache_data(show_spinner=False)
def load_groups() -> dict[str, list[str]]:
    import yaml
    with open(ROOT / "configs/wc2026_groups.yaml") as f:
        return yaml.safe_load(f)["groups"]


@st.cache_data(show_spinner=False)
def load_model_metadata() -> dict:
    with open(ROOT / "models/model_metadata.json") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_model_pipeline():
    import joblib
    pipe      = joblib.load(ROOT / "models/best_model.pkl")
    feat_cols = joblib.load(ROOT / "models/feature_columns.pkl")
    return pipe, feat_cols


# ── Team helpers ──────────────────────────────────────────────────────────────

def all_wc_teams() -> list[str]:
    groups = load_groups()
    return sorted(t for grp in groups.values() for t in grp)


def team_group(team: str) -> str:
    for grp, teams in load_groups().items():
        if team in teams:
            return grp
    return "?"


def team_stats(team: str) -> dict:
    df   = load_dataset()
    home = df[df["home_team"] == team]
    away = df[df["away_team"] == team]
    total = len(home) + len(away)
    if total == 0:
        return {}
    wins   = (home["result"] == 2).sum() + (away["result"] == 0).sum()
    draws  = (home["result"] == 1).sum() + (away["result"] == 1).sum()
    losses = (home["result"] == 0).sum() + (away["result"] == 2).sum()
    gf     = home["home_score"].sum() + away["away_score"].sum()
    ga     = home["away_score"].sum() + away["home_score"].sum()
    elo    = load_elo()
    h_rank = home[["home_rank"]].dropna().tail(1)
    a_rank = away[["away_rank"]].dropna().tail(1)
    rank   = float(h_rank["home_rank"].iloc[0]) if len(h_rank) else \
             (float(a_rank["away_rank"].iloc[0]) if len(a_rank) else None)
    return {
        "total_matches": int(total),
        "wins":  int(wins),
        "draws": int(draws),
        "losses": int(losses),
        "win_pct":  round(wins / total * 100, 1),
        "goals_for":     int(gf),
        "goals_against": int(ga),
        "goal_diff":     int(gf - ga),
        "goals_per_game": round(gf / total, 2),
        "elo": round(elo.get(team, 1500), 1),
        "rank": rank,
        "group": team_group(team),
    }


def confederation_map() -> dict[str, str]:
    import yaml
    with open(ROOT / "configs/wc2026_groups.yaml") as f:
        data = yaml.safe_load(f)
    mapping = {}
    for conf, teams in data.get("confederations", {}).items():
        for t in teams:
            mapping[t] = conf
    return mapping


# ── HTML component helpers ────────────────────────────────────────────────────

def kpi(value: str, label: str, delta: str = "", accent: str = "#38BDF8",
        delta_color: str = "") -> str:
    """Return a dark glassmorphism KPI card HTML string."""
    delta_html = ""
    if delta:
        dc = delta_color or ("#34D399" if not delta.startswith("-") else "#F87171")
        delta_html = f"<div style='font-size:0.75rem;color:{dc};margin-top:4px;font-weight:600;'>{delta}</div>"
    return f"""
    <div style='
        background:linear-gradient(135deg,rgba(56,189,248,0.07) 0%,rgba(14,165,233,0.03) 100%);
        border:1px solid {accent}33;
        border-top:2px solid {accent};
        border-radius:12px;
        padding:18px 20px 14px;
        text-align:center;
        box-shadow:0 4px 24px rgba(0,0,0,0.4),inset 0 1px 0 rgba(255,255,255,0.04);
        height:100%;
    '>
        <div style='font-size:1.9rem;font-weight:800;color:#F1F5F9;
                    letter-spacing:-0.5px;line-height:1.1;'>{value}</div>
        <div style='font-size:0.7rem;font-weight:700;color:#64748B;
                    text-transform:uppercase;letter-spacing:0.1em;margin-top:6px;'>{label}</div>
        {delta_html}
    </div>"""


def section_title(title: str, subtitle: str = "") -> None:
    """Render a dark section header with accent left border."""
    sub_html = f"<div style='font-size:0.8rem;color:#64748B;margin-top:3px;'>{subtitle}</div>" if subtitle else ""
    st.markdown(f"""
    <div style='border-left:3px solid #38BDF8;padding-left:14px;margin:1.8rem 0 1rem;'>
        <div style='font-size:1.1rem;font-weight:700;color:#E2E8F0;
                    letter-spacing:-0.2px;'>{title}</div>
        {sub_html}
    </div>""", unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    """Render the top-of-page hero banner."""
    sub_html = f"<p style='margin:6px 0 0;opacity:0.65;font-size:0.9rem;font-weight:400;'>{subtitle}</p>" if subtitle else ""
    st.markdown(f"""
    <div style='
        background:linear-gradient(135deg,#0B1728 0%,#0F2040 40%,#0A1830 100%);
        border:1px solid #1C2E4A;
        border-left:4px solid #38BDF8;
        border-radius:14px;
        padding:22px 28px;
        margin-bottom:1.5rem;
        box-shadow:0 8px 32px rgba(0,0,0,0.5);
    '>
        <h2 style='margin:0;font-size:1.5rem;font-weight:800;
                   background:linear-gradient(90deg,#F1F5F9,#38BDF8);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                   letter-spacing:-0.5px;'>⚽ {title}</h2>
        {sub_html}
    </div>""", unsafe_allow_html=True)


def prob_bar(label: str, pct: float, color: str = "#38BDF8") -> str:
    """Render a dark probability bar (pct is 0–100)."""
    bar_w = max(2, min(100, int(pct)))
    return f"""
    <div style='margin:10px 0;'>
        <div style='display:flex;justify-content:space-between;
                    font-size:0.82rem;font-weight:600;margin-bottom:5px;'>
            <span style='color:#CBD5E1;'>{label}</span>
            <span style='color:{color};'>{pct:.1f}%</span>
        </div>
        <div style='background:rgba(255,255,255,0.06);border-radius:6px;height:8px;'>
            <div style='width:{bar_w}%;height:100%;border-radius:6px;
                        background:linear-gradient(90deg,{color}99,{color});
                        box-shadow:0 0 8px {color}66;'></div>
        </div>
    </div>"""


# ── CSS injection ─────────────────────────────────────────────────────────────

def inject_css() -> None:
    st.markdown("""
    <style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* ── Base ── */
    html, body, .stApp {
        background-color: #080C14 !important;
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    }
    .stApp > header {
        background: rgba(8,12,20,0.95) !important;
        border-bottom: 1px solid #1C2E4A;
        backdrop-filter: blur(12px);
    }
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg,#0B1120 0%,#060A12 100%) !important;
        border-right: 1px solid #1C2E4A !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
    }
    [data-testid="stSidebar"] hr {
        border-color: #1C2E4A !important;
        margin: 0.6rem 0;
    }
    /* Sidebar radio labels */
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        color: #94A3B8 !important;
        padding: 6px 8px !important;
        border-radius: 8px;
        transition: color 0.15s;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        color: #38BDF8 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #475569 !important;
        font-size: 0.75rem !important;
    }

    /* ── Streamlit metric cards → dark glass ── */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg,rgba(56,189,248,0.06),rgba(14,165,233,0.02)) !important;
        border: 1px solid rgba(56,189,248,0.18) !important;
        border-top: 2px solid #38BDF8 !important;
        border-radius: 12px !important;
        padding: 16px 18px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04) !important;
    }
    [data-testid="metric-container"] label {
        color: #64748B !important;
        font-size: 0.68rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
    }
    [data-testid="stMetricValue"] {
        color: #F1F5F9 !important;
        font-size: 1.65rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.76rem !important;
        font-weight: 600 !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #0EA5E9, #0284C7) !important;
        color: #F0F9FF !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.02em !important;
        padding: 10px 22px !important;
        box-shadow: 0 4px 16px rgba(14,165,233,0.35) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        box-shadow: 0 6px 24px rgba(14,165,233,0.5) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button[kind="secondary"] {
        background: rgba(255,255,255,0.06) !important;
        color: #94A3B8 !important;
        box-shadow: none !important;
        border: 1px solid #1C2E4A !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid #1C2E4A !important;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 8px 8px 0 0 !important;
        color: #64748B !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
        border: none !important;
        transition: color 0.15s !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #94A3B8 !important;
        background: rgba(255,255,255,0.04) !important;
    }
    .stTabs [aria-selected="true"] {
        color: #38BDF8 !important;
        background: rgba(56,189,248,0.08) !important;
        border-bottom: 2px solid #38BDF8 !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1.2rem !important;
    }

    /* ── Selectbox / dropdowns ── */
    .stSelectbox > div > div {
        background: #0D1526 !important;
        border: 1px solid #1C2E4A !important;
        border-radius: 10px !important;
        color: #E2E8F0 !important;
    }
    .stSelectbox > div > div:focus-within {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 0 2px rgba(56,189,248,0.2) !important;
    }

    /* ── Number input / text input ── */
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        background: #0D1526 !important;
        border: 1px solid #1C2E4A !important;
        border-radius: 8px !important;
        color: #E2E8F0 !important;
    }

    /* ── Checkboxes / sliders ── */
    .stCheckbox label span { color: #94A3B8 !important; }
    .stSlider [data-testid="stSlider"] > div { color: #94A3B8 !important; }

    /* ── DataFrames (Streamlit table) ── */
    .stDataFrame {
        border: 1px solid #1C2E4A !important;
        border-radius: 10px !important;
        overflow: hidden;
    }
    .stDataFrame [data-testid="stDataFrameResizable"] {
        background: #0D1526 !important;
    }
    iframe[title="st_aggrid.agGrid"] {
        background: #0D1526 !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: #0D1526 !important;
        border: 1px solid #1C2E4A !important;
        border-radius: 10px !important;
        color: #94A3B8 !important;
        font-weight: 600 !important;
    }
    .streamlit-expanderContent {
        background: #080C14 !important;
        border: 1px solid #1C2E4A !important;
        border-top: none !important;
    }

    /* ── Info / warning / error banners ── */
    [data-testid="stAlert"] {
        background: rgba(56,189,248,0.08) !important;
        border: 1px solid rgba(56,189,248,0.25) !important;
        border-radius: 10px !important;
        color: #94A3B8 !important;
    }

    /* ── Divider ── */
    hr { border-color: #1C2E4A !important; }

    /* ── Spinner ── */
    [data-testid="stSpinner"] { color: #38BDF8 !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #080C14; }
    ::-webkit-scrollbar-thumb { background: #1C2E4A; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #2A4470; }

    /* ── Custom HTML components (used in pages) ── */

    /* Glass card */
    .glass-card {
        background: linear-gradient(135deg,rgba(56,189,248,0.06) 0%,rgba(14,165,233,0.02) 100%);
        border: 1px solid #1C2E4A;
        border-radius: 14px;
        padding: 20px 22px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    }
    .glass-card:hover {
        border-color: #2A4470;
        box-shadow: 0 6px 32px rgba(0,0,0,0.5);
    }

    /* Champion reveal */
    .champion-banner {
        background: linear-gradient(135deg,#0B1728 0%,#1A0F2A 50%,#0A1828 100%);
        border: 1px solid rgba(252,211,77,0.35);
        border-top: 3px solid #FCD34D;
        border-radius: 16px;
        padding: 32px 40px;
        text-align: center;
        box-shadow: 0 8px 48px rgba(0,0,0,0.6), 0 0 40px rgba(252,211,77,0.08);
        margin: 1rem 0;
    }

    /* Bracket match card */
    .match-card {
        background: #0D1526;
        border: 1px solid #1C2E4A;
        border-radius: 10px;
        padding: 12px 18px;
        margin: 6px 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: border-color 0.15s;
    }
    .match-card:hover { border-color: #2A4470; }
    .match-winner { font-weight: 700; color: #F1F5F9; font-size: 0.9rem; }
    .match-loser  { color: #475569; font-size: 0.9rem; }
    .match-score  { font-size: 1.1rem; font-weight: 800;
                    color: #38BDF8; letter-spacing: 2px; }

    /* Leaderboard row */
    .lb-row {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 10px 16px;
        border-radius: 10px;
        margin: 3px 0;
        transition: background 0.15s;
    }
    .lb-row:hover { background: rgba(255,255,255,0.04); }
    .lb-rank {
        font-size: 0.85rem;
        font-weight: 800;
        color: #475569;
        width: 24px;
        text-align: center;
        flex-shrink: 0;
    }
    .lb-rank.gold   { color: #FCD34D; }
    .lb-rank.silver { color: #94A3B8; }
    .lb-rank.bronze { color: #FB923C; }
    .lb-team  { font-size: 0.9rem; font-weight: 600; color: #E2E8F0; flex: 1; }
    .lb-value { font-size: 0.95rem; font-weight: 700; color: #38BDF8; }

    /* Group standing table */
    .grp-header {
        font-size: 0.65rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #38BDF8;
        padding: 6px 12px 4px;
        border-bottom: 1px solid #1C2E4A;
    }
    .grp-row {
        display: flex;
        align-items: center;
        padding: 7px 12px;
        font-size: 0.78rem;
        border-bottom: 1px solid rgba(28,46,74,0.5);
        gap: 8px;
    }
    .grp-row.advance { background: rgba(52,211,153,0.07); }
    .grp-row.best3   { background: rgba(56,189,248,0.07); }
    .grp-name { font-weight: 600; color: #E2E8F0; flex: 1; }
    .grp-stat { color: #64748B; width: 26px; text-align: center; }
    .grp-pts  { font-weight: 800; color: #F1F5F9; width: 26px; text-align: center; }

    /* VS separator */
    .vs-badge {
        background: linear-gradient(135deg,#0EA5E9,#7C3AED);
        color: white;
        font-size: 0.7rem;
        font-weight: 900;
        letter-spacing: 0.15em;
        padding: 4px 10px;
        border-radius: 20px;
    }

    /* Conf badge */
    .conf-badge {
        font-size: 0.65rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 20px;
        letter-spacing: 0.08em;
        display: inline-block;
    }

    /* Stat pill */
    .stat-pill {
        background: rgba(255,255,255,0.05);
        border: 1px solid #1C2E4A;
        border-radius: 8px;
        padding: 8px 14px;
        text-align: center;
    }
    .stat-pill .val { font-size: 1.25rem; font-weight: 800; color: #F1F5F9; }
    .stat-pill .lbl { font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
                      letter-spacing: 0.1em; color: #475569; margin-top: 2px; }
    </style>
    """, unsafe_allow_html=True)
