"""
Shared utilities, data loaders, and constants for the WC2026 dashboard.
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

# ── Colour palette ────────────────────────────────────────────────────────────
COLORS = {
    "primary":   "#1E3A5F",   # deep navy
    "secondary": "#C8102E",   # FIFA red
    "gold":      "#FFD700",
    "accent":    "#00A651",   # green
    "light":     "#F0F4F8",
    "text":      "#1A1A2E",
    "muted":     "#6B7280",
}

# Stage colours for charts
STAGE_COLORS = {
    "champion":      "#FFD700",
    "runner_up":     "#C0C0C0",
    "semifinal":     "#CD7F32",
    "quarterfinal":  "#1E3A5F",
    "round_of_16":   "#2563EB",
    "round_of_32":   "#60A5FA",
    "group_stage":   "#BFDBFE",
}

CONF_COLORS = {
    "UEFA":     "#1565C0",
    "CONMEBOL": "#2E7D32",
    "CONCACAF": "#F57F17",
    "CAF":      "#BF360C",
    "AFC":      "#6A1B9A",
    "OFC":      "#00695C",
}

# ── Data loaders (cached) ─────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data/processed/model_dataset.csv", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_raw_results() -> pd.DataFrame:
    path = ROOT / "data/raw/results.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
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
    """Compute team-level summary stats from the processed dataset."""
    df = load_dataset()
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
    # Last known rank
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


# ── CSS injection ─────────────────────────────────────────────────────────────

def inject_css() -> None:
    st.markdown("""
    <style>
        /* Main background */
        .stApp { background-color: #F8FAFC; }

        /* Metric cards */
        div[data-testid="metric-container"] {
            background: white;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }
        div[data-testid="metric-container"] label {
            color: #64748B !important;
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
            color: #1E3A5F !important;
            font-size: 1.6rem !important;
            font-weight: 700 !important;
        }

        /* Section headers */
        .section-header {
            color: #1E3A5F;
            font-size: 1.3rem;
            font-weight: 700;
            border-left: 4px solid #C8102E;
            padding-left: 12px;
            margin: 1.5rem 0 0.8rem 0;
        }

        /* Team badge */
        .team-badge {
            background: linear-gradient(135deg, #1E3A5F, #2563EB);
            color: white;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-block;
        }

        /* Probability bars */
        .prob-bar-container { margin: 8px 0; }
        .prob-label { font-size: 0.9rem; color: #374151; font-weight: 600; }
        .prob-bar-bg {
            background: #E5E7EB;
            border-radius: 6px;
            height: 22px;
            overflow: hidden;
            margin-top: 4px;
        }
        .prob-bar-fill {
            height: 100%;
            border-radius: 6px;
            display: flex;
            align-items: center;
            padding-left: 10px;
            font-size: 0.82rem;
            font-weight: 700;
            color: white;
        }
        .win-bar  { background: linear-gradient(90deg, #1E3A5F, #2563EB); }
        .draw-bar { background: linear-gradient(90deg, #6B7280, #9CA3AF); }
        .lose-bar { background: linear-gradient(90deg, #C8102E, #EF4444); }

        /* Champion callout */
        .champion-card {
            background: linear-gradient(135deg, #1E3A5F 0%, #C8102E 100%);
            border-radius: 16px;
            padding: 24px 32px;
            color: white;
            text-align: center;
            margin: 1rem 0;
        }
        .champion-card h1 { font-size: 2rem; margin: 0; }
        .champion-card p  { opacity: 0.85; margin: 4px 0 0 0; }

        /* Bracket match row */
        .bracket-match {
            background: white;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 8px 16px;
            margin: 4px 0;
            font-size: 0.88rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .bracket-winner { font-weight: 700; color: #1E3A5F; }
        .bracket-loser  { color: #9CA3AF; }

        /* Sidebar nav */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1E3A5F 0%, #0F2035 100%);
        }
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] label { color: #CBD5E1 !important; }
        [data-testid="stSidebar"] .stRadio label { color: #E2E8F0 !important; font-size: 0.95rem; }

        /* General metric card */
        .metric-card {
            background: white;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 16px 14px;
            margin-bottom: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }

        /* Page title stripe */
        .page-title {
            background: linear-gradient(90deg, #1E3A5F 0%, #C8102E 100%);
            color: white;
            padding: 20px 28px;
            border-radius: 12px;
            margin-bottom: 1.5rem;
        }
        .page-title h2 { margin: 0; font-size: 1.6rem; }
        .page-title p  { margin: 4px 0 0; opacity: 0.85; font-size: 0.9rem; }
    </style>
    """, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"""
    <div class="page-title">
        <h2>⚽ {title}</h2>
        {"<p>" + subtitle + "</p>" if subtitle else ""}
    </div>
    """, unsafe_allow_html=True)
