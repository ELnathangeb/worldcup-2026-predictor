"""
FIFA World Cup 2026 — Prediction Dashboard
Entry point: streamlit run dashboard/streamlit_app.py
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="WC2026 Prediction Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils import inject_css
inject_css()

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px;'>
        <div style='font-size:2.4rem;'>⚽</div>
        <div style='color:white; font-size:1.1rem; font-weight:700; line-height:1.3;'>
            WC 2026<br>Prediction Model
        </div>
        <div style='color:#94A3B8; font-size:0.75rem; margin-top:4px;'>
            Powered by LightGBM + Elo
        </div>
    </div>
    <hr style='border-color:#2D4A6B; margin:12px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🏠  Home",
            "🌍  Team Explorer",
            "⚡  Match Predictor",
            "🏆  Tournament Simulator",
            "📊  Championship Odds",
            "🔍  Data Insights",
        ],
        label_visibility="collapsed",
    )

    st.markdown("""
    <hr style='border-color:#2D4A6B; margin:12px 0;'>
    <div style='color:#64748B; font-size:0.72rem; text-align:center; padding-bottom:8px;'>
        FIFA World Cup 2026<br>USA · Canada · Mexico<br>June – July 2026
    </div>
    """, unsafe_allow_html=True)

# ── Route to page ─────────────────────────────────────────────────────────────
page_key = page.split("  ", 1)[1].strip()

if page_key == "Home":
    from pages.home import render
elif page_key == "Team Explorer":
    from pages.team_explorer import render
elif page_key == "Match Predictor":
    from pages.match_predictor import render
elif page_key == "Tournament Simulator":
    from pages.tournament_simulator import render
elif page_key == "Championship Odds":
    from pages.championship_odds import render
else:
    from pages.data_insights import render

render()
