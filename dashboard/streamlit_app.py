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
    page_title="WC2026 Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils import inject_css
inject_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────
NAV_ITEMS = [
    ("🏠", "Home",                "Overview & favourites"),
    ("🌍", "Team Explorer",       "Form, stats & H2H"),
    ("⚡", "Match Predictor",     "ML + Elo probabilities"),
    ("🏆", "Tournament Simulator","Full bracket simulation"),
    ("📊", "Championship Odds",   "10k Monte Carlo results"),
    ("🔍", "Data Insights",       "Model & dataset analysis"),
]

with st.sidebar:
    # Logo block
    st.markdown("""
    <div style='padding:18px 12px 12px;'>
        <div style='display:flex;align-items:center;gap:12px;'>
            <div style='width:42px;height:42px;background:linear-gradient(135deg,#0EA5E9,#7C3AED);
                        border-radius:10px;display:flex;align-items:center;justify-content:center;
                        font-size:1.3rem;box-shadow:0 4px 16px rgba(14,165,233,0.4);'>⚽</div>
            <div>
                <div style='font-size:0.95rem;font-weight:800;color:#F1F5F9;
                            letter-spacing:-0.3px;line-height:1.2;'>WC 2026</div>
                <div style='font-size:0.68rem;color:#475569;font-weight:500;
                            letter-spacing:0.05em;'>ANALYTICS PLATFORM</div>
            </div>
        </div>
    </div>
    <hr/>
    <div style='padding:4px 12px 8px;'>
        <div style='font-size:0.62rem;font-weight:700;color:#334155;
                    text-transform:uppercase;letter-spacing:0.12em;
                    margin-bottom:8px;'>Navigation</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav",
        [f"{icon}  {label}" for icon, label, _ in NAV_ITEMS],
        label_visibility="collapsed",
    )

    # Description of selected page
    selected_idx = next(
        i for i, (icon, label, _) in enumerate(NAV_ITEMS)
        if f"{icon}  {label}" == page
    )
    _, _, desc = NAV_ITEMS[selected_idx]
    st.markdown(f"""
    <div style='margin:4px 12px 0;padding:8px 12px;background:rgba(56,189,248,0.06);
                border-radius:8px;border:1px solid rgba(56,189,248,0.15);'>
        <span style='font-size:0.72rem;color:#38BDF8;'>{desc}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<hr/>""", unsafe_allow_html=True)

    # Bottom info block
    st.markdown("""
    <div style='padding:8px 12px 4px;'>
        <div style='font-size:0.68rem;color:#334155;line-height:1.7;'>
            <div style='color:#475569;font-weight:600;margin-bottom:4px;'>Stack</div>
            LightGBM · Elo · Poisson<br>
            Streamlit · Plotly
        </div>
        <div style='margin-top:10px;font-size:0.65rem;color:#1E3A5F;'>
            USA · Canada · Mexico · 2026
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Route ─────────────────────────────────────────────────────────────────────
page_label = page.split("  ", 1)[1].strip()

if page_label == "Home":
    from pages.home import render
elif page_label == "Team Explorer":
    from pages.team_explorer import render
elif page_label == "Match Predictor":
    from pages.match_predictor import render
elif page_label == "Tournament Simulator":
    from pages.tournament_simulator import render
elif page_label == "Championship Odds":
    from pages.championship_odds import render
else:
    from pages.data_insights import render

render()
