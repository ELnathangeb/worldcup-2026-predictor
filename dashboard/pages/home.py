"""Page 1 — Home: Project overview, dataset stats, model summary."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from utils import (
    load_dataset, load_monte_carlo, load_elo, load_model_metadata,
    load_groups, page_header, COLORS,
)


def render() -> None:
    page_header(
        "FIFA World Cup 2026 — Prediction Dashboard",
        "Machine-learning match prediction · Tournament simulation · Championship odds",
    )

    # ── Top KPI strip ─────────────────────────────────────────────────────────
    df   = load_dataset()
    mc   = load_monte_carlo()
    meta = load_model_metadata()
    elo  = load_elo()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Matches in Dataset",  f"{len(df):,}")
    c2.metric("Teams Covered",       f"{len(elo):,}")
    c3.metric("WC 2026 Teams",       "48")
    c4.metric("Best Model",          meta.get("model_name", "LightGBM"))
    c5.metric("Model Accuracy",
              f"{meta.get('tune_results', {}).get('test_accuracy', 0)*100:.1f}%")

    st.markdown("---")

    # ── Two-column layout ─────────────────────────────────────────────────────
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown('<div class="section-header">About This Project</div>',
                    unsafe_allow_html=True)
        st.markdown("""
This dashboard presents a **production-quality machine learning system** for predicting
FIFA World Cup 2026 match outcomes and simulating the entire tournament.

**Pipeline overview:**
1. 🗂️ **Data** — 23,958 international matches (2000 – 2026) from Kaggle
2. 🔧 **Features** — Elo ratings, rolling form (5/10 games), FIFA rankings,
   head-to-head records, goal averages
3. 🤖 **Model** — 5 classifiers trained chronologically; LightGBM selected
   (F1-macro = 0.527 on 2024–2026 test set)
4. 🏆 **Simulation** — Full WC bracket with Elo-based neutral-venue probabilities
   and Poisson score generation; 10,000 Monte Carlo runs
        """)

        st.markdown('<div class="section-header">Top 5 Championship Favourites</div>',
                    unsafe_allow_html=True)
        top5 = mc.head(5)[["team", "pct_champion", "pct_semifinal", "pct_round_of_32"]]
        fig = px.bar(
            top5[::-1],
            x="pct_champion", y="team", orientation="h",
            color="pct_champion",
            color_continuous_scale=[[0, "#1E3A5F"], [0.5, "#C8102E"], [1, "#FFD700"]],
            text="pct_champion",
            labels={"pct_champion": "Championship %", "team": ""},
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(
            height=260, margin=dict(l=0, r=40, t=10, b=0),
            showlegend=False, coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(tickfont=dict(size=13, color="#1E3A5F")),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section-header">Model Performance</div>',
                    unsafe_allow_html=True)
        tune = meta.get("tune_results", {})
        st.metric("Test Accuracy",  f"{tune.get('test_accuracy', 0)*100:.1f}%")
        st.metric("F1 Macro",       f"{tune.get('test_f1_macro', 0):.4f}")
        st.metric("Log Loss",       f"{tune.get('test_log_loss', 0):.4f}")
        st.metric("CV F1 (tuning)", f"{tune.get('best_cv_f1', 0):.4f}")

        st.markdown('<div class="section-header">Dataset Split</div>',
                    unsafe_allow_html=True)
        split_df = pd.DataFrame({
            "Split":  ["Train", "Validation", "Test"],
            "Period": ["2000 – 2021", "2022 – 2023", "2024 – 2026"],
        })
        st.dataframe(split_df, hide_index=True, use_container_width=True)

        st.markdown('<div class="section-header">Match Outcome Distribution</div>',
                    unsafe_allow_html=True)
        vc = df["result"].value_counts().sort_index()
        labels = {0: "Away Win", 1: "Draw", 2: "Home Win"}
        fig2 = go.Figure(go.Pie(
            labels=[labels[k] for k in vc.index],
            values=vc.values,
            hole=0.5,
            marker_colors=[COLORS["secondary"], "#6B7280", COLORS["primary"]],
            textinfo="label+percent",
            textfont_size=12,
        ))
        fig2.update_layout(
            height=220, margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ── Groups at a glance ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">WC 2026 Groups</div>',
                unsafe_allow_html=True)
    groups = load_groups()
    elo_ratings = load_elo()

    cols = st.columns(6)
    for idx, (grp, teams) in enumerate(groups.items()):
        with cols[idx % 6]:
            team_lines = ""
            for t in teams:
                e = elo_ratings.get(t, 1500)
                bar_w = max(8, int((e - 1400) / 10))
                team_lines += f"""
                <div style='display:flex; align-items:center; gap:6px; margin:3px 0;'>
                    <div style='font-size:0.8rem; width:100px; white-space:nowrap;
                                overflow:hidden; text-overflow:ellipsis;
                                color:#1E3A5F; font-weight:600;'>{t}</div>
                    <div style='font-size:0.72rem; color:#6B7280;'>{e:.0f}</div>
                </div>"""
            st.markdown(f"""
            <div style='background:white; border:1px solid #E2E8F0; border-radius:10px;
                        padding:12px; margin-bottom:10px; box-shadow:0 1px 3px rgba(0,0,0,0.06);'>
                <div style='font-size:0.75rem; font-weight:800; color:#C8102E;
                            letter-spacing:0.1em; margin-bottom:8px;'>GROUP {grp}</div>
                {team_lines}
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.caption(
        "Data sources: martj42/international_results (Kaggle) · "
        "samuraitruong/fifa-ranking-data · "
        "Official WC2026 draw (Dec 2024)  |  "
        "Model: LightGBM · Simulation: Elo-based Poisson · N=10,000 Monte Carlo"
    )
