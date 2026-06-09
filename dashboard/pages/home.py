"""Home — overview, favourites leaderboard, groups grid."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from utils import (
    load_dataset, load_monte_carlo, load_elo, load_model_metadata,
    load_groups, page_header, section_title, kpi, COLORS, CHART, CONF_COLORS,
    confederation_map,
)


def render() -> None:
    page_header(
        "FIFA World Cup 2026 — Analytics Platform",
        "Machine learning · Elo simulation · 10,000 Monte Carlo runs",
    )

    df   = load_dataset()
    mc   = load_monte_carlo()
    meta = load_model_metadata()
    elo  = load_elo()
    tune = meta.get("tune_results", {})

    # ── KPI strip ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(kpi(f"{len(df):,}", "Historical Matches", accent="#38BDF8"), unsafe_allow_html=True)
    c2.markdown(kpi("48",            "WC 2026 Teams",      accent="#34D399"), unsafe_allow_html=True)
    c3.markdown(kpi("46",            "ML Features",        accent="#A78BFA"), unsafe_allow_html=True)
    c4.markdown(kpi(f"{tune.get('test_accuracy',0)*100:.1f}%", "Model Accuracy",
                    delta="vs 33% random baseline", accent="#FCD34D"), unsafe_allow_html=True)
    c5.markdown(kpi("10,000",        "MC Simulations",     accent="#F87171"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two-column main layout ─────────────────────────────────────────────────
    left, right = st.columns([3, 2], gap="large")

    with left:
        # Championship leaderboard
        section_title("Championship Favourites", "Probability of winning the 2026 World Cup")

        conf_map = confederation_map()
        conf_css = {
            "UEFA":"#38BDF8","CONMEBOL":"#34D399","CONCACAF":"#FCD34D",
            "CAF":"#F87171","AFC":"#A78BFA","OFC":"#FB923C",
        }
        rank_classes = {1:"gold", 2:"silver", 3:"bronze"}

        rows_html = ""
        for i, (_, row) in enumerate(mc.head(12).iterrows(), 1):
            rc = rank_classes.get(i, "")
            cc = conf_css.get(conf_map.get(row["team"], ""), "#475569")
            bar_w = max(2, int(row["pct_champion"] / mc.iloc[0]["pct_champion"] * 100))
            flag  = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else ""))
            rows_html += f"""
            <div class='lb-row' style='position:relative;'>
                <div class='lb-rank {rc}'>{flag or i}</div>
                <div class='lb-team'>{row['team']}</div>
                <div style='flex:2;padding:0 12px;'>
                    <div style='background:rgba(255,255,255,0.05);border-radius:4px;height:6px;'>
                        <div style='width:{bar_w}%;height:100%;border-radius:4px;
                                    background:linear-gradient(90deg,#0EA5E9aa,#38BDF8);
                                    box-shadow:0 0 6px #38BDF866;'></div>
                    </div>
                </div>
                <div style='display:flex;align-items:center;gap:10px;'>
                    <div class='lb-value'>{row['pct_champion']:.1f}%</div>
                    <div style='font-size:0.65rem;color:#475569;min-width:34px;
                                text-align:right;'>{row['pct_semifinal']:.0f}% SF</div>
                    <div class='conf-badge' style='background:{cc}22;color:{cc};
                                                   border:1px solid {cc}44;'>
                        {conf_map.get(row['team'],'')[:4]}
                    </div>
                </div>
            </div>"""

        st.markdown(f"""
        <div class='glass-card' style='padding:8px 0;'>
            <div style='display:flex;padding:6px 16px 4px;
                        font-size:0.62rem;font-weight:700;color:#334155;
                        text-transform:uppercase;letter-spacing:0.1em;'>
                <div style='width:24px;text-align:center;'>#</div>
                <div style='flex:1;margin-left:14px;'>Team</div>
                <div style='flex:2;padding:0 12px;'>Odds</div>
                <div style='min-width:200px;text-align:right;'>Champion % · SF · Conf</div>
            </div>
            {rows_html}
        </div>""", unsafe_allow_html=True)

    with right:
        # Model card
        section_title("Model Performance")
        st.markdown(f"""
        <div class='glass-card'>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;'>
                <div class='stat-pill'>
                    <div class='val'>{tune.get('test_accuracy',0)*100:.1f}%</div>
                    <div class='lbl'>Test Accuracy</div>
                </div>
                <div class='stat-pill'>
                    <div class='val'>{tune.get('test_f1_macro',0):.3f}</div>
                    <div class='lbl'>F1 Macro</div>
                </div>
                <div class='stat-pill'>
                    <div class='val'>{tune.get('test_log_loss',0):.3f}</div>
                    <div class='lbl'>Log Loss</div>
                </div>
                <div class='stat-pill'>
                    <div class='val'>46</div>
                    <div class='lbl'>Features</div>
                </div>
            </div>
            <div style='margin-top:16px;padding-top:14px;border-top:1px solid #1C2E4A;'>
                <div style='font-size:0.68rem;font-weight:700;color:#334155;
                            text-transform:uppercase;letter-spacing:0.1em;
                            margin-bottom:8px;'>Training Split</div>
                <div style='font-size:0.8rem;color:#64748B;line-height:2;'>
                    <span style='color:#34D399;font-weight:600;'>Train</span>
                    &nbsp;2000 – 2021<br>
                    <span style='color:#FCD34D;font-weight:600;'>Validate</span>
                    &nbsp;2022 – 2023<br>
                    <span style='color:#F87171;font-weight:600;'>Test</span>
                    &nbsp;2024 – 2026
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Outcome distribution donut
        section_title("Historical Outcomes")
        vc = df["result"].value_counts().sort_index()
        fig = go.Figure(go.Pie(
            labels=["Away Win", "Draw", "Home Win"],
            values=[vc.get(0, 0), vc.get(1, 0), vc.get(2, 0)],
            hole=0.6,
            marker_colors=["#F87171", "#475569", "#38BDF8"],
            textinfo="percent",
            textfont=dict(size=11, color="white"),
            hovertemplate="<b>%{label}</b><br>%{value:,} matches<br>%{percent}<extra></extra>",
        ))
        fig.add_annotation(
            text=f"<b>{len(df):,}</b><br><span style='font-size:10px'>matches</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#94A3B8"),
        )
        fig.update_layout(
            **{**CHART, "height": 220, "showlegend": True,
               "legend": dict(orientation="h", yanchor="bottom", y=-0.15,
                              font=dict(size=10, color="#64748B"), bgcolor="rgba(0,0,0,0)"),
               "margin": dict(l=0, r=0, t=10, b=0)},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Groups grid ────────────────────────────────────────────────────────────
    section_title("WC 2026 Groups", "Official draw — December 2024  ·  48 teams  ·  12 groups")

    groups   = load_groups()
    elo_all  = load_elo()
    conf_map = confederation_map()

    # Sort groups by Elo sum for visual interest
    group_items = sorted(
        groups.items(),
        key=lambda kv: sum(elo_all.get(t, 1500) for t in kv[1]),
        reverse=True,
    )

    rows = [group_items[i:i+4] for i in range(0, 12, 4)]
    for row in rows:
        cols = st.columns(4)
        for col, (grp, teams) in zip(cols, row):
            teams_sorted = sorted(teams, key=lambda t: elo_all.get(t, 1500), reverse=True)
            rows_html = ""
            for rank_in_grp, t in enumerate(teams_sorted):
                e   = elo_all.get(t, 1500)
                cc  = conf_css.get(conf_map.get(t, ""), "#475569")
                dot = f"<span style='color:{cc};font-size:0.5rem;'>●</span>"
                rows_html += f"""
                <div class='grp-row'>
                    <span class='grp-name'>{dot} {t}</span>
                    <span style='font-size:0.7rem;color:#475569;'>{e:.0f}</span>
                </div>"""
            col.markdown(f"""
            <div class='glass-card' style='padding:0;overflow:hidden;margin-bottom:8px;'>
                <div class='grp-header'>Group {grp}</div>
                {rows_html}
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center;padding:16px 0 4px;
                font-size:0.72rem;color:#334155;'>
        Data: martj42/international-football-results · samuraitruong/fifa-ranking-data ·
        LightGBM · Elo · Poisson simulation
    </div>""", unsafe_allow_html=True)
