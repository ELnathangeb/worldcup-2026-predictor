"""Championship Odds — Monte Carlo leaderboard and charts."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from utils import (
    load_monte_carlo, load_elo, page_header, section_title, kpi,
    COLORS, CHART, CONF_COLORS, confederation_map,
)

_CONF_CSS = {
    "UEFA":"#38BDF8","CONMEBOL":"#34D399","CONCACAF":"#FCD34D",
    "CAF":"#F87171","AFC":"#A78BFA","OFC":"#FB923C",
}


def render() -> None:
    page_header("Championship Odds", "10,000 independent full-tournament Monte Carlo simulations")

    mc   = load_monte_carlo()
    elo  = load_elo()
    conf = confederation_map()

    if mc.empty:
        st.error("Run `python src/monte_carlo.py` first to generate results.")
        return

    mc = mc.copy()
    mc["elo"]  = mc["team"].map(lambda t: elo.get(t, 1500))
    mc["conf"] = mc["team"].map(lambda t: conf.get(t, "OFC"))
    mc["pct_final"] = mc.get("pct_final", mc.get("pct_runner_up", 0.0))

    top1 = mc.iloc[0]

    # ── KPI strip ──────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.html(kpi("10,000",         "Simulations",     accent="#38BDF8"))
    k2.html(kpi(top1["team"],      "Top Favourite",   delta=f"{top1['pct_champion']:.1f}% odds",
                    accent="#FCD34D"))
    k3.html(kpi(f"{top1['pct_semifinal']:.0f}%", f"{top1['team'][:10]} SF Rate",
                    accent="#A78BFA"))
    k4.html(kpi(f"{100/48:.1f}%", "Uniform Baseline", delta="per team if equal",
                    accent="#475569"))
    k5.html(kpi(str(len(mc)),     "WC Teams",        accent="#34D399"))

    st.html("<br>")

    tabs = st.tabs([
        "🥇 Leaderboard", "📊 All Stages", "🗺️ Confederation", "📈 Elo vs Odds", "📋 Full Table"
    ])

    # ── Tab 1: Leaderboard ─────────────────────────────────────────────────────
    with tabs[0]:
        col_main, col_podium = st.columns([3, 2], gap="large")

        with col_main:
            section_title("Championship Probability Ranking")
            n_show = st.slider("Show top N teams", 5, 48, 20, key="odds_n")
            top_n  = mc.head(n_show)
            max_pct = top_n["pct_champion"].max()

            rank_meta = {1:("🥇","#FCD34D"), 2:("🥈","#94A3B8"), 3:("🥉","#FB923C")}

            rows_html = ""
            for i, (_, row) in enumerate(top_n.iterrows(), 1):
                icon, hl = rank_meta.get(i, ("", "#1C2E4A"))
                cc       = _CONF_CSS.get(row["conf"], "#475569")
                bar_w    = max(2, int(row["pct_champion"] / max_pct * 100))
                rows_html += f"""
                <div class='lb-row'>
                    <div class='lb-rank {"gold" if i==1 else "silver" if i==2 else "bronze" if i==3 else ""}'>
                        {icon or i}
                    </div>
                    <div class='lb-team'>{row['team']}</div>
                    <div style='flex:2;padding:0 12px;'>
                        <div style='background:rgba(255,255,255,0.05);border-radius:4px;height:6px;'>
                            <div style='width:{bar_w}%;height:100%;border-radius:4px;
                                        background:linear-gradient(90deg,{cc}88,{cc});
                                        box-shadow:0 0 6px {cc}44;'></div>
                        </div>
                    </div>
                    <div style='display:flex;align-items:center;gap:10px;min-width:180px;
                                justify-content:flex-end;'>
                        <span class='lb-value' style='color:{cc};'>{row['pct_champion']:.1f}%</span>
                        <span style='font-size:0.68rem;color:#334155;min-width:52px;'>
                            {row['pct_semifinal']:.0f}% SF
                        </span>
                        <span class='conf-badge' style='background:{cc}22;color:{cc};
                                                        border:1px solid {cc}44;font-size:0.6rem;'>
                            {row['conf'][:4]}
                        </span>
                    </div>
                </div>"""

            st.html(f"""
            <div class='glass-card' style='padding:8px 0;'>
                <div style='display:flex;padding:6px 16px 6px;font-size:0.6rem;font-weight:700;
                            color:#334155;text-transform:uppercase;letter-spacing:0.1em;'>
                    <div style='width:24px;text-align:center;'>#</div>
                    <div style='flex:1;margin-left:14px;'>Team</div>
                    <div style='flex:2;padding:0 12px;'>Odds bar</div>
                    <div style='min-width:180px;text-align:right;'>Champion · SF · Conf</div>
                </div>
                {rows_html}
            </div>""")

        with col_podium:
            section_title("Podium")
            medals = [
                ("🥇", mc.iloc[0], "#FCD34D", "Champion Favourite"),
                ("🥈", mc.iloc[1], "#94A3B8", "2nd Favourite"),
                ("🥉", mc.iloc[2], "#FB923C", "3rd Favourite"),
            ]
            for medal, row, color, sublabel in medals:
                st.html(f"""
                <div style='
                    background:linear-gradient(135deg,rgba(255,255,255,0.03),rgba(255,255,255,0.01));
                    border:1px solid {color}44;border-top:2px solid {color};
                    border-radius:14px;padding:18px 20px;margin-bottom:10px;
                    box-shadow:0 0 24px {color}11;
                '>
                    <div style='display:flex;align-items:center;gap:14px;'>
                        <div style='font-size:2rem;'>{medal}</div>
                        <div style='flex:1;'>
                            <div style='font-size:1.05rem;font-weight:800;color:#F1F5F9;'>{row["team"]}</div>
                            <div style='font-size:0.7rem;color:#475569;margin-top:2px;'>{sublabel}</div>
                        </div>
                        <div style='text-align:right;'>
                            <div style='font-size:1.6rem;font-weight:900;color:{color};'>{row["pct_champion"]:.1f}%</div>
                            <div style='font-size:0.68rem;color:#334155;'>SF: {row["pct_semifinal"]:.0f}%</div>
                        </div>
                    </div>
                    <div style='margin-top:10px;padding-top:8px;border-top:1px solid #1C2E4A;
                                display:flex;gap:8px;flex-wrap:wrap;'>
                        <span style='font-size:0.7rem;color:#475569;'>
                            Elo: <span style='color:#64748B;font-weight:600;'>{row["elo"]:.0f}</span>
                        </span>
                        <span style='font-size:0.7rem;color:#475569;'>
                            Final: <span style='color:#64748B;font-weight:600;'>{row.get("pct_final", row.get("pct_runner_up",0)):.0f}%</span>
                        </span>
                        <span style='font-size:0.7rem;color:#475569;'>
                            R32: <span style='color:#64748B;font-weight:600;'>{row["pct_round_of_32"]:.0f}%</span>
                        </span>
                    </div>
                </div>""")

    # ── Tab 2: All stages ──────────────────────────────────────────────────────
    with tabs[1]:
        section_title("All Stages — Top 20 Teams")
        top20 = mc.head(20)
        stages_cfg = [
            ("pct_champion",     "Champion",    "#FCD34D"),
            ("pct_final",        "Final",       "#94A3B8"),
            ("pct_semifinal",    "Semifinal",   "#A78BFA"),
            ("pct_quarterfinal", "QF",          "#38BDF8"),
            ("pct_round_of_16",  "R16",         "#34D399"),
            ("pct_round_of_32",  "R32",         "#1C4A2E"),
        ]
        fig2 = go.Figure()
        for col, label, color in stages_cfg:
            if col in top20.columns:
                fig2.add_trace(go.Bar(
                    name=label, x=top20["team"], y=top20[col],
                    marker=dict(color=color, line=dict(width=0)),
                    hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.1f}}%<extra></extra>",
                ))
        fig2.update_layout(**{**CHART, "height": 420, "barmode": "group",
                               "xaxis": dict(tickangle=-35, tickfont=dict(size=9)),
                               "yaxis": dict(title="Probability (%)", gridcolor="rgba(255,255,255,0.04)"),
                               "legend": dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)")})
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 3: Confederation ──────────────────────────────────────────────────
    with tabs[2]:
        conf_grp = mc.groupby("conf").agg(
            total_champion=("pct_champion","sum"),
            avg_sf=("pct_semifinal","mean"),
            n_teams=("team","count"),
        ).reset_index().sort_values("total_champion", ascending=False)

        col_l, col_r = st.columns(2)
        with col_l:
            section_title("Total Championship Share")
            fig_cb = go.Figure(go.Bar(
                x=conf_grp["conf"],
                y=conf_grp["total_champion"],
                marker=dict(
                    color=[_CONF_CSS.get(c,"#475569") for c in conf_grp["conf"]],
                    line=dict(width=0),
                ),
                text=[f"{v:.1f}%" for v in conf_grp["total_champion"]],
                textposition="outside", textfont=dict(color="#64748B"),
                hovertemplate="<b>%{x}</b><br>%{y:.1f}% total share<extra></extra>",
            ))
            fig_cb.update_layout(**{**CHART, "height": 300, "showlegend": False,
                                    "yaxis": dict(title="% share", gridcolor="rgba(255,255,255,0.04)"),
                                    "xaxis": dict(tickfont=dict(size=12, color="#64748B"))})
            st.plotly_chart(fig_cb, use_container_width=True)

        with col_r:
            section_title("Share by Confederation")
            fig_dp = go.Figure(go.Pie(
                labels=conf_grp["conf"],
                values=conf_grp["total_champion"],
                hole=0.5,
                marker_colors=[_CONF_CSS.get(c,"#475569") for c in conf_grp["conf"]],
                textinfo="label+percent",
                textfont=dict(size=11, color="white"),
                hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
            ))
            fig_dp.update_layout(**{**CHART, "height": 300, "showlegend": False})
            st.plotly_chart(fig_dp, use_container_width=True)

        # Per-conf best team
        section_title("Top Team per Confederation")
        cols_c = st.columns(len(conf_grp))
        for i, (_, row) in enumerate(conf_grp.iterrows()):
            best = mc[mc["conf"] == row["conf"]].head(1).iloc[0]
            cc   = _CONF_CSS.get(row["conf"], "#475569")
            with cols_c[i]:
                st.html(f"""
                <div style='background:rgba(255,255,255,0.02);border:1px solid {cc}33;
                            border-top:2px solid {cc};border-radius:12px;
                            padding:14px;text-align:center;'>
                    <div style='font-size:0.62rem;font-weight:700;color:{cc};
                                letter-spacing:0.1em;'>{row['conf']}</div>
                    <div style='font-size:0.9rem;font-weight:800;color:#F1F5F9;
                                margin:6px 0 2px;'>{best['team']}</div>
                    <div style='font-size:1.2rem;color:{cc};font-weight:900;'>
                        {best['pct_champion']:.1f}%
                    </div>
                    <div style='font-size:0.65rem;color:#334155;margin-top:4px;'>
                        {int(row['n_teams'])} teams
                    </div>
                </div>""")

    # ── Tab 4: Elo vs Odds ────────────────────────────────────────────────────
    with tabs[3]:
        section_title("Elo Rating vs Championship Probability")
        fig_sc = px.scatter(
            mc, x="elo", y="pct_champion",
            size="pct_semifinal",
            color="conf",
            color_discrete_map={k: v for k, v in _CONF_CSS.items()},
            hover_name="team",
            labels={"elo":"Elo Rating","pct_champion":"Champion %","conf":"Confederation"},
            trendline="ols",
        )
        fig_sc.update_traces(
            marker=dict(line=dict(width=0)),
            selector=dict(mode="markers"),
        )
        fig_sc.update_layout(**{**CHART, "height": 460,
                                 "legend": dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)")})
        st.plotly_chart(fig_sc, use_container_width=True)
        st.caption("Bubble size = semifinal qualification rate.  OLS trendline shown.")

    # ── Tab 5: Full table ──────────────────────────────────────────────────────
    with tabs[4]:
        section_title("Complete Results — All 48 WC Teams")
        cols_show = ["team","conf","elo","pct_champion","pct_final","pct_runner_up",
                     "pct_semifinal","pct_quarterfinal","pct_round_of_16","pct_round_of_32"]
        existing  = [c for c in cols_show if c in mc.columns]
        show      = mc[existing].copy()
        show.columns = [c.replace("pct_","").replace("_"," ").title() for c in existing]

        def _hl(val):
            if isinstance(val, float) and val >= 5.0:
                return "background-color:#1a1400;color:#FCD34D;font-weight:700"
            if isinstance(val, float) and val >= 2.0:
                return "background-color:#130f20;color:#A78BFA"
            return ""

        st.dataframe(
            show.style.applymap(_hl, subset=["Champion"] if "Champion" in show.columns else []),
            hide_index=True, use_container_width=True, height=560,
        )
        st.caption("Sorted by championship probability.  10,000 simulations.  "
                   "All % = share of simulations where team reached that stage.")