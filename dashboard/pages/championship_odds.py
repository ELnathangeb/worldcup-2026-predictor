"""Page 5 — Championship Odds: Monte Carlo results visualised."""
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

from utils import load_monte_carlo, load_elo, page_header, COLORS, CONF_COLORS, confederation_map


def render() -> None:
    page_header(
        "Championship Odds",
        "Monte Carlo simulation results — 10,000 full tournament runs",
    )

    mc   = load_monte_carlo()
    elo  = load_elo()
    conf = confederation_map()

    if mc.empty:
        st.error("Monte Carlo results not found at `outputs/monte_carlo_results.csv`. "
                 "Run `python src/monte_carlo.py` first.")
        return

    # Enrich with Elo and confederation
    mc["elo"]  = mc["team"].map(lambda t: elo.get(t, 1500))
    mc["conf"] = mc["team"].map(lambda t: conf.get(t, "OFC"))

    # ── Summary KPIs ─────────────────────────────────────────────────────────
    top1  = mc.iloc[0]
    top3  = mc.head(3)
    n_sim = 10_000  # stored assumption

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Simulations Run",     f"{n_sim:,}")
    k2.metric("Top Favourite",       top1["team"],    delta=f"{top1['pct_champion']:.1f}% chance")
    k3.metric("Avg Teams Past R32",  f"{(mc['pct_round_of_32'] > 0).sum()}")
    k4.metric("Avg Champion Odds",   f"{100/len(mc):.2f}% (uniform)")
    k5.metric("Most SF Appearances", mc.nlargest(1,"pct_semifinal").iloc[0]["team"],
              delta=f"{mc.nlargest(1,'pct_semifinal').iloc[0]['pct_semifinal']:.1f}% SF rate")

    st.markdown("---")

    # ── Main chart + table ────────────────────────────────────────────────────
    tabs = st.tabs([
        "🥇 Championship Odds", "🌍 All Stages", "🗺️ By Confederation",
        "📊 Elo vs Odds", "📋 Full Table",
    ])

    # ── Tab 1: Top N championship bar ─────────────────────────────────────────
    with tabs[0]:
        n_show = st.slider("Show top N teams", min_value=5, max_value=48, value=20, step=1)
        top_n = mc.head(n_show).copy()

        # Medal colouring: gold / silver / bronze / rest
        def _medal_color(idx):
            if idx == 0: return "#FFD700"
            if idx == 1: return "#C0C0C0"
            if idx == 2: return "#CD7F32"
            return COLORS["primary"]
        top_n["color"] = [_medal_color(i) for i in range(len(top_n))]

        fig = go.Figure(go.Bar(
            x=top_n["pct_champion"][::-1],
            y=top_n["team"][::-1],
            orientation="h",
            marker_color=top_n["color"][::-1].tolist(),
            text=[f"{v:.1f}%" for v in top_n["pct_champion"][::-1]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Championship: %{x:.2f}%<extra></extra>",
        ))
        fig.update_layout(
            height=max(340, n_show * 22),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=80, t=20, b=0),
            xaxis=dict(title="Championship Probability (%)", showgrid=True, gridcolor="#F0F0F0"),
            yaxis=dict(tickfont=dict(size=11, color="#1E3A5F")),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Podium cards
        st.markdown('<div class="section-header">Podium Favourites</div>',
                    unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        medals = ["🥇", "🥈", "🥉"]
        for col, medal, (_, row) in zip([p1, p2, p3], medals, mc.head(3).iterrows()):
            col.markdown(f"""
            <div class="metric-card" style='text-align:center;'>
                <div style='font-size:2.5rem;'>{medal}</div>
                <div style='font-size:1.3rem; font-weight:800; color:#1E3A5F;'>{row['team']}</div>
                <div style='font-size:1.6rem; font-weight:700; color:#C8102E; margin:4px 0;'>
                    {row['pct_champion']:.1f}%
                </div>
                <div style='font-size:0.78rem; color:#6B7280;'>
                    SF: {row['pct_semifinal']:.1f}% &nbsp;|&nbsp;
                    Final: {row['pct_final']:.1f}%
                </div>
                <div style='font-size:0.75rem; color:#9CA3AF; margin-top:4px;'>
                    Elo: {row['elo']:.0f}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Tab 2: All stages stacked bar ─────────────────────────────────────────
    with tabs[1]:
        top20 = mc.head(20).copy()
        stages = [
            ("pct_champion",     "Champion",     "#FFD700"),
            ("pct_final",        "Final",        "#C8102E"),
            ("pct_semifinal",    "Semifinal",    "#1D4ED8"),
            ("pct_quarterfinal", "Quarterfinal", "#3B82F6"),
            ("pct_round_of_16",  "Round of 16",  "#60A5FA"),
            ("pct_round_of_32",  "Round of 32",  "#BFDBFE"),
        ]
        fig2 = go.Figure()
        for col, label, color in stages:
            if col in top20.columns:
                fig2.add_trace(go.Bar(
                    name=label,
                    x=top20["team"],
                    y=top20[col],
                    marker_color=color,
                    hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.1f}}%<extra></extra>",
                ))
        fig2.update_layout(
            barmode="group",
            height=440,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(tickangle=-35, tickfont=dict(size=10)),
            yaxis=dict(title="Probability (%)", showgrid=True, gridcolor="#F0F0F0"),
            legend=dict(orientation="h", yanchor="bottom", y=1),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 3: Confederation view ──────────────────────────────────────────────
    with tabs[2]:
        conf_grp = mc.groupby("conf").agg(
            avg_champion=("pct_champion", "mean"),
            avg_sf=("pct_semifinal", "mean"),
            n_teams=("team", "count"),
            total_champion=("pct_champion", "sum"),
        ).reset_index().sort_values("total_champion", ascending=False)

        col_l, col_r = st.columns(2)
        with col_l:
            fig_c = px.bar(
                conf_grp, x="conf", y="total_champion",
                color="conf",
                color_discrete_map={k: v for k, v in CONF_COLORS.items()},
                title="Total Championship Share by Confederation",
                text="total_champion",
                labels={"total_champion": "Total Champ %", "conf": "Confederation"},
            )
            fig_c.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_c.update_layout(
                height=340, showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_c, use_container_width=True)

        with col_r:
            fig_pie = go.Figure(go.Pie(
                labels=conf_grp["conf"],
                values=conf_grp["total_champion"],
                hole=0.45,
                textinfo="label+percent",
                marker_colors=[CONF_COLORS.get(c, "#9CA3AF") for c in conf_grp["conf"]],
            ))
            fig_pie.update_layout(
                title="Championship Share (Pie)",
                height=340, showlegend=False,
                margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Per-confederation top teams
        st.markdown('<div class="section-header">Top Team per Confederation</div>',
                    unsafe_allow_html=True)
        cols_conf = st.columns(len(conf_grp))
        for i, (_, row) in enumerate(conf_grp.iterrows()):
            best = mc[mc["conf"] == row["conf"]].head(1).iloc[0]
            conf_color = CONF_COLORS.get(row["conf"], "#1E3A5F")
            with cols_conf[i]:
                st.markdown(f"""
                <div class="metric-card" style='border-top:3px solid {conf_color};'>
                    <div style='font-size:0.75rem; font-weight:700; color:{conf_color};
                                letter-spacing:0.05em;'>{row['conf']}</div>
                    <div style='font-size:1rem; font-weight:800; color:#1E3A5F;
                                margin:4px 0;'>{best['team']}</div>
                    <div style='font-size:1.2rem; color:#C8102E; font-weight:700;'>
                        {best['pct_champion']:.1f}%
                    </div>
                    <div style='font-size:0.7rem; color:#9CA3AF;'>{int(row['n_teams'])} teams</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Tab 4: Elo vs champion odds scatter ───────────────────────────────────
    with tabs[3]:
        fig_s = px.scatter(
            mc, x="elo", y="pct_champion",
            size="pct_semifinal",
            color="conf",
            color_discrete_map={k: v for k, v in CONF_COLORS.items()},
            hover_name="team",
            title="Elo Rating vs Championship Probability",
            labels={"elo": "Elo Rating", "pct_champion": "Championship %", "conf": "Confederation"},
            trendline="ols",
        )
        fig_s.update_layout(
            height=480,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_s, use_container_width=True)
        st.caption("Bubble size = semifinal qualification rate. Trendline = OLS regression.")

        # Biggest over/under performers vs Elo rank
        mc_sorted_elo  = mc.sort_values("elo", ascending=False).reset_index(drop=True)
        mc_sorted_odds = mc.sort_values("pct_champion", ascending=False).reset_index(drop=True)
        mc_sorted_elo["elo_rank"]   = mc_sorted_elo.index + 1
        mc_sorted_odds["odds_rank"] = mc_sorted_odds.index + 1
        merged = mc_sorted_elo.merge(mc_sorted_odds[["team","odds_rank"]], on="team")
        merged["rank_diff"] = merged["elo_rank"] - merged["odds_rank"]
        overperformers  = merged.nlargest(5, "rank_diff")[["team","elo_rank","odds_rank","rank_diff"]]
        underperformers = merged.nsmallest(5, "rank_diff")[["team","elo_rank","odds_rank","rank_diff"]]

        col_o, col_u = st.columns(2)
        with col_o:
            st.markdown("**📈 Biggest Over-performers vs Elo**")
            overperformers.columns = ["Team","Elo Rank","Odds Rank","Rank Jump"]
            st.dataframe(overperformers, hide_index=True, use_container_width=True)
        with col_u:
            st.markdown("**📉 Biggest Under-performers vs Elo**")
            underperformers.columns = ["Team","Elo Rank","Odds Rank","Rank Drop"]
            underperformers["Rank Drop"] = underperformers["Rank Drop"].abs()
            st.dataframe(underperformers, hide_index=True, use_container_width=True)

    # ── Tab 5: Full table ──────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown("**Full Monte Carlo Results — all 48 WC teams**")
        display_cols = ["team","conf","elo","pct_champion","pct_final","pct_runner_up",
                        "pct_semifinal","pct_quarterfinal","pct_round_of_16","pct_round_of_32"]
        existing = [c for c in display_cols if c in mc.columns]
        show_mc = mc[existing].copy()
        show_mc.columns = [c.replace("pct_","").replace("_"," ").title() for c in existing]

        def highlight_champ(val):
            if isinstance(val, float) and val >= 5.0:
                return "background-color:#FEF9C3; font-weight:700"
            if isinstance(val, float) and val >= 2.0:
                return "background-color:#FFF7ED"
            return ""

        st.dataframe(
            show_mc.style.applymap(highlight_champ, subset=["Champion"]) if "Champion" in show_mc.columns
            else show_mc,
            hide_index=True, use_container_width=True, height=540,
        )
        st.caption(f"Sorted by championship probability. {n_sim:,} simulations run. "
                   "All percentages = % of simulations where team reached that stage.")
