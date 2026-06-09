"""Page 2 — Team Explorer: stats, form, and trends for any national team."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from utils import (
    load_dataset, load_elo, load_monte_carlo,
    all_wc_teams, team_stats, team_group,
    page_header, COLORS, confederation_map,
)


def render() -> None:
    page_header("Team Explorer", "Analyse any WC 2026 team — form, goals, rankings, and more")

    df  = load_dataset()
    elo = load_elo()
    mc  = load_monte_carlo()

    # ── Team selector ─────────────────────────────────────────────────────────
    wc_teams = all_wc_teams()
    # All teams in dataset (sorted by Elo)
    all_ds_teams = sorted(
        set(df["home_team"].unique()) | set(df["away_team"].unique()),
        key=lambda t: elo.get(t, 0), reverse=True
    )

    col_sel, col_vs = st.columns([2, 3])
    with col_sel:
        team = st.selectbox(
            "Select a national team",
            options=wc_teams + ["─── All teams ───"] + [t for t in all_ds_teams if t not in wc_teams],
            index=0,
        )
        if "───" in team:
            st.info("Please select a team above.")
            return

    stats = team_stats(team)
    if not stats:
        st.warning(f"No historical data found for **{team}**.")
        return

    # ── KPI strip ─────────────────────────────────────────────────────────────
    conf_map = confederation_map()
    conf     = conf_map.get(team, "OFC")
    grp      = stats["group"]
    is_wc    = team in wc_teams

    st.markdown(f"""
    <div style='display:flex; align-items:center; gap:16px; margin:8px 0 16px;'>
        <div style='background:linear-gradient(135deg,#1E3A5F,#2563EB);
                    color:white; padding:8px 20px; border-radius:24px;
                    font-size:1.4rem; font-weight:800;'>{team}</div>
        {"<span style='background:#C8102E;color:white;padding:4px 12px;border-radius:16px;font-size:0.8rem;font-weight:700;'>WC 2026 · Group " + grp + "</span>" if is_wc else ""}
        <span style='background:#E2E8F0;color:#374151;padding:4px 12px;border-radius:16px;font-size:0.8rem;font-weight:600;'>{conf}</span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Elo Rating",    f"{stats['elo']:.0f}")
    c2.metric("FIFA Rank",     f"#{stats['rank']:.0f}" if stats["rank"] else "N/A")
    c3.metric("Win %",         f"{stats['win_pct']}%")
    c4.metric("Goals / Game",  f"{stats['goals_per_game']}")
    c5.metric("Total Matches", f"{stats['total_matches']}")
    c6.metric("Goal Diff",     f"{stats['goal_diff']:+d}")

    # MC row (if WC team)
    mc_row = mc[mc["team"] == team]
    if len(mc_row):
        r = mc_row.iloc[0]
        st.markdown('<div class="section-header">Monte Carlo Championship Odds</div>',
                    unsafe_allow_html=True)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Champion",    f"{r['pct_champion']:.1f}%")
        m2.metric("Runner-Up",   f"{r['pct_runner_up']:.1f}%")
        m3.metric("Semifinal",   f"{r['pct_semifinal']:.1f}%")
        m4.metric("Quarterfinal",f"{r['pct_quarterfinal']:.1f}%")
        m5.metric("R32 (Advance)",f"{r['pct_round_of_32']:.1f}%")

    st.markdown("---")

    # ── Match history for this team ───────────────────────────────────────────
    home_df = df[df["home_team"] == team].copy()
    home_df["team_score"] = home_df["home_score"]
    home_df["opp_score"]  = home_df["away_score"]
    home_df["opponent"]   = home_df["away_team"]
    home_df["venue"]      = "Home"
    home_df["result_label"] = home_df["result"].map({2:"Win", 1:"Draw", 0:"Loss"})

    away_df = df[df["away_team"] == team].copy()
    away_df["team_score"] = away_df["away_score"]
    away_df["opp_score"]  = away_df["home_score"]
    away_df["opponent"]   = away_df["home_team"]
    away_df["venue"]      = "Away"
    away_df["result_label"] = away_df["result"].map({0:"Win", 1:"Draw", 2:"Loss"})

    hist = pd.concat([home_df, away_df]).sort_values("date").reset_index(drop=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Form & Goals", "🏅 W/D/L Breakdown", "📋 Match History", "🆚 Head-to-Head"])

    # ── Tab 1: Form & Goals ───────────────────────────────────────────────────
    with tab1:
        # Rolling 12-match goal trend + form
        hist["rolling_gf"] = hist["team_score"].rolling(12, min_periods=3).mean()
        hist["rolling_ga"] = hist["opp_score"].rolling(12, min_periods=3).mean()

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=("Goals Scored & Conceded (12-match rolling avg)", "Result per Match"),
            vertical_spacing=0.12, row_heights=[0.65, 0.35],
        )
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["rolling_gf"],
            name="Goals Scored", line=dict(color=COLORS["accent"], width=2.5),
            fill="tozeroy", fillcolor="rgba(0,166,81,0.12)",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["rolling_ga"],
            name="Goals Conceded", line=dict(color=COLORS["secondary"], width=2),
            fill="tozeroy", fillcolor="rgba(200,16,46,0.08)",
        ), row=1, col=1)

        color_map = {"Win": COLORS["accent"], "Draw": "#9CA3AF", "Loss": COLORS["secondary"]}
        for label, color in color_map.items():
            sub = hist[hist["result_label"] == label]
            fig.add_trace(go.Scatter(
                x=sub["date"], y=[label]*len(sub),
                mode="markers",
                marker=dict(color=color, size=7, symbol="circle"),
                name=label, showlegend=True,
            ), row=2, col=1)

        fig.update_layout(
            height=480, margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig.update_yaxes(showgrid=True, gridcolor="#F0F0F0", row=1, col=1)
        st.plotly_chart(fig, use_container_width=True)

        # Recent 10 matches table
        st.markdown('<div class="section-header">Recent 10 Matches</div>',
                    unsafe_allow_html=True)
        recent = hist.tail(10)[["date","opponent","venue","team_score","opp_score","result_label","tournament"]].copy()
        recent["date"] = recent["date"].dt.strftime("%Y-%m-%d")
        recent.columns = ["Date","Opponent","Venue","GF","GA","Result","Tournament"]
        def color_result(val):
            if val == "Win":  return "background-color:#D1FAE5; color:#065F46; font-weight:700"
            if val == "Loss": return "background-color:#FEE2E2; color:#991B1B; font-weight:700"
            return "background-color:#F3F4F6; color:#374151"
        st.dataframe(
            recent.style.applymap(color_result, subset=["Result"]),
            hide_index=True, use_container_width=True,
        )

    # ── Tab 2: W/D/L Breakdown ────────────────────────────────────────────────
    with tab2:
        col_l, col_r = st.columns(2)
        with col_l:
            wdl_vals = [stats["wins"], stats["draws"], stats["losses"]]
            fig_wdl = go.Figure(go.Pie(
                labels=["Wins", "Draws", "Losses"],
                values=wdl_vals,
                hole=0.5,
                marker_colors=[COLORS["accent"], "#9CA3AF", COLORS["secondary"]],
                textinfo="label+percent+value",
                textfont_size=13,
            ))
            fig_wdl.update_layout(
                title=dict(text="All-time W/D/L", font_size=15, x=0.5),
                height=320, margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(fig_wdl, use_container_width=True)

        with col_r:
            # Annual win rate
            hist["year"] = hist["date"].dt.year
            annual = hist.groupby("year").apply(
                lambda g: pd.Series({
                    "win_rate": (g["result_label"] == "Win").mean() * 100,
                    "matches":  len(g),
                })
            ).reset_index()
            annual = annual[annual["matches"] >= 3]
            fig_annual = px.bar(
                annual, x="year", y="win_rate",
                title="Annual Win Rate (%)",
                color="win_rate",
                color_continuous_scale=[[0, COLORS["secondary"]], [0.5, "#9CA3AF"], [1, COLORS["accent"]]],
                labels={"win_rate": "Win %", "year": "Year"},
            )
            fig_annual.update_layout(
                height=320, margin=dict(l=0, r=0, t=40, b=0),
                showlegend=False, coloraxis_showscale=False,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_annual, use_container_width=True)

        # By tournament type
        ttype = hist.copy()
        ttype["is_friendly"] = ttype["tournament"].str.contains("Friendly", case=False, na=False)
        ttype_grp = ttype.groupby(["tournament","result_label"]).size().reset_index(name="count")
        top_tourns = ttype_grp.groupby("tournament")["count"].sum().nlargest(8).index
        ttype_grp = ttype_grp[ttype_grp["tournament"].isin(top_tourns)]
        fig_t = px.bar(
            ttype_grp, x="tournament", y="count", color="result_label",
            color_discrete_map={"Win": COLORS["accent"], "Draw": "#9CA3AF", "Loss": COLORS["secondary"]},
            title="Results by Tournament Type (top 8)",
            labels={"count": "Matches", "tournament": "", "result_label": ""},
        )
        fig_t.update_layout(
            height=340, margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis_tickangle=-30, legend_title="",
        )
        st.plotly_chart(fig_t, use_container_width=True)

    # ── Tab 3: Match History ──────────────────────────────────────────────────
    with tab3:
        st.markdown("**Full match history** (most recent first)")
        show = hist[["date","opponent","venue","team_score","opp_score","result_label","tournament"]].copy()
        show = show.sort_values("date", ascending=False)
        show["date"] = show["date"].dt.strftime("%Y-%m-%d")
        show.columns = ["Date","Opponent","Venue","GF","GA","Result","Tournament"]
        st.dataframe(
            show.style.applymap(
                lambda v: (
                    "background-color:#D1FAE5; color:#065F46; font-weight:700" if v == "Win"
                    else "background-color:#FEE2E2; color:#991B1B; font-weight:700" if v == "Loss"
                    else ""
                ),
                subset=["Result"]
            ),
            height=480, hide_index=True, use_container_width=True,
        )

    # ── Tab 4: Head-to-Head ───────────────────────────────────────────────────
    with tab4:
        opponent = st.selectbox(
            "Compare against",
            options=[t for t in wc_teams if t != team],
            key="h2h_opp",
        )
        h2h_mask = (
            ((hist["team"] == team) if "team" in hist.columns else pd.Series([True]*len(hist))) |
            (hist["opponent"] == opponent)
        )
        h2h = hist[hist["opponent"] == opponent].sort_values("date", ascending=False)

        if len(h2h) == 0:
            st.info(f"No historical matches found between **{team}** and **{opponent}**.")
        else:
            wins_a   = (h2h["result_label"] == "Win").sum()
            draws_h2 = (h2h["result_label"] == "Draw").sum()
            wins_b   = (h2h["result_label"] == "Loss").sum()
            total_h2 = len(h2h)

            hc1, hc2, hc3, hc4 = st.columns(4)
            hc1.metric("Total Meetings", total_h2)
            hc2.metric(f"{team} Wins", wins_a)
            hc3.metric("Draws", draws_h2)
            hc4.metric(f"{opponent} Wins", wins_b)

            fig_h2h = go.Figure(go.Bar(
                x=[f"{team} Wins", "Draws", f"{opponent} Wins"],
                y=[wins_a, draws_h2, wins_b],
                marker_color=[COLORS["primary"], "#9CA3AF", COLORS["secondary"]],
                text=[wins_a, draws_h2, wins_b],
                textposition="outside",
            ))
            fig_h2h.update_layout(
                height=260, showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=20, b=0),
                yaxis=dict(showgrid=False, showticklabels=False),
            )
            st.plotly_chart(fig_h2h, use_container_width=True)

            show_h2h = h2h[["date","venue","team_score","opp_score","result_label","tournament"]].copy()
            show_h2h["date"] = show_h2h["date"].dt.strftime("%Y-%m-%d")
            show_h2h.columns = ["Date","Venue","GF","GA","Result","Tournament"]
            st.dataframe(show_h2h, hide_index=True, use_container_width=True)
