"""Team Explorer — dark sports profile page."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

from utils import (
    load_dataset, load_elo, load_monte_carlo,
    all_wc_teams, team_stats, team_group,
    page_header, section_title, kpi, prob_bar,
    COLORS, CHART, CONF_COLORS, confederation_map,
)

_CONF_CSS = {
    "UEFA":    "#38BDF8", "CONMEBOL": "#34D399",
    "CONCACAF":"#FCD34D", "CAF":      "#F87171",
    "AFC":     "#A78BFA", "OFC":      "#FB923C",
}


def _result_color(v):
    if v == "Win":  return "background-color:#052e1a;color:#34D399;font-weight:700"
    if v == "Loss": return "background-color:#2d0f0f;color:#F87171;font-weight:700"
    return "color:#64748B"


def render() -> None:
    page_header("Team Explorer", "Deep-dive stats, form, and head-to-head for any WC 2026 squad")

    df   = load_dataset()
    elo  = load_elo()
    mc   = load_monte_carlo()

    wc_teams = all_wc_teams()
    all_ds   = sorted(
        set(df["home_team"].unique()) | set(df["away_team"].unique()),
        key=lambda t: elo.get(t, 0), reverse=True,
    )

    col_sel, _ = st.columns([2, 3])
    with col_sel:
        team = st.selectbox(
            "Select team",
            options=wc_teams + ["─── All teams ───"] + [t for t in all_ds if t not in wc_teams],
            index=0, key="te_team",
        )
    if "───" in team:
        st.info("Select a team above to begin.")
        return

    stats = team_stats(team)
    if not stats:
        st.warning(f"No data for **{team}**.")
        return

    conf_map = confederation_map()
    conf     = conf_map.get(team, "OFC")
    cc       = _CONF_CSS.get(conf, "#475569")
    grp      = stats["group"]
    is_wc    = team in wc_teams

    # ── Team header ────────────────────────────────────────────────────────────
    st.html(f"""
    <div style='
        background:linear-gradient(135deg,#0B1728 0%,#0F2040 60%,#0A1828 100%);
        border:1px solid #1C2E4A;border-left:4px solid {cc};
        border-radius:14px;padding:20px 28px;margin-bottom:1.2rem;
        box-shadow:0 6px 32px rgba(0,0,0,0.5);
        display:flex;align-items:center;gap:20px;
    '>
        <div style='width:60px;height:60px;background:linear-gradient(135deg,{cc}33,{cc}11);
                    border:2px solid {cc}55;border-radius:12px;
                    display:flex;align-items:center;justify-content:center;
                    font-size:1.6rem;font-weight:900;color:{cc};flex-shrink:0;'>
            {team[:2].upper()}
        </div>
        <div style='flex:1;'>
            <div style='font-size:1.5rem;font-weight:800;color:#F1F5F9;
                        letter-spacing:-0.5px;'>{team}</div>
            <div style='display:flex;gap:8px;margin-top:6px;flex-wrap:wrap;'>
                <span class='conf-badge' style='background:{cc}22;color:{cc};
                                                border:1px solid {cc}44;'>
                    {conf}
                </span>
                {"<span class='conf-badge' style='background:#34D39922;color:#34D399;border:1px solid #34D39944;'>WC 2026 · Group " + grp + "</span>" if is_wc else ""}
                <span class='conf-badge' style='background:rgba(255,255,255,0.05);
                                                color:#64748B;border:1px solid #1C2E4A;'>
                    Elo {stats['elo']:.0f}
                </span>
                {"<span class='conf-badge' style='background:rgba(255,255,255,0.05);color:#64748B;border:1px solid #1C2E4A;'>Rank #" + str(int(stats['rank'])) + "</span>" if stats.get('rank') else ""}
            </div>
        </div>
        <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:10px;min-width:320px;'>
            <div class='stat-pill'><div class='val'>{stats["win_pct"]}%</div><div class='lbl'>Win Rate</div></div>
            <div class='stat-pill'><div class='val'>{stats["goals_per_game"]}</div><div class='lbl'>Goals/Game</div></div>
            <div class='stat-pill'><div class='val'>{stats["total_matches"]}</div><div class='lbl'>Matches</div></div>
        </div>
    </div>""")

    # ── MC odds strip ──────────────────────────────────────────────────────────
    mc_row = mc[mc["team"] == team]
    if len(mc_row):
        r = mc_row.iloc[0]
        stages = [
            ("Champion",     r["pct_champion"],     "#FCD34D"),
            ("Final",        r.get("pct_final", r["pct_runner_up"]), "#94A3B8"),
            ("Semifinal",    r["pct_semifinal"],    "#A78BFA"),
            ("Quarterfinal", r["pct_quarterfinal"], "#38BDF8"),
            ("Rd of 16",     r["pct_round_of_16"],  "#34D399"),
            ("Rd of 32",     r["pct_round_of_32"],  "#6EE7B7"),
        ]
        cols = st.columns(6)
        for col, (label, val, accent) in zip(cols, stages):
            col.html(kpi(f"{val:.1f}%", label, accent=accent))
        st.html("<br>")

    # ── Build match history ────────────────────────────────────────────────────
    home_df = df[df["home_team"] == team].copy()
    home_df["team_score"]   = home_df["home_score"]
    home_df["opp_score"]    = home_df["away_score"]
    home_df["opponent"]     = home_df["away_team"]
    home_df["venue"]        = "Home"
    home_df["result_label"] = home_df["result"].map({2:"Win", 1:"Draw", 0:"Loss"})

    away_df = df[df["away_team"] == team].copy()
    away_df["team_score"]   = away_df["away_score"]
    away_df["opp_score"]    = away_df["home_score"]
    away_df["opponent"]     = away_df["home_team"]
    away_df["venue"]        = "Away"
    away_df["result_label"] = away_df["result"].map({0:"Win", 1:"Draw", 2:"Loss"})

    hist = pd.concat([home_df, away_df]).sort_values("date").reset_index(drop=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Form & Goals", "🏅 W/D/L Breakdown", "📋 Match History", "🆚 Head-to-Head"]
    )

    # ── Tab 1: Form ────────────────────────────────────────────────────────────
    with tab1:
        hist["rolling_gf"] = hist["team_score"].rolling(12, min_periods=3).mean()
        hist["rolling_ga"] = hist["opp_score"].rolling(12, min_periods=3).mean()

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=("12-match rolling avg — Goals Scored vs Conceded", "Match Results"),
            vertical_spacing=0.10, row_heights=[0.68, 0.32],
        )
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["rolling_gf"], name="Scored",
            line=dict(color="#38BDF8", width=2.5),
            fill="tozeroy", fillcolor="rgba(56,189,248,0.08)",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["rolling_ga"], name="Conceded",
            line=dict(color="#F87171", width=2),
            fill="tozeroy", fillcolor="rgba(248,113,113,0.06)",
        ), row=1, col=1)
        c_map = {"Win":"#34D399","Draw":"#475569","Loss":"#F87171"}
        for label, color in c_map.items():
            sub = hist[hist["result_label"] == label]
            fig.add_trace(go.Scatter(
                x=sub["date"], y=[label]*len(sub), mode="markers",
                marker=dict(color=color, size=7, symbol="circle",
                            line=dict(width=0)),
                name=label,
            ), row=2, col=1)
        fig.update_layout(**{**CHART, "height": 420,
                             "legend": dict(orientation="h", y=1.04, bgcolor="rgba(0,0,0,0)",
                                            font=dict(size=11, color="#64748B"))})
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)", row=1, col=1)
        st.plotly_chart(fig, use_container_width=True)

        # Recent 10 table
        section_title("Last 10 Matches")
        recent = hist.tail(10)[["date","opponent","venue","team_score","opp_score","result_label","tournament"]].copy()
        recent["date"] = recent["date"].dt.strftime("%Y-%m-%d")
        recent.columns = ["Date","Opponent","Venue","GF","GA","Result","Tournament"]
        st.dataframe(
            recent.style.applymap(_result_color, subset=["Result"]),
            hide_index=True, use_container_width=True,
        )

    # ── Tab 2: W/D/L ──────────────────────────────────────────────────────────
    with tab2:
        col_l, col_r = st.columns(2)
        with col_l:
            fig_pie = go.Figure(go.Pie(
                labels=["Wins","Draws","Losses"],
                values=[stats["wins"], stats["draws"], stats["losses"]],
                hole=0.55,
                marker_colors=["#34D399","#475569","#F87171"],
                textinfo="label+percent",
                textfont=dict(size=12, color="white"),
                hovertemplate="<b>%{label}</b><br>%{value} matches<extra></extra>",
            ))
            fig_pie.add_annotation(
                text=f"<b>{stats['win_pct']}%</b><br>win rate",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=13, color="#94A3B8"),
            )
            fig_pie.update_layout(**{**CHART, "height": 300, "showlegend": False,
                                     "title": dict(text="All-time W/D/L", font=dict(color="#64748B", size=13))})
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_r:
            hist["year"] = hist["date"].dt.year
            annual = hist.groupby("year").apply(
                lambda g: pd.Series({
                    "win_rate": (g["result_label"] == "Win").mean() * 100,
                    "matches":  len(g),
                })
            ).reset_index()
            annual = annual[annual["matches"] >= 3]
            fig_yr = go.Figure(go.Bar(
                x=annual["year"], y=annual["win_rate"],
                marker=dict(
                    color=annual["win_rate"],
                    colorscale=[[0,"#2d0f0f"],[0.4,"#475569"],[1,"#052e1a"]],
                    line=dict(width=0),
                ),
                hovertemplate="<b>%{x}</b><br>Win rate: %{y:.1f}%<extra></extra>",
            ))
            fig_yr.update_layout(**{**CHART, "height": 300,
                                    "title": dict(text="Annual Win Rate", font=dict(color="#64748B", size=13)),
                                    "yaxis": dict(title="Win %", gridcolor="rgba(255,255,255,0.04)"),
                                    "xaxis": dict(tickangle=-30)})
            st.plotly_chart(fig_yr, use_container_width=True)

        # By tournament
        ttype_grp = hist.groupby(["tournament","result_label"]).size().reset_index(name="count")
        top_t     = ttype_grp.groupby("tournament")["count"].sum().nlargest(8).index
        ttype_grp = ttype_grp[ttype_grp["tournament"].isin(top_t)]
        fig_t = px.bar(
            ttype_grp, x="tournament", y="count", color="result_label",
            color_discrete_map={"Win":"#34D399","Draw":"#475569","Loss":"#F87171"},
            labels={"count":"Matches","tournament":"","result_label":""},
        )
        fig_t.update_layout(**{**CHART, "height": 320, "barmode": "group",
                                "title": dict(text="Results by Tournament (top 8)", font=dict(color="#64748B",size=13)),
                                "xaxis": dict(tickangle=-30),
                                "legend": dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)")})
        st.plotly_chart(fig_t, use_container_width=True)

    # ── Tab 3: Match History ───────────────────────────────────────────────────
    with tab3:
        show = hist[["date","opponent","venue","team_score","opp_score","result_label","tournament"]].copy()
        show = show.sort_values("date", ascending=False)
        show["date"] = show["date"].dt.strftime("%Y-%m-%d")
        show.columns = ["Date","Opponent","Venue","GF","GA","Result","Tournament"]
        st.dataframe(
            show.style.applymap(_result_color, subset=["Result"]),
            height=500, hide_index=True, use_container_width=True,
        )

    # ── Tab 4: H2H ────────────────────────────────────────────────────────────
    with tab4:
        opp = st.selectbox("Compare against", [t for t in wc_teams if t != team], key="h2h_opp")
        h2h = hist[hist["opponent"] == opp].sort_values("date", ascending=False)

        if len(h2h) == 0:
            st.info(f"No historical meetings found between **{team}** and **{opp}**.")
        else:
            wins_a = (h2h["result_label"] == "Win").sum()
            draws  = (h2h["result_label"] == "Draw").sum()
            wins_b = (h2h["result_label"] == "Loss").sum()

            h1, h2c, h3, h4 = st.columns(4)
            h1.html(kpi(str(len(h2h)), "Meetings",         accent="#38BDF8"))
            h2c.html(kpi(str(wins_a),  f"{team[:12]} Wins", accent="#34D399"))
            h3.html(kpi(str(draws),    "Draws",             accent="#475569"))
            h4.html(kpi(str(wins_b),   f"{opp[:12]} Wins",  accent="#F87171"))

            st.html("<br>")

            fig_h2h = go.Figure(go.Bar(
                x=[f"{team} Wins", "Draws", f"{opp} Wins"],
                y=[wins_a, draws, wins_b],
                marker_color=["#34D399", "#475569", "#F87171"],
                marker_line_width=0,
                text=[wins_a, draws, wins_b], textposition="outside",
                textfont=dict(color="#94A3B8"),
                hovertemplate="%{x}: <b>%{y}</b><extra></extra>",
            ))
            fig_h2h.update_layout(**{**CHART, "height": 260, "showlegend": False,
                                     "yaxis": dict(showgrid=False, showticklabels=False),
                                     "xaxis": dict(tickfont=dict(size=13, color="#94A3B8"))})
            st.plotly_chart(fig_h2h, use_container_width=True)

            show_h2h = h2h[["date","venue","team_score","opp_score","result_label","tournament"]].copy()
            show_h2h["date"] = show_h2h["date"].dt.strftime("%Y-%m-%d")
            show_h2h.columns = ["Date","Venue","GF","GA","Result","Tournament"]
            st.dataframe(
                show_h2h.style.applymap(_result_color, subset=["Result"]),
                hide_index=True, use_container_width=True,
            )