"""Page 4 — Tournament Simulator: simulate one full WC and display the bracket."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from utils import load_groups, load_elo, page_header, COLORS


def _run_tournament(seed: int | None = None) -> dict:
    """Run a single simulation (cached by seed)."""
    from src.predict_match import precompute_matchups
    from src.simulation.tournament import simulate_world_cup

    groups = load_groups()
    all_teams = [t for g in groups.values() for t in g]
    precompute_matchups(all_teams)

    rng = np.random.default_rng(seed)
    return simulate_world_cup(groups, rng=rng, verbose=False)


def _group_table_df(standings) -> pd.DataFrame:
    rows = []
    for tr in standings:
        rows.append({
            "Team": tr.name,
            "P": tr.played, "W": tr.wins, "D": tr.draws, "L": tr.losses,
            "GF": tr.gf, "GA": tr.ga, "GD": f"{tr.gd:+d}",
            "Pts": tr.pts,
        })
    return pd.DataFrame(rows)


def _ko_df(results) -> pd.DataFrame:
    rows = []
    for mr in results:
        flag = " (pens)" if mr.after_pens else (" (aet)" if mr.after_et else "")
        rows.append({
            "Team A": mr.team_a,
            "Score": mr.score_str(),
            "Team B": mr.team_b,
            "Winner": mr.winner + flag,
        })
    return pd.DataFrame(rows)


def _bracket_chart(result: dict) -> go.Figure:
    """Simple linear bracket visualisation using scatter + text."""
    rounds = ["R32", "R16", "QF", "SF", "Final"]
    keys   = ["r32_results", "r16_results", "qf_results", "sf_results", "final_result"]

    fig = go.Figure()
    x_pos, y_pos, texts, colors_ = [], [], [], []

    for r_idx, (rnd, key) in enumerate(zip(rounds, keys)):
        res = result[key]
        if not isinstance(res, list):
            res = [res]
        for m_idx, mr in enumerate(res):
            winner_a = mr.winner == mr.team_a
            # Team A
            x_pos.append(r_idx * 2)
            y_pos.append(m_idx * 2 + 0.3)
            texts.append(f"  {mr.team_a}")
            colors_.append(COLORS["accent"] if winner_a else "#D1D5DB")
            # Score
            x_pos.append(r_idx * 2 + 0.85)
            y_pos.append(m_idx * 2)
            texts.append(mr.score_str())
            colors_.append("#374151")
            # Team B
            x_pos.append(r_idx * 2)
            y_pos.append(m_idx * 2 - 0.3)
            texts.append(f"  {mr.team_b}")
            colors_.append(COLORS["secondary"] if not winner_a else "#D1D5DB")

    fig.add_trace(go.Scatter(
        x=x_pos, y=y_pos, mode="text",
        text=texts,
        textfont=dict(size=9, color=colors_),
        showlegend=False,
    ))
    for r_idx, rnd in enumerate(rounds):
        fig.add_annotation(
            x=r_idx * 2, y=max(y_pos) + 1.2,
            text=f"<b>{rnd}</b>", showarrow=False,
            font=dict(size=11, color=COLORS["primary"]),
            bgcolor="white",
        )

    fig.update_layout(
        height=max(400, len(result["r32_results"]) * 35),
        margin=dict(l=0, r=0, t=60, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, autorange="reversed"),
    )
    return fig


def render() -> None:
    page_header("Tournament Simulator", "Simulate the full WC 2026 bracket — group stage through the Final")

    # ── Controls ──────────────────────────────────────────────────────────────
    col_btn, col_seed = st.columns([2, 1])
    with col_seed:
        use_seed = st.checkbox("Use fixed seed (reproducible)", value=False)
        seed_val = st.number_input("Seed", value=42, step=1, disabled=not use_seed)
    with col_btn:
        st.markdown("")
        simulate_btn = st.button("🏆  Simulate World Cup 2026", type="primary",
                                 use_container_width=True)

    if simulate_btn:
        with st.spinner("Simulating full tournament (group stage → final)..."):
            seed = int(seed_val) if use_seed else None
            result = _run_tournament(seed)
        st.session_state["tournament_result"] = result
        st.success("Simulation complete! Scroll down to see the full bracket.")

    result = st.session_state.get("tournament_result")
    if result is None:
        st.info("👆 Click **Simulate World Cup 2026** to run a full tournament simulation.")
        return

    # ── Champion banner ────────────────────────────────────────────────────────
    champ = result["champion"]
    ru    = result["runner_up"]
    fr    = result["final_result"]
    sf    = result["semifinalists"]

    st.markdown(f"""
    <div class="champion-card">
        <div style='font-size:3rem;'>🏆</div>
        <h1>{champ}</h1>
        <p>World Cup 2026 Champion</p>
        <div style='margin-top:12px; font-size:1rem; opacity:0.9;'>
            Final: {fr.team_a} <strong>{fr.score_str()}</strong> {fr.team_b}
        </div>
        <div style='margin-top:8px; font-size:0.9rem; opacity:0.75;'>
            🥈 Runner-up: {ru}  &nbsp;|&nbsp;
            4th-place finalists: {", ".join(t for t in sf if t not in [champ, ru])}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Group Stage ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Group Stage Standings</div>',
                unsafe_allow_html=True)

    best_thirds = set(result.get("best_thirds", []))
    groups      = load_groups()

    # Display groups in 3 columns (4 groups each)
    grp_items = list(result["group_standings"].items())
    for row_start in range(0, 12, 4):
        cols = st.columns(4)
        for col_idx, (grp, standings) in enumerate(grp_items[row_start:row_start+4]):
            with cols[col_idx]:
                st.markdown(f"**Group {grp}**")
                df_grp = _group_table_df(standings)

                def highlight_grp(row):
                    pos = row.name
                    if pos == 0 or pos == 1:
                        return ["background-color:#D1FAE5; font-weight:700"] * len(row)
                    if standings[pos].name in best_thirds:
                        return ["background-color:#FEF9C3; font-weight:600"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    df_grp.style.apply(highlight_grp, axis=1),
                    hide_index=True, use_container_width=True, height=178,
                )

    st.caption("🟢 Top 2 advance directly  |  🟡 Best 8 third-place teams also advance")
    st.markdown(f"**Best 8 third-place qualifiers:** "
                + ", ".join(f"**{t}**" for t in result.get("best_thirds", [])))

    st.markdown("---")

    # ── Knockout Rounds ────────────────────────────────────────────────────────
    tabs = st.tabs(["🥊 Round of 32", "⚔️ Round of 16", "🎯 Quarterfinals",
                    "🌟 Semifinals", "🏆 Final", "📋 Full Bracket"])

    with tabs[0]:
        st.dataframe(
            _ko_df(result["r32_results"]).style.apply(
                lambda r: ["font-weight:700; color:#1E3A5F" if r["Winner"].startswith(r["Team A"])
                           else "" for _ in r], axis=1
            ),
            hide_index=True, use_container_width=True,
        )
    with tabs[1]:
        df_r16 = _ko_df(result["r16_results"])
        st.dataframe(df_r16, hide_index=True, use_container_width=True)

    with tabs[2]:
        df_qf = _ko_df(result["qf_results"])
        _render_ko_cards(result["qf_results"])

    with tabs[3]:
        _render_ko_cards(result["sf_results"])

    with tabs[4]:
        fr_df = _ko_df([result["final_result"]])
        _render_final(result["final_result"])

    with tabs[5]:
        st.markdown("#### Complete Tournament Bracket")
        # Progression table
        adv = result["advancement"]
        prog_rows = []
        stage_order = {"champion":7,"runner_up":6,"final":5,"semifinal":4,
                       "quarterfinal":3,"r16":2,"r32":1,"group_stage":0}
        for team, stage in sorted(adv.items(), key=lambda x: -stage_order.get(x[1],0)):
            prog_rows.append({"Team": team, "Furthest Stage": stage.replace("_"," ").title()})
        prog_df = pd.DataFrame(prog_rows)
        stage_colors_map = {
            "Champion":      "#FEF9C3",
            "Runner Up":     "#F0F4FF",
            "Final":         "#E0E7FF",
            "Semifinal":     "#DBEAFE",
            "Quarterfinal":  "#EFF6FF",
            "R16":           "#F0FDF4",
            "R32":           "#F9FAFB",
            "Group Stage":   "#FFFFFF",
        }
        def stage_color(val):
            return f"background-color:{stage_colors_map.get(val, 'white')}"
        st.dataframe(
            prog_df.style.applymap(stage_color, subset=["Furthest Stage"]),
            hide_index=True, use_container_width=True, height=520,
        )

    # ── Advancement funnel ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">Tournament Funnel</div>',
                unsafe_allow_html=True)
    adv_counts = {
        "Group Stage (48)":  48,
        "Round of 32 (32)":  32,
        "Round of 16 (16)":  16,
        "Quarterfinals (8)":  8,
        "Semifinals (4)":     4,
        "Final (2)":          2,
        "Champion (1)":       1,
    }
    fig_funnel = go.Figure(go.Funnel(
        y=list(adv_counts.keys()),
        x=list(adv_counts.values()),
        textinfo="value+percent initial",
        marker=dict(color=[
            "#BFDBFE","#93C5FD","#60A5FA","#3B82F6","#1D4ED8","#C8102E","#FFD700"
        ]),
    ))
    fig_funnel.update_layout(
        height=360, margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_funnel, use_container_width=True)


def _render_ko_cards(results) -> None:
    for mr in results:
        winner_a = mr.winner == mr.team_a
        flag = " 🔫 *pens*" if mr.after_pens else (" ⏱️ *aet*" if mr.after_et else "")
        st.markdown(f"""
        <div class="bracket-match">
            <span class='{"bracket-winner" if winner_a else "bracket-loser"}'>{mr.team_a}</span>
            <span style='font-size:1.1rem; font-weight:800; color:#1E3A5F;'>{mr.score_str()}</span>
            <span class='{"bracket-loser" if winner_a else "bracket-winner"}'>{mr.team_b}</span>
            <span style='font-size:0.8rem; color:#C8102E; font-weight:600;'>
                → {mr.winner}{flag}
            </span>
        </div>
        """, unsafe_allow_html=True)


def _render_final(mr) -> None:
    winner_a = mr.winner == mr.team_a
    flag = " (penalties)" if mr.after_pens else (" (aet)" if mr.after_et else ""
    )
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#1E3A5F,#C8102E);
                border-radius:16px; padding:28px 40px; text-align:center;
                color:white; margin:1rem 0;'>
        <div style='font-size:0.9rem; opacity:0.8; margin-bottom:16px;
                    letter-spacing:0.15em;'>🏆 WORLD CUP 2026 FINAL 🏆</div>
        <div style='display:flex; justify-content:center; align-items:center; gap:32px;'>
            <div style='font-size:1.6rem; font-weight:800;
                        {"" if winner_a else "opacity:0.55;"}'>
                {mr.team_a}{"  🏆" if winner_a else ""}
            </div>
            <div style='font-size:2.5rem; font-weight:900; letter-spacing:4px;'>
                {mr.score_str()}
            </div>
            <div style='font-size:1.6rem; font-weight:800;
                        {"" if not winner_a else "opacity:0.55;"}'>
                {"🏆  " if not winner_a else ""}{mr.team_b}
            </div>
        </div>
        {"<div style='margin-top:10px;font-size:0.85rem;opacity:0.8;'>" + flag + "</div>" if flag else ""}
    </div>
    """, unsafe_allow_html=True)
