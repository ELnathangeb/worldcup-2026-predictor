"""Tournament Simulator — simulate a full WC 2026 bracket."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from utils import (
    load_groups, load_elo, page_header, section_title, kpi,
    COLORS, CHART, confederation_map,
)

_CONF_CSS = {
    "UEFA":"#38BDF8","CONMEBOL":"#34D399","CONCACAF":"#FCD34D",
    "CAF":"#F87171","AFC":"#A78BFA","OFC":"#FB923C",
}


def _run_tournament(seed):
    from src.predict_match import precompute_matchups
    from src.simulation.tournament import simulate_world_cup
    groups   = load_groups()
    all_teams = [t for g in groups.values() for t in g]
    precompute_matchups(all_teams)
    rng = np.random.default_rng(seed)
    return simulate_world_cup(groups, rng=rng, verbose=False)


def _group_rows(standings, best_thirds) -> str:
    rows = ""
    for pos, tr in enumerate(standings):
        cls = "advance" if pos < 2 else ("best3" if tr.name in best_thirds else "")
        dot = "🟢" if pos < 2 else ("🔵" if tr.name in best_thirds else "⚫")
        rows += f"""
        <div class='grp-row {cls}'>
            <span style='font-size:0.65rem;'>{dot}</span>
            <span class='grp-name'>{tr.name}</span>
            <span class='grp-stat'>{tr.pts}</span>
            <span class='grp-stat' style='color:#64748B;font-size:0.7rem;'>{tr.gd:+d}</span>
            <span class='grp-stat' style='color:#64748B;font-size:0.7rem;'>{tr.played}</span>
        </div>"""
    return rows


def _match_card_html(mr) -> str:
    winner_a = mr.winner == mr.team_a
    flag = " · pens" if mr.after_pens else (" · aet" if mr.after_et else "")
    wname_a = f"<span class='match-winner'>{mr.team_a}</span>" if winner_a \
              else f"<span class='match-loser'>{mr.team_a}</span>"
    wname_b = f"<span class='match-winner'>{mr.team_b}</span>" if not winner_a \
              else f"<span class='match-loser'>{mr.team_b}</span>"
    winner_label_color = "#34D399"
    return f"""
    <div class='match-card'>
        <div style='flex:1;'>{wname_a}</div>
        <div class='match-score'>{mr.score_str()}</div>
        <div style='flex:1;text-align:right;'>{wname_b}</div>
        <div style='font-size:0.65rem;color:#334155;min-width:50px;
                    text-align:right;margin-left:8px;'>{flag.strip(' ·') or ''}</div>
    </div>"""


def render() -> None:
    page_header("Tournament Simulator", "Simulate the full 48-team WC 2026 bracket from groups to final")

    # ── Controls ──────────────────────────────────────────────────────────────
    c_btn, c_seed = st.columns([3, 1])
    with c_seed:
        use_seed = st.checkbox("Fixed seed", value=False, key="ts_fix")
        seed_val = st.number_input("Seed value", value=42, step=1, key="ts_seed",
                                   disabled=not use_seed, label_visibility="collapsed")
    with c_btn:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        run_btn = st.button("🏆  Simulate World Cup 2026", type="primary",
                            use_container_width=True)

    if run_btn:
        with st.spinner("Simulating tournament…"):
            seed   = int(seed_val) if use_seed else None
            result = _run_tournament(seed)
        st.session_state["ts_result"] = result

    result = st.session_state.get("ts_result")
    if result is None:
        st.markdown("""
        <div style='text-align:center;padding:48px 0;'>
            <div style='font-size:2.5rem;margin-bottom:12px;'>🏆</div>
            <div style='color:#334155;font-size:0.9rem;'>
                Click <strong style='color:#38BDF8;'>Simulate World Cup 2026</strong>
                to run a full bracket simulation
            </div>
        </div>""", unsafe_allow_html=True)
        return

    # ── Champion banner ────────────────────────────────────────────────────────
    champ  = result["champion"]
    ru     = result["runner_up"]
    fr     = result["final_result"]
    sf     = result["semifinalists"]
    others = [t for t in sf if t not in [champ, ru]]

    flag_et = " (penalties)" if fr.after_pens else (" (aet)" if fr.after_et else "")

    st.markdown(f"""
    <div class='champion-banner'>
        <div style='font-size:0.65rem;font-weight:800;letter-spacing:0.2em;
                    color:#FCD34D88;text-transform:uppercase;margin-bottom:10px;'>
            🏆 WORLD CUP 2026 CHAMPION
        </div>
        <div style='font-size:2.6rem;font-weight:900;color:#FCD34D;
                    letter-spacing:-1px;text-shadow:0 0 30px #FCD34D66;'>{champ}</div>
        <div style='margin-top:16px;font-size:1rem;color:#94A3B8;'>
            Final: <strong style='color:#F1F5F9;'>{fr.team_a}</strong>
            <span style='color:#38BDF8;font-weight:800;font-size:1.2rem;
                         letter-spacing:3px;margin:0 10px;'>{fr.score_str()}</span>
            <strong style='color:#F1F5F9;'>{fr.team_b}</strong>
            <span style='color:#475569;font-size:0.8rem;'>{flag_et}</span>
        </div>
        <div style='margin-top:10px;font-size:0.8rem;color:#475569;'>
            🥈 Runner-up: <span style='color:#94A3B8;'>{ru}</span>
            &nbsp;·&nbsp; 4th: <span style='color:#475569;'>{" · ".join(others)}</span>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Group standings ────────────────────────────────────────────────────────
    section_title("Group Stage", "🟢 Top 2 qualify directly  ·  🔵 Best 8 third-place also advance")

    best_thirds  = set(result.get("best_thirds", []))
    grp_items    = list(result["group_standings"].items())
    elo          = load_elo()

    for row_start in range(0, 12, 4):
        cols = st.columns(4)
        for col_idx, (grp, standings) in enumerate(grp_items[row_start:row_start+4]):
            with cols[col_idx]:
                rows_html = _group_rows(standings, best_thirds)
                st.markdown(f"""
                <div class='glass-card' style='padding:0;overflow:hidden;margin-bottom:8px;'>
                    <div class='grp-header' style='display:flex;justify-content:space-between;'>
                        <span>GROUP {grp}</span>
                        <span style='color:#334155;font-weight:500;
                                     text-transform:none;letter-spacing:0;'>PTS  GD  P</span>
                    </div>
                    {rows_html}
                </div>""", unsafe_allow_html=True)

    if best_thirds:
        thirds_html = " · ".join(
            f"<span style='color:#38BDF8;font-weight:600;'>{t}</span>"
            for t in result["best_thirds"]
        )
        st.markdown(
            f"<div style='font-size:0.78rem;color:#64748B;margin-top:4px;'>"
            f"Best 8 third-place qualifiers: {thirds_html}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Knockout tabs ──────────────────────────────────────────────────────────
    section_title("Knockout Stage")

    round_data = [
        ("🥊 Round of 32 (16)",  "r32_results",  16),
        ("⚔️ Round of 16 (8)",   "r16_results",   8),
        ("🎯 Quarterfinals (4)", "qf_results",    4),
        ("🌟 Semifinals (2)",    "sf_results",    2),
        ("🏆 Final",             "final_result",  1),
    ]
    tab_labels = [r[0] for r in round_data]
    tabs       = st.tabs(tab_labels)

    for tab, (label, key, _) in zip(tabs, round_data):
        with tab:
            res = result[key]
            if not isinstance(res, list):
                res = [res]

            if label.startswith("🏆"):
                # Full-width final card
                winner_a = res[0].winner == res[0].team_a
                fe = " (penalties)" if res[0].after_pens else (" (aet)" if res[0].after_et else "")
                st.markdown(f"""
                <div style='
                    background:linear-gradient(135deg,#0B1728,#1A0F2A,#0A1828);
                    border:1px solid rgba(252,211,77,0.3);border-top:3px solid #FCD34D;
                    border-radius:14px;padding:28px;margin:8px 0;
                '>
                    <div style='text-align:center;font-size:0.65rem;font-weight:700;
                                color:#FCD34D88;letter-spacing:0.2em;
                                text-transform:uppercase;margin-bottom:16px;'>
                        🏆 FINAL
                    </div>
                    <div style='display:flex;align-items:center;justify-content:center;gap:24px;'>
                        <div style='font-size:1.5rem;font-weight:800;text-align:right;
                                    color:{"#F1F5F9" if winner_a else "#334155"};flex:1;'>
                            {res[0].team_a}
                            {"<span style='color:#FCD34D;'> 🏆</span>" if winner_a else ""}
                        </div>
                        <div style='font-size:2.8rem;font-weight:900;color:#FCD34D;
                                    letter-spacing:4px;text-shadow:0 0 20px #FCD34D44;'>
                            {res[0].score_str()}
                        </div>
                        <div style='font-size:1.5rem;font-weight:800;text-align:left;
                                    color:{"#F1F5F9" if not winner_a else "#334155"};flex:1;'>
                            {"<span style='color:#FCD34D;'>🏆 </span>" if not winner_a else ""}
                            {res[0].team_b}
                        </div>
                    </div>
                    {"<div style='text-align:center;margin-top:10px;font-size:0.8rem;color:#475569;'>" + fe + "</div>" if fe else ""}
                </div>""", unsafe_allow_html=True)
            else:
                # Two-column grid of match cards
                half = len(res) // 2 or 1
                col1, col2 = st.columns(2)
                for mr in res[:half]:
                    col1.markdown(_match_card_html(mr), unsafe_allow_html=True)
                for mr in res[half:]:
                    col2.markdown(_match_card_html(mr), unsafe_allow_html=True)

    st.markdown("---")

    # ── Full progression table ─────────────────────────────────────────────────
    section_title("Full Progression", "Every team's furthest stage reached")

    adv = result["advancement"]
    stage_order = {
        "champion":7,"runner_up":6,"final":5,"semifinal":4,
        "quarterfinal":3,"r16":2,"r32":1,"group_stage":0,
    }
    stage_color = {
        "Champion":    "#FCD34D","Runner Up": "#94A3B8","Final":"#A78BFA",
        "Semifinal":   "#38BDF8","Quarterfinal":"#34D399","R16":"#6EE7B7",
        "R32":         "#1C3A2A","Group Stage":"#1C2E4A",
    }
    prog_rows = [
        {"Team": team, "Stage": stage.replace("_"," ").title()}
        for team, stage in sorted(adv.items(), key=lambda x: -stage_order.get(x[1], 0))
    ]
    prog_df = pd.DataFrame(prog_rows)

    def _sc(val):
        return f"background-color:{stage_color.get(val,'#0D1526')};color:#F1F5F9;font-weight:600"

    st.dataframe(
        prog_df.style.applymap(_sc, subset=["Stage"]),
        hide_index=True, use_container_width=True, height=500,
    )

    # ── Funnel ────────────────────────────────────────────────────────────────
    section_title("Tournament Funnel")
    labels_f = ["Group Stage","Round of 32","Round of 16","Quarterfinals","Semifinals","Final","Champion"]
    values_f = [48, 32, 16, 8, 4, 2, 1]
    fig_f = go.Figure(go.Funnel(
        y=labels_f, x=values_f,
        textinfo="value+percent initial",
        textfont=dict(color="white", size=11),
        marker=dict(color=["#1C2E4A","#1C3A2A","#0E4429","#0C3D5F","#1B1240","#7C3AED","#FCD34D"]),
        connector=dict(line=dict(color="#1C2E4A", width=1)),
    ))
    fig_f.update_layout(**{**CHART, "height": 340, "margin": dict(l=0,r=0,t=20,b=0),
                           "paper_bgcolor":"rgba(0,0,0,0)"})
    st.plotly_chart(fig_f, use_container_width=True)
