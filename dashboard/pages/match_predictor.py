"""Match Predictor — head-to-head prediction panel."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
import numpy as np

from utils import (
    load_elo, all_wc_teams,
    page_header, section_title, kpi, prob_bar,
    COLORS, CHART, team_group,
)

_CONF_CSS = {
    "UEFA":"#38BDF8","CONMEBOL":"#34D399","CONCACAF":"#FCD34D",
    "CAF":"#F87171","AFC":"#A78BFA","OFC":"#FB923C",
}


@st.cache_data(show_spinner=False)
def _get_prediction(team_a: str, team_b: str) -> dict:
    from src.predict_match import predict_match, _elo_neutral_probs, _expected_goals
    from src.predict_match import _load_dataset, _load_recent_team_stats, _get_rank_info
    _load_dataset.cache_clear()
    _load_recent_team_stats.cache_clear()
    _get_rank_info.cache_clear()
    result = predict_match(team_a, team_b)
    pa, pd_, pb = _elo_neutral_probs(team_a, team_b)
    result["elo_a_win"] = round(pa, 4)
    result["elo_draw"]  = round(pd_, 4)
    result["elo_b_win"] = round(pb, 4)
    return result


def render() -> None:
    page_header("Match Predictor", "ML model + Elo engine probabilities for any WC 2026 fixture")

    elo = load_elo()
    wc  = all_wc_teams()

    # ── Team selection ─────────────────────────────────────────────────────────
    col_a, col_vs, col_b = st.columns([5, 1, 5])

    with col_a:
        team_a = st.selectbox(
            "Team A", wc,
            index=wc.index("Spain") if "Spain" in wc else 0,
            key="pred_ta",
        )
    with col_vs:
        st.html(
            "<div style='text-align:center;padding-top:34px;'>"
            "<span class='vs-badge'>VS</span></div>",

        )
    with col_b:
        default_b = "France" if "France" in wc else wc[1]
        team_b = st.selectbox(
            "Team B", wc,
            index=wc.index(default_b),
            key="pred_tb",
        )

    if team_a == team_b:
        st.warning("Select two different teams.")
        return

    # ── Context strip ──────────────────────────────────────────────────────────
    elo_a = elo.get(team_a, 1500)
    elo_b = elo.get(team_b, 1500)
    diff  = elo_a - elo_b
    grp_a, grp_b = team_group(team_a), team_group(team_b)

    st.html(f"""
    <div style='display:flex;gap:12px;margin:12px 0;align-items:stretch;'>
        <div class='glass-card' style='flex:1;text-align:center;padding:14px;'>
            <div style='font-size:1.2rem;font-weight:800;color:#F1F5F9;'>{team_a}</div>
            <div style='font-size:1.6rem;font-weight:900;color:#38BDF8;margin:4px 0;'>{elo_a:.0f}</div>
            <div style='font-size:0.68rem;color:#475569;text-transform:uppercase;
                        letter-spacing:0.1em;'>Elo · Group {grp_a}</div>
        </div>
        <div style='display:flex;align-items:center;'>
            <div style='font-size:0.75rem;color:#334155;text-align:center;padding:0 8px;'>
                <div style='font-size:1rem;font-weight:800;
                            color:{"#34D399" if diff > 20 else "#F87171" if diff < -20 else "#FCD34D"};'>
                    {diff:+.0f}
                </div>
                <div>Elo diff</div>
            </div>
        </div>
        <div class='glass-card' style='flex:1;text-align:center;padding:14px;'>
            <div style='font-size:1.2rem;font-weight:800;color:#F1F5F9;'>{team_b}</div>
            <div style='font-size:1.6rem;font-weight:900;color:#F87171;margin:4px 0;'>{elo_b:.0f}</div>
            <div style='font-size:0.68rem;color:#475569;text-transform:uppercase;
                        letter-spacing:0.1em;'>Elo · Group {grp_b}</div>
        </div>
    </div>""")

    # ── Predict button ─────────────────────────────────────────────────────────
    predict_btn = st.button("⚡  Predict Match", type="primary", use_container_width=True)

    if not predict_btn and "last_pred" not in st.session_state:
        st.html("""
        <div style='text-align:center;padding:32px;color:#334155;font-size:0.9rem;'>
            Select two teams and click <strong style='color:#38BDF8;'>Predict Match</strong>
        </div>""")
        return

    if predict_btn:
        with st.spinner("Running prediction..."):
            result = _get_prediction(team_a, team_b)
        st.session_state["last_pred"] = result
        st.session_state["pred_ta"]   = team_a
        st.session_state["pred_tb"]   = team_b

    result = st.session_state.get("last_pred", {})
    ta     = st.session_state.get("pred_ta", team_a)
    tb     = st.session_state.get("pred_tb", team_b)
    if not result:
        return

    p_a = result["team_a_win"]
    p_d = result["draw"]
    p_b = result["team_b_win"]
    ea  = result["elo_a_win"]
    ed  = result["elo_draw"]
    eb  = result["elo_b_win"]
    conf = result["confidence"]
    winner = ta if p_a > p_b else (tb if p_b > p_a else "Draw")
    exp_a  = result["expected_goals_a"]
    exp_b  = result["expected_goals_b"]

    st.html("---")

    # ── Results layout ─────────────────────────────────────────────────────────
    left, right = st.columns([3, 2], gap="large")

    with left:
        section_title("Outcome Probabilities", f"{ta} vs {tb}  ·  Neutral venue")

        # ML probs
        st.html(
            f"<div style='font-size:0.72rem;font-weight:700;color:#38BDF8;"
            f"text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;'>"
            f"ML Model (LightGBM)</div>",

        )
        st.html(
            prob_bar(f"🔵 {ta} Win", p_a * 100, "#38BDF8") +
            prob_bar("⬜ Draw",       p_d * 100, "#64748B") +
            prob_bar(f"🔴 {tb} Win", p_b * 100, "#F87171"),

        )

        st.html("<div style='height:14px;'></div>")

        # Elo probs
        st.html(
            f"<div style='font-size:0.72rem;font-weight:700;color:#A78BFA;"
            f"text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;'>"
            f"Elo Simulation Engine</div>",

        )
        st.html(
            prob_bar(f"🔵 {ta} Win", ea * 100, "#A78BFA") +
            prob_bar("⬜ Draw",       ed * 100, "#64748B") +
            prob_bar(f"🔴 {tb} Win", eb * 100, "#F87171"),

        )

        st.html("<div style='height:16px;'></div>")

        # ML vs Elo bar chart
        section_title("Model Comparison")
        outcomes = [f"{ta[:10]} Win", "Draw", f"{tb[:10]} Win"]
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(
            name="ML Model", x=outcomes, y=[p_a*100, p_d*100, p_b*100],
            marker=dict(color=["#38BDF8","#475569","#F87171"], line=dict(width=0)),
            text=[f"{v*100:.1f}%" for v in [p_a, p_d, p_b]],
            textposition="outside", textfont=dict(color="#64748B", size=11),
        ))
        fig_cmp.add_trace(go.Bar(
            name="Elo Engine", x=outcomes, y=[ea*100, ed*100, eb*100],
            marker=dict(color=["rgba(56,189,248,0.4)","rgba(71,85,105,0.4)","rgba(248,113,113,0.4)"],
                        line=dict(width=1, color=["#38BDF8","#475569","#F87171"])),
            text=[f"{v*100:.1f}%" for v in [ea, ed, eb]],
            textposition="outside", textfont=dict(color="#64748B", size=11),
        ))
        fig_cmp.update_layout(**{**CHART, "height": 260, "barmode": "group",
                                  "yaxis": dict(range=[0, 100], title="Probability (%)",
                                                gridcolor="rgba(255,255,255,0.04)"),
                                  "xaxis": dict(tickfont=dict(size=12)),
                                  "legend": dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)")})
        st.plotly_chart(fig_cmp, use_container_width=True)

    with right:
        # Confidence gauge
        conf_color = "#34D399" if conf > 0.6 else ("#FCD34D" if conf > 0.4 else "#F87171")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=conf * 100,
            number={"suffix":"%", "font":{"size":26,"color":"#F1F5F9"}},
            gauge={
                "axis": {"range":[0,100],"tickwidth":1,"tickcolor":"#1C2E4A",
                         "tickfont":{"size":9,"color":"#334155"}},
                "bar":  {"color": conf_color, "thickness":0.3},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range":[0,40],  "color":"rgba(248,113,113,0.12)"},
                    {"range":[40,65], "color":"rgba(252,211,77,0.10)"},
                    {"range":[65,100],"color":"rgba(52,211,153,0.12)"},
                ],
            },
            title={"text":"Confidence","font":{"size":12,"color":"#64748B"}},
        ))
        fig_gauge.update_layout(
            height=200, margin=dict(l=20,r=20,t=40,b=10),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Score prediction card
        winner_color = "#38BDF8" if winner == ta else ("#F87171" if winner == tb else "#FCD34D")
        st.html(f"""
        <div style='
            background:linear-gradient(135deg,#0B1728,#0F2040);
            border:1px solid #1C2E4A;border-top:3px solid {winner_color};
            border-radius:14px;padding:20px;text-align:center;
            box-shadow:0 4px 24px rgba(0,0,0,0.4);
        '>
            <div style='font-size:0.65rem;font-weight:700;color:#334155;
                        text-transform:uppercase;letter-spacing:0.12em;
                        margin-bottom:10px;'>Expected Score</div>
            <div style='font-size:2.8rem;font-weight:900;color:#F1F5F9;
                        letter-spacing:4px;line-height:1;'>
                {result["expected_score"]}
            </div>
            <div style='font-size:0.72rem;color:#475569;margin-top:6px;'>
                xG: {exp_a:.2f} — {exp_b:.2f}
            </div>
            <div style='margin-top:14px;padding:8px;
                        background:rgba(255,255,255,0.04);border-radius:8px;
                        font-size:0.82rem;color:{winner_color};font-weight:700;'>
                Predicted winner: {winner}
            </div>
        </div>""")

        # Score probability heatmap
        st.html("<div style='height:16px;'></div>")
        section_title("Score Matrix", "Poisson probability distribution")

        from src.predict_match import _elo_neutral_probs, _expected_goals
        from scipy.stats import poisson

        pa_e, pd_e, pb_e = _elo_neutral_probs(ta, tb)
        λ_a, λ_b = _expected_goals(pa_e, pd_e, pb_e)

        N = 6
        matrix = np.array([
            [poisson.pmf(i, λ_a) * poisson.pmf(j, λ_b) * 100
             for j in range(N)] for i in range(N)
        ])

        fig_hm = go.Figure(go.Heatmap(
            z=matrix,
            x=[str(j) for j in range(N)],
            y=[str(i) for i in range(N)],
            colorscale=[[0,"#0B1728"],[0.4,"#0EA5E930"],[1,"#38BDF8"]],
            text=[[f"{v:.1f}%" for v in row] for row in matrix],
            texttemplate="%{text}",
            textfont={"size": 9, "color": "white"},
            hovertemplate=f"{ta} %{{y}}–%{{x}} {tb}: <b>%{{z:.2f}}%</b><extra></extra>",
            showscale=False,
        ))
        fig_hm.update_layout(
            **{**CHART, "height": 280,
               "xaxis": dict(title=f"{tb[:12]} goals", tickfont=dict(size=10),
                             gridcolor="rgba(0,0,0,0)"),
               "yaxis": dict(title=f"{ta[:12]} goals", autorange="reversed",
                             tickfont=dict(size=10), gridcolor="rgba(0,0,0,0)"),
               "margin": dict(l=4, r=4, t=10, b=4)},
        )
        st.plotly_chart(fig_hm, use_container_width=True)
        st.caption(f"λ: {ta[:12]}={λ_a:.2f}, {tb[:12]}={λ_b:.2f}  ·  "
                   "Upper-left = Team A wins, diagonal = draws")