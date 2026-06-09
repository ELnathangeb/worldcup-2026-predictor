"""Page 3 — Match Predictor: predict any WC 2026 match."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import math

from utils import (
    load_elo, all_wc_teams, load_dataset,
    page_header, COLORS, team_group,
)


@st.cache_data(show_spinner=False)
def _get_prediction(team_a: str, team_b: str) -> dict:
    from src.predict_match import predict_match, _elo_neutral_probs, _expected_goals
    from src.predict_match import _load_recent_team_stats, _get_rank_info, _load_dataset, _load_elo
    _load_dataset.cache_clear()
    _load_recent_team_stats.cache_clear()
    _get_rank_info.cache_clear()
    result = predict_match(team_a, team_b)
    # Also compute Elo-based probs for comparison
    pa, pd_, pb = _elo_neutral_probs(team_a, team_b)
    result["elo_a_win"] = round(pa, 4)
    result["elo_draw"]  = round(pd_, 4)
    result["elo_b_win"] = round(pb, 4)
    return result


def _prob_bar(label: str, prob: float, color_class: str, color_hex: str) -> str:
    pct = prob * 100
    bar_w = max(4, int(pct))
    return f"""
    <div class="prob-bar-container">
        <div class="prob-label">{label}</div>
        <div class="prob-bar-bg">
            <div class="prob-bar-fill {color_class}"
                 style="width:{bar_w}%; background:{color_hex};">
                {pct:.1f}%
            </div>
        </div>
    </div>"""


def _confidence_gauge(conf: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=conf * 100,
        number={"suffix": "%", "font": {"size": 28, "color": COLORS["primary"]}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1,
                     "tickcolor": "#CBD5E1", "tickfont": {"size": 10}},
            "bar": {"color": COLORS["primary"], "thickness": 0.35},
            "bgcolor": "white",
            "steps": [
                {"range": [0, 40],  "color": "#FEE2E2"},
                {"range": [40, 65], "color": "#FEF9C3"},
                {"range": [65, 100],"color": "#D1FAE5"},
            ],
            "threshold": {
                "line": {"color": COLORS["secondary"], "width": 3},
                "thickness": 0.85,
                "value": conf * 100,
            },
        },
        title={"text": "Prediction Confidence", "font": {"size": 14, "color": "#6B7280"}},
    ))
    fig.update_layout(
        height=200, margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render() -> None:
    page_header("Match Predictor", "Predict the outcome of any WC 2026 match")

    elo = load_elo()
    wc  = all_wc_teams()

    # ── Team selection ─────────────────────────────────────────────────────────
    c_a, c_vs, c_b = st.columns([5, 1, 5])
    with c_a:
        st.markdown("#### 🔵 Team A")
        team_a = st.selectbox("Team A", wc, index=wc.index("Spain") if "Spain" in wc else 0,
                              key="pred_ta", label_visibility="collapsed")
    with c_vs:
        st.markdown("<div style='text-align:center;font-size:1.8rem;margin-top:32px;'>⚡</div>",
                    unsafe_allow_html=True)
    with c_b:
        st.markdown("#### 🔴 Team B")
        default_b = "France" if "France" in wc else wc[1]
        team_b = st.selectbox("Team B", wc, index=wc.index(default_b),
                              key="pred_tb", label_visibility="collapsed")

    if team_a == team_b:
        st.warning("Please select two different teams.")
        return

    # ── Elo context ────────────────────────────────────────────────────────────
    elo_a, elo_b = elo.get(team_a, 1500), elo.get(team_b, 1500)
    diff = elo_a - elo_b

    cc1, cc2, cc3 = st.columns(3)
    cc1.metric(f"{team_a} Elo", f"{elo_a:.0f}", delta=f"{diff:+.0f} vs opponent")
    cc2.metric("Group",
               f"{'Same' if team_group(team_a)==team_group(team_b) else 'Different'} group",
               f"A: {team_group(team_a)} · B: {team_group(team_b)}")
    cc3.metric(f"{team_b} Elo", f"{elo_b:.0f}", delta=f"{-diff:+.0f} vs opponent")

    st.markdown("")
    predict_btn = st.button("⚡  Predict Match", type="primary", use_container_width=True)

    if not predict_btn and "last_pred" not in st.session_state:
        st.info("👆 Select two teams and click **Predict Match** to see the result.")
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

    st.markdown("---")
    st.markdown(f"### Prediction: **{ta}** vs **{tb}** *(neutral venue)*")

    # ── Main prediction display ────────────────────────────────────────────────
    left, right = st.columns([3, 2])

    with left:
        p_a = result["team_a_win"]
        p_d = result["draw"]
        p_b = result["team_b_win"]

        st.markdown("#### Outcome Probabilities (ML Model)")
        st.markdown(
            _prob_bar(f"🔵 {ta} Win", p_a, "win-bar", "#1E3A5F") +
            _prob_bar("⬜ Draw",       p_d, "draw-bar", "#6B7280") +
            _prob_bar(f"🔴 {tb} Win", p_b, "lose-bar", "#C8102E"),
            unsafe_allow_html=True,
        )

        st.markdown("#### Elo-Based Probabilities (Simulation Engine)")
        ea, ed, eb = result["elo_a_win"], result["elo_draw"], result["elo_b_win"]
        st.markdown(
            _prob_bar(f"🔵 {ta} Win", ea, "win-bar", "#1E3A5F") +
            _prob_bar("⬜ Draw",       ed, "draw-bar", "#6B7280") +
            _prob_bar(f"🔴 {tb} Win", eb, "lose-bar", "#C8102E"),
            unsafe_allow_html=True,
        )

    with right:
        # Gauge
        st.plotly_chart(_confidence_gauge(result["confidence"]),
                        use_container_width=True)

        # Score prediction
        exp_a = result["expected_goals_a"]
        exp_b = result["expected_goals_b"]
        winner = ta if p_a > p_b else (tb if p_b > p_a else "Draw")
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#1E3A5F,#2563EB);
                    border-radius:14px; padding:20px; text-align:center; color:white;
                    margin-top:8px;'>
            <div style='font-size:0.85rem; opacity:0.8; margin-bottom:4px;'>Expected Score</div>
            <div style='font-size:2.2rem; font-weight:800; letter-spacing:2px;'>
                {result['expected_score']}
            </div>
            <div style='font-size:0.78rem; opacity:0.7; margin-top:4px;'>
                xG: {exp_a:.2f} — {exp_b:.2f}
            </div>
            <div style='margin-top:12px; font-size:0.9rem; background:rgba(255,255,255,0.15);
                        border-radius:8px; padding:6px;'>
                Predicted winner: <strong>{winner}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Comparison radar / bar ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Probability Breakdown — ML vs Elo")
    fig = go.Figure()
    outcomes = [f"{ta} Win", "Draw", f"{tb} Win"]
    ml_vals  = [p_a*100, p_d*100, p_b*100]
    elo_vals = [ea*100, ed*100, eb*100]

    fig.add_trace(go.Bar(
        name="ML Model", x=outcomes, y=ml_vals,
        marker_color=COLORS["primary"], text=[f"{v:.1f}%" for v in ml_vals],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Elo Model", x=outcomes, y=elo_vals,
        marker_color=COLORS["secondary"], text=[f"{v:.1f}%" for v in elo_vals],
        textposition="outside",
    ))
    fig.update_layout(
        barmode="group", height=320,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(title="Probability (%)", showgrid=True, gridcolor="#F0F0F0", range=[0,100]),
        legend=dict(orientation="h", yanchor="bottom", y=1),
        xaxis=dict(tickfont=dict(size=13)),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Score distribution heatmap ─────────────────────────────────────────────
    st.markdown("#### Score Probability Matrix (Poisson)")
    from src.predict_match import _elo_neutral_probs, _expected_goals
    pa_e, pd_e, pb_e = _elo_neutral_probs(ta, tb)
    λ_a, λ_b = _expected_goals(pa_e, pd_e, pb_e)

    max_goals = 5
    from scipy.stats import poisson  # type: ignore
    matrix = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            matrix[i][j] = poisson.pmf(i, λ_a) * poisson.pmf(j, λ_b) * 100

    import plotly.figure_factory as ff
    text_matrix = [[f"{matrix[i][j]:.1f}%" for j in range(max_goals+1)]
                   for i in range(max_goals+1)]
    fig_heat = ff.create_annotated_heatmap(
        z=matrix,
        x=[str(j) for j in range(max_goals+1)],
        y=[str(i) for i in range(max_goals+1)],
        annotation_text=text_matrix,
        colorscale="Blues",
        showscale=True,
    )
    fig_heat.update_layout(
        height=380, margin=dict(l=40, r=0, t=60, b=40),
        title=dict(
            text=f"Score probabilities (Poisson)  —  xG: {ta}={λ_a:.2f}, {tb}={λ_b:.2f}",
            font_size=13,
        ),
        xaxis=dict(title=f"{tb} goals", side="bottom"),
        yaxis=dict(title=f"{ta} goals", autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption("Each cell = P(score A–B) from independent Poisson distributions. "
               "Diagonal = draws, upper-left = Team A wins, lower-right = Team B wins.")
