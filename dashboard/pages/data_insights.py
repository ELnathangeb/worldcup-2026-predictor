"""Data Insights — feature importance, goal trends, correlations, score distribution."""
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
import numpy as np

from utils import (
    load_dataset, load_model_metadata, load_model_pipeline,
    page_header, section_title, kpi, COLORS, CHART,
)


@st.cache_data(show_spinner=False)
def _feature_importance() -> pd.DataFrame:
    try:
        result    = load_model_pipeline()
        pipe      = result[0] if isinstance(result, (tuple, list)) else result
        feat_cols = result[1] if isinstance(result, (tuple, list)) else None
        model     = (pipe.named_steps.get("clf")
                     or pipe.named_steps.get("model")
                     or list(pipe.named_steps.values())[-1])
        imps      = model.feature_importances_

        if feat_cols is not None and len(feat_cols) == len(imps):
            names = list(feat_cols)
        elif hasattr(model, "feature_name_"):
            names = model.feature_name_()
        else:
            try:
                prev = list(pipe.named_steps.values())[-2]
                if hasattr(prev, "get_feature_names_out"):
                    names = list(prev.get_feature_names_out())
                elif hasattr(prev, "feature_names_in_"):
                    names = list(prev.feature_names_in_)
                else:
                    names = [f"feature_{i}" for i in range(len(imps))]
            except Exception:
                names = [f"feature_{i}" for i in range(len(imps))]

        _rename = {
            "home_elo":"Home Elo","away_elo":"Away Elo","elo_diff":"Elo Difference",
            "home_rank":"Home Rank","away_rank":"Away Rank",
            "home_form5":"Home Form 5g","away_form5":"Away Form 5g",
            "home_form10":"Home Form 10g","away_form10":"Away Form 10g",
            "home_avg_gf5":"Home Goals 5g","away_avg_gf5":"Away Goals 5g",
            "home_win_rate5":"Home WR 5g","away_win_rate5":"Away WR 5g",
            "neutral":"Neutral Venue","is_world_cup":"World Cup Match",
            "h2h_home_wins":"H2H Wins","h2h_away_wins":"H2H Away Wins",
            "h2h_draws":"H2H Draws","rank_diff":"Rank Diff",
            "home_gf_avg5":"Home GF Avg 5g","away_gf_avg5":"Away GF Avg 5g",
            "home_ga_avg5":"Home GA Avg 5g","away_ga_avg5":"Away GA Avg 5g",
        }
        df = pd.DataFrame({"feature": names, "importance": imps})
        df["feature"] = df["feature"].map(lambda f: _rename.get(f, f))
        return df.sort_values("importance", ascending=False).head(25).reset_index(drop=True)
    except Exception as e:
        return pd.DataFrame({"feature": [f"Error loading model: {e}"], "importance": [0]})


@st.cache_data(show_spinner=False)
def _goal_trends(_df: pd.DataFrame) -> pd.DataFrame:
    df = _df.copy()
    df["year"] = df["date"].dt.year
    return df.groupby("year").agg(
        avg_home=("home_score","mean"),
        avg_away=("away_score","mean"),
        avg_total=("home_score", lambda x: (x + _df.loc[x.index,"away_score"]).mean()),
        matches=("home_score","count"),
    ).reset_index().query("matches >= 20")


@st.cache_data(show_spinner=False)
def _corr_matrix(_df: pd.DataFrame) -> pd.DataFrame:
    num_cols = ["home_elo","away_elo","elo_diff","home_rank","away_rank",
                "home_form5","away_form5","home_avg_gf5","away_avg_gf5",
                "home_score","away_score","result"]
    existing = [c for c in num_cols if c in _df.columns]
    return _df[existing].corr()


@st.cache_data(show_spinner=False)
def _top_teams(_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    home = _df.groupby("home_team").agg(
        hw=("result", lambda x: (x==2).sum()), hm=("result","count")
    ).rename_axis("team")
    away = _df.groupby("away_team").agg(
        aw=("result", lambda x: (x==0).sum()), am=("result","count")
    ).rename_axis("team")
    c = home.join(away, how="outer").fillna(0)
    c["wins"]    = c["hw"] + c["aw"]
    c["matches"] = c["hm"] + c["am"]
    c["win_pct"] = c["wins"] / c["matches"] * 100
    return c.reset_index().nlargest(top_n, "wins")


def render() -> None:
    page_header("Data Insights", "Feature importance, goal trends, correlations & score distribution")

    df   = load_dataset()
    meta = load_model_metadata()
    tune = meta.get("tune_results", {})

    tabs = st.tabs([
        "🧠 Feature Importance", "⚽ Goal Trends", "🔥 Correlations",
        "🏆 Top Teams", "📐 Score Distribution",
    ])

    # ── Tab 1: Feature importance ──────────────────────────────────────────────
    with tabs[0]:
        fi = _feature_importance()
        if "Error" in fi["feature"].iloc[0]:
            st.warning(fi["feature"].iloc[0])
        else:
            section_title("LightGBM Feature Importance", "Gain-based importance across 46 features")

            total = fi["importance"].sum()
            fig_fi = go.Figure(go.Bar(
                x=fi["importance"][::-1],
                y=fi["feature"][::-1],
                orientation="h",
                marker=dict(
                    color=fi["importance"][::-1],
                    colorscale=[[0,"#0C3D5F"],[0.4,"#0EA5E9"],[1,"#38BDF8"]],
                    line=dict(width=0),
                ),
                text=[f"{v/total*100:.1f}%" for v in fi["importance"][::-1]],
                textposition="outside",
                textfont=dict(color="#475569", size=10),
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.0f}<br>"
                              "Share: %{text}<extra></extra>",
            ))
            fig_fi.update_layout(**{**CHART, "height": 560, "showlegend": False,
                                    "margin": dict(l=4, r=60, t=20, b=4),
                                    "yaxis": dict(tickfont=dict(size=10, color="#94A3B8")),
                                    "xaxis": dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)",
                                                  title="Importance (gain)")})
            st.plotly_chart(fig_fi, use_container_width=True)

            # Top 5 callout
            section_title("Top 5 Most Predictive Features")
            cols = st.columns(5)
            for col, (_, row) in zip(cols, fi.head(5).iterrows()):
                pct = row["importance"] / total * 100
                col.markdown(kpi(f"{pct:.1f}%", row["feature"][:18], accent="#38BDF8"),
                             unsafe_allow_html=True)

        st.markdown("---")
        section_title("Model Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(kpi(f"{tune.get('test_accuracy',0)*100:.2f}%", "Test Accuracy", accent="#34D399"),
                    unsafe_allow_html=True)
        c2.markdown(kpi(f"{tune.get('test_f1_macro',0):.4f}", "F1 Macro",     accent="#38BDF8"),
                    unsafe_allow_html=True)
        c3.markdown(kpi(f"{tune.get('test_log_loss',0):.4f}", "Log Loss",     accent="#F87171"),
                    unsafe_allow_html=True)
        c4.markdown(kpi(f"{tune.get('best_cv_f1',0):.4f}",   "Best CV F1",   accent="#A78BFA"),
                    unsafe_allow_html=True)

    # ── Tab 2: Goal trends ─────────────────────────────────────────────────────
    with tabs[1]:
        annual = _goal_trends(df)

        section_title("Average Goals per Match (by Year)", "Rolling yearly averages")
        fig_g = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=("Goals per Match", "Matches per Year"),
            vertical_spacing=0.1, row_heights=[0.65, 0.35],
        )
        fig_g.add_trace(go.Scatter(
            x=annual["year"], y=annual["avg_home"], name="Home",
            line=dict(color="#38BDF8", width=2.5),
            fill="tozeroy", fillcolor="rgba(56,189,248,0.07)",
        ), row=1, col=1)
        fig_g.add_trace(go.Scatter(
            x=annual["year"], y=annual["avg_away"], name="Away",
            line=dict(color="#F87171", width=2),
            fill="tozeroy", fillcolor="rgba(248,113,113,0.05)",
        ), row=1, col=1)
        fig_g.add_trace(go.Scatter(
            x=annual["year"], y=annual["avg_total"], name="Total",
            line=dict(color="#FCD34D", width=2, dash="dot"),
        ), row=1, col=1)
        fig_g.add_trace(go.Bar(
            x=annual["year"], y=annual["matches"],
            name="Matches", marker_color="#1C2E4A", marker_line_width=0,
        ), row=2, col=1)
        fig_g.update_layout(**{**CHART, "height": 460,
                                "legend": dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)")})
        fig_g.update_yaxes(gridcolor="rgba(255,255,255,0.04)")
        st.plotly_chart(fig_g, use_container_width=True)

        # Home advantage
        section_title("Home Advantage Over Time")
        df2 = df.copy()
        df2["year"] = df2["date"].dt.year
        ha = df2.groupby("year").apply(lambda g: pd.Series({
            "home_win_pct": (g["result"]==2).mean()*100,
            "draw_pct":     (g["result"]==1).mean()*100,
            "away_win_pct": (g["result"]==0).mean()*100,
            "n": len(g),
        })).reset_index().query("n >= 30")

        fig_ha = go.Figure()
        fig_ha.add_trace(go.Scatter(
            x=ha["year"], y=ha["home_win_pct"], name="Home Win",
            stackgroup="one", fillcolor="rgba(56,189,248,0.25)",
            line=dict(color="#38BDF8", width=1.5),
        ))
        fig_ha.add_trace(go.Scatter(
            x=ha["year"], y=ha["draw_pct"], name="Draw",
            stackgroup="one", fillcolor="rgba(71,85,105,0.4)",
            line=dict(color="#475569", width=1.5),
        ))
        fig_ha.add_trace(go.Scatter(
            x=ha["year"], y=ha["away_win_pct"], name="Away Win",
            stackgroup="one", fillcolor="rgba(248,113,113,0.2)",
            line=dict(color="#F87171", width=1.5),
        ))
        fig_ha.update_layout(**{**CHART, "height": 300,
                                 "yaxis": dict(title="%", gridcolor="rgba(255,255,255,0.04)"),
                                 "legend": dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)")})
        st.plotly_chart(fig_ha, use_container_width=True)

    # ── Tab 3: Correlations ────────────────────────────────────────────────────
    with tabs[2]:
        corr = _corr_matrix(df)
        if not corr.empty:
            section_title("Feature Correlation Matrix")
            fig_c = go.Figure(go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                colorscale=[[0,"#2d0f0f"],[0.25,"#1C2030"],[0.5,"#0D1526"],
                            [0.75,"#0C2E40"],[1,"#38BDF8"]],
                zmid=0, zmin=-1, zmax=1,
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
                textfont={"size": 8, "color": "rgba(255,255,255,0.6)"},
                hovertemplate="%{y} × %{x}: <b>%{z:.3f}</b><extra></extra>",
                showscale=True,
                colorbar=dict(
                    tickfont=dict(color="#64748B", size=10),
                    bgcolor="rgba(0,0,0,0)",
                    bordercolor="#1C2E4A",
                ),
            ))
            fig_c.update_layout(**{**CHART, "height": 500,
                                   "xaxis": dict(tickangle=-35, tickfont=dict(size=9, color="#64748B")),
                                   "yaxis": dict(tickfont=dict(size=9, color="#64748B"), autorange="reversed"),
                                   "margin": dict(l=4, r=4, t=20, b=4)})
            st.plotly_chart(fig_c, use_container_width=True)

            if "result" in corr.columns:
                section_title("Top Correlates with Match Result")
                rc = corr["result"].drop("result").abs().sort_values(ascending=False)
                fig_rc = go.Figure(go.Bar(
                    x=rc.values[:12],
                    y=rc.index[:12],
                    orientation="h",
                    marker=dict(
                        color=rc.values[:12],
                        colorscale=[[0,"#0C2E40"],[1,"#38BDF8"]],
                        line=dict(width=0),
                    ),
                    text=[f"{v:.3f}" for v in rc.values[:12]],
                    textposition="outside",
                    textfont=dict(color="#475569", size=10),
                ))
                fig_rc.update_layout(**{**CHART, "height": 320, "showlegend": False,
                                        "margin": dict(l=4, r=50, t=20, b=4),
                                        "xaxis": dict(title="|Correlation|",
                                                      gridcolor="rgba(255,255,255,0.04)"),
                                        "yaxis": dict(tickfont=dict(size=10, color="#94A3B8"))})
                st.plotly_chart(fig_rc, use_container_width=True)

    # ── Tab 4: Top teams ───────────────────────────────────────────────────────
    with tabs[3]:
        top_n_val = st.slider("Show top N teams", 5, 40, 20, key="dt_topn")
        top_teams = _top_teams(df, top_n_val)

        section_title(f"Top {top_n_val} Teams by Total Wins (All-time)")
        fig_top = go.Figure(go.Bar(
            x=top_teams.sort_values("wins")["wins"],
            y=top_teams.sort_values("wins")["team"],
            orientation="h",
            marker=dict(
                color=top_teams.sort_values("wins")["win_pct"],
                colorscale=[[0,"#0C2E40"],[0.5,"#0EA5E9"],[1,"#38BDF8"]],
                line=dict(width=0),
            ),
            text=[f"{v:.0f}%" for v in top_teams.sort_values("wins")["win_pct"]],
            textposition="outside",
            textfont=dict(color="#475569", size=10),
            hovertemplate="<b>%{y}</b><br>%{x} wins<br>Win rate: %{text}<extra></extra>",
        ))
        fig_top.update_layout(**{**CHART, "height": max(340, top_n_val*22),
                                  "showlegend": False,
                                  "margin": dict(l=4, r=50, t=20, b=4),
                                  "xaxis": dict(title="Total Wins", gridcolor="rgba(255,255,255,0.04)"),
                                  "yaxis": dict(tickfont=dict(size=10, color="#94A3B8"))})
        st.plotly_chart(fig_top, use_container_width=True)

    # ── Tab 5: Score distribution ──────────────────────────────────────────────
    with tabs[4]:
        section_title("Goal Distribution")
        col_l, col_r = st.columns(2)

        with col_l:
            home_sc = df["home_score"].clip(0,10).astype(int).value_counts().sort_index()
            away_sc = df["away_score"].clip(0,10).astype(int).value_counts().sort_index()
            fig_d = go.Figure()
            fig_d.add_trace(go.Bar(
                x=home_sc.index, y=home_sc.values, name="Home",
                marker=dict(color="#38BDF8", line=dict(width=0)), opacity=0.85,
            ))
            fig_d.add_trace(go.Bar(
                x=away_sc.index, y=away_sc.values, name="Away",
                marker=dict(color="#F87171", line=dict(width=0)), opacity=0.85,
            ))
            fig_d.update_layout(**{**CHART, "height": 300, "barmode": "group",
                                   "xaxis": dict(title="Goals", dtick=1),
                                   "yaxis": dict(title="Matches", gridcolor="rgba(255,255,255,0.04)"),
                                   "legend": dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)")})
            st.plotly_chart(fig_d, use_container_width=True)

        with col_r:
            df_s    = df.copy()
            df_s["score"] = (df_s["home_score"].astype(int).astype(str)
                             + "–" + df_s["away_score"].astype(int).astype(str))
            top_sc  = df_s["score"].value_counts().head(15)
            fig_sc  = go.Figure(go.Bar(
                x=top_sc.values[::-1],
                y=top_sc.index[::-1],
                orientation="h",
                marker=dict(
                    color=top_sc.values[::-1],
                    colorscale=[[0,"#0C2E40"],[1,"#38BDF8"]],
                    line=dict(width=0),
                ),
                hovertemplate="<b>%{y}</b>: %{x} matches<extra></extra>",
            ))
            fig_sc.update_layout(**{**CHART, "height": 300, "showlegend": False,
                                    "margin": dict(l=4, r=4, t=20, b=4),
                                    "xaxis": dict(title="Frequency", gridcolor="rgba(255,255,255,0.04)"),
                                    "yaxis": dict(tickfont=dict(size=11, color="#94A3B8"))})
            st.plotly_chart(fig_sc, use_container_width=True)

        section_title("Historical Score Heatmap", "% of all matches ending with each exact scoreline")
        df_h = df.copy()
        df_h["hs"] = df_h["home_score"].clip(0,7).astype(int)
        df_h["as_"] = df_h["away_score"].clip(0,7).astype(int)
        pivot     = df_h.groupby(["hs","as_"]).size().unstack(fill_value=0)
        pivot_pct = pivot / pivot.values.sum() * 100

        fig_hm = go.Figure(go.Heatmap(
            z=pivot_pct.values,
            x=[str(c) for c in pivot_pct.columns],
            y=[str(i) for i in pivot_pct.index],
            colorscale=[[0,"#0B1728"],[0.3,"#0C3D5F"],[0.7,"#0EA5E9"],[1,"#38BDF8"]],
            text=np.round(pivot_pct.values, 1),
            texttemplate="%{text}%",
            textfont={"size": 9, "color": "rgba(255,255,255,0.7)"},
            hovertemplate="Home %{y}–Away %{x}: <b>%{z:.2f}%</b><extra></extra>",
            showscale=True,
            colorbar=dict(tickfont=dict(color="#64748B", size=10), bgcolor="rgba(0,0,0,0)"),
        ))
        fig_hm.update_layout(**{**CHART, "height": 420,
                                 "xaxis": dict(title="Away Goals", tickfont=dict(size=11)),
                                 "yaxis": dict(title="Home Goals"),
                                 "margin": dict(l=4, r=4, t=20, b=4)})
        st.plotly_chart(fig_hm, use_container_width=True)
        st.caption("Scores clipped at 7. Cells = % of all historical matches.")
