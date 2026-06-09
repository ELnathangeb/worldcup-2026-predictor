"""Page 6 — Data Insights: feature importance, goal trends, correlation heatmap."""
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
    load_dataset, load_model_metadata, load_model_pipeline,
    page_header, COLORS,
)


@st.cache_data(show_spinner=False)
def _feature_importance() -> pd.DataFrame:
    """Extract feature importances from the saved LightGBM model."""
    try:
        result = load_model_pipeline()
        # load_model_pipeline returns (pipe, feat_cols) tuple
        pipe      = result[0] if isinstance(result, (tuple, list)) else result
        feat_cols = result[1] if isinstance(result, (tuple, list)) else None

        model = pipe.named_steps.get("clf") or pipe.named_steps.get("model") or list(pipe.named_steps.values())[-1]
        importances = model.feature_importances_

        # Try feature names: passed feat_cols > model attribute > fallback
        if feat_cols is not None and len(feat_cols) == len(importances):
            names = list(feat_cols)
        elif hasattr(model, "feature_name_"):
            names = model.feature_name_()
        else:
            try:
                prev_step = list(pipe.named_steps.values())[-2]
                if hasattr(prev_step, "get_feature_names_out"):
                    names = prev_step.get_feature_names_out()
                elif hasattr(prev_step, "feature_names_in_"):
                    names = prev_step.feature_names_in_
                else:
                    names = [f"feature_{i}" for i in range(len(importances))]
            except Exception:
                names = [f"feature_{i}" for i in range(len(importances))]

        df = pd.DataFrame({"feature": names, "importance": importances})
        df = df.sort_values("importance", ascending=False).head(30).reset_index(drop=True)
        # Friendly renaming
        rename_map = {
            "home_elo": "Home Elo", "away_elo": "Away Elo",
            "elo_diff": "Elo Difference",
            "home_rank": "Home FIFA Rank", "away_rank": "Away FIFA Rank",
            "home_form5": "Home Form (5g)", "away_form5": "Away Form (5g)",
            "home_form10": "Home Form (10g)", "away_form10": "Away Form (10g)",
            "home_avg_gf5": "Home Avg Goals (5g)", "away_avg_gf5": "Away Avg Goals (5g)",
            "home_win_rate5": "Home Win Rate (5g)", "away_win_rate5": "Away Win Rate (5g)",
            "neutral": "Neutral Venue",
            "h2h_home_wins": "H2H Home Wins", "h2h_away_wins": "H2H Away Wins",
            "h2h_draws": "H2H Draws",
        }
        df["feature"] = df["feature"].map(lambda f: rename_map.get(f, f))
        return df
    except Exception as e:
        return pd.DataFrame({"feature": [f"Error: {e}"], "importance": [0]})


@st.cache_data(show_spinner=False)
def _goal_trends(_df: pd.DataFrame) -> pd.DataFrame:
    df = _df.copy()
    df["year"] = df["date"].dt.year
    annual = df.groupby("year").agg(
        avg_home_goals=("home_score", "mean"),
        avg_away_goals=("away_score", "mean"),
        avg_total=("home_score", lambda x: (x + df.loc[x.index, "away_score"]).mean()),
        matches=("home_score", "count"),
    ).reset_index()
    return annual[annual["matches"] >= 20]


@st.cache_data(show_spinner=False)
def _corr_matrix(_df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [
        "home_elo", "away_elo", "elo_diff",
        "home_rank", "away_rank",
        "home_form5", "away_form5",
        "home_avg_gf5", "away_avg_gf5",
        "home_score", "away_score",
        "result",
    ]
    existing = [c for c in num_cols if c in _df.columns]
    return _df[existing].corr()


@st.cache_data(show_spinner=False)
def _top_teams(_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    home = _df.groupby("home_team").agg(
        home_wins=("result", lambda x: (x == 2).sum()),
        home_matches=("result", "count"),
    ).rename_axis("team")
    away = _df.groupby("away_team").agg(
        away_wins=("result", lambda x: (x == 0).sum()),
        away_matches=("result", "count"),
    ).rename_axis("team")
    combined = home.join(away, how="outer").fillna(0)
    combined["total_wins"]    = combined["home_wins"] + combined["away_wins"]
    combined["total_matches"] = combined["home_matches"] + combined["away_matches"]
    combined["win_pct"]       = combined["total_wins"] / combined["total_matches"] * 100
    return combined.reset_index().nlargest(top_n, "total_wins")


def render() -> None:
    page_header("Data Insights", "Feature importance, goal trends, correlations, and team stats")

    df   = load_dataset()
    meta = load_model_metadata()

    tabs = st.tabs([
        "🧠 Feature Importance", "⚽ Goal Trends", "🔥 Correlation Matrix",
        "🏆 Top Teams", "📐 Score Distribution",
    ])

    # ── Tab 1: Feature Importance ──────────────────────────────────────────────
    with tabs[0]:
        st.markdown('<div class="section-header">LightGBM Feature Importance (top 30)</div>',
                    unsafe_allow_html=True)
        fi = _feature_importance()

        if "Error" in fi["feature"].iloc[0]:
            st.warning(f"Could not load model: {fi['feature'].iloc[0]}")
        else:
            fig_fi = px.bar(
                fi[::-1], x="importance", y="feature", orientation="h",
                color="importance",
                color_continuous_scale=[[0, "#BFDBFE"], [0.5, "#1D4ED8"], [1, "#1E3A5F"]],
                labels={"importance": "Importance Score", "feature": ""},
                title="Feature Importance — LightGBM (gain)",
            )
            fig_fi.update_layout(
                height=640,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=40, t=40, b=0),
                coloraxis_showscale=False,
                xaxis=dict(showgrid=True, gridcolor="#F0F0F0"),
                yaxis=dict(tickfont=dict(size=10, color="#1E3A5F")),
            )
            st.plotly_chart(fig_fi, use_container_width=True)

            # Top 5 callout
            top5 = fi.head(5)
            st.markdown("**Top 5 most predictive features:**")
            for _, row in top5.iterrows():
                pct = row["importance"] / fi["importance"].sum() * 100
                st.markdown(f"- **{row['feature']}** — `{row['importance']:.0f}` ({pct:.1f}% of total)")

        # Model card
        st.markdown("---")
        st.markdown('<div class="section-header">Model Performance Summary</div>',
                    unsafe_allow_html=True)
        tune = meta.get("tune_results", {})
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Test Accuracy",  f"{tune.get('test_accuracy', 0)*100:.2f}%")
        col_b.metric("F1 Macro",       f"{tune.get('test_f1_macro', 0):.4f}")
        col_c.metric("Log Loss",       f"{tune.get('test_log_loss', 0):.4f}")
        col_d.metric("Best CV F1",     f"{tune.get('best_cv_f1', 0):.4f}")

        st.markdown("""
        > **Interpretation note:** The model uses the ML classifier for individual match
        > predictions in the *Match Predictor* page. For tournament simulation, an
        > Elo-based neutral-venue model is used instead, as WC training labels encode
        > a home/away assignment artifact that inverts predictions at neutral venues.
        """)

    # ── Tab 2: Goal Trends ─────────────────────────────────────────────────────
    with tabs[1]:
        annual = _goal_trends(df)
        fig_g = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=("Average Goals per Match (by Year)", "Total Matches per Year"),
            vertical_spacing=0.12, row_heights=[0.65, 0.35],
        )
        fig_g.add_trace(go.Scatter(
            x=annual["year"], y=annual["avg_home_goals"],
            name="Home Goals", line=dict(color=COLORS["primary"], width=2.5),
            fill="tozeroy", fillcolor="rgba(30,58,95,0.08)",
        ), row=1, col=1)
        fig_g.add_trace(go.Scatter(
            x=annual["year"], y=annual["avg_away_goals"],
            name="Away Goals", line=dict(color=COLORS["secondary"], width=2.5),
            fill="tozeroy", fillcolor="rgba(200,16,46,0.06)",
        ), row=1, col=1)
        fig_g.add_trace(go.Scatter(
            x=annual["year"], y=annual["avg_total"],
            name="Total Goals", line=dict(color=COLORS["accent"], width=2, dash="dash"),
        ), row=1, col=1)
        fig_g.add_trace(go.Bar(
            x=annual["year"], y=annual["matches"],
            name="Matches",
            marker_color="#E2E8F0",
            showlegend=True,
        ), row=2, col=1)

        fig_g.update_layout(
            height=480, margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig_g.update_yaxes(showgrid=True, gridcolor="#F0F0F0")
        st.plotly_chart(fig_g, use_container_width=True)

        # Home advantage trend
        st.markdown("---")
        st.markdown('<div class="section-header">Home Advantage Over Time</div>',
                    unsafe_allow_html=True)
        df["year"] = df["date"].dt.year
        home_adv = df.groupby("year").apply(lambda g: pd.Series({
            "home_win_pct": (g["result"] == 2).mean() * 100,
            "draw_pct":     (g["result"] == 1).mean() * 100,
            "away_win_pct": (g["result"] == 0).mean() * 100,
            "n": len(g),
        })).reset_index()
        home_adv = home_adv[home_adv["n"] >= 30]

        fig_ha = px.area(
            home_adv, x="year",
            y=["home_win_pct", "draw_pct", "away_win_pct"],
            color_discrete_map={
                "home_win_pct": COLORS["primary"],
                "draw_pct":     "#9CA3AF",
                "away_win_pct": COLORS["secondary"],
            },
            title="Match Outcome % by Year (stacked area)",
            labels={"value": "%", "variable": "Outcome", "year": "Year"},
        )
        fig_ha.update_layout(
            height=340, margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        fig_ha.for_each_trace(lambda t: t.update(
            name={"home_win_pct": "Home Win", "draw_pct": "Draw", "away_win_pct": "Away Win"}
            .get(t.name, t.name)
        ))
        st.plotly_chart(fig_ha, use_container_width=True)

    # ── Tab 3: Correlation Matrix ──────────────────────────────────────────────
    with tabs[2]:
        corr = _corr_matrix(df)
        if not corr.empty:
            fig_corr = go.Figure(go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                colorscale="RdBu_r",
                zmid=0, zmin=-1, zmax=1,
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
                textfont={"size": 9},
                hovertemplate="%{x} × %{y}: %{z:.3f}<extra></extra>",
            ))
            fig_corr.update_layout(
                title="Feature Correlation Matrix",
                height=520,
                margin=dict(l=0, r=0, t=50, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(tickangle=-35, tickfont=dict(size=9)),
                yaxis=dict(tickfont=dict(size=9), autorange="reversed"),
            )
            st.plotly_chart(fig_corr, use_container_width=True)

            # Strongest correlations with result
            if "result" in corr.columns:
                st.markdown("**Correlations with match result:**")
                result_corr = corr["result"].drop("result").abs().sort_values(ascending=False)
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(
                        pd.DataFrame({
                            "Feature": result_corr.index,
                            "|Correlation|": result_corr.values.round(4),
                        }),
                        hide_index=True, use_container_width=True,
                    )
                with col2:
                    fig_rc = px.bar(
                        x=result_corr.values[:10],
                        y=result_corr.index[:10],
                        orientation="h",
                        color=result_corr.values[:10],
                        color_continuous_scale=[[0,"#BFDBFE"],[1,"#1E3A5F"]],
                        labels={"x": "|Correlation|", "y": ""},
                    )
                    fig_rc.update_layout(
                        height=320, showlegend=False, coloraxis_showscale=False,
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=0, r=0, t=10, b=0),
                    )
                    st.plotly_chart(fig_rc, use_container_width=True)

    # ── Tab 4: Most Successful Teams ──────────────────────────────────────────
    with tabs[3]:
        top_n_val = st.slider("Show top N teams", 5, 40, 20)
        top_teams = _top_teams(df, top_n_val)

        fig_top = px.bar(
            top_teams.sort_values("total_wins"), x="total_wins", y="team",
            orientation="h",
            color="win_pct",
            color_continuous_scale=[[0,"#93C5FD"],[0.5,"#1D4ED8"],[1,"#1E3A5F"]],
            hover_data={"total_matches": True, "win_pct": ":.1f"},
            labels={"total_wins": "Total Wins", "team": "", "win_pct": "Win %"},
            title=f"Top {top_n_val} Most Successful Teams (All-time)",
        )
        fig_top.update_layout(
            height=max(340, top_n_val*22),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=40, t=40, b=0),
            coloraxis_colorbar=dict(title="Win %"),
            xaxis=dict(showgrid=True, gridcolor="#F0F0F0"),
        )
        st.plotly_chart(fig_top, use_container_width=True)

    # ── Tab 5: Score Distribution ─────────────────────────────────────────────
    with tabs[4]:
        col_l, col_r = st.columns(2)
        with col_l:
            # Goal distribution histogram
            home_scores = df["home_score"].dropna().clip(0, 10).astype(int).value_counts().sort_index()
            away_scores = df["away_score"].dropna().clip(0, 10).astype(int).value_counts().sort_index()
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Bar(
                x=home_scores.index, y=home_scores.values,
                name="Home Goals",
                marker_color=COLORS["primary"],
                opacity=0.8,
            ))
            fig_dist.add_trace(go.Bar(
                x=away_scores.index, y=away_scores.values,
                name="Away Goals",
                marker_color=COLORS["secondary"],
                opacity=0.8,
            ))
            fig_dist.update_layout(
                title="Goals per Match Distribution",
                barmode="group", height=320,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=40, b=0),
                xaxis=dict(title="Goals", dtick=1),
                yaxis=dict(title="Matches"),
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        with col_r:
            # Top 15 most common exact scores
            df_s = df.copy()
            df_s["score"] = df_s["home_score"].astype(int).astype(str) + "–" + df_s["away_score"].astype(int).astype(str)
            top_scores = df_s["score"].value_counts().head(15)
            fig_sc = px.bar(
                x=top_scores.values[::-1],
                y=top_scores.index[::-1],
                orientation="h",
                title="15 Most Common Exact Scores",
                labels={"x": "Frequency", "y": "Score"},
                color=top_scores.values[::-1],
                color_continuous_scale=[[0,"#BFDBFE"],[1,"#1E3A5F"]],
            )
            fig_sc.update_layout(
                height=320,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=40, b=0),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        # 2D score heatmap
        st.markdown('<div class="section-header">Score Probability Heatmap (Historical)</div>',
                    unsafe_allow_html=True)
        df_h = df.copy()
        df_h["home_score_clip"] = df_h["home_score"].clip(0, 7).astype(int)
        df_h["away_score_clip"] = df_h["away_score"].clip(0, 7).astype(int)
        pivot = df_h.groupby(["home_score_clip","away_score_clip"]).size().unstack(fill_value=0)
        pivot_pct = pivot / pivot.values.sum() * 100

        fig_hm = go.Figure(go.Heatmap(
            z=pivot_pct.values,
            x=[str(c) for c in pivot_pct.columns],
            y=[str(i) for i in pivot_pct.index],
            colorscale="Blues",
            text=np.round(pivot_pct.values, 1),
            texttemplate="%{text}%",
            textfont={"size": 9},
            hovertemplate="Home %{y}–%{x} Away: %{z:.2f}%<extra></extra>",
        ))
        fig_hm.update_layout(
            height=420,
            title="% of all historical matches ending with each scoreline",
            margin=dict(l=0, r=0, t=50, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Away Goals"),
            yaxis=dict(title="Home Goals"),
        )
        st.plotly_chart(fig_hm, use_container_width=True)
        st.caption("Scores clipped at 7 goals. Cells show % of total historical matches.")
