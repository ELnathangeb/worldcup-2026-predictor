"""
Generate feature documentation as a printed report and save to outputs/.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import numpy as np

FEATURE_DOCS: dict[str, dict] = {
    # --- Rankings & Elo ---
    "home_rank": {
        "group": "FIFA Rankings",
        "description": "Most recent FIFA world ranking of the home team before the match date.",
        "source": "FIFA official rankings (samuraitruong dataset); forward-filled from last known value.",
        "leakage_safe": True,
        "missing_strategy": "Column median",
    },
    "away_rank": {
        "group": "FIFA Rankings",
        "description": "Most recent FIFA world ranking of the away team before the match date.",
        "source": "Same as home_rank.",
        "leakage_safe": True,
        "missing_strategy": "Column median",
    },
    "home_fifa_pts": {
        "group": "FIFA Rankings",
        "description": "FIFA ranking points of the home team at the most recent ranking release before the match.",
        "source": "FIFA official points; forward-filled.",
        "leakage_safe": True,
        "missing_strategy": "Column median",
    },
    "away_fifa_pts": {
        "group": "FIFA Rankings",
        "description": "FIFA ranking points of the away team at the most recent ranking release before the match.",
        "source": "Same as home_fifa_pts.",
        "leakage_safe": True,
        "missing_strategy": "Column median",
    },
    "rank_diff": {
        "group": "FIFA Rankings",
        "description": "home_rank minus away_rank. Negative = home team is ranked higher (better). "
                       "Key predictor of match outcome.",
        "source": "Derived from home_rank and away_rank.",
        "leakage_safe": True,
        "missing_strategy": "Column median",
    },
    "fifa_pts_diff": {
        "group": "FIFA Rankings",
        "description": "home_fifa_pts minus away_fifa_pts. Positive = home team has more points.",
        "source": "Derived from home_fifa_pts and away_fifa_pts.",
        "leakage_safe": True,
        "missing_strategy": "Column median",
    },
    "home_elo": {
        "group": "Elo Ratings",
        "description": "Elo rating of the home team immediately before this match (computed chronologically "
                       "from all prior results in the dataset). K-factor scales with tournament importance "
                       "and margin of victory.",
        "source": "Computed from scratch using results.csv (chronological order).",
        "leakage_safe": True,
        "missing_strategy": "Initialised at 1500 for first appearance",
    },
    "away_elo": {
        "group": "Elo Ratings",
        "description": "Elo rating of the away team immediately before this match.",
        "source": "Same computation as home_elo.",
        "leakage_safe": True,
        "missing_strategy": "Initialised at 1500 for first appearance",
    },
    "elo_diff": {
        "group": "Elo Ratings",
        "description": "home_elo minus away_elo. Strong continuous predictor of win probability. "
                       "An Elo difference of 200 implies ~75% win probability for the stronger team.",
        "source": "Derived.",
        "leakage_safe": True,
        "missing_strategy": "N/A (always computable)",
    },
    # --- Rolling Form (5-match) ---
    "home_form5": {
        "group": "Rolling Form",
        "description": "Average points per game for the home team in their last 5 matches before this one "
                       "(3=win, 1=draw, 0=loss). Captures recent momentum.",
        "source": "Computed from results.csv per team, strictly before match date.",
        "leakage_safe": True,
        "missing_strategy": "Global mean (teams with < 5 prior matches)",
    },
    "away_form5": {
        "group": "Rolling Form",
        "description": "Same as home_form5 for the away team.",
        "source": "Same.",
        "leakage_safe": True,
        "missing_strategy": "Global mean",
    },
    "home_form10": {
        "group": "Rolling Form",
        "description": "Average points per game in last 10 matches. Smooths out variance vs form5.",
        "source": "Computed from results.csv.",
        "leakage_safe": True,
        "missing_strategy": "Global mean",
    },
    "away_form10": {"group": "Rolling Form", "description": "Same as home_form10 for away team.",
                    "source": "Same.", "leakage_safe": True, "missing_strategy": "Global mean"},
    "home_win_rate5": {"group": "Rolling Form", "description": "Win rate in last 5 matches (home team).",
                       "source": "Computed.", "leakage_safe": True, "missing_strategy": "Global mean"},
    "away_win_rate5": {"group": "Rolling Form", "description": "Win rate in last 5 matches (away team).",
                       "source": "Computed.", "leakage_safe": True, "missing_strategy": "Global mean"},
    "home_draw_rate5": {"group": "Rolling Form", "description": "Draw rate in last 5 matches (home).",
                        "source": "Computed.", "leakage_safe": True, "missing_strategy": "Global mean"},
    "away_draw_rate5": {"group": "Rolling Form", "description": "Draw rate in last 5 matches (away).",
                        "source": "Computed.", "leakage_safe": True, "missing_strategy": "Global mean"},
    "home_win_rate10": {"group": "Rolling Form", "description": "Win rate in last 10 matches (home).",
                        "source": "Computed.", "leakage_safe": True, "missing_strategy": "Global mean"},
    "away_win_rate10": {"group": "Rolling Form", "description": "Win rate in last 10 matches (away).",
                        "source": "Computed.", "leakage_safe": True, "missing_strategy": "Global mean"},
    "home_draw_rate10": {"group": "Rolling Form", "description": "Draw rate in last 10 matches (home).",
                         "source": "Computed.", "leakage_safe": True, "missing_strategy": "Global mean"},
    "away_draw_rate10": {"group": "Rolling Form", "description": "Draw rate in last 10 matches (away).",
                         "source": "Computed.", "leakage_safe": True, "missing_strategy": "Global mean"},
    # --- Goals ---
    "home_gf_avg5": {"group": "Goals (Rolling)", "description": "Goals scored per game, last 5 (home).",
                     "source": "Computed.", "leakage_safe": True, "missing_strategy": "Global mean"},
    "away_gf_avg5": {"group": "Goals (Rolling)", "description": "Goals scored per game, last 5 (away).",
                     "source": "Computed.", "leakage_safe": True, "missing_strategy": "Global mean"},
    "home_ga_avg5": {"group": "Goals (Rolling)", "description": "Goals conceded per game, last 5 (home).",
                     "source": "Computed.", "leakage_safe": True, "missing_strategy": "Global mean"},
    "away_ga_avg5": {"group": "Goals (Rolling)", "description": "Goals conceded per game, last 5 (away).",
                     "source": "Computed.", "leakage_safe": True, "missing_strategy": "Global mean"},
    "home_gd_avg5": {"group": "Goals (Rolling)", "description": "Goal difference per game, last 5 (home).",
                     "source": "Computed.", "leakage_safe": True, "missing_strategy": "Global mean"},
    "away_gd_avg5": {"group": "Goals (Rolling)", "description": "Goal difference per game, last 5 (away).",
                     "source": "Computed.", "leakage_safe": True, "missing_strategy": "Global mean"},
    "home_gf_avg10": {"group": "Goals (Rolling)", "description": "Goals scored per game, last 10 (home).",
                      "source": "Computed.", "leakage_safe": True, "missing_strategy": "Global mean"},
    "away_gf_avg10": {"group": "Goals (Rolling)", "description": "Goals scored per game, last 10 (away).",
                      "source": "Computed.", "leakage_safe": True, "missing_strategy": "Global mean"},
    "home_ga_avg10": {"group": "Goals (Rolling)", "description": "Goals conceded per game, last 10 (home).",
                      "source": "Computed.", "leakage_safe": True, "missing_strategy": "Global mean"},
    "away_ga_avg10": {"group": "Goals (Rolling)", "description": "Goals conceded per game, last 10 (away).",
                      "source": "Computed.", "leakage_safe": True, "missing_strategy": "Global mean"},
    "home_gd_avg10": {"group": "Goals (Rolling)", "description": "Goal diff per game, last 10 (home).",
                      "source": "Computed.", "leakage_safe": True, "missing_strategy": "Global mean"},
    "away_gd_avg10": {"group": "Goals (Rolling)", "description": "Goal diff per game, last 10 (away).",
                      "source": "Computed.", "leakage_safe": True, "missing_strategy": "Global mean"},
    # --- H2H ---
    "h2h_home_wins": {"group": "Head-to-Head", "description": "Win rate of home team in last 10 H2H meetings.",
                      "source": "Computed from results.csv.", "leakage_safe": True,
                      "missing_strategy": "0.333 (uniform prior for unseen matchups)"},
    "h2h_draws": {"group": "Head-to-Head", "description": "Draw rate in last 10 H2H meetings.",
                  "source": "Computed.", "leakage_safe": True, "missing_strategy": "0.333"},
    "h2h_away_wins": {"group": "Head-to-Head", "description": "Away win rate in last 10 H2H meetings.",
                      "source": "Computed.", "leakage_safe": True, "missing_strategy": "0.333"},
    "h2h_home_gf_avg": {"group": "Head-to-Head", "description": "Average goals scored by home team in H2H.",
                         "source": "Computed.", "leakage_safe": True, "missing_strategy": "Column median"},
    "h2h_away_gf_avg": {"group": "Head-to-Head", "description": "Average goals scored by away team in H2H.",
                         "source": "Computed.", "leakage_safe": True, "missing_strategy": "Column median"},
    "h2h_total_games": {"group": "Head-to-Head", "description": "Number of H2H meetings in dataset (0 = first meeting).",
                         "source": "Computed.", "leakage_safe": True, "missing_strategy": "0"},
    # --- Contextual ---
    "home_advantage": {"group": "Context", "description": "1 if the home team plays at their home ground (not neutral venue).",
                       "source": "Derived from neutral column.", "leakage_safe": True, "missing_strategy": "N/A"},
    "is_neutral": {"group": "Context", "description": "1 if the match is played at a neutral venue.",
                   "source": "results.csv neutral column.", "leakage_safe": True, "missing_strategy": "N/A"},
    "tournament_weight": {"group": "Context", "description": "Competitiveness weight 0.20–1.0. "
                           "World Cup=1.0, Friendly=0.20.",
                           "source": "Manual mapping in cleaner.py.", "leakage_safe": True, "missing_strategy": "0.40"},
    "tournament_tier": {"group": "Context", "description": "Ordinal bucket 0–4 from tournament_weight. "
                         "4=Major tournament, 0=Friendly.",
                         "source": "Derived.", "leakage_safe": True, "missing_strategy": "N/A"},
    "is_world_cup": {"group": "Context", "description": "1 if this is a FIFA World Cup match.",
                     "source": "Derived from tournament column.", "leakage_safe": True, "missing_strategy": "N/A"},
    "home_days_rest": {"group": "Context", "description": "Days since the home team's last match. "
                        "Proxy for squad freshness.",
                        "source": "Computed from results.", "leakage_safe": True, "missing_strategy": "Column median"},
    "away_days_rest": {"group": "Context", "description": "Days since the away team's last match.",
                       "source": "Computed from results.", "leakage_safe": True, "missing_strategy": "Column median"},
    # --- Target ---
    "result": {"group": "Target", "description": "0 = home team loss, 1 = draw, 2 = home team win. "
                "Multi-class classification target.",
                "source": "Derived from home_score vs away_score.", "leakage_safe": True, "missing_strategy": "N/A"},
}


def print_feature_docs(df: pd.DataFrame | None = None) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print("  FEATURE DOCUMENTATION")
    print(sep)

    groups: dict[str, list[str]] = {}
    for feat, meta in FEATURE_DOCS.items():
        groups.setdefault(meta["group"], []).append(feat)

    for grp, feats in groups.items():
        print(f"\n  [{grp}]")
        for f in feats:
            m = FEATURE_DOCS[f]
            safe = "✓" if m["leakage_safe"] else "✗"
            desc = m["description"][:80]
            print(f"    {safe} {f:<30s} — {desc}")

    if df is not None:
        print(f"\n{sep}")
        print("  MISSING VALUE SUMMARY (after imputation)")
        print(f"  Dataset shape: {df.shape}")
        ml_cols = [c for c in FEATURE_DOCS if c in df.columns]
        nulls = df[ml_cols].isnull().sum()
        nulls = nulls[nulls > 0]
        if nulls.empty:
            print("  No missing values in feature columns.")
        else:
            for col, n in nulls.items():
                print(f"  {col:<35s} {n:6,} nulls ({n/len(df)*100:.1f}%)")

    print(f"\n{sep}\n")


def save_feature_docs(df: pd.DataFrame, output_dir: str | Path = "outputs") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a clean table
    rows = []
    for feat, meta in FEATURE_DOCS.items():
        present = feat in df.columns if df is not None else None
        null_count = int(df[feat].isna().sum()) if (df is not None and feat in df.columns) else None
        rows.append({
            "feature": feat,
            "group": meta["group"],
            "description": meta["description"],
            "leakage_safe": meta["leakage_safe"],
            "missing_strategy": meta["missing_strategy"],
            "in_dataset": present,
            "null_count": null_count,
        })

    doc_df = pd.DataFrame(rows)
    out_path = output_dir / "feature_documentation.csv"
    doc_df.to_csv(out_path, index=False)
    print(f"Feature documentation saved → {out_path}")
    return out_path
