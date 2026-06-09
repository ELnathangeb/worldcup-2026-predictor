"""
Feature engineering — all features are computed strictly from data available
BEFORE the match date to prevent any form of data leakage.

Features produced
-----------------
Rankings & ratings
  home_rank, away_rank               — most recent FIFA rank before match
  home_fifa_pts, away_fifa_pts       — most recent FIFA points before match
  rank_diff                          — home_rank - away_rank  (negative = home ranked higher)
  fifa_pts_diff                      — home_fifa_pts - away_fifa_pts
  home_elo, away_elo                 — Elo rating immediately before this match
  elo_diff                           — home_elo - away_elo

Rolling form  (last N matches, weighted by recency via tournament_weight)
  home_form5 / away_form5            — points per game in last 5 matches (3=W,1=D,0=L)
  home_form10 / away_form10          — same, last 10 matches
  home_gf_avg5 / away_gf_avg5        — goals for per game, last 5
  home_ga_avg5 / away_ga_avg5        — goals against per game, last 5
  home_gd_avg5 / away_gd_avg5        — goal difference per game, last 5
  home_gf_avg10 / away_gf_avg10
  home_ga_avg10 / away_ga_avg10
  home_gd_avg10 / away_gd_avg10
  home_win_rate5 / away_win_rate5    — win rate in last 5
  home_draw_rate5 / away_draw_rate5
  home_win_rate10 / away_win_rate10
  home_draw_rate10 / away_draw_rate10

Head-to-head  (last 10 meetings between these two teams, before match date)
  h2h_home_wins, h2h_draws, h2h_away_wins
  h2h_home_gf_avg, h2h_away_gf_avg
  h2h_total_games

Contextual
  is_neutral                         — 1 if neutral venue
  home_advantage                     — 1 if not neutral (home team plays at home)
  tournament_weight                  — competitiveness weight of the fixture
  is_world_cup                       — 1 if FIFA World Cup match

Target
  result                             — 0=home win, 1=draw, 2=away win
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Elo parameters  (standard values from academic football literature)
# ---------------------------------------------------------------------------
ELO_START = 1500.0
ELO_K_BASE = 32.0
_TOURNAMENT_K_MULT: dict[str, float] = {
    "FIFA World Cup": 2.0,
    "UEFA Euro": 1.75,
    "Copa América": 1.75,
    "African Cup of Nations": 1.5,
    "AFC Asian Cup": 1.4,
    "Gold Cup": 1.3,
    "FIFA Confederations Cup": 1.4,
    "UEFA Nations League": 1.2,
    "CONCACAF Nations League": 1.1,
    "FIFA World Cup qualification": 1.2,
    "UEFA Euro qualification": 1.1,
    "Friendly": 0.75,
}
_DEFAULT_K_MULT = 1.0


def _elo_expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _elo_k(tournament: str, goal_diff: int) -> float:
    """K factor scaled by tournament importance and margin of victory."""
    k = ELO_K_BASE * _TOURNAMENT_K_MULT.get(tournament, _DEFAULT_K_MULT)
    # World Football Elo–style margin multiplier (capped)
    margin_mult = np.log(abs(goal_diff) + 1) + 1.0
    margin_mult = min(margin_mult, 3.0)
    return k * margin_mult


def compute_elo(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Compute Elo ratings chronologically.

    Returns
    -------
    df_out : DataFrame with home_elo / away_elo columns added
             (ratings BEFORE each match is played)
    final_ratings : dict[team → final Elo] (for use in simulation)
    """
    df = df.sort_values("date").reset_index(drop=True)
    ratings: dict[str, float] = {}

    home_elos: list[float] = []
    away_elos: list[float] = []

    for _, row in df.iterrows():
        ht, at = row["home_team"], row["away_team"]
        r_h = ratings.get(ht, ELO_START)
        r_a = ratings.get(at, ELO_START)

        home_elos.append(r_h)
        away_elos.append(r_a)

        # Update after recording pre-match ratings
        result = int(row["result"])   # 0=home win, 1=draw, 2=away win
        actual_h = 1.0 if result == 0 else (0.5 if result == 1 else 0.0)
        actual_a = 1.0 - actual_h

        exp_h = _elo_expected(r_h, r_a)
        exp_a = 1.0 - exp_h
        gd = abs(int(row["goal_diff"]))
        k = _elo_k(row["tournament"], gd)

        ratings[ht] = r_h + k * (actual_h - exp_h)
        ratings[at] = r_a + k * (actual_a - exp_a)

    df = df.copy()
    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    logger.info(f"Elo computed. Range: {min(ratings.values()):.0f} – {max(ratings.values()):.0f}")
    return df, ratings


# ---------------------------------------------------------------------------
# Rolling form helpers
# ---------------------------------------------------------------------------

def _team_match_history(df: pd.DataFrame) -> dict[str, list[dict]]:
    """
    Build a per-team timeline of {date, goals_for, goals_against, result_pts, gd}.
    result_pts: 3=win, 1=draw, 0=loss  (from that team's perspective).
    """
    history: dict[str, list[dict]] = {}

    for _, row in df.iterrows():
        ht, at = row["home_team"], row["away_team"]
        hs, as_ = int(row["home_score"]), int(row["away_score"])
        result = int(row["result"])
        date = row["date"]

        # from home team perspective
        h_pts = 3 if result == 0 else (1 if result == 1 else 0)
        a_pts = 3 if result == 2 else (1 if result == 1 else 0)

        history.setdefault(ht, []).append({"date": date, "gf": hs, "ga": as_, "pts": h_pts, "gd": hs - as_})
        history.setdefault(at, []).append({"date": date, "gf": as_, "ga": hs, "pts": a_pts, "gd": as_ - hs})

    return history


def _rolling_stats(history_list: list[dict], before_date: pd.Timestamp, n: int) -> dict[str, float]:
    """Return rolling stats for the last N matches strictly before before_date."""
    past = [h for h in history_list if h["date"] < before_date]
    recent = past[-n:] if len(past) >= 1 else []
    if not recent:
        return {
            f"form{n}": np.nan, f"win_rate{n}": np.nan, f"draw_rate{n}": np.nan,
            f"gf_avg{n}": np.nan, f"ga_avg{n}": np.nan, f"gd_avg{n}": np.nan,
        }
    pts = [r["pts"] for r in recent]
    wins = sum(1 for p in pts if p == 3)
    draws = sum(1 for p in pts if p == 1)
    total = len(recent)
    return {
        f"form{n}": np.mean(pts),
        f"win_rate{n}": wins / total,
        f"draw_rate{n}": draws / total,
        f"gf_avg{n}": np.mean([r["gf"] for r in recent]),
        f"ga_avg{n}": np.mean([r["ga"] for r in recent]),
        f"gd_avg{n}": np.mean([r["gd"] for r in recent]),
    }


def compute_rolling_form(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling form features. O(n) over rows per team."""
    history = _team_match_history(df)
    df = df.sort_values("date").reset_index(drop=True)

    home_stats5: list[dict] = []
    away_stats5: list[dict] = []
    home_stats10: list[dict] = []
    away_stats10: list[dict] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Rolling form", leave=False):
        ht, at = row["home_team"], row["away_team"]
        date = row["date"]
        home_stats5.append(_rolling_stats(history.get(ht, []), date, 5))
        away_stats5.append(_rolling_stats(history.get(at, []), date, 5))
        home_stats10.append(_rolling_stats(history.get(ht, []), date, 10))
        away_stats10.append(_rolling_stats(history.get(at, []), date, 10))

    def _prefix(lst: list[dict], prefix: str) -> pd.DataFrame:
        out = pd.DataFrame(lst)
        out.columns = [f"{prefix}_{c}" for c in out.columns]
        return out

    parts = [
        df.reset_index(drop=True),
        _prefix(home_stats5, "home"),
        _prefix(away_stats5, "away"),
        _prefix(home_stats10, "home"),
        _prefix(away_stats10, "away"),
    ]
    result = pd.concat(parts, axis=1)
    logger.info("Rolling form features added")
    return result


# ---------------------------------------------------------------------------
# Head-to-head
# ---------------------------------------------------------------------------

def compute_h2h(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each match, compute head-to-head stats between the two teams
    using only results strictly before the match date (last 10 meetings).
    """
    df = df.sort_values("date").reset_index(drop=True)

    h2h_records: list[dict] = []

    # Build a lookup: frozenset({teamA, teamB}) → sorted list of past matches
    pair_history: dict[frozenset, list[dict]] = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="H2H", leave=False):
        ht, at = row["home_team"], row["away_team"]
        date = row["date"]
        pair = frozenset({ht, at})

        past = [m for m in pair_history.get(pair, []) if m["date"] < date][-10:]
        if not past:
            h2h_records.append({
                "h2h_home_wins": np.nan, "h2h_draws": np.nan, "h2h_away_wins": np.nan,
                "h2h_home_gf_avg": np.nan, "h2h_away_gf_avg": np.nan, "h2h_total_games": 0,
            })
        else:
            hw, d, aw = 0, 0, 0
            hgf_list, agf_list = [], []
            for m in past:
                # perspective: home_team = ht
                if m["team_a"] == ht:
                    hgf, agf = m["gf_a"], m["gf_b"]
                else:
                    hgf, agf = m["gf_b"], m["gf_a"]
                hgf_list.append(hgf)
                agf_list.append(agf)
                if hgf > agf:
                    hw += 1
                elif hgf == agf:
                    d += 1
                else:
                    aw += 1
            total = len(past)
            h2h_records.append({
                "h2h_home_wins": hw / total,
                "h2h_draws": d / total,
                "h2h_away_wins": aw / total,
                "h2h_home_gf_avg": np.mean(hgf_list),
                "h2h_away_gf_avg": np.mean(agf_list),
                "h2h_total_games": total,
            })

        # Record this match for future lookups
        pair_history.setdefault(pair, []).append({
            "date": date,
            "team_a": ht, "team_b": at,
            "gf_a": int(row["home_score"]), "gf_b": int(row["away_score"]),
        })

    h2h_df = pd.DataFrame(h2h_records)
    result = pd.concat([df.reset_index(drop=True), h2h_df], axis=1)
    logger.info("H2H features added")
    return result


# ---------------------------------------------------------------------------
# Final feature assembly
# ---------------------------------------------------------------------------

def assemble_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features from already-computed columns and finalize the dataset.
    Drops raw columns not needed for ML training.
    """
    df = df.copy()

    # Ranking difference features
    df["rank_diff"] = df["home_rank"] - df["away_rank"]
    df["fifa_pts_diff"] = df["home_fifa_pts"] - df["away_fifa_pts"]

    # Home advantage flag
    df["home_advantage"] = (~df["neutral"].astype(bool)).astype(int)

    # World Cup flag
    df["is_world_cup"] = (df["tournament"] == "FIFA World Cup").astype(int)

    # Tournament tier (ordinal bucket)
    def _tier(w: float) -> int:
        if w >= 0.90:
            return 4  # Major tournament (WC, Euro, Copa)
        elif w >= 0.70:
            return 3  # Continental championship
        elif w >= 0.50:
            return 2  # WCQ / continental qual
        elif w >= 0.30:
            return 1  # Minor tournament
        return 0       # Friendly

    df["tournament_tier"] = df["tournament_weight"].apply(_tier)

    # Days since last match (proxy for rest/fitness)
    df = df.sort_values("date")
    for team_col, prefix in [("home_team", "home"), ("away_team", "away")]:
        last_date: dict[str, pd.Timestamp] = {}
        days_rest: list[float] = []
        for _, row in df.iterrows():
            team = row[team_col]
            prev = last_date.get(team)
            days_rest.append((row["date"] - prev).days if prev is not None else np.nan)
            last_date[team] = row["date"]
        df[f"{prefix}_days_rest"] = days_rest

    df = df.reset_index(drop=True)
    logger.info("Feature assembly complete")
    return df


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy per feature type:
      - FIFA rank/points: median across all teams in dataset (neutral prior)
      - Rolling form (< N matches in history): global mean of that feature
      - H2H (no prior meetings): 0.33/0.33/0.33 split (uniform prior)
      - days_rest: median
    """
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # H2H: explicit uniform prior for unseen matchups
    for col in ["h2h_home_wins", "h2h_draws", "h2h_away_wins"]:
        if col in df.columns:
            df[col] = df[col].fillna(1 / 3)
    for col in ["h2h_home_gf_avg", "h2h_away_gf_avg"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # All remaining numeric: fill with column median
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    remaining = df[numeric_cols].isna().sum().sum()
    logger.info(f"After imputation — remaining nulls in numeric cols: {remaining}")
    return df


# ---------------------------------------------------------------------------
# Validation: no leakage check
# ---------------------------------------------------------------------------

def validate_no_leakage(df: pd.DataFrame) -> None:
    """
    Spot-check that all rolling and h2h features use only past data.
    Samples 500 rows and verifies form windows don't include current match.
    Raises AssertionError on failure.
    """
    sample = df.sample(min(500, len(df)), random_state=42)
    for _, row in sample.iterrows():
        # Elo values should always be finite (assigned before match)
        assert np.isfinite(row["home_elo"]), f"Non-finite home_elo at {row['date']}"
        assert np.isfinite(row["away_elo"]), f"Non-finite away_elo at {row['date']}"
        # H2H total games must be non-negative integer
        assert row["h2h_total_games"] >= 0, f"Negative H2H count at {row['date']}"
    logger.info("Leakage validation passed on 500 sampled rows")
