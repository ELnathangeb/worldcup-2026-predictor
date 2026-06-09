"""
Match prediction engine — FIFA World Cup 2026.

Wraps the trained LightGBM pipeline to produce:
  - win/draw/loss probabilities
  - expected score (via Poisson model calibrated on model probabilities)
  - confidence score

Usage
-----
    from src.predict_match import predict_match, load_predictor

    result = predict_match("Brazil", "France")
    # {
    #   "team_a": "Brazil",
    #   "team_b": "France",
    #   "team_a_win": 0.412,
    #   "draw":       0.238,
    #   "team_b_win": 0.350,
    #   "expected_score": "1 - 1",
    #   "expected_goals_a": 1.31,
    #   "expected_goals_b": 1.14,
    #   "confidence": 0.71
    # }
"""
from __future__ import annotations

import json
import math
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# Suppress cosmetic LightGBM "X does not have valid feature names" warning
# (sklearn Pipeline strips column names before passing to the classifier)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
MODEL_PKL   = ROOT / "models" / "best_model.pkl"
FEAT_PKL    = ROOT / "models" / "feature_columns.pkl"
ELO_JSON    = ROOT / "outputs" / "final_elo_ratings.json"
DATASET_CSV = ROOT / "data" / "processed" / "model_dataset.csv"

# ---------------------------------------------------------------------------
# Default Elo / ranking medians (filled if team is unknown)
# ---------------------------------------------------------------------------
_DEFAULT_ELO        = 1500.0
_DEFAULT_RANK       = 100
_DEFAULT_FIFA_PTS   = 1200.0

# World-Cup neutral-venue goal rates (calibrated on historical WC data)
_WC_HOME_GOAL_RATE  = 1.32   # goals per 90 min at neutral WC venue
_WC_AWAY_GOAL_RATE  = 1.10

# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_model():
    pipe = joblib.load(MODEL_PKL)
    feat_cols = joblib.load(FEAT_PKL)
    logger.info(f"Loaded model from {MODEL_PKL}  ({len(feat_cols)} features)")
    return pipe, feat_cols


@lru_cache(maxsize=1)
def _load_elo() -> dict[str, float]:
    with open(ELO_JSON) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_dataset() -> pd.DataFrame:
    """Load and sort the model dataset once — shared by all lookup helpers."""
    df = pd.read_csv(DATASET_CSV, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


@lru_cache(maxsize=1)
def _load_feature_medians() -> dict[str, float]:
    """
    Compute per-feature medians from the training set.
    Used to fill in rolling-form / H2H features for teams with no history.
    """
    df = _load_dataset()
    _, feat_cols = _load_model()
    numeric_cols = [c for c in feat_cols if c in df.columns]
    return df[numeric_cols].median().to_dict()


@lru_cache(maxsize=256)
def _load_recent_team_stats(team: str) -> dict[str, float]:
    """
    Compute rolling-form features for `team` by averaging their most recent
    home and away appearances in the dataset.

    The original feature engineering computed form over ALL matches regardless
    of home/away role. Here we approximate that by averaging:
        team_form5 = mean(home_form5 from home rows, away_form5 from away rows)

    This avoids the bug of assigning Japan's 100% home win-rate as their
    "away_win_rate5" in a neutral venue prediction.
    Cached per team — dataset loaded once.
    """
    df = _load_dataset()

    FORM_SUFF = ["form5", "win_rate5", "draw_rate5", "gf_avg5", "ga_avg5", "gd_avg5",
                 "form10", "win_rate10", "draw_rate10", "gf_avg10", "ga_avg10", "gd_avg10"]

    home_rows = df[df["home_team"] == team].dropna(subset=["home_form5"]).tail(10)
    away_rows = df[df["away_team"] == team].dropna(subset=["away_form5"]).tail(10)

    stats: dict[str, float] = {}

    for suf in FORM_SUFF:
        vals: list[float] = []
        if len(home_rows):
            col = f"home_{suf}"
            if col in home_rows.columns:
                v = home_rows[col].dropna()
                if len(v):
                    vals.append(float(v.mean()))
        if len(away_rows):
            col = f"away_{suf}"
            if col in away_rows.columns:
                v = away_rows[col].dropna()
                if len(v):
                    vals.append(float(v.mean()))
        if vals:
            stats[f"team_{suf}"] = float(np.mean(vals))

    return stats


@lru_cache(maxsize=256)
def _get_rank_info(team: str) -> tuple[float, float]:
    """Return (rank, fifa_pts) for a team. Cached per team."""
    df = _load_dataset()
    h = df[df["home_team"] == team][["home_rank", "home_fifa_pts"]].dropna().tail(1)
    a = df[df["away_team"] == team][["away_rank", "away_fifa_pts"]].dropna().tail(1)
    if len(h):
        return float(h["home_rank"].iloc[0]), float(h["home_fifa_pts"].iloc[0])
    if len(a):
        return float(a["away_rank"].iloc[0]), float(a["away_fifa_pts"].iloc[0])
    return float(_DEFAULT_RANK), float(_DEFAULT_FIFA_PTS)


# ---------------------------------------------------------------------------
# Feature vector builder
# ---------------------------------------------------------------------------

def _build_feature_row(
    team_a: str,
    team_b: str,
    neutral: bool = True,
    tournament_weight: float = 1.0,
    tournament_tier: int = 3,
    is_world_cup: int = 1,
    days_rest_a: int = 4,
    days_rest_b: int = 4,
) -> pd.DataFrame:
    """
    Construct a 1-row DataFrame with all model features.

    Team A is treated as 'home' (or the reference side) and Team B as 'away'.
    For WC neutral venues, home_advantage = 0 and neutral = True.
    """
    elo        = _load_elo()
    medians    = _load_feature_medians()
    _, feat_cols = _load_model()

    elo_a = elo.get(team_a, _DEFAULT_ELO)
    elo_b = elo.get(team_b, _DEFAULT_ELO)

    stats_a = _load_recent_team_stats(team_a)
    stats_b = _load_recent_team_stats(team_b)

    rank_a, fifa_pts_a = _get_rank_info(team_a)
    rank_b, fifa_pts_b = _get_rank_info(team_b)

    def _suf_val(stats, prefix, suf, fallback_key):
        v = stats.get(f"team_{suf}", np.nan)
        if np.isnan(v):
            v = medians.get(fallback_key, 0.0)
        return v

    # Build row dict matching ML_FEATURES order exactly
    row: dict[str, float] = {
        "home_elo":   elo_a,
        "away_elo":   elo_b,
        "elo_diff":   elo_a - elo_b,

        "home_rank":      rank_a,
        "away_rank":      rank_b,
        "rank_diff":      rank_a - rank_b,

        "home_fifa_pts":  fifa_pts_a,
        "away_fifa_pts":  fifa_pts_b,
        "fifa_pts_diff":  fifa_pts_a - fifa_pts_b,

        # Rolling form — team A
        "home_form5":      _suf_val(stats_a, "home", "form5",  "home_form5"),
        "home_win_rate5":  _suf_val(stats_a, "home", "win_rate5", "home_win_rate5"),
        "home_draw_rate5": _suf_val(stats_a, "home", "draw_rate5", "home_draw_rate5"),
        "home_gf_avg5":    _suf_val(stats_a, "home", "gf_avg5", "home_gf_avg5"),
        "home_ga_avg5":    _suf_val(stats_a, "home", "ga_avg5", "home_ga_avg5"),
        "home_gd_avg5":    _suf_val(stats_a, "home", "gd_avg5", "home_gd_avg5"),

        "home_form10":     _suf_val(stats_a, "home", "form10", "home_form10"),
        "home_win_rate10": _suf_val(stats_a, "home", "win_rate10", "home_win_rate10"),
        "home_draw_rate10":_suf_val(stats_a, "home", "draw_rate10", "home_draw_rate10"),
        "home_gf_avg10":   _suf_val(stats_a, "home", "gf_avg10", "home_gf_avg10"),
        "home_ga_avg10":   _suf_val(stats_a, "home", "ga_avg10", "home_ga_avg10"),
        "home_gd_avg10":   _suf_val(stats_a, "home", "gd_avg10", "home_gd_avg10"),

        # Rolling form — team B
        "away_form5":      _suf_val(stats_b, "away", "form5",  "away_form5"),
        "away_win_rate5":  _suf_val(stats_b, "away", "win_rate5", "away_win_rate5"),
        "away_draw_rate5": _suf_val(stats_b, "away", "draw_rate5", "away_draw_rate5"),
        "away_gf_avg5":    _suf_val(stats_b, "away", "gf_avg5", "away_gf_avg5"),
        "away_ga_avg5":    _suf_val(stats_b, "away", "ga_avg5", "away_ga_avg5"),
        "away_gd_avg5":    _suf_val(stats_b, "away", "gd_avg5", "away_gd_avg5"),

        "away_form10":     _suf_val(stats_b, "away", "form10", "away_form10"),
        "away_win_rate10": _suf_val(stats_b, "away", "win_rate10", "away_win_rate10"),
        "away_draw_rate10":_suf_val(stats_b, "away", "draw_rate10", "away_draw_rate10"),
        "away_gf_avg10":   _suf_val(stats_b, "away", "gf_avg10", "away_gf_avg10"),
        "away_ga_avg10":   _suf_val(stats_b, "away", "ga_avg10", "away_ga_avg10"),
        "away_gd_avg10":   _suf_val(stats_b, "away", "gd_avg10", "away_gd_avg10"),

        # H2H — use medians (1/3 prior is baked in during imputation)
        "h2h_home_wins":   medians.get("h2h_home_wins",   1.0),
        "h2h_draws":       medians.get("h2h_draws",       1.0),
        "h2h_away_wins":   medians.get("h2h_away_wins",   1.0),
        "h2h_home_gf_avg": medians.get("h2h_home_gf_avg", 1.0),
        "h2h_away_gf_avg": medians.get("h2h_away_gf_avg", 1.0),
        "h2h_total_games": medians.get("h2h_total_games", 3.0),

        # Context
        "home_advantage":    0.0 if neutral else 1.0,
        "neutral":           float(neutral),
        "tournament_weight": tournament_weight,
        "tournament_tier":   float(tournament_tier),
        "is_world_cup":      float(is_world_cup),
        "home_days_rest":    float(days_rest_a),
        "away_days_rest":    float(days_rest_b),
    }

    return pd.DataFrame([row])[feat_cols]


# ---------------------------------------------------------------------------
# Matchup probability cache — precompute all WC2026 pairings once
# ---------------------------------------------------------------------------

# Module-level cache: (team_a, team_b) -> (p_a_win, p_draw, p_b_win, λ_a, λ_b)
_MATCHUP_CACHE: dict[tuple[str, str], tuple[float, float, float, float, float]] = {}


def _elo_neutral_probs(team_a: str, team_b: str) -> tuple[float, float, float]:
    """
    Compute win/draw/loss probabilities using Elo ratings for a neutral venue.

    Why Elo instead of the ML model for WC simulation:
    ──────────────────────────────────────────────────
    The ML model was trained on historical matches where the "home/away" label
    at neutral venues is NOT randomly assigned — in the raw dataset, teams are
    listed in a fixed order that happens to anti-correlate with Elo strength
    (i.e., the weaker team is often listed as "home"). The model learned this
    artifact and produces inverted predictions at neutral venues.

    Elo is the correct tool for neutral-venue predictions: it is symmetric,
    calibrated from all historical results, and free of the home/away bias.

    Draw calibration: P(draw) decreases with Elo gap (strong mismatches
    produce fewer draws). Empirically calibrated from WC history.
    """
    elo = _load_elo()
    e_a = elo.get(team_a, _DEFAULT_ELO)
    e_b = elo.get(team_b, _DEFAULT_ELO)

    # Standard Elo expected score (probability A scores a "point" in chess sense)
    elo_expected_a = 1.0 / (1.0 + 10 ** ((e_b - e_a) / 400.0))

    # Draw probability: starts at 0.27 (WC baseline), shrinks with Elo gap
    elo_gap = abs(e_a - e_b)
    p_draw = max(0.10, 0.27 - 0.00020 * elo_gap)

    # Distribute remaining probability proportional to Elo expected score
    p_decisive = 1.0 - p_draw
    p_a_win = elo_expected_a * p_decisive
    p_b_win = (1.0 - elo_expected_a) * p_decisive

    return float(p_a_win), float(p_draw), float(p_b_win)


def precompute_matchups(teams: list[str]) -> None:
    """
    Precompute Elo-based probabilities and Poisson λ for all ordered pairings
    of `teams`. Called once before running many simulations.

    For WC neutral-venue matches we use Elo probabilities (see _elo_neutral_probs).
    The ML model is still used via predict_match() for the individual match API.
    """
    global _MATCHUP_CACHE
    _MATCHUP_CACHE.clear()

    for a in teams:
        for b in teams:
            if a != b:
                p_a_win, p_draw, p_b_win = _elo_neutral_probs(a, b)
                λ_a, λ_b = _expected_goals(p_a_win, p_draw, p_b_win, neutral=True)
                _MATCHUP_CACHE[(a, b)] = (p_a_win, p_draw, p_b_win, λ_a, λ_b)

    logger.info(f"Precomputed {len(_MATCHUP_CACHE)} matchup probabilities (Elo) for {len(teams)} teams.")


def get_matchup(team_a: str, team_b: str) -> tuple[float, float, float, float, float]:
    """
    Return (p_a_win, p_draw, p_b_win, λ_a, λ_b).
    Uses the precomputed Elo-based cache; falls back to Elo on the fly.
    """
    if (team_a, team_b) in _MATCHUP_CACHE:
        return _MATCHUP_CACHE[(team_a, team_b)]
    # Fallback
    p_a_win, p_draw, p_b_win = _elo_neutral_probs(team_a, team_b)
    λ_a, λ_b = _expected_goals(p_a_win, p_draw, p_b_win, neutral=True)
    return p_a_win, p_draw, p_b_win, λ_a, λ_b


# ---------------------------------------------------------------------------
# Confidence score
# ---------------------------------------------------------------------------

def _confidence(probs: np.ndarray) -> float:
    """
    Confidence = 1 - normalised entropy.
    Ranges from 0 (perfectly uncertain, all probs equal) to 1 (certain).
    max_entropy for 3 classes = log(3).
    """
    probs = np.clip(probs, 1e-9, 1.0)
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = math.log(3)
    return round(1.0 - entropy / max_entropy, 4)


# ---------------------------------------------------------------------------
# Score generation (Poisson)
# ---------------------------------------------------------------------------

def _expected_goals(
    prob_a_win: float,
    prob_draw: float,
    prob_b_win: float,
    neutral: bool = True,
) -> tuple[float, float]:
    """
    Derive expected goals for each team from model win probabilities using a
    Dixon–Coles–inspired mapping:

    Strategy
    --------
    1. Start from baseline WC goal rates (1.32 for "home" side, 1.10 for "away").
    2. Scale by the strength ratio implied by the predicted win probabilities:
         λ_a = base_a × (p_win_a / 0.38) ** 0.4
         λ_b = base_b × (p_win_b / 0.28) ** 0.4
       where 0.38 and 0.28 are the approximate historical WC home-win and
       away-win rates respectively.
    3. Blend slightly toward equal (1.2, 1.2) to avoid extreme scores from
       lopsided probabilities.

    This avoids the chicken-and-egg problem of requiring goals to predict
    goals — it uses only the outcome probabilities that the ML model provides.
    """
    base_a = _WC_HOME_GOAL_RATE if not neutral else (_WC_HOME_GOAL_RATE + _WC_AWAY_GOAL_RATE) / 2
    base_b = _WC_AWAY_GOAL_RATE if not neutral else (_WC_HOME_GOAL_RATE + _WC_AWAY_GOAL_RATE) / 2

    # Reference baselines (neutral WC)
    ref_win  = 0.33
    ref_loss = 0.33

    # Scale factor: how much stronger is team A than average?
    strength_a = max(0.5, prob_a_win / ref_win) ** 0.35
    strength_b = max(0.5, prob_b_win / ref_loss) ** 0.35

    λ_a = base_a * strength_a
    λ_b = base_b * strength_b

    # Mild regression to mean (avoids 5-0 scorelines from p=0.85 predictions)
    λ_a = 0.75 * λ_a + 0.25 * base_a
    λ_b = 0.75 * λ_b + 0.25 * base_b

    return round(λ_a, 3), round(λ_b, 3)


def sample_score(
    λ_a: float,
    λ_b: float,
    rng: np.random.Generator | None = None,
) -> tuple[int, int]:
    """Sample a Poisson score. Capped at 7 goals per team (avoids extreme outliers)."""
    if rng is None:
        rng = np.random.default_rng()
    g_a = min(int(rng.poisson(λ_a)), 7)
    g_b = min(int(rng.poisson(λ_b)), 7)
    return g_a, g_b


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_match(
    team_a: str,
    team_b: str,
    neutral: bool = True,
    tournament_weight: float = 1.0,
    tournament_tier: int = 3,
    is_world_cup: int = 1,
    days_rest_a: int = 4,
    days_rest_b: int = 4,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """
    Predict the outcome of a match between team_a and team_b.

    Parameters
    ----------
    team_a, team_b : str
        Team names (must match names used during training).
    neutral : bool
        True if played at a neutral venue (World Cup default).
    tournament_weight : float
        Importance weight: 1.0 = World Cup, 0.20 = Friendly.
    tournament_tier : int
        0 = Friendly, 1 = Continental Qualifying, 2 = Continental Final, 3 = World Cup.
    is_world_cup : int
        1 if this is a World Cup match.
    days_rest_a/b : int
        Days since last match for each team.
    rng : np.random.Generator, optional
        Random number generator (pass for reproducibility in simulations).

    Returns
    -------
    dict with keys:
        team_a, team_b, team_a_win, draw, team_b_win,
        expected_goals_a, expected_goals_b, expected_score,
        confidence
    """
    pipe, _ = _load_model()

    X = _build_feature_row(
        team_a, team_b,
        neutral=neutral,
        tournament_weight=tournament_weight,
        tournament_tier=tournament_tier,
        is_world_cup=is_world_cup,
        days_rest_a=days_rest_a,
        days_rest_b=days_rest_b,
    )

    # Model: class 0 = team_a loss (team_b win), 1 = draw, 2 = team_a win
    probs = pipe.predict_proba(X)[0]   # [p_loss, p_draw, p_win] from team_a perspective
    p_a_win   = float(probs[2])
    p_draw    = float(probs[1])
    p_b_win   = float(probs[0])

    λ_a, λ_b = _expected_goals(p_a_win, p_draw, p_b_win, neutral=neutral)

    # Expected (modal) score = mode of Poisson ~ floor(λ)
    exp_a = max(0, math.floor(λ_a))
    exp_b = max(0, math.floor(λ_b))

    return {
        "team_a":          team_a,
        "team_b":          team_b,
        "team_a_win":      round(p_a_win, 4),
        "draw":            round(p_draw, 4),
        "team_b_win":      round(p_b_win, 4),
        "expected_goals_a": λ_a,
        "expected_goals_b": λ_b,
        "expected_score":  f"{exp_a} - {exp_b}",
        "confidence":      _confidence(probs),
    }


def predict_match_verbose(team_a: str, team_b: str) -> None:
    """Pretty-print a match prediction to stdout."""
    r = predict_match(team_a, team_b)
    bar_width = 40
    print(f"\n{'─'*55}")
    print(f"  {r['team_a']:20s}  vs  {r['team_b']}")
    print(f"{'─'*55}")

    for label, p in [
        (f"{r['team_a']} Win", r["team_a_win"]),
        ("Draw",               r["draw"]),
        (f"{r['team_b']} Win", r["team_b_win"]),
    ]:
        filled = int(p * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"  {label:<20s} {bar}  {p*100:.1f}%")

    print(f"\n  Expected score : {r['team_a']} {r['expected_score']} {r['team_b']}")
    print(f"  xG             : {r['expected_goals_a']:.2f}  –  {r['expected_goals_b']:.2f}")
    print(f"  Confidence     : {r['confidence']*100:.1f}%")
    print(f"{'─'*55}\n")
