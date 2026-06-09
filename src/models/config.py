"""
Central model configuration — feature lists, splits, model definitions.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Feature columns used for training (45 features, zero nulls after Phase 2)
# ---------------------------------------------------------------------------
ML_FEATURES: list[str] = [
    # Elo ratings
    "home_elo", "away_elo", "elo_diff",
    # FIFA rankings
    "home_rank", "away_rank", "rank_diff",
    "home_fifa_pts", "away_fifa_pts", "fifa_pts_diff",
    # Rolling form — last 5
    "home_form5", "away_form5",
    "home_win_rate5", "away_win_rate5",
    "home_draw_rate5", "away_draw_rate5",
    "home_gf_avg5", "away_gf_avg5",
    "home_ga_avg5", "away_ga_avg5",
    "home_gd_avg5", "away_gd_avg5",
    # Rolling form — last 10
    "home_form10", "away_form10",
    "home_win_rate10", "away_win_rate10",
    "home_draw_rate10", "away_draw_rate10",
    "home_gf_avg10", "away_gf_avg10",
    "home_ga_avg10", "away_ga_avg10",
    "home_gd_avg10", "away_gd_avg10",
    # Head-to-head
    "h2h_home_wins", "h2h_draws", "h2h_away_wins",
    "h2h_home_gf_avg", "h2h_away_gf_avg", "h2h_total_games",
    # Context
    "home_advantage", "neutral",
    "tournament_weight", "tournament_tier", "is_world_cup",
    "home_days_rest", "away_days_rest",
]

TARGET = "result"

# ---------------------------------------------------------------------------
# Chronological splits
# ---------------------------------------------------------------------------
TRAIN_END  = "2021-12-31"   # train: 2000–2021
VAL_END    = "2023-12-31"   # val:   2022–2023
# test: 2024–present

# Class labels
CLASSES = {0: "Home Win", 1: "Draw", 2: "Away Win"}

# ---------------------------------------------------------------------------
# Hyperparameter search spaces
# ---------------------------------------------------------------------------
XGBOOST_PARAM_GRID = {
    "n_estimators":    [200, 400, 600],
    "max_depth":       [3, 4, 5, 6],
    "learning_rate":   [0.02, 0.05, 0.1],
    "subsample":       [0.7, 0.85, 1.0],
    "colsample_bytree":[0.7, 0.85, 1.0],
    "min_child_weight":[1, 3, 5],
    "reg_alpha":       [0, 0.1, 0.5],
    "reg_lambda":      [1, 2, 5],
}

LGBM_PARAM_GRID = {
    "n_estimators":    [200, 400, 600],
    "max_depth":       [4, 6, 8, -1],
    "learning_rate":   [0.02, 0.05, 0.1],
    "num_leaves":      [31, 63, 127],
    "subsample":       [0.7, 0.85, 1.0],
    "colsample_bytree":[0.7, 0.85, 1.0],
    "reg_alpha":       [0, 0.1, 0.5],
    "reg_lambda":      [1, 2, 5],
}
