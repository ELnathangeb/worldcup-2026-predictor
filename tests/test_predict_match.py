"""Unit tests for prediction and simulation logic."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Elo probability tests ─────────────────────────────────────────────────────

def test_elo_probs_sum_to_one():
    from src.predict_match import _elo_neutral_probs
    pa, pd_, pb = _elo_neutral_probs("Spain", "Japan")
    assert abs(pa + pd_ + pb - 1.0) < 1e-9


def test_elo_probs_stronger_team_favoured():
    """Spain (Elo ~2300) should beat Japan (Elo ~2066) more often than not."""
    from src.predict_match import _elo_neutral_probs
    pa, pd_, pb = _elo_neutral_probs("Spain", "Japan")
    assert pa > pb, f"Expected Spain win prob ({pa:.3f}) > Japan ({pb:.3f})"


def test_elo_probs_symmetric():
    """Swapping teams should swap win probs; draw should be unchanged."""
    from src.predict_match import _elo_neutral_probs
    pa, pd_, pb = _elo_neutral_probs("France", "Morocco")
    pb2, pd2, pa2 = _elo_neutral_probs("Morocco", "France")
    assert abs(pa - pa2) < 1e-9
    assert abs(pb - pb2) < 1e-9
    assert abs(pd_ - pd2) < 1e-9


def test_elo_equal_teams_near_even():
    """Two equal-Elo teams should each win with roughly equal probability."""
    from src.predict_match import _elo_neutral_probs, _load_elo
    elo_ratings = _load_elo()
    # Pick any two teams with similar Elo
    teams = sorted(elo_ratings.items(), key=lambda x: x[1])
    # Find a pair within 5 Elo points
    found = False
    for i in range(len(teams) - 1):
        ta, ea = teams[i]
        tb, eb = teams[i + 1]
        if abs(ea - eb) < 5:
            pa, pd_, pb = _elo_neutral_probs(ta, tb)
            assert abs(pa - pb) < 0.02, f"{ta} vs {tb}: win probs differ by more than 2pp"
            found = True
            break
    assert found, "No two teams within 5 Elo points found — test needs update"


def test_draw_prob_minimum():
    """Draw probability should always be at least 10% per model design."""
    from src.predict_match import _elo_neutral_probs
    _, pd_, _ = _elo_neutral_probs("Spain", "Guatemala")
    assert pd_ >= 0.10, f"Draw prob {pd_:.3f} below minimum 0.10"


# ── precompute_matchups ────────────────────────────────────────────────────────

def test_precompute_fills_cache():
    from src.predict_match import precompute_matchups, _MATCHUP_CACHE
    teams = ["Spain", "France", "Brazil"]
    precompute_matchups(teams)
    for a in teams:
        for b in teams:
            if a != b:
                assert (a, b) in _MATCHUP_CACHE


def test_precompute_cache_values_valid():
    from src.predict_match import precompute_matchups, _MATCHUP_CACHE
    teams = ["Germany", "England"]
    precompute_matchups(teams)
    pa, pd_, pb, la, lb = _MATCHUP_CACHE[("Germany", "England")]
    assert abs(pa + pd_ + pb - 1.0) < 1e-9
    assert la > 0 and lb > 0


# ── Group config ──────────────────────────────────────────────────────────────

def test_groups_48_teams():
    import yaml
    with open("configs/wc2026_groups.yaml") as f:
        d = yaml.safe_load(f)
    teams = [t for grp in d["groups"].values() for t in grp]
    assert len(teams) == 48


def test_groups_12_groups_of_4():
    import yaml
    with open("configs/wc2026_groups.yaml") as f:
        d = yaml.safe_load(f)
    groups = d["groups"]
    assert len(groups) == 12
    for grp, teams in groups.items():
        assert len(teams) == 4, f"Group {grp} has {len(teams)} teams, expected 4"


def test_confederation_covers_all_teams():
    import yaml
    with open("configs/wc2026_groups.yaml") as f:
        d = yaml.safe_load(f)
    all_teams = set(t for grp in d["groups"].values() for t in grp)
    conf_teams = set(t for teams in d["confederations"].values() for t in teams)
    missing = all_teams - conf_teams
    assert not missing, f"Teams missing from confederations: {missing}"


# ── Tournament smoke test ─────────────────────────────────────────────────────

def test_tournament_completes():
    """Full tournament simulation should run without error and return a champion."""
    import yaml
    from src.predict_match import precompute_matchups
    from src.simulation.tournament import simulate_world_cup

    with open("configs/wc2026_groups.yaml") as f:
        groups = yaml.safe_load(f)["groups"]

    all_teams = [t for g in groups.values() for t in g]
    precompute_matchups(all_teams)

    rng = np.random.default_rng(99)
    result = simulate_world_cup(groups, rng=rng, verbose=False)

    assert result["champion"] in all_teams
    assert result["runner_up"] in all_teams
    assert result["champion"] != result["runner_up"]
    assert len(result["semifinalists"]) == 4
    assert len(result["group_standings"]) == 12
    assert len(result["r32_results"]) == 16
    assert len(result["r16_results"]) == 8
    assert len(result["qf_results"]) == 4
    assert len(result["sf_results"]) == 2
    assert result["final_result"] is not None
