"""
Single-tournament simulator for FIFA World Cup 2026.

Format
------
- 48 teams, 12 groups of 4
- Each group: 6 round-robin matches (every team plays 3)
- Advancement: top 2 from each group (24 teams) + best 8 third-place teams = 32
- Knockout: R32 → R16 → QF → SF → Final

Scoring rules (FIFA)
--------------------
- Win = 3 pts, Draw = 1 pt, Loss = 0 pts
- Tiebreak order:
    1. Points
    2. Goal difference
    3. Goals scored
    4. H2H points
    5. H2H goal difference
    6. H2H goals scored
    7. Random draw (via RNG)

Terminology: inside the simulator, every "match" is played from the perspective of
team_a (home side) vs team_b (away side), both at a neutral venue.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.predict_match import predict_match, sample_score, _expected_goals, get_matchup
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TeamRecord:
    name: str
    played: int = 0
    wins:   int = 0
    draws:  int = 0
    losses: int = 0
    gf:     int = 0     # goals for
    ga:     int = 0     # goals against
    pts:    int = 0

    # h2h records against specific opponents {opponent: [pts_for_us]}
    h2h: dict[str, list[int]] = field(default_factory=dict)
    h2h_gf: dict[str, int]    = field(default_factory=dict)
    h2h_ga: dict[str, int]    = field(default_factory=dict)

    @property
    def gd(self) -> int:
        return self.gf - self.ga

    def sort_key(self) -> tuple:
        return (self.pts, self.gd, self.gf)


@dataclass
class MatchResult:
    team_a: str
    team_b: str
    goals_a: int
    goals_b: int
    after_et: bool = False       # required extra time
    after_pens: bool = False     # decided by penalties
    pen_winner: str | None = None

    @property
    def winner(self) -> str | None:
        if self.goals_a > self.goals_b:
            return self.team_a
        if self.goals_b > self.goals_a:
            return self.team_b
        return self.pen_winner  # None if group stage draw

    @property
    def loser(self) -> str | None:
        w = self.winner
        if w is None:
            return None
        return self.team_b if w == self.team_a else self.team_a

    def score_str(self) -> str:
        suffix = ""
        if self.after_pens:
            suffix = " (pens)"
        elif self.after_et:
            suffix = " (aet)"
        return f"{self.goals_a}–{self.goals_b}{suffix}"


# ---------------------------------------------------------------------------
# Core simulation helpers
# ---------------------------------------------------------------------------

def _simulate_group_match(
    team_a: str,
    team_b: str,
    rng: np.random.Generator,
) -> MatchResult:
    """Simulate one group-stage match (draw is a valid outcome)."""
    _, _, _, λ_a, λ_b = get_matchup(team_a, team_b)
    g_a, g_b = sample_score(λ_a, λ_b, rng=rng)
    return MatchResult(team_a=team_a, team_b=team_b, goals_a=g_a, goals_b=g_b)


def _simulate_knockout_match(
    team_a: str,
    team_b: str,
    rng: np.random.Generator,
) -> MatchResult:
    """
    Knockout match — must have a winner.
    After 90 min draw → extra time coin flip (slight momentum effect) →
    if still level → penalty shootout (50/50 with slight skill tilt).
    """
    _, _, _, λ_a, λ_b = get_matchup(team_a, team_b)
    g_a, g_b  = sample_score(λ_a, λ_b, rng=rng)

    after_et   = False
    after_pens = False
    pen_winner = None

    if g_a == g_b:
        after_et = True
        # Extra time: scale Poisson rates to 30 min
        et_λ_a = λ_a * (30 / 90)
        et_λ_b = λ_b * (30 / 90)
        et_a, et_b = sample_score(et_λ_a, et_λ_b, rng=rng)
        g_a += et_a
        g_b += et_b

        if g_a == g_b:
            # Penalty shootout — slightly biased by Elo strength
            from src.predict_match import _load_elo
            elo = _load_elo()
            e_a = elo.get(team_a, 1500.0)
            e_b = elo.get(team_b, 1500.0)
            # Elo-weighted penalty win probability (compress toward 50%)
            p_a_pens = 0.5 + 0.5 * (e_a - e_b) / (e_a + e_b - 2 * 1000)
            p_a_pens = float(np.clip(p_a_pens, 0.30, 0.70))
            after_pens = True
            pen_winner = team_a if rng.random() < p_a_pens else team_b

    return MatchResult(
        team_a=team_a, team_b=team_b,
        goals_a=g_a, goals_b=g_b,
        after_et=after_et, after_pens=after_pens,
        pen_winner=pen_winner,
    )


# ---------------------------------------------------------------------------
# Group stage
# ---------------------------------------------------------------------------

def simulate_group(
    group_name: str,
    teams: list[str],
    rng: np.random.Generator,
) -> tuple[list[TeamRecord], list[MatchResult]]:
    """
    Simulate all 6 round-robin matches in a group.
    Returns (sorted standings, match results).
    """
    records = {t: TeamRecord(name=t) for t in teams}
    results: list[MatchResult] = []

    for team_a, team_b in itertools.combinations(teams, 2):
        mr = _simulate_group_match(team_a, team_b, rng)
        results.append(mr)

        ra, rb = records[team_a], records[team_b]
        ra.played += 1
        rb.played += 1
        ra.gf += mr.goals_a; ra.ga += mr.goals_b
        rb.gf += mr.goals_b; rb.ga += mr.goals_a

        if mr.goals_a > mr.goals_b:
            ra.wins += 1; ra.pts += 3
            rb.losses += 1
        elif mr.goals_a < mr.goals_b:
            rb.wins += 1; rb.pts += 3
            ra.losses += 1
        else:
            ra.draws += 1; ra.pts += 1
            rb.draws += 1; rb.pts += 1

        # Track H2H
        pts_a = 3 if mr.goals_a > mr.goals_b else (1 if mr.goals_a == mr.goals_b else 0)
        pts_b = 3 - pts_a if pts_a != 1 else 1
        ra.h2h.setdefault(team_b, []).append(pts_a)
        rb.h2h.setdefault(team_a, []).append(pts_b)
        ra.h2h_gf[team_b] = ra.h2h_gf.get(team_b, 0) + mr.goals_a
        ra.h2h_ga[team_b] = ra.h2h_ga.get(team_b, 0) + mr.goals_b
        rb.h2h_gf[team_a] = rb.h2h_gf.get(team_a, 0) + mr.goals_b
        rb.h2h_ga[team_a] = rb.h2h_ga.get(team_a, 0) + mr.goals_a

    standings = _sort_group(list(records.values()), rng)
    return standings, results


def _sort_group(
    records: list[TeamRecord],
    rng: np.random.Generator,
) -> list[TeamRecord]:
    """
    Sort group standings by FIFA tiebreak rules.
    """
    def tiebreak_key(tr: TeamRecord) -> tuple:
        return (tr.pts, tr.gd, tr.gf)

    records.sort(key=tiebreak_key, reverse=True)

    # Resolve ties between exactly-tied teams using H2H
    result: list[TeamRecord] = []
    i = 0
    while i < len(records):
        j = i + 1
        while j < len(records) and tiebreak_key(records[j]) == tiebreak_key(records[i]):
            j += 1
        tied_group = records[i:j]

        if len(tied_group) > 1:
            tied_group = _resolve_h2h_tie(tied_group, rng)

        result.extend(tied_group)
        i = j

    return result


def _resolve_h2h_tie(
    tied: list[TeamRecord],
    rng: np.random.Generator,
) -> list[TeamRecord]:
    """Apply H2H tiebreaks within a tied group, then random draw."""
    def h2h_key(tr: TeamRecord) -> tuple:
        opponents = [t.name for t in tied if t.name != tr.name]
        h2h_pts = sum(sum(tr.h2h.get(o, [0])) for o in opponents)
        h2h_gd  = sum((tr.h2h_gf.get(o, 0) - tr.h2h_ga.get(o, 0)) for o in opponents)
        h2h_gf  = sum(tr.h2h_gf.get(o, 0) for o in opponents)
        return (h2h_pts, h2h_gd, h2h_gf)

    tied.sort(key=h2h_key, reverse=True)

    # Random final tie-break (coin flip / draw)
    i = 0
    result: list[TeamRecord] = []
    while i < len(tied):
        j = i + 1
        while j < len(tied) and h2h_key(tied[j]) == h2h_key(tied[i]):
            j += 1
        sub = tied[i:j]
        if len(sub) > 1:
            rng.shuffle(sub)
        result.extend(sub)
        i = j

    return result


# ---------------------------------------------------------------------------
# Best third-place logic
# ---------------------------------------------------------------------------

_BEST_THIRD_SORT = ("pts", "gd", "gf")


def select_best_third_place(
    third_place_teams: list[TeamRecord],
    rng: np.random.Generator,
) -> list[TeamRecord]:
    """
    Select 8 best third-place teams (one per group) from 12.
    Sort by pts → gd → gf → random.
    """
    def sort_key(tr: TeamRecord) -> tuple:
        return (tr.pts, tr.gd, tr.gf)

    sorted_thirds = sorted(third_place_teams, key=sort_key, reverse=True)

    # Break ties randomly
    result: list[TeamRecord] = []
    i = 0
    while i < len(sorted_thirds):
        j = i + 1
        while j < len(sorted_thirds) and sort_key(sorted_thirds[j]) == sort_key(sorted_thirds[i]):
            j += 1
        sub = sorted_thirds[i:j]
        if len(sub) > 1:
            rng.shuffle(sub)
        result.extend(sub)
        i = j

    return result[:8]


# ---------------------------------------------------------------------------
# Bracket construction (Round of 32)
# ---------------------------------------------------------------------------

# Official WC2026 R32 seeding pairing (Group winner vs 3rd-place / Group runner-up vs 3rd)
# We'll use a simplified ordered pairing matching the announced bracket structure.
# Groups A-L:  winner[A] vs runner[B], runner[A] vs winner[B], etc.
# The official bracket pairs group positions in a specific pattern to avoid
# same-group opponents in R32.  We use a standard serpentine pairing.

def _build_r32_bracket(
    group_winners: list[str],    # 12 teams (one per group A-L)
    group_runners: list[str],    # 12 teams
    best_thirds: list[str],      # 8 teams
) -> list[tuple[str, str]]:
    """
    Build the Round of 32 matchups.

    Official WC2026 bracket (simplified for simulation):
    - The 8 third-place qualifiers are slotted into pre-determined positions
      based on which groups they came from.  For simulation purposes we use
      a seeded pairing: winners[i] vs thirds[i] for i in 0..7,
      and runners[i] vs runners[11-i] for i in 0..5,
      keeping winners away from same-group runners.

    Returns list of 16 (team_a, team_b) pairs.
    """
    matches: list[tuple[str, str]] = []

    # Pair group winners with best 8 third-place (8 matches)
    for i in range(8):
        matches.append((group_winners[i], best_thirds[i]))

    # Pair remaining 4 group winners vs 4 group runners-up (cross-bracket)
    # Winners 8-11 face runners 0-3
    for i in range(4):
        matches.append((group_winners[8 + i], group_runners[i]))

    # Runners 4-11 pair up among themselves (4 matches)
    for i in range(4):
        matches.append((group_runners[4 + i], group_runners[8 + i]))

    return matches


# ---------------------------------------------------------------------------
# Knockout rounds
# ---------------------------------------------------------------------------

def simulate_knockout_round(
    matchups: list[tuple[str, str]],
    rng: np.random.Generator,
    round_name: str = "Knockout",
) -> tuple[list[str], list[MatchResult]]:
    """
    Simulate a single knockout round.
    Returns (winners list, match results).
    """
    winners: list[str] = []
    results: list[MatchResult] = []

    for team_a, team_b in matchups:
        mr = _simulate_knockout_match(team_a, team_b, rng)
        results.append(mr)
        winners.append(mr.winner)
        logger.debug(
            f"  [{round_name}] {team_a} {mr.score_str()} {team_b} → {mr.winner}"
        )

    return winners, results


def _make_matchups(teams: list[str]) -> list[tuple[str, str]]:
    """Pair sequential teams: [A,B,C,D] → [(A,B),(C,D)]."""
    return [(teams[i], teams[i + 1]) for i in range(0, len(teams), 2)]


# ---------------------------------------------------------------------------
# Full tournament
# ---------------------------------------------------------------------------

def simulate_world_cup(
    groups: dict[str, list[str]],
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Simulate the entire FIFA World Cup 2026.

    Parameters
    ----------
    groups : dict  {group_letter: [team1, team2, team3, team4]}
    rng    : np.random.Generator (optional, for reproducibility)
    verbose: bool  print progress

    Returns
    -------
    dict with keys:
        group_standings   : {group: [TeamRecord, ...]}
        group_results     : {group: [MatchResult, ...]}
        r32_bracket       : list of (team_a, team_b) pairs
        r32_results       : list of MatchResult
        r16_results       : list of MatchResult
        qf_results        : list of MatchResult
        sf_results        : list of MatchResult
        final_result      : MatchResult
        champion          : str
        runner_up         : str
        semifinalists     : list[str] (4 teams, includes finalists)
        quarterfinalists  : list[str] (8 teams)
        r16_teams         : list[str] (16 teams that reached R16)
        r32_teams         : list[str] (32 teams in R32)
        advancement       : dict[team → furthest_round]
    """
    if rng is None:
        rng = np.random.default_rng()

    # ── Group Stage ────────────────────────────────────────────────────────
    group_standings: dict[str, list[TeamRecord]] = {}
    group_results:   dict[str, list[MatchResult]] = {}
    group_winners:   list[str] = []
    group_runners:   list[str] = []
    third_place_records: list[TeamRecord] = []

    for grp, teams in groups.items():
        standings, results = simulate_group(grp, teams, rng)
        group_standings[grp] = standings
        group_results[grp]   = results

        group_winners.append(standings[0].name)
        group_runners.append(standings[1].name)
        third_place_records.append(standings[2])

        if verbose:
            print(f"\nGroup {grp}:")
            for i, tr in enumerate(standings[:3], 1):
                tag = " ✓" if i <= 2 else ""
                print(f"  {i}. {tr.name:22s} Pts:{tr.pts}  GD:{tr.gd:+d}  GF:{tr.gf}{tag}")

    # ── Best 8 Third-place teams ───────────────────────────────────────────
    best_thirds_records = select_best_third_place(third_place_records, rng)
    best_thirds = [t.name for t in best_thirds_records]

    if verbose:
        print(f"\nBest third-place qualifiers: {', '.join(best_thirds)}")

    # Track advancement
    advancement: dict[str, str] = {}
    for grp, standings in group_standings.items():
        for tr in standings:
            advancement[tr.name] = "group_stage"
    for t in group_winners + group_runners:
        advancement[t] = "r32"
    for t in best_thirds:
        advancement[t] = "r32"

    # ── Round of 32 ────────────────────────────────────────────────────────
    r32_matchups = _build_r32_bracket(group_winners, group_runners, best_thirds)
    r32_winners, r32_results = simulate_knockout_round(r32_matchups, rng, "R32")

    for t in r32_winners:
        advancement[t] = "r16"

    if verbose:
        print("\nRound of 32 winners:")
        _print_ko_results(r32_results)

    # ── Round of 16 ────────────────────────────────────────────────────────
    r16_matchups = _make_matchups(r32_winners)
    r16_winners, r16_results = simulate_knockout_round(r16_matchups, rng, "R16")

    for t in r16_winners:
        advancement[t] = "quarterfinal"

    if verbose:
        print("\nRound of 16 winners:")
        _print_ko_results(r16_results)

    # ── Quarterfinals ──────────────────────────────────────────────────────
    qf_matchups = _make_matchups(r16_winners)
    qf_winners, qf_results = simulate_knockout_round(qf_matchups, rng, "QF")

    for t in qf_winners:
        advancement[t] = "semifinal"

    if verbose:
        print("\nQuarterfinal winners:")
        _print_ko_results(qf_results)

    # ── Semifinals ─────────────────────────────────────────────────────────
    sf_matchups = _make_matchups(qf_winners)
    sf_winners, sf_results = simulate_knockout_round(sf_matchups, rng, "SF")

    for t in sf_winners:
        advancement[t] = "final"

    if verbose:
        print("\nSemifinal winners:")
        _print_ko_results(sf_results)

    # ── Final ──────────────────────────────────────────────────────────────
    final_mr = _simulate_knockout_match(sf_winners[0], sf_winners[1], rng)
    champion  = final_mr.winner
    runner_up = final_mr.loser
    advancement[champion]  = "champion"
    advancement[runner_up] = "runner_up"

    if verbose:
        print(f"\n{'═'*50}")
        print(f"  🏆  FINAL:  {sf_winners[0]}  {final_mr.score_str()}  {sf_winners[1]}")
        print(f"  🏆  CHAMPION: {champion}")
        print(f"{'═'*50}\n")

    return {
        "group_standings":  group_standings,
        "group_results":    group_results,
        "r32_bracket":      r32_matchups,
        "r32_results":      r32_results,
        "r16_results":      r16_results,
        "qf_results":       qf_results,
        "sf_results":       sf_results,
        "final_result":     final_mr,
        "champion":         champion,
        "runner_up":        runner_up,
        "semifinalists":    qf_winners,          # 4 teams
        "quarterfinalists": r16_winners,          # 8 teams
        "r16_teams":        r32_winners,          # 16 teams
        "r32_teams":        group_winners + group_runners + best_thirds,  # 32 teams
        "best_thirds":      best_thirds,
        "advancement":      advancement,
    }


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _print_ko_results(results: list[MatchResult]) -> None:
    for mr in results:
        flag = " (pens)" if mr.after_pens else (" (aet)" if mr.after_et else "")
        print(f"  {mr.team_a:22s} {mr.score_str():10s} {mr.team_b:22s}  → {mr.winner}{flag}")


def print_bracket(result: dict[str, Any]) -> None:
    """Pretty-print a full tournament bracket."""
    print(f"\n{'═'*60}")
    print("  FIFA WORLD CUP 2026 — TOURNAMENT BRACKET")
    print(f"{'═'*60}")

    print("\n── GROUP STAGE ──────────────────────────────────────────────")
    for grp, standings in result["group_standings"].items():
        print(f"\n  Group {grp}:")
        print(f"  {'Team':<22} {'P':>2} {'W':>2} {'D':>2} {'L':>2} {'GF':>3} {'GA':>3} {'GD':>4} {'Pts':>4}")
        for i, tr in enumerate(standings):
            advance = "✓" if i < 2 else ("*" if tr.name in result["best_thirds"] else " ")
            print(f"  {advance} {tr.name:<21} {tr.played:>2} {tr.wins:>2} {tr.draws:>2} {tr.losses:>2} "
                  f"{tr.gf:>3} {tr.ga:>3} {tr.gd:>+4} {tr.pts:>4}")

    for round_name, key in [
        ("ROUND OF 32", "r32_results"),
        ("ROUND OF 16", "r16_results"),
        ("QUARTERFINALS", "qf_results"),
        ("SEMIFINALS", "sf_results"),
    ]:
        print(f"\n── {round_name} {'─'*(50-len(round_name))}")
        _print_ko_results(result[key])

    fr = result["final_result"]
    print(f"\n{'═'*60}")
    print(f"  🏆  FINAL:  {fr.team_a}  {fr.score_str()}  {fr.team_b}")
    print(f"  🏆  CHAMPION:  {result['champion']}")
    print(f"  🥈  RUNNER-UP: {result['runner_up']}")
    print(f"{'═'*60}\n")
