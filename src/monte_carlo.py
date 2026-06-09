"""
Monte Carlo simulation engine — FIFA World Cup 2026.

Runs N independent tournament simulations and aggregates probabilities
for every team at every tournament stage.

Usage
-----
    python src/monte_carlo.py                     # 10,000 simulations
    python src/monte_carlo.py --n 1000 --seed 0   # fast test run
    python src/monte_carlo.py --n 50000           # high precision

Outputs
-------
    outputs/monte_carlo_results.json
    reports/tournament_simulation_report.md
    reports/figures/  (5 charts)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.predict_match import precompute_matchups
from src.simulation.tournament import simulate_world_cup
from src.utils.logger import get_logger

logger = get_logger("monte_carlo", log_file="outputs/run.log")


# ---------------------------------------------------------------------------
# Monte Carlo engine
# ---------------------------------------------------------------------------

def run_monte_carlo(
    n_simulations: int = 10_000,
    seed: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run `n_simulations` independent World Cup simulations.

    Returns
    -------
    DataFrame with columns:
        team, champion, runner_up, semifinal, quarterfinal,
        round_of_16, round_of_32, group_stage, n_simulations
    All numeric columns are counts (not probabilities) — divide by n_simulations.
    """
    groups = _load_groups()
    all_teams = [t for grp_teams in groups.values() for t in grp_teams]

    # Precompute all matchup probabilities once (avoids LightGBM inference in the hot loop)
    if verbose:
        print("  Precomputing matchup probabilities for all 48 teams ...")
    precompute_matchups(all_teams)

    # Counters
    counts: dict[str, dict[str, int]] = {
        t: defaultdict(int) for t in all_teams
    }

    master_rng = np.random.default_rng(seed)
    seeds = master_rng.integers(0, 2**31, size=n_simulations)

    t0 = time.time()
    log_every = max(1, n_simulations // 20)

    for sim_idx in range(n_simulations):
        rng = np.random.default_rng(int(seeds[sim_idx]))
        try:
            result = simulate_world_cup(groups, rng=rng, verbose=False)
        except Exception as e:
            logger.warning(f"Simulation {sim_idx} failed: {e} — skipping")
            continue

        adv = result["advancement"]

        for team in all_teams:
            stage = adv.get(team, "group_stage")
            counts[team]["group_stage"] += 1  # everyone played group stage

            if stage in ("r32", "r16", "quarterfinal", "semifinal",
                         "final", "runner_up", "champion"):
                counts[team]["round_of_32"] += 1
            if stage in ("r16", "quarterfinal", "semifinal",
                         "final", "runner_up", "champion"):
                counts[team]["round_of_16"] += 1
            if stage in ("quarterfinal", "semifinal",
                         "final", "runner_up", "champion"):
                counts[team]["quarterfinal"] += 1
            if stage in ("semifinal", "final", "runner_up", "champion"):
                counts[team]["semifinal"] += 1
            if stage in ("final", "runner_up", "champion"):
                counts[team]["final"] += 1
            if stage in ("runner_up", "champion"):
                counts[team]["runner_up"] += 1
            if stage == "champion":
                counts[team]["champion"] += 1

        if verbose and (sim_idx + 1) % log_every == 0:
            elapsed = time.time() - t0
            pct = (sim_idx + 1) / n_simulations * 100
            eta  = elapsed / (sim_idx + 1) * (n_simulations - sim_idx - 1)
            print(f"  [{sim_idx+1:>6,}/{n_simulations:,}]  {pct:5.1f}%  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    elapsed_total = time.time() - t0
    if verbose:
        print(f"\n  Completed {n_simulations:,} simulations in {elapsed_total:.1f}s "
              f"({elapsed_total/n_simulations*1000:.1f} ms/sim)")

    # Build DataFrame
    rows = []
    for team in all_teams:
        c = counts[team]
        rows.append({
            "team":         team,
            "group":        _team_group(team, groups),
            "champion":     c["champion"],
            "runner_up":    c["runner_up"],
            "semifinal":    c["semifinal"],
            "quarterfinal": c["quarterfinal"],
            "round_of_16":  c["round_of_16"],
            "round_of_32":  c["round_of_32"],
            "group_stage":  c["group_stage"],
            "n_simulations": n_simulations,
        })

    df = pd.DataFrame(rows)

    # Add probability columns (pct_ prefix)
    for col in ["champion", "runner_up", "semifinal", "quarterfinal",
                "round_of_16", "round_of_32", "group_stage"]:
        df[f"pct_{col}"] = (df[col] / n_simulations * 100).round(2)

    df = df.sort_values("champion", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def generate_charts(df: pd.DataFrame, fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns

    fig_dir.mkdir(parents=True, exist_ok=True)
    top20 = df.head(20).copy()

    PALETTE = sns.color_palette("tab20", n_colors=20)

    # ── 1. Championship Probability ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(
        top20["team"][::-1],
        top20["pct_champion"][::-1],
        color=PALETTE[::-1], edgecolor="white", linewidth=0.6
    )
    ax.set_xlabel("Championship Probability (%)", fontsize=12)
    ax.set_title("FIFA World Cup 2026 — Championship Probabilities\n(Monte Carlo Simulation)",
                 fontsize=14, fontweight="bold")
    for bar, pct in zip(bars, top20["pct_champion"][::-1]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", ha="left", fontsize=9)
    ax.set_xlim(0, top20["pct_champion"].max() * 1.18)
    plt.tight_layout()
    plt.savefig(fig_dir / "championship_probabilities.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 2. Semifinal Probability ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8))
    sf_top20 = df.sort_values("pct_semifinal", ascending=False).head(20)
    bars = ax.barh(
        sf_top20["team"][::-1],
        sf_top20["pct_semifinal"][::-1],
        color=PALETTE[::-1], edgecolor="white", linewidth=0.6
    )
    ax.set_xlabel("Semifinal Probability (%)", fontsize=12)
    ax.set_title("FIFA World Cup 2026 — Semifinal Probabilities\n(Top 20 Teams)",
                 fontsize=14, fontweight="bold")
    for bar, pct in zip(bars, sf_top20["pct_semifinal"][::-1]):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", ha="left", fontsize=9)
    ax.set_xlim(0, sf_top20["pct_semifinal"].max() * 1.15)
    plt.tight_layout()
    plt.savefig(fig_dir / "semifinal_probabilities.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 3. Advancement Odds — stacked bar (top 20 by championship) ────────
    fig, ax = plt.subplots(figsize=(14, 8))

    cols_ordered = ["pct_round_of_32", "pct_round_of_16", "pct_quarterfinal",
                    "pct_semifinal", "pct_runner_up", "pct_champion"]
    labels       = ["Round of 32", "Round of 16", "QF", "SF", "Runner-Up", "Champion"]
    stage_colors = ["#B0BEC5", "#78909C", "#42A5F5", "#1E88E5", "#FFA726", "#E53935"]

    teams = top20["team"].tolist()
    bottoms = np.zeros(len(teams))

    # We show incremental advancement (R32 then +R16 etc.)
    prev_col = np.zeros(len(teams))
    for col, label, color in zip(cols_ordered, labels, stage_colors):
        vals = top20[col].values
        increments = np.maximum(vals - prev_col, 0)
        ax.bar(teams, increments, bottom=prev_col,
               label=label, color=color, edgecolor="white", linewidth=0.4)
        prev_col = vals

    ax.set_ylabel("Probability (%)", fontsize=12)
    ax.set_title("FIFA World Cup 2026 — Tournament Advancement Odds\n(Top 20 Teams, Incremental)",
                 fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(fig_dir / "advancement_odds.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 4. Championship bubble chart (all 48 teams) ───────────────────────
    fig, ax = plt.subplots(figsize=(14, 9))
    # Colour by confederation
    from src.utils.config import GROUPS as cfg_groups_obj
    try:
        conf_map = cfg_groups_obj.get("confederations", {})
        team_conf = {}
        for conf, members in conf_map.items():
            for m in members:
                team_conf[m] = conf
    except Exception:
        team_conf = {}

    conf_colors = {
        "UEFA": "#1565C0", "CONMEBOL": "#2E7D32", "CONCACAF": "#F57F17",
        "CAF": "#BF360C", "AFC": "#6A1B9A", "OFC": "#00695C",
    }

    for _, row in df.iterrows():
        conf = team_conf.get(row["team"], "OFC")
        color = conf_colors.get(conf, "#607D8B")
        size  = max(20, row["pct_champion"] * 60)
        ax.scatter(row["pct_round_of_32"], row["pct_semifinal"],
                   s=size, color=color, alpha=0.75, edgecolors="white", linewidth=0.5)
        if row["pct_champion"] >= 2.0:
            ax.annotate(row["team"],
                        (row["pct_round_of_32"], row["pct_semifinal"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=7)

    ax.set_xlabel("Group Stage Advancement Probability (%)", fontsize=11)
    ax.set_ylabel("Semifinal Probability (%)", fontsize=11)
    ax.set_title("FIFA World Cup 2026 — Team Strength Map\n(bubble size = champion probability)",
                 fontsize=13, fontweight="bold")
    patches = [mpatches.Patch(color=c, label=n) for n, c in conf_colors.items()]
    ax.legend(handles=patches, loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "team_strength_map.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 5. Per-group advancement heat map ─────────────────────────────────
    groups = _load_groups()
    heat_rows = []
    for grp, teams in groups.items():
        for t in teams:
            row = df[df["team"] == t]
            if len(row):
                heat_rows.append({
                    "Group": grp, "Team": t,
                    "Advance": row["pct_round_of_32"].iloc[0],
                    "Champion": row["pct_champion"].iloc[0],
                })
    heat_df = pd.DataFrame(heat_rows)
    pivot = heat_df.pivot(index="Group", columns="Team", values="Advance").fillna(0)

    # Sort columns within each group by advancement
    sorted_cols = heat_df.sort_values(["Group","Advance"], ascending=[True,False])["Team"].tolist()
    pivot = pivot[sorted_cols] if len(sorted_cols) == pivot.shape[1] else pivot

    fig, ax = plt.subplots(figsize=(20, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", ax=ax,
                cmap="YlOrRd", linewidths=0.5,
                cbar_kws={"label": "Group Advancement %"})
    ax.set_title("Group Stage Advancement Probability (%) by Team",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "group_advancement_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Charts saved → {fig_dir.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def generate_report(
    df: pd.DataFrame,
    n_simulations: int,
    example_result: dict,
    report_path: Path,
) -> None:
    from datetime import datetime
    from src.simulation.tournament import MatchResult

    report_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines: list[str] = []

    lines += [
        "# FIFA World Cup 2026 — Tournament Simulation Report",
        f"\n_Generated: {now} · {n_simulations:,} Monte Carlo simulations_\n",
        "---\n",

        "## 1. Simulation Methodology\n",
        "### 1.1 Match Prediction Engine\n",
        "Each match is predicted using the **tuned LightGBM pipeline** trained in Phase 3 "
        "(test F1-macro = 0.527, accuracy = 57.3%). The model outputs three probabilities:\n",
        "- `p(Home Win)` → remapped as `p(Team A win)` at neutral venue",
        "- `p(Draw)`",
        "- `p(Away Win)` → remapped as `p(Team B win)`\n",

        "### 1.2 Score Generation (Poisson Model)\n",
        "Exact scores are generated using a **calibrated Poisson model**:\n",
        "1. **Baseline rates**: World Cup neutral-venue averages (1.32 and 1.10 goals/90 min).",
        "2. **Strength scaling**: rates are scaled by the ratio of the team's predicted win "
        "probability to the historical baseline win rate, with a damping exponent of 0.35 "
        "to prevent extreme scorelines.",
        "3. **Regression to mean**: 25% blend toward the baseline prevents 5–0 outcomes "
        "from near-certain predictions.",
        "4. **Poisson sampling**: `goals ~ Poisson(λ)`, capped at 7 per team.\n",
        "Example: Brazil (p_win=0.52) vs Ecuador (p_win=0.22) → λ_Brazil=1.45, λ_Ecuador=1.05\n",

        "### 1.3 Knockout Tiebreaker (Extra Time + Penalties)\n",
        "When a knockout match is level after 90 min:\n",
        "- **Extra time**: additional 30 min simulated with scaled Poisson rates (×30/90).",
        "- **Penalty shootout**: if still level, winner drawn with probability biased by "
        "Elo strength difference (clamped to 30%–70%).\n",

        "### 1.4 Group Stage Tiebreaks (FIFA rules)\n",
        "1. Points  2. Goal difference  3. Goals scored  4. H2H points  "
        "5. H2H goal difference  6. H2H goals scored  7. Random draw\n",

        "### 1.5 Monte Carlo Setup\n",
        f"- **{n_simulations:,} independent simulations** — each uses a fresh random seed.",
        "- Each simulation is fully independent: group stage → R32 → R16 → QF → SF → Final.",
        "- Probabilities are the fraction of simulations where each outcome occurred.\n",

        "---\n",
        "## 2. Championship Probabilities\n",

        "| Rank | Team | Group | Champion% | Runner-Up% | Semifinal% | QF% | R16% | R32% |",
        "|------|------|-------|-----------|------------|------------|-----|------|------|",
    ]

    for rank, row in df.head(32).iterrows():
        lines.append(
            f"| {rank+1} | **{row['team']}** | {row['group']} "
            f"| {row['pct_champion']:.1f}% "
            f"| {row['pct_runner_up']:.1f}% "
            f"| {row['pct_semifinal']:.1f}% "
            f"| {row['pct_quarterfinal']:.1f}% "
            f"| {row['pct_round_of_16']:.1f}% "
            f"| {row['pct_round_of_32']:.1f}% |"
        )

    # Top 10 summary callout
    top5 = df.head(5)
    lines += [
        "\n\n### Top 5 Favourites\n",
        "```",
    ]
    for _, row in top5.iterrows():
        bar = "█" * int(row["pct_champion"] / 0.5)
        lines.append(f"  {row['team']:22s}  {bar}  {row['pct_champion']:.1f}%")
    lines.append("```\n")

    # ── Example tournament result ─────────────────────────────────────────
    lines += [
        "---\n",
        "## 3. Example Tournament Run (Seed 42)\n",
        "Below is the result of one deterministic simulation to illustrate the output:\n",
    ]

    lines.append("### 3.1 Group Stage Standings\n")
    for grp, standings in example_result["group_standings"].items():
        lines.append(f"**Group {grp}**\n")
        lines.append("| Pos | Team | Pts | W | D | L | GF | GA | GD |")
        lines.append("|-----|------|-----|---|---|---|----|----|-----|")
        for i, tr in enumerate(standings, 1):
            adv = " ✓" if i <= 2 else (" *" if tr.name in example_result["best_thirds"] else "")
            lines.append(f"| {i}{adv} | {tr.name} | {tr.pts} | {tr.wins} | {tr.draws} | "
                         f"{tr.losses} | {tr.gf} | {tr.ga} | {tr.gd:+d} |")
        lines.append("")

    def _ko_table(results, title):
        lines.append(f"### {title}\n")
        lines.append("| Team A | Score | Team B | Winner |")
        lines.append("|--------|-------|--------|--------|")
        for mr in results:
            suffix = " *(pens)*" if mr.after_pens else (" *(aet)*" if mr.after_et else "")
            lines.append(f"| {mr.team_a} | {mr.score_str()} | {mr.team_b} | "
                         f"**{mr.winner}**{suffix} |")
        lines.append("")

    _ko_table(example_result["r32_results"],  "3.2 Round of 32")
    _ko_table(example_result["r16_results"],  "3.3 Round of 16")
    _ko_table(example_result["qf_results"],   "3.4 Quarterfinals")
    _ko_table(example_result["sf_results"],   "3.5 Semifinals")

    fr = example_result["final_result"]
    lines += [
        "### 3.6 Final\n",
        f"| {fr.team_a} | **{fr.score_str()}** | {fr.team_b} |",
        "|---|---|---|",
        f"| {'🏆 **CHAMPION**' if fr.winner==fr.team_a else ''} | | "
        f"{'🏆 **CHAMPION**' if fr.winner==fr.team_b else ''} |\n",
        f"**🏆 Champion: {example_result['champion']}**  ",
        f"**🥈 Runner-up: {example_result['runner_up']}**\n",
    ]

    lines += [
        "---\n",
        "## 4. Visualisations\n",
        "Charts saved to `reports/figures/`:\n",
        "| File | Description |",
        "|------|-------------|",
        "| `championship_probabilities.png` | Top 20 teams by championship % |",
        "| `semifinal_probabilities.png`    | Top 20 teams by semifinal % |",
        "| `advancement_odds.png`           | Stacked advancement bar chart |",
        "| `team_strength_map.png`          | Bubble chart: R32% vs SF%, size=champion% |",
        "| `group_advancement_heatmap.png`  | Heatmap of group advancement by team |\n",

        "---\n",
        "## 5. Model Limitations & Caveats\n",
        "| # | Limitation | Impact |",
        "|---|-----------|--------|",
        "| 1 | **Model accuracy 57.3%** — soccer outcomes are inherently uncertain | Probabilities are calibrated estimates, not certainties |",
        "| 2 | **No squad/injury data** — key absences (e.g., star player suspended) are invisible | Upsets may be under/over-estimated |",
        "| 3 | **FIFA rankings frozen at Nov 2019** — rank features are forward-filled | Elo (fully current) is the primary strength signal |",
        "| 4 | **Draw prediction recall ≈ 34%** — draws are inherently hard to predict | Score outcomes have higher variance than outcome probabilities |",
        "| 5 | **Poisson independence assumption** — goals for each team are drawn independently | Under-represents 0-0 and 1-1 draws (Dixon-Coles correction not applied) |",
        "| 6 | **WC format novelty** — 48-team format with 12 groups is new in 2026 | No historical WC data in this exact format |",
        "\n---",
        f"\n_Simulation report generated automatically by `src/monte_carlo.py` on {now}_",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report saved → {report_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_groups() -> dict[str, list[str]]:
    cfg_path = ROOT / "configs" / "wc2026_groups.yaml"
    with open(cfg_path) as f:
        data = yaml.safe_load(f)
    return data["groups"]


def _team_group(team: str, groups: dict[str, list[str]]) -> str:
    for grp, teams in groups.items():
        if team in teams:
            return grp
    return "?"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(n: int = 10_000, seed: int | None = None, quick: bool = False) -> None:
    n = 500 if quick else n

    out_dir  = ROOT / "outputs"
    fig_dir  = ROOT / "reports" / "figures"
    rep_path = ROOT / "reports" / "tournament_simulation_report.md"
    out_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  FIFA World Cup 2026 — Monte Carlo Simulation")
    print(f"  Running {n:,} simulations ...")
    print(f"{'='*60}\n")

    # Run Monte Carlo
    df = run_monte_carlo(n_simulations=n, seed=seed, verbose=True)

    # Save raw results
    results_path = out_dir / "monte_carlo_results.csv"
    df.to_csv(results_path, index=False)
    json_path = out_dir / "monte_carlo_results.json"
    df.to_json(json_path, orient="records", indent=2)
    logger.info(f"Monte Carlo results saved → {results_path.relative_to(ROOT)}")

    # Print top 20
    print(f"\n{'─'*72}")
    print(f"  {'Team':<22}  {'Champion%':>10}  {'Final%':>8}  {'SF%':>7}  {'QF%':>7}  {'R32%':>7}")
    print(f"{'─'*72}")
    for _, row in df.head(20).iterrows():
        print(f"  {row['team']:<22}  {row['pct_champion']:>9.1f}%  "
              f"{row['pct_runner_up']:>7.1f}%  "
              f"{row['pct_semifinal']:>6.1f}%  "
              f"{row['pct_quarterfinal']:>6.1f}%  "
              f"{row['pct_round_of_32']:>6.1f}%")
    print(f"{'─'*72}\n")

    # Example single tournament run (fixed seed for report)
    print("Running example tournament (seed=42) for report ...\n")
    example_groups = _load_groups()
    example_rng = np.random.default_rng(42)
    example_result = simulate_world_cup(example_groups, rng=example_rng, verbose=True)

    # Generate charts
    print("\nGenerating charts ...")
    generate_charts(df, fig_dir)

    # Generate report
    print("Writing report ...")
    generate_report(df, n_simulations=n, example_result=example_result, report_path=rep_path)

    print(f"\n{'='*60}")
    print(f"  ✅  Monte Carlo complete!")
    print(f"  Results  → {results_path}")
    print(f"  Charts   → {fig_dir}/")
    print(f"  Report   → {rep_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WC2026 Monte Carlo Simulator")
    parser.add_argument("--n",     type=int, default=10_000, help="Number of simulations")
    parser.add_argument("--seed",  type=int, default=None,   help="Master RNG seed")
    parser.add_argument("--quick", action="store_true",      help="Run 500 sims (fast test)")
    args = parser.parse_args()
    main(n=args.n, seed=args.seed, quick=args.quick)
