"""
Data quality audit — produces a printed report and returns a structured dict.

Checks:
  - Shape, dtypes, null counts per dataset
  - Duplicate rows
  - Score anomalies (nulls, negatives, implausible high scores)
  - Team name mismatches between results and rankings
  - Rankings date coverage gap (ends 2019, matches run to 2026)
  - Upcoming / future matches with no scores (not bugs — expected)
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _null_summary(df: pd.DataFrame, label: str) -> dict:
    nulls = df.isnull().sum()
    pct = (nulls / len(df) * 100).round(2)
    issues = nulls[nulls > 0]
    return {
        "dataset": label,
        "rows": len(df),
        "cols": len(df.columns),
        "null_counts": issues.to_dict(),
        "null_pct": pct[issues.index].to_dict(),
    }


def run(raw_dir: str | Path = "data/raw") -> dict:
    raw_dir = Path(raw_dir)
    report: dict = {}

    # ------------------------------------------------------------------ load
    results = pd.read_csv(raw_dir / "results.csv", parse_dates=["date"])
    rankings = pd.read_csv(raw_dir / "fifa_rankings.csv")
    rankings["date"] = pd.to_datetime(rankings["date"], utc=True).dt.tz_localize(None)
    shootouts = pd.read_csv(raw_dir / "shootouts.csv", parse_dates=["date"])
    goalscorers = pd.read_csv(raw_dir / "goalscorers.csv", parse_dates=["date"])

    # ------------------------------------------------------------------ nulls
    report["nulls"] = {
        ds: _null_summary(df, ds)
        for ds, df in [
            ("results", results),
            ("rankings", rankings),
            ("shootouts", shootouts),
            ("goalscorers", goalscorers),
        ]
    }

    # ------------------------------------------------------------------ duplicates
    dup_results = results.duplicated(subset=["date", "home_team", "away_team"]).sum()
    dup_rankings = rankings.duplicated(subset=["date", "country"]).sum()
    report["duplicates"] = {"results": int(dup_results), "rankings": int(dup_rankings)}

    # ------------------------------------------------------------------ score anomalies
    future_mask = results["home_score"].isna()
    negative_mask = (results["home_score"] < 0) | (results["away_score"] < 0)
    high_mask = (results["home_score"] > 20) | (results["away_score"] > 20)
    report["score_issues"] = {
        "null_scores_future_matches": int(future_mask.sum()),
        "negative_scores": int(negative_mask.dropna().sum()),
        "implausibly_high_scores": int(high_mask.dropna().sum()),
        "high_score_examples": results[high_mask][["date", "home_team", "away_team",
                                                    "home_score", "away_score"]].head(5).to_dict("records"),
    }

    # ------------------------------------------------------------------ team name mismatches
    rank_names = set(rankings["country"].unique())
    post2000 = results[results["date"].dt.year >= 2000]
    result_names = set(post2000["home_team"].unique()) | set(post2000["away_team"].unique())
    not_in_rankings = sorted(result_names - rank_names)
    not_in_results = sorted(rank_names - result_names)
    report["name_mismatches"] = {
        "in_results_not_rankings": not_in_rankings,
        "in_rankings_not_results": not_in_results,
        "mismatch_count": len(not_in_rankings),
    }

    # ------------------------------------------------------------------ rankings coverage
    last_ranking_date = rankings["date"].max()
    last_match_date = results[results["home_score"].notna()]["date"].max()
    report["rankings_coverage"] = {
        "rankings_end": str(last_ranking_date.date()),
        "matches_end": str(last_match_date.date()),
        "gap_days": int((last_match_date - last_ranking_date).days),
        "note": "Rankings end 2019; FIFA points for 2020-2026 will use last-known forward-fill",
    }

    # ------------------------------------------------------------------ tournament breakdown
    post2000_played = post2000.dropna(subset=["home_score"])
    report["tournament_breakdown"] = post2000_played["tournament"].value_counts().head(20).to_dict()

    # ------------------------------------------------------------------ print
    _print_report(report)
    return report


def _print_report(r: dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print("  DATA QUALITY REPORT")
    print(sep)

    print("\n[1] NULL VALUES")
    for ds, info in r["nulls"].items():
        print(f"  {ds:15s} — {info['rows']:,} rows × {info['cols']} cols")
        if info["null_counts"]:
            for col, cnt in info["null_counts"].items():
                print(f"    • {col:30s} {cnt:6,} nulls ({info['null_pct'][col]:.1f}%)")
        else:
            print("    • No nulls")

    print("\n[2] DUPLICATES")
    for ds, cnt in r["duplicates"].items():
        print(f"  {ds:15s} — {cnt} duplicates")

    print("\n[3] SCORE ANOMALIES")
    si = r["score_issues"]
    print(f"  Null scores (future matches) : {si['null_scores_future_matches']}")
    print(f"  Negative scores              : {si['negative_scores']}")
    print(f"  Implausibly high (>20)       : {si['implausibly_high_scores']}")
    if si["high_score_examples"]:
        for ex in si["high_score_examples"]:
            print(f"    → {ex['date']} {ex['home_team']} {ex['home_score']:.0f}–{ex['away_score']:.0f} {ex['away_team']}")

    print("\n[4] TEAM NAME MISMATCHES (results 2000+ vs rankings)")
    mm = r["name_mismatches"]
    print(f"  Teams in results NOT in rankings: {mm['mismatch_count']}")
    print(f"  Key examples: {mm['in_results_not_rankings'][:10]}")

    print("\n[5] RANKINGS DATE COVERAGE")
    rc = r["rankings_coverage"]
    print(f"  Rankings end : {rc['rankings_end']}")
    print(f"  Matches end  : {rc['matches_end']}")
    print(f"  Gap          : {rc['gap_days']} days")
    print(f"  Strategy     : {rc['note']}")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    run()
