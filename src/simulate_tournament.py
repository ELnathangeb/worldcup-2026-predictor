"""
Single World Cup simulation — entry point.

Usage
-----
    python src/simulate_tournament.py
    python src/simulate_tournament.py --seed 42
    python src/simulate_tournament.py --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.simulation.tournament import simulate_world_cup, print_bracket
from src.utils.logger import get_logger

logger = get_logger("simulate_tournament", log_file="outputs/run.log")


def load_groups() -> dict[str, list[str]]:
    cfg_path = ROOT / "configs" / "wc2026_groups.yaml"
    with open(cfg_path) as f:
        data = yaml.safe_load(f)
    return data["groups"]


def run_single_simulation(seed: int | None = None, verbose: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    groups = load_groups()

    logger.info(f"Starting single tournament simulation (seed={seed})")
    result = simulate_world_cup(groups, rng=rng, verbose=verbose)

    if verbose:
        print_bracket(result)

    # Summary
    logger.info(f"Champion : {result['champion']}")
    logger.info(f"Runner-up: {result['runner_up']}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate a single WC2026 tournament")
    parser.add_argument("--seed",    type=int, default=None, help="Random seed")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    result = run_single_simulation(seed=args.seed, verbose=args.verbose)

    # Save bracket summary to JSON
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    bracket_path = out_dir / "single_simulation_bracket.json"

    def _serialise(obj):
        if hasattr(obj, "__dict__"):
            d = {k: v for k, v in obj.__dict__.items()
                 if not k.startswith("_") and k not in ("h2h", "h2h_gf", "h2h_ga")}
            return d
        raise TypeError(f"Not serialisable: {type(obj)}")

    summary = {
        "champion":       result["champion"],
        "runner_up":      result["runner_up"],
        "semifinalists":  result["semifinalists"],
        "quarterfinalists": result["quarterfinalists"],
        "best_thirds":    result["best_thirds"],
        "advancement":    result["advancement"],
    }
    with open(bracket_path, "w") as f:
        json.dump(summary, f, indent=2, default=_serialise)

    print(f"\nBracket summary saved → {bracket_path}")
