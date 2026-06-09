"""
Clean and filter raw data, then save to data/processed/.

Steps:
  1. Filter results to training_start_year–present
  2. Drop friendlies played during COVID bubble (2020-06 to 2021-03) — anomalous
  3. Standardise team name spelling (e.g. "Korea Republic" → "South Korea")
  4. Add a 'result' column: 0=home win, 1=draw, 2=away win
  5. Merge nearest FIFA ranking for each team at match date
  6. Save processed CSVs
"""

from pathlib import Path
import numpy as np
import pandas as pd

from src.utils.config import CONFIG
from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_DIR = Path(CONFIG["paths"]["raw_data"])
PROC_DIR = Path(CONFIG["paths"]["processed_data"])
START_YEAR = CONFIG["data"]["training_start_year"]

# Canonical name map — maps source spellings → project standard
TEAM_NAME_MAP: dict[str, str] = {
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "IR Iran": "Iran",
    "China PR": "China",
    "Congo DR": "DR Congo",
    "Cape Verde Islands": "Cape Verde",
    "Ivory Coast": "Ivory Coast",
    "Czech Republic": "Czech Republic",
    "Northern Ireland": "Northern Ireland",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "São Tomé and Príncipe": "Sao Tome and Principe",
}


def _standardise_names(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = df[col].replace(TEAM_NAME_MAP)
    return df


def _add_result_column(df: pd.DataFrame) -> pd.DataFrame:
    """0 = home win, 1 = draw, 2 = away win.  Rows with NaN scores are dropped."""
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["result"] = np.select(
        [df["home_score"] > df["away_score"], df["home_score"] == df["away_score"]],
        [0, 1],
        default=2,
    )
    return df


def preprocess_results(results: pd.DataFrame) -> pd.DataFrame:
    df = results.copy()

    # Filter by year
    df = df[df["date"].dt.year >= START_YEAR].copy()

    # Drop COVID-bubble anomaly window
    covid_mask = (df["date"] >= "2020-06-01") & (df["date"] <= "2021-03-31")
    n_dropped = covid_mask.sum()
    if n_dropped:
        logger.info(f"Dropping {n_dropped} COVID-window matches")
    df = df[~covid_mask]

    df = _standardise_names(df, ["home_team", "away_team"])
    df = _add_result_column(df)
    df = df.drop_duplicates(subset=["date", "home_team", "away_team"])
    df = df.sort_values("date").reset_index(drop=True)

    logger.info(f"Processed results: {len(df):,} rows")
    return df


def merge_rankings(matches: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    """
    For each match, attach the most recent FIFA ranking points for both teams.
    Uses a backwards merge-asof on date.
    """
    rankings = _standardise_names(rankings.copy(), ["team"])
    rankings = rankings.sort_values("date")

    def _get_points(team_col: str, prefix: str) -> pd.DataFrame:
        merged = pd.merge_asof(
            matches[["date", team_col]].sort_values("date"),
            rankings[["date", "team", "rank", "fifa_points"]].rename(
                columns={"team": team_col, "rank": f"{prefix}_rank", "fifa_points": f"{prefix}_fifa_points"}
            ),
            on="date",
            by=team_col,
            direction="backward",
        )
        return merged[[f"{prefix}_rank", f"{prefix}_fifa_points"]]

    home_pts = _get_points("home_team", "home")
    away_pts = _get_points("away_team", "away")

    result = pd.concat([matches.reset_index(drop=True), home_pts, away_pts], axis=1)
    logger.info("Merged FIFA rankings into match data")
    return result


def save_processed(df: pd.DataFrame, filename: str) -> Path:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out = PROC_DIR / filename
    df.to_csv(out, index=False)
    logger.info(f"Saved processed file → {out}")
    return out
