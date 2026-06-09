"""
Load raw CSVs into clean pandas DataFrames.

Each loader performs:
  - type coercion (dates, ints)
  - column renaming to project-standard names
  - basic sanity assertions
"""

from pathlib import Path
import pandas as pd

from src.utils.config import CONFIG
from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_DIR = Path(CONFIG["paths"]["raw_data"])


def load_results() -> pd.DataFrame:
    """
    Load international match results.

    Columns returned:
        date, home_team, away_team, home_score, away_score,
        tournament, city, country, neutral
    """
    path = RAW_DIR / "results.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    assert {"date", "home_team", "away_team", "home_score", "away_score"}.issubset(df.columns)
    logger.info(f"Loaded results: {len(df):,} rows ({df['date'].min().year}–{df['date'].max().year})")
    return df


def load_shootouts() -> pd.DataFrame:
    """Load penalty shootout results (date, home_team, away_team, winner)."""
    path = RAW_DIR / "shootouts.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    logger.info(f"Loaded shootouts: {len(df):,} rows")
    return df


def load_goalscorers() -> pd.DataFrame:
    """Load goalscorer-level events (used to compute goals-per-game features)."""
    path = RAW_DIR / "goalscorers.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    logger.info(f"Loaded goalscorers: {len(df):,} rows")
    return df


def load_fifa_rankings() -> pd.DataFrame:
    """
    Load FIFA world rankings snapshot.

    Columns returned:
        rank_date, country_full, rank, total_points
    """
    path = RAW_DIR / "fifa_rankings.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Source: samuraitruong/fifa-ranking-data (columns: date, rank, country, ct, points, ...)
    rename_map = {
        "country": "team",
        "points": "fifa_points",
        # legacy alternate names
        "country_full": "team",
        "total_points": "fifa_points",
        "rank_date": "date",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Drop unnamed index column if present
    df = df.loc[:, ~df.columns.str.match(r"^unnamed")]

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)

    logger.info(f"Loaded FIFA rankings: {len(df):,} rows")
    return df
