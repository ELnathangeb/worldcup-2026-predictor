"""
Data quality checks run after download.

Raises ValueError on critical failures; logs warnings for soft issues.
"""

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def validate_results(df: pd.DataFrame) -> None:
    errors: list[str] = []

    if df.empty:
        errors.append("results.csv is empty")
    if df["date"].isna().any():
        errors.append(f"{df['date'].isna().sum()} rows have null date")
    if (df["home_score"] < 0).any() or (df["away_score"] < 0).any():
        errors.append("Negative scores detected")
    if df.duplicated(subset=["date", "home_team", "away_team"]).any():
        n = df.duplicated(subset=["date", "home_team", "away_team"]).sum()
        logger.warning(f"{n} duplicate match rows — will be de-duplicated downstream")

    if errors:
        raise ValueError("results.csv validation failed:\n" + "\n".join(f"  • {e}" for e in errors))

    logger.info("results.csv validation passed")


def validate_rankings(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("fifa_rankings.csv is empty")
    if "team" not in df.columns:
        raise ValueError("FIFA rankings missing 'team' column after normalisation")
    logger.info("fifa_rankings.csv validation passed")
