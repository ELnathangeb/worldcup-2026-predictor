"""
Comprehensive data cleaning:
  1. Canonical team-name map (results ↔ rankings ↔ WC2026 draw)
  2. Drop future matches (no scores yet)
  3. Drop true duplicates; keep first of soft duplicates
  4. Filter to post-2000 (sufficient history without ancient noise)
  5. Drop COVID bubble (June 2020 – March 2021)
  6. Strip non-FIFA-member matches (CONIFA / stateless nations) for training
  7. Forward-fill FIFA rankings to cover the 2020–2026 gap
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Canonical name map: source spelling → project standard
# Covers results.csv, rankings (country col), and wc2026_groups.yaml
# ---------------------------------------------------------------------------
NAME_MAP: dict[str, str] = {
    # FIFA official → project standard
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "IR Iran": "Iran",
    "China PR": "China",
    "Congo DR": "DR Congo",
    "Congo, DR": "DR Congo",
    "Cape Verde Islands": "Cape Verde",
    "Cape Verde": "Cape Verde",
    "Cabo Verde": "Cape Verde",
    "Ivory Coast": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "CÃ´te d'Ivoire": "Ivory Coast",
    "Czech Republic": "Czech Republic",
    "Czechia": "Czech Republic",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "São Tomé e Príncipe": "Sao Tome and Principe",
    "Sao Tome e Principe": "Sao Tome and Principe",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "St. Lucia": "Saint Lucia",
    "St. Vincent / Grenadines": "Saint Vincent and the Grenadines",
    "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "Swaziland": "Eswatini",
    "FYR Macedonia": "North Macedonia",
    "Republic of Ireland": "Ireland",
    "Trinidad & Tobago": "Trinidad and Tobago",
    "Kyrgyz Republic": "Kyrgyzstan",
    "Chinese Taipei": "Taiwan",
    "USA": "United States",
    "United States": "United States",
    "Curacao": "Curaçao",
    "Curaçao": "Curaçao",
    "Guinea Bissau": "Guinea-Bissau",
    "Antigua & Barbuda": "Antigua and Barbuda",
    "Serbia and Montenegro": "Serbia",  # dissolved 2006
    "Netherlands Antilles": "Curaçao",  # successor state for football
    "US Virgin Islands": "US Virgin Islands",
    "Brunei Darussalam": "Brunei",
    "Brunei": "Brunei",
}

# Teams that are NOT FIFA members (CONIFA / stateless): exclude from ML training
# They have no FIFA rankings and will never appear in the World Cup
_NON_FIFA = {
    "Abkhazia", "Alderney", "Ambazonia", "Andalusia", "Arameans Suryoye",
    "Artsakh", "Aymara", "Barawa", "Basque Country", "Biafra", "Bonaire",
    "Brittany", "Canary Islands", "Cascadia", "Catalonia", "Chagos Islands",
    "Chameria", "Chechnya", "Cilento", "Corsica", "County of Nice", "Crimea",
    "Darfur", "Donetsk PR", "Délvidék", "East Turkestan", "Elba Island",
    "Ellan Vannin", "Falkland Islands", "Felvidék", "Franconia",
    "French Guiana", "Frøya", "Galicia", "Gotland", "Gozo", "Jersey",
    "Guernsey", "Isle of Man", "Karpatalya", "Kosovo (before FIFA)",
    "Lakota Nation", "Liqeni", "Matabeleland", "Monaco", "Occitania",
    "Padania", "Panjab", "Provence", "Romani People", "Ryukyu",
    "Sahrawi Arab Democratic Republic", "Sapmi", "Sequoyah",
    "South Ossetia", "Székely Land", "Tamil Eelam", "Tuvalu",
    "Two Sicilies", "United Koreans in Japan", "Vatican City",
    "Western Armenia", "Western Sahara", "Yorkshire",
}

# Tournament weight: higher = more competitive, used as a feature
TOURNAMENT_WEIGHT: dict[str, float] = {
    "FIFA World Cup": 1.0,
    "UEFA Euro": 0.95,
    "Copa América": 0.90,
    "African Cup of Nations": 0.85,
    "AFC Asian Cup": 0.80,
    "Gold Cup": 0.75,
    "FIFA Confederations Cup": 0.80,
    "UEFA Nations League": 0.70,
    "CONCACAF Nations League": 0.65,
    "FIFA World Cup qualification": 0.65,
    "UEFA Euro qualification": 0.60,
    "African Cup of Nations qualification": 0.55,
    "AFC Asian Cup qualification": 0.50,
    "Copa América qualification": 0.55,
    "Friendly": 0.20,
}
_DEFAULT_WEIGHT = 0.40  # for regional cups / minor tournaments


def apply_name_map(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = df[col].replace(NAME_MAP)
    return df


def clean_results(results: pd.DataFrame, start_year: int = 2000) -> pd.DataFrame:
    df = results.copy()

    # 1. Standardise names
    df = apply_name_map(df, ["home_team", "away_team"])

    # 2. Drop rows with no scores (future/upcoming matches)
    future_mask = df["home_score"].isna() | df["away_score"].isna()
    logger.info(f"Dropping {future_mask.sum()} future/no-score matches")
    df = df[~future_mask].copy()

    # 3. Drop rows with negative scores (data error)
    bad = (df["home_score"] < 0) | (df["away_score"] < 0)
    if bad.any():
        logger.warning(f"Dropping {bad.sum()} rows with negative scores")
        df = df[~bad]

    # 4. Filter year range
    df = df[df["date"].dt.year >= start_year].copy()

    # 5. Drop COVID bubble (anomalous conditions, behind-closed-doors)
    covid = (df["date"] >= "2020-06-01") & (df["date"] <= "2021-03-31")
    logger.info(f"Dropping {covid.sum()} COVID-bubble matches")
    df = df[~covid]

    # 6. Remove CONIFA/non-FIFA teams
    non_fifa_mask = df["home_team"].isin(_NON_FIFA) | df["away_team"].isin(_NON_FIFA)
    logger.info(f"Dropping {non_fifa_mask.sum()} matches involving non-FIFA teams")
    df = df[~non_fifa_mask]

    # 7. De-duplicate (keep first occurrence)
    dup_mask = df.duplicated(subset=["date", "home_team", "away_team"])
    if dup_mask.any():
        logger.warning(f"Dropping {dup_mask.sum()} duplicate match rows")
        df = df[~dup_mask]

    # 8. Add result label  0=home win, 1=draw, 2=away win
    df["result"] = np.select(
        [df["home_score"] > df["away_score"], df["home_score"] == df["away_score"]],
        [0, 1],
        default=2,
    )

    # 9. Add goal difference
    df["goal_diff"] = (df["home_score"] - df["away_score"]).astype(int)

    # 10. Add tournament weight
    df["tournament_weight"] = df["tournament"].map(TOURNAMENT_WEIGHT).fillna(_DEFAULT_WEIGHT)

    df = df.sort_values("date").reset_index(drop=True)
    logger.info(f"Clean results: {len(df):,} rows")
    return df


def clean_rankings(rankings: pd.DataFrame) -> pd.DataFrame:
    df = rankings.copy()
    df.columns = df.columns.str.lower().str.strip()
    df = df.rename(columns={"country": "team", "points": "fifa_points"})
    df = df.loc[:, ~df.columns.str.match(r"^unnamed")]
    df = apply_name_map(df, ["team"])
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df = df[["date", "team", "rank", "fifa_points"]].dropna(subset=["team", "fifa_points"])
    df = df.drop_duplicates(subset=["date", "team"]).sort_values(["team", "date"]).reset_index(drop=True)
    logger.info(f"Clean rankings: {len(df):,} rows, {df['team'].nunique()} teams, "
                f"{df['date'].min().date()} → {df['date'].max().date()}")
    return df


def merge_rankings_safe(matches: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    """
    Merge nearest past FIFA ranking points onto each match.
    Uses merge_asof (direction='backward') — guarantees no future data leakage.
    Matches with no known ranking get NaN (handled later by imputation in features).
    """
    rankings_sorted = rankings.sort_values("date")
    matches_sorted = matches.sort_values("date").reset_index(drop=True)

    def _attach(team_col: str, prefix: str) -> pd.DataFrame:
        left = matches_sorted[["date", team_col]].copy()
        right = rankings_sorted.rename(columns={
            "team": team_col,
            "rank": f"{prefix}_rank",
            "fifa_points": f"{prefix}_fifa_pts",
        })
        merged = pd.merge_asof(left, right, on="date", by=team_col, direction="backward")
        return merged[[f"{prefix}_rank", f"{prefix}_fifa_pts"]]

    home_cols = _attach("home_team", "home")
    away_cols = _attach("away_team", "away")

    result = pd.concat([matches_sorted, home_cols, away_cols], axis=1)
    pct_h = result["home_fifa_pts"].notna().mean() * 100
    pct_a = result["away_fifa_pts"].notna().mean() * 100
    logger.info(f"FIFA points coverage — home: {pct_h:.1f}%  away: {pct_a:.1f}%")
    return result
