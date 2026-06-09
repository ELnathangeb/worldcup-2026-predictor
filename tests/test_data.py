"""Unit tests for data loading and preprocessing."""

import pandas as pd
import pytest

from src.data.preprocessor import _add_result_column, _standardise_names, TEAM_NAME_MAP
from src.data.validator import validate_results


def _make_matches(**kwargs) -> pd.DataFrame:
    defaults = dict(
        date=pd.to_datetime(["2022-11-20"]),
        home_team=["Brazil"],
        away_team=["Serbia"],
        home_score=[2],
        away_score=[0],
        tournament=["FIFA World Cup"],
        neutral=[False],
    )
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


def test_result_home_win():
    df = _make_matches(home_score=[2], away_score=[0])
    df = _add_result_column(df)
    assert df["result"].iloc[0] == 0


def test_result_draw():
    df = _make_matches(home_score=[1], away_score=[1])
    df = _add_result_column(df)
    assert df["result"].iloc[0] == 1


def test_result_away_win():
    df = _make_matches(home_score=[0], away_score=[3])
    df = _add_result_column(df)
    assert df["result"].iloc[0] == 2


def test_standardise_names():
    df = pd.DataFrame({"home_team": ["Korea Republic", "IR Iran"], "away_team": ["China PR", "Brazil"]})
    out = _standardise_names(df, ["home_team", "away_team"])
    assert out["home_team"].tolist() == ["South Korea", "Iran"]
    assert out["away_team"].tolist() == ["China", "Brazil"]


def test_validate_results_passes():
    df = _make_matches()
    df["city"] = ["Doha"]
    df["country"] = ["Qatar"]
    validate_results(df)  # should not raise


def test_validate_results_negative_score():
    df = _make_matches(home_score=[-1], away_score=[0])
    with pytest.raises(ValueError, match="Negative scores"):
        validate_results(df)
