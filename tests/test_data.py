"""Input validation and source alias handling."""

from pathlib import Path

import pandas as pd
import pytest

from src.data import prepare_results


def sample_results() -> pd.DataFrame:
    """Small real-schema fixture; tied scores represent a draw even on penalties."""
    return pd.DataFrame({
        "date": ["2006-07-09", "2006-07-10"],
        "home_team": ["Italy", "France"], "away_team": ["France", "Italy"],
        "home_score": [1, 0], "away_score": [1, 1], "tournament": ["FIFA World Cup"] * 2,
        "city": ["Berlin"] * 2, "country": ["Germany"] * 2, "neutral": [True, "FALSE"],
    })


def test_draw_label_excludes_penalty_winner() -> None:
    matches, audit = prepare_results(sample_results())
    assert matches.outcome.tolist() == ["draw", "loss"]
    assert matches.neutral.tolist() == [True, False]
    assert audit["completed_matches"] == 2


def test_aliases_respect_dates(tmp_path: Path) -> None:
    path = tmp_path / "former_names.csv"
    path.write_text("current,former,start_date,end_date\nItaly,Old Italy,2006-07-09,2006-07-09\n")
    frame = sample_results()
    frame.loc[0, "home_team"] = "Old Italy"
    frame.loc[1, "away_team"] = "Old Italy"
    matches, audit = prepare_results(frame, path)
    assert matches.loc[0, "home_team"] == "Italy"
    assert matches.loc[1, "away_team"] == "Old Italy"
    assert audit["team_name_cells_normalised"] == 1


def test_invalid_neutral_fails() -> None:
    frame = sample_results()
    frame.loc[1, "neutral"] = "maybe"
    with pytest.raises(ValueError, match="neutral"):
        prepare_results(frame)


def test_unplayed_future_and_duplicate_rows_are_audited() -> None:
    frame = sample_results()
    frame.loc[1, "home_score"] = float("nan")
    matches, audit = prepare_results(pd.concat([frame, frame.iloc[[0]]]), as_of="2006-07-09")
    assert len(matches) == 1
    assert audit["missing_scores_removed"] == 1
    assert audit["exact_duplicates_removed"] == 1
