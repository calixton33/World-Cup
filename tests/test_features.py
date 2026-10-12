"""Leakage guardrails: future, own-match, and same-date results are unavailable."""

import pandas as pd
import pytest

from src.data import prepare_results
from src.features import ALL_FEATURES, build_features, match_features, recent_form


def history() -> pd.DataFrame:
    """Include repeated same-day appearances to expose accidental sequential leaks."""
    records = [
        ("2000-01-01", "A", "B", 2, 0),
        ("2000-01-02", "A", "C", 0, 1),
        ("2000-01-02", "B", "A", 5, 0),
        ("2000-01-03", "C", "B", 1, 1),
        ("2001-01-01", "A", "B", 0, 8),
    ]
    frame = pd.DataFrame(records, columns=["date", "home_team", "away_team", "home_score", "away_score"])
    frame["tournament"] = "Friendly"
    frame["city"] = "Test"
    frame["country"] = "Test"
    frame["neutral"] = True
    return prepare_results(frame)[0]


def test_features_identical_with_or_without_future_rows() -> None:
    matches = history()
    cutoff = pd.Timestamp("2000-01-02")
    full, _, _ = build_features(matches)
    prefix, _, _ = build_features(matches.loc[matches.date.le(cutoff)])
    pd.testing.assert_frame_equal(
        full.loc[full.date.le(cutoff), list(ALL_FEATURES)].reset_index(drop=True),
        prefix[list(ALL_FEATURES)].reset_index(drop=True),
    )


def test_own_and_other_same_day_results_do_not_affect_features() -> None:
    matches = history()
    original, _, _ = build_features(matches)
    altered = matches.copy()
    altered.loc[altered.date.eq("2000-01-02"), ["home_score", "away_score"]] = [9, 0]
    altered.loc[altered.date.eq("2000-01-02"), "outcome"] = "win"
    changed, _, _ = build_features(altered)
    mask = original.date.le("2000-01-02")
    pd.testing.assert_frame_equal(original.loc[mask, list(ALL_FEATURES)], changed.loc[mask, list(ALL_FEATURES)])
    assert not original.loc[~mask, list(ALL_FEATURES)].equals(changed.loc[~mask, list(ALL_FEATURES)])


def test_form_uses_only_previous_matches_and_available_count() -> None:
    features, _, _ = build_features(history())
    row = features.loc[features.home_team.eq("A") & features.date.eq("2000-01-02")].iloc[0]
    assert row.home_elo == pytest.approx(1515)  # K20 * G1.5 * .5 = 15 after yesterday.
    assert row.home_ppg_5 == 3
    assert row.home_gf_pg_5 == 2
    assert row.home_ga_pg_5 == 0
    assert row.home_n_10 == 1
    assert features.iloc[0].home_n_5 == 0
    assert features.iloc[0].home_ppg_5 == 1


def test_form_window_bounds() -> None:
    form = recent_form([(3, 2, 0)] * 5 + [(0, 0, 2)] * 5)
    assert form["ppg_5"] == 0
    assert form["ppg_10"] == 1.5
    assert form["n_5"] == 5
    assert form["n_10"] == 10


def test_live_features_equal_next_historical_match() -> None:
    matches = history()
    prior = matches.loc[matches.date.lt("2001-01-01")]
    _, states, _ = build_features(prior)
    state = states.set_index("team")
    live = match_features(state.loc["A"], state.loc["B"], True, 20)
    full, _, _ = build_features(matches)
    expected = full.loc[full.date.eq("2001-01-01"), list(ALL_FEATURES)].iloc[0].to_dict()
    assert live == expected
