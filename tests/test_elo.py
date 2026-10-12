"""Hand-worked Elo guardrails independent of production arithmetic."""

import pytest

from src.elo import expected_result, goal_multiplier, tournament_k, update_ratings


def test_equal_neutral_world_cup_two_goal_win() -> None:
    """1500 vs 1500: We=.5; delta=60*1.5*(1-.5)=45 -> 1545,1455."""
    assert update_ratings(1500, 1500, 2, 0, True, 60) == (1545, 1455)


def test_non_neutral_home_bonus_and_draw() -> None:
    """100-point home bonus gives We=.6400649998 and a -2.8012999961 draw delta."""
    assert expected_result(1500, 1500, False) == pytest.approx(0.6400649998028851)
    home, away = update_ratings(1500, 1500, 1, 1, False, 20)
    assert home == pytest.approx(1497.1987000039423)
    assert away == pytest.approx(1502.8012999960577)


@pytest.mark.parametrize("margin,expected", [(0, 1), (1, 1), (2, 1.5), (3, 1.75), (-5, 2)])
def test_goal_multiplier(margin: int, expected: float) -> None:
    assert goal_multiplier(margin) == expected


def test_unknown_tournament_requires_review() -> None:
    with pytest.raises(ValueError, match="Unclassified"):
        tournament_k("Unreviewed Cup")
