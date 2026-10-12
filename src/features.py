"""Causal Elo and recent form with all same-date outcomes withheld."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence

import pandas as pd

from src.elo import INITIAL_RATING, compute_elo, tournament_k

FORM_WINDOWS = (5, 10)
FORM_NAMES = tuple(f"{name}_{window}" for window in FORM_WINDOWS
                   for name in ("ppg", "gf_pg", "ga_pg", "n"))
ELO_FEATURES = ("elo_diff", "home_advantage")
ALL_FEATURES = ("home_elo", "away_elo", *ELO_FEATURES, "k_tier", *(
    f"{side}_{name}" for side in ("home", "away") for name in FORM_NAMES
))
Result = tuple[float, float, float]


def recent_form(results: Sequence[Result]) -> dict[str, float]:
    """Use available games up to each window; no games means a fixed unit prior.

    The cold-start prior is one point and one goal for/against per game. Counts
    expose the support of each estimate to the full models.
    """
    features: dict[str, float] = {}
    for window in FORM_WINDOWS:
        recent = list(results)[-window:]
        size = len(recent)
        for index, name in enumerate(("ppg", "gf_pg", "ga_pg")):
            features[f"{name}_{window}"] = (
                sum(game[index] for game in recent) / size if size else 1.0
            )
        features[f"n_{window}"] = float(size)
    return features


def match_features(
    home: Mapping[str, float], away: Mapping[str, float], neutral: bool, k: int,
) -> dict[str, float]:
    """Share the identical feature schema between historical and live inference."""
    values = {
        "home_elo": float(home["elo"]), "away_elo": float(away["elo"]),
        "elo_diff": float(home["elo"] - away["elo"]),
        "home_advantage": float(not neutral), "k_tier": float(k),
    }
    for side, state in (("home", home), ("away", away)):
        values.update({f"{side}_{name}": float(state[name]) for name in FORM_NAMES})
    return values


def build_features(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return historical pre-match rows, current team states, and Elo history.

    Elo and form advance after an entire date's features have been computed.
    Full history warms up both; a later training-year filter is separate.
    """
    before, elo_history, ratings = compute_elo(matches)
    elo = before.set_index("match_id").to_dict("index")
    form: dict[str, deque[Result]] = defaultdict(lambda: deque(maxlen=max(FORM_WINDOWS)))
    counts: dict[str, int] = defaultdict(int)
    last_dates: dict[str, pd.Timestamp] = {}
    rows: list[dict[str, object]] = []
    for date, day in matches.groupby("date", sort=True):
        teams = set(day.home_team) | set(day.away_team)
        prior = {team: recent_form(list(form[team])) for team in teams}
        for match in day.itertuples(index=False):
            row = elo[match.match_id]
            values = match_features(
                {"elo": row["home_elo"], **prior[match.home_team]},
                {"elo": row["away_elo"], **prior[match.away_team]},
                match.neutral, tournament_k(match.tournament),
            )
            rows.append({
                "match_id": match.match_id, "date": date, "home_team": match.home_team,
                "away_team": match.away_team, "tournament": match.tournament,
                "neutral": match.neutral, "outcome": match.outcome, **values,
            })
        for match in day.itertuples(index=False):
            home_points = 3.0 if match.outcome == "win" else 1.0 if match.outcome == "draw" else 0.0
            away_points = 3.0 if match.outcome == "loss" else 1.0 if match.outcome == "draw" else 0.0
            form[match.home_team].append((home_points, match.home_score, match.away_score))
            form[match.away_team].append((away_points, match.away_score, match.home_score))
            for team in (match.home_team, match.away_team):
                counts[team] += 1
                last_dates[team] = date
    states = pd.DataFrame([
        {"team": team, "elo": ratings.get(team, INITIAL_RATING),
         "last_match_date": last_dates[team], "matches_played": counts[team],
         **recent_form(list(form[team]))}
        for team in sorted(form)
    ])
    return pd.DataFrame(rows), states, elo_history
