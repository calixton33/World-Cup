"""World Football Elo style equations with explicit tournament importance."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import pandas as pd

INITIAL_RATING = 1500.0
HOME_ADVANTAGE = 100.0

# Sole tournament-to-K config. Historical continental predecessors and
# Confederations Cup / Finalissima / Pan American Championship use 50.
# Nations Leagues, subregional qualifiers, bilateral cups, and all other
# named invitation or non-FIFA tournaments use 30 by judgement.
TOURNAMENT_K: dict[str, int] = {
    'ABCS Tournament': 30,
    'AFC Asian Cup': 50,
    'AFC Asian Cup qualification': 40,
    'AFC Challenge Cup': 30,
    'AFC Challenge Cup qualification': 30,
    'AFC Solidarity Cup': 30,
    'AFF Championship': 30,
    'AFF Championship qualification': 30,
    'ASEAN Championship': 30,
    'ASEAN Championship qualification': 30,
    'Afghanistan Republic Day Festival Cup': 30,
    'African Cup of Nations': 50,
    'African Cup of Nations qualification': 40,
    'African Friendship Games': 30,
    'Afro-Asian Games': 30,
    'Al Ain International Cup': 30,
    'All-African Games': 30,
    'Amílcar Cabral Cup': 30,
    'Arab Cup': 30,
    'Arab Cup qualification': 30,
    'Asian Games': 30,
    'Atlantic Cup': 30,
    'Atlantic Heritage Cup': 30,
    'Australia Bicentenary Gold Cup': 30,
    'Balkan Cup': 30,
    'Baltic Cup': 30,
    'Bangabandhu Cup': 30,
    "Bangladesh President's Gold Cup": 30,
    'Beijing International Friendship Tournament': 30,
    'Benedikt Fontana Cup': 30,
    'Bolivarian Games': 30,
    'Brazil Independence Cup': 30,
    'British Home Championship': 30,
    'CAFA Nations Cup': 30,
    'CCCF Championship': 50,
    'CECAFA Cup': 30,
    'CFU Caribbean Cup': 30,
    'CFU Caribbean Cup qualification': 30,
    'CONCACAF Championship': 50,
    'CONCACAF Championship qualification': 40,
    'CONCACAF Nations League': 30,
    'CONCACAF Nations League qualification': 30,
    'CONCACAF Series': 30,
    'CONIFA Africa Football Cup': 30,
    'CONIFA Asia Cup': 30,
    'CONIFA European Football Cup': 30,
    'CONIFA South America Football Cup': 30,
    'CONIFA World Cup qualification': 30,
    'CONIFA World Football Cup': 30,
    'CONIFA World Football Cup qualification': 30,
    'CONMEBOL–UEFA Cup of Champions': 50,
    'COSAFA Cup': 30,
    'COSAFA Cup qualification': 30,
    'CTFA International Tournament': 30,
    'Canadian Shield': 30,
    'Central American and Caribbean Games': 30,
    'Central European International Cup': 30,
    'China Cup': 30,
    'Columbus Cup': 30,
    'ConIFA Challenger Cup': 30,
    'Confederations Cup': 50,
    'Copa América': 50,
    'Copa América qualification': 40,
    'Copa Artigas': 30,
    "Copa Bernardo O'Higgins": 30,
    'Copa Carlos Dittborn': 30,
    'Copa Chevallier Boutell': 30,
    'Copa Confraternidad': 30,
    'Copa Félix Bogado': 30,
    'Copa Juan Pinto Durán': 30,
    'Copa Lipton': 30,
    'Copa Newton': 30,
    'Copa Oswaldo Cruz': 30,
    'Copa Paz del Chaco': 30,
    'Copa Premio Honor Argentino': 30,
    'Copa Premio Honor Uruguayo': 30,
    'Copa Ramón Castilla': 30,
    'Copa Rio Branco': 30,
    'Copa Roca': 30,
    'Copa del Pacífico': 30,
    'Corsica Cup': 30,
    "Coupe de l'Outre-Mer": 30,
    'Cup of Ancient Civilizations': 30,
    'Cyprus International Tournament': 30,
    'Dakar Tournament': 30,
    'Diamond Jubilee International Football Tournament': 30,
    'Dragon Cup': 30,
    'Dunhill Cup': 30,
    'Dynasty Cup': 30,
    'EAFF Championship': 30,
    'EAFF Championship qualification': 30,
    'ELF Cup': 30,
    'East Asian Games': 30,
    'FAS Tri-Nations Series': 30,
    'FIFA 75th Anniversary Cup': 30,
    'FIFA ASEAN Challenge Cup': 30,
    'FIFA ASEAN Cup': 30,
    'FIFA Confederations Cup': 50,
    'FIFA Series': 30,
    'FIFA World Cup': 60,
    'FIFA World Cup qualification': 40,
    'FIFI Wild Cup': 30,
    'Far Eastern Championship Games': 30,
    'Four Nations Tournament': 30,
    "Four Nations' Cup": 30,
    'Friendly': 20,
    'Fuchs International Tournament': 30,
    'GaNEFo': 30,
    'Gold Cup': 50,
    'Gold Cup qualification': 40,
    'Great Wall Cup': 30,
    'Guangzhou International Friendship Tournament': 30,
    'Gulf Cup': 30,
    'Hungary Heritage Cup': 30,
    'Indian Ocean Island Games': 30,
    'Indonesian Independence Cup': 30,
    'Inter Games': 30,
    'Inter-Allied Games': 30,
    'Intercontinental Cup': 30,
    'International Friendship Championship': 30,
    'International Tournament of Peoples, Cultures and Tribes': 30,
    'Iran International Tournament': 30,
    'Island Games': 30,
    'Jakarta Anniversary Tournament': 30,
    'Joe Robbie Cup': 30,
    'Jordan International Tournament': 30,
    'KTFF 50th Anniversary Cup': 30,
    'King Hassan II Tournament': 30,
    "King's Cup": 30,
    'Kirin Challenge Cup': 30,
    'Kirin Cup': 30,
    'Korea Cup': 30,
    'Kuneitra Cup': 30,
    'LG Cup': 30,
    'Los Angeles Nations Cup': 30,
    'Lunar New Year Cup': 30,
    "MSG Prime Minister's Cup": 30,
    'Mahinda Rajapaksa Cup': 30,
    'Malta International Tournament': 30,
    'Mapinduzi Cup': 30,
    'Maple Cup': 30,
    'Marianas Cup': 30,
    'Marlboro Cup': 30,
    'Matthews Cup': 30,
    'Mauritius Four Nations Cup': 30,
    'Mediterranean Cup': 30,
    'Melanesia Cup': 30,
    'Merdeka Tournament': 30,
    'Merlion Cup': 30,
    'Miami Cup': 30,
    'Millennium Cup': 30,
    'Morocco, Capital of African Football': 30,
    'Mukuru 4 Nations': 30,
    'Mundialito': 30,
    'Muratti Vase': 30,
    'NAFC Championship': 50,
    'NAFU Championship': 30,
    'Nations Cup': 30,
    'Navruz Cup': 30,
    'Nehru Cup': 30,
    'Niamh Challenge Cup': 30,
    'Nile Basin Tournament': 30,
    'Nordic Championship': 30,
    'OSN Cup': 30,
    'Oceania Nations Cup': 50,
    'Oceania Nations Cup qualification': 40,
    'Olympic Games': 30,
    'Open International Championship': 30,
    'Outrigger Challenge Cup': 30,
    'Pacific Games': 30,
    'Pacific Mini Games': 30,
    'Palestine Cup': 30,
    'Palestine International Championship': 30,
    'Pan American Championship': 50,
    'Peace Cup': 30,
    'Philippine Peace Cup': 30,
    'Phillip Seaga Cup': 30,
    "President's Gold Cup International Tournament": 30,
    "Prime Minister's Cup": 30,
    'Quaid-e-Azam International Tournament': 30,
    'Real Madrid 75th Anniversary Cup': 30,
    'Rous Cup': 30,
    'SAFF Cup': 30,
    'SKN Football Festival': 30,
    'Scania 100 Tournament': 30,
    'Simba Tournament': 30,
    'Soccer Ashes': 30,
    'South Asian Games': 30,
    'South Asian Super Cup': 30,
    'South Pacific Games': 30,
    'South Pacific Mini Games': 30,
    'Southeast Asian Games': 30,
    'Southeast Asian Peninsular Games': 30,
    'Superclásico de las Américas': 30,
    'TIFOCO Tournament': 30,
    'Taça das Nações': 30,
    'The Other Final': 30,
    'Three Nations Cup': 30,
    'Tournament Burkina Faso': 30,
    'Tournoi de France': 30,
    'Trans-Tasman Cup': 30,
    'Tri Nation Tournament': 30,
    'Tri-Nation Series': 30,
    'Tri-Nations Cup': 30,
    'Tri-Nations Series': 30,
    'Tynwald Hill Tournament': 30,
    'U.S.A. Bicentennial Cup Tournament': 30,
    'UDEAC Cup': 30,
    'UEFA Euro': 50,
    'UEFA Euro qualification': 40,
    'UEFA Nations League': 30,
    'UNCAF Cup': 30,
    'UNIFFAC Cup': 30,
    'USA Cup': 30,
    'Umbro Cup': 30,
    'United Arab Emirates Friendship Tournament': 30,
    'Unity Cup': 30,
    'VFF Cup': 30,
    'Vietnam Independence Cup': 30,
    'Viva World Cup': 30,
    'WAFF Championship': 30,
    'West African Cup': 30,
    'Windward Islands Tournament': 30,
    'World Unity Cup': 30,
    'Zambian Independence Tournament': 30,
    'Évence Coppée Trophy': 30,
}

CORE_TOURNAMENTS = frozenset({
    "FIFA World Cup", "UEFA Euro", "Copa América", "African Cup of Nations", "AFC Asian Cup",
    "CONCACAF Championship", "Gold Cup", "Oceania Nations Cup", "Friendly",
    "FIFA World Cup qualification", "UEFA Euro qualification", "Copa América qualification",
    "African Cup of Nations qualification", "AFC Asian Cup qualification",
    "CONCACAF Championship qualification", "Gold Cup qualification", "Oceania Nations Cup qualification",
})


def classification_audit(tournaments: Iterable[str]) -> list[dict[str, Any]]:
    """List every observed name and explicitly flag judgement-based importance."""
    rows: list[dict[str, Any]] = []
    for name in sorted(tournaments):
        k = tournament_k(name)
        judged = name not in CORE_TOURNAMENTS
        reason = "Named principal finals, qualifiers, or friendlies"
        if judged:
            reason = ("Historic continental predecessor or major intercontinental finals"
                      if k == 50 else "Other named competition; regional qualifiers and Nations Leagues also use this tier")
        rows.append({"tournament": name, "k": k, "judgement": judged, "reason": reason})
    return rows


def tournament_k(tournament: str) -> int:
    """Return a reviewed K tier; require a conscious decision for unknown names."""
    if tournament not in TOURNAMENT_K:
        raise ValueError(f"Unclassified tournament {tournament!r}; add it to TOURNAMENT_K")
    return TOURNAMENT_K[tournament]


def expected_result(home_rating: float, away_rating: float, neutral: bool) -> float:
    """Expected first-listed result, with a home bonus only at nonneutral venues."""
    difference = home_rating - away_rating + (0.0 if neutral else HOME_ADVANTAGE)
    return float(1.0 / (10.0 ** (-difference / 400.0) + 1.0))


def goal_multiplier(margin: int) -> float:
    """Scale a result by the absolute score margin, including extra time."""
    margin = abs(margin)
    if margin <= 1:
        return 1.0
    return 1.5 if margin == 2 else (11.0 + margin) / 8.0


def update_ratings(
    home_rating: float, away_rating: float, home_score: int, away_score: int,
    neutral: bool, k: int,
) -> tuple[float, float]:
    """Apply equal and opposite Elo changes after the match has finished."""
    result = 1.0 if home_score > away_score else 0.0 if home_score < away_score else 0.5
    change = k * goal_multiplier(home_score - away_score) * (
        result - expected_result(home_rating, away_rating, neutral)
    )
    return home_rating + change, away_rating - change


def compute_elo(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Freeze ratings per date, then sum that day's post-match updates.

    Kickoff order is unavailable. All games on a date therefore use the prior
    day's state, even if a team plays more than once that day.
    """
    ratings: dict[str, float] = {}
    before: list[dict[str, float | int]] = []
    history: list[dict[str, object]] = []
    for date, day in matches.groupby("date", sort=True):
        changes: dict[str, float] = defaultdict(float)
        for match in day.itertuples(index=False):
            home = ratings.get(match.home_team, INITIAL_RATING)
            away = ratings.get(match.away_team, INITIAL_RATING)
            before.append({"match_id": match.match_id, "home_elo": home, "away_elo": away})
            new_home, new_away = update_ratings(
                home, away, match.home_score, match.away_score,
                match.neutral, tournament_k(match.tournament),
            )
            changes[match.home_team] += new_home - home
            changes[match.away_team] += new_away - away
        for team, change in sorted(changes.items()):
            ratings[team] = ratings.get(team, INITIAL_RATING) + change
            history.append({"date": date, "team": team, "elo": ratings[team]})
    return pd.DataFrame(before), pd.DataFrame(history), ratings
