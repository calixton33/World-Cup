"""Validated, date-scoped loading of the martj42 international results."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "date", "home_team", "away_team", "home_score", "away_score",
    "tournament", "city", "country", "neutral",
}


def file_sha256(path: Path) -> str:
    """Fingerprint a local source file for reproducible reports."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalise_names(matches: pd.DataFrame, names_path: Path | None) -> tuple[pd.DataFrame, int]:
    """Apply source aliases only within their inclusive historical date windows."""
    result = matches.copy()
    if names_path is None or not names_path.exists():
        return result, 0
    names = pd.read_csv(names_path, parse_dates=["start_date", "end_date"])
    if not {"current", "former", "start_date", "end_date"}.issubset(names.columns):
        raise ValueError("former_names.csv requires current, former, start_date, end_date")
    renamed = 0
    for alias in names.itertuples(index=False):
        in_period = result["date"].between(alias.start_date, alias.end_date)
        for column in ("home_team", "away_team"):
            mask = in_period & result[column].eq(alias.former)
            renamed += int(mask.sum())
            result.loc[mask, column] = alias.current
    return result, renamed


def prepare_results(
    frame: pd.DataFrame, names_path: Path | None = None, as_of: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate completed matches, remove exact duplicates, and sort by match date."""
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"results.csv is missing columns: {sorted(missing)}")
    data = frame.copy()
    input_rows = len(data)
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    if data["date"].isna().any():
        raise ValueError("Match dates cannot be missing")
    # Missing scores are unplayed fixtures, never target imputation candidates.
    missing_scores = int(data[["home_score", "away_score"]].isna().any(axis=1).sum())
    data = data.dropna(subset=["home_score", "away_score"])
    for column in ("home_score", "away_score"):
        scores = pd.to_numeric(data[column], errors="raise")
        if not (np.isfinite(scores) & scores.ge(0) & scores.eq(np.floor(scores))).all():
            raise ValueError(f"{column} must contain nonnegative integer scores")
        data[column] = scores.astype(int)
    for column in ("home_team", "away_team", "tournament"):
        if data[column].isna().any() or data[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} cannot be blank")
        data[column] = data[column].str.strip()
    neutral = data["neutral"].astype(str).str.lower().map({"true": True, "false": False})
    if neutral.isna().any():
        raise ValueError("neutral must be TRUE or FALSE")
    data["neutral"] = neutral.astype(bool)
    future_rows = 0
    if as_of is not None:
        future_rows = int(data["date"].gt(pd.Timestamp(as_of)).sum())
        data = data.loc[data["date"].le(pd.Timestamp(as_of))].copy()
    duplicates = int(data.duplicated().sum())
    data = data.drop_duplicates()
    data, renamed = normalise_names(data, names_path)
    if data["home_team"].eq(data["away_team"]).any():
        raise ValueError("A team cannot play itself after name normalisation")
    data = data.sort_values(["date", "home_team", "away_team", "tournament"], kind="stable")
    data = data.reset_index(drop=True)
    if data.empty:
        raise ValueError("No completed matches available")
    data["match_id"] = data.index
    data["outcome"] = np.select(
        [data.home_score.gt(data.away_score), data.home_score.lt(data.away_score)],
        ["win", "loss"], default="draw",
    )
    audit = {
        "input_rows": input_rows, "completed_matches": len(data),
        "missing_scores_removed": missing_scores, "future_rows_removed": future_rows,
        "exact_duplicates_removed": duplicates, "team_name_cells_normalised": renamed,
        "former_names_available": bool(names_path is not None and names_path.exists()),
        "first_match": data.date.min().date().isoformat(),
        "last_match": data.date.max().date().isoformat(),
        "teams": int(pd.concat([data.home_team, data.away_team]).nunique()),
        "as_of": as_of,
    }
    return data, audit


def load_results(raw_dir: Path, as_of: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load results and optional aliases, returning the data-quality audit."""
    path = raw_dir / "results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Place results.csv in {raw_dir} or run scripts/download_data.py")
    matches, audit = prepare_results(pd.read_csv(path), raw_dir / "former_names.csv", as_of)
    audit["source_sha256"] = {
        name: file_sha256(raw_dir / name)
        for name in ("results.csv", "former_names.csv", "shootouts.csv")
        if (raw_dir / name).exists()
    }
    return matches, audit
