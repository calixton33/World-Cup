"""Validate the hosted app bundle without access to raw data or any training."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference import load_assets, predict_match


def main() -> None:
    """Load artifacts, make a prediction, and fail deployment on inconsistency."""
    assets = load_assets(ROOT)
    teams = list(assets.ratings.index)
    probabilities = predict_match(assets, teams[0], teams[1], True)
    assert abs(sum(probabilities.values()) - 1) < 1e-9
    print(f"Deployment preflight passed: {assets.model.model_id}, {len(teams)} teams, no raw data required.")


if __name__ == "__main__":
    main()
