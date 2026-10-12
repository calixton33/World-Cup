"""Read-only loading and inference for the app and deployment checks."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import sklearn

from src.data import file_sha256
from src.features import match_features
from src.models import CLASSES, FittedModel

REQUIRED_ARTIFACTS = (
    "models/final_model.pkl", "models/metadata.json", "models/ratings.csv", "models/elo_history.csv",
    "reports/metrics.json", "reports/metrics.csv", "reports/calibration.vl.json",
    "reports/model_comparison.vl.json",
)


@dataclass
class PredictionAssets:
    """Precomputed bundle; loading never builds features or fits a model."""

    model: FittedModel
    ratings: pd.DataFrame
    elo_history: pd.DataFrame
    metrics: dict[str, Any]
    metadata: dict[str, Any]


def load_assets(root: Path) -> PredictionAssets:
    """Check bundle consistency before loading the project's saved estimator."""
    missing = [name for name in REQUIRED_ARTIFACTS if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing artifacts: {', '.join(missing)}. Run scripts/run_all.py.")
    metadata = json.loads((root / "models/metadata.json").read_text())
    metrics = json.loads((root / "reports/metrics.json").read_text())
    if metadata["environment"]["scikit-learn"] != sklearn.__version__:
        raise ValueError("The saved model requires the scikit-learn version pinned in requirements.txt.")
    if metadata["source_sha256"] != metrics["data"]["source_sha256"] or metadata["model"] != metrics["selection"]["model"]:
        raise ValueError("Model and reports disagree. Run the full pipeline to rebuild a consistent bundle.")
    for name, expected in metadata["artifact_sha256"].items():
        if file_sha256(root / "models" / name) != expected:
            raise ValueError(f"Stale {name}. Run the full pipeline to rebuild a consistent bundle.")
    model: FittedModel = pickle.loads((root / "models/final_model.pkl").read_bytes())
    ratings = pd.read_csv(root / "models/ratings.csv", parse_dates=["last_match_date"]).set_index("team")
    history = pd.read_csv(root / "models/elo_history.csv", parse_dates=["date"])
    return PredictionAssets(model, ratings, history, metrics, metadata)


def predict_match(
    assets: PredictionAssets, home_team: str, away_team: str, neutral: bool, k: int = 60,
) -> dict[str, float]:
    """Forecast from the latest saved states in the first-listed team's perspective."""
    if home_team == away_team:
        raise ValueError("Choose two different teams")
    if home_team not in assets.ratings.index or away_team not in assets.ratings.index:
        raise ValueError("Both teams need a saved rating and form state")
    if k not in assets.metrics["config"]["k_tiers"]:
        raise ValueError("Unknown match importance")
    features = match_features(assets.ratings.loc[home_team], assets.ratings.loc[away_team], neutral, k)
    probabilities = assets.model.predict_proba(pd.DataFrame([features]))[0]
    return {label: float(probabilities[index]) for index, label in enumerate(CLASSES)}
