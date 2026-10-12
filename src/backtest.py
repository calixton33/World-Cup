"""World Cup rolling-origin evaluation with saved out-of-sample probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.models import CLASSES, MODEL_NAMES, fit_model

FIRST_TEST_YEAR = 2006
EXPECTED_MATCHES = {2006: 64, 2010: 64, 2014: 64, 2018: 64, 2022: 64, 2026: 104}


@dataclass
class WorldCupSplit:
    """Date-auditable expanding training window and one final tournament."""

    year: int
    cutoff: pd.Timestamp
    training: pd.DataFrame
    test: pd.DataFrame


def world_cup_splits(features: pd.DataFrame, train_start_year: int = 1990) -> list[WorldCupSplit]:
    """Find each men's final tournament in the available completed match history."""
    cups = features.loc[features.tournament.eq("FIFA World Cup")
                        & features.date.dt.year.ge(FIRST_TEST_YEAR)]
    splits: list[WorldCupSplit] = []
    for year, test in cups.groupby(cups.date.dt.year, sort=True):
        cutoff = test.date.min()
        training = features.loc[features.date.dt.year.ge(train_start_year) & features.date.lt(cutoff)]
        if training.empty:
            raise ValueError(f"No training matches before World Cup {year}")
        if not training.date.max() < test.date.min():
            raise ValueError("Temporal split overlaps")
        splits.append(WorldCupSplit(int(year), cutoff, training, test))
    if not splits:
        raise ValueError("No completed FIFA World Cup matches from the first test year onward")
    return splits


def outcome_metrics(labels: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    """Natural-log loss, unscaled multiclass Brier (sum over classes), and accuracy."""
    if probabilities.shape != (len(labels), len(CLASSES)):
        raise ValueError("Probability shape does not match labels and classes")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Invalid probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("Probabilities must sum to one")
    target_indices = np.asarray([CLASSES.index(label) for label in labels])
    targets = np.eye(len(CLASSES))[target_indices]
    observed_probability = probabilities[np.arange(len(labels)), target_indices]
    return {
        "log_loss": float(-np.log(np.clip(observed_probability, 1e-15, 1.0)).mean()),
        "brier_score": float(np.square(probabilities - targets).sum(axis=1).mean()),
        "accuracy": float((probabilities.argmax(axis=1) == target_indices).mean()),
    }


def higher_elo_accuracy(test: pd.DataFrame) -> float:
    """Pick the higher raw pre-match rating; an exactly equal rating picks a draw."""
    picks = np.where(test.elo_diff.gt(0), "win", np.where(test.elo_diff.lt(0), "loss", "draw"))
    return float((picks == test.outcome).mean())


def run_backtest(
    features: pd.DataFrame, train_start_year: int = 1990, calibration_years: int = 4,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Compare fixed candidates at each origin and rank equal-weight tournament means."""
    rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    origins: list[dict[str, Any]] = []
    for split in world_cup_splits(features, train_start_year):
        origins.append({
            "year": split.year, "cutoff": split.cutoff.date().isoformat(),
            "train_matches": len(split.training),
            "max_train_date": split.training.date.max().date().isoformat(),
            "min_test_date": split.test.date.min().date().isoformat(),
            "max_test_date": split.test.date.max().date().isoformat(),
            "test_matches": len(split.test), "expected_matches": EXPECTED_MATCHES.get(split.year),
            "complete": len(split.test) == EXPECTED_MATCHES.get(split.year),
        })
        for model_id in MODEL_NAMES:
            fitted = fit_model(model_id, split.training, split.cutoff, calibration_years)
            probabilities = fitted.predict_proba(split.test)
            scores = outcome_metrics(split.test.outcome, probabilities)
            rows.append({"year": split.year, "model": model_id, "name": MODEL_NAMES[model_id],
                         "test_matches": len(split.test), **scores, **fitted.training_info})
            output = split.test[["match_id", "date", "home_team", "away_team", "outcome", "elo_diff"]].copy()
            output["year"], output["model"] = split.year, model_id
            output["prediction"] = np.asarray(CLASSES)[probabilities.argmax(axis=1)]
            for index, label in enumerate(CLASSES):
                output[f"p_{label}"] = probabilities[:, index]
            predictions.append(output)
            print(f"{split.year} {model_id}: log loss {scores['log_loss']:.4f}", flush=True)
        rows.append({"year": split.year, "model": "Elo pick", "name": "Pick higher Elo",
                     "test_matches": len(split.test), "log_loss": None, "brier_score": None,
                     "accuracy": higher_elo_accuracy(split.test)})
    per_tournament = pd.DataFrame(rows)
    mean_rows: list[dict[str, Any]] = []
    for model_id, group in per_tournament.groupby("model", sort=False):
        means = {name: None if group[name].isna().all() else float(group[name].mean())
                 for name in ("log_loss", "brier_score", "accuracy")}
        mean_rows.append({"model": model_id, "name": group["name"].iloc[0], **means})
    ranked = sorted((row for row in mean_rows if row["log_loss"] is not None), key=lambda row: row["log_loss"])
    winner, runner_up = ranked[:2]
    pooled = pd.concat(predictions, ignore_index=True)
    diagnostics: dict[str, Any] = {}
    for model_id, group in pooled.groupby("model"):
        draws = group.outcome.eq("draw")
        diagnostics[model_id] = {
            "actual_draws": int(draws.sum()), "predicted_draws": int(group.prediction.eq("draw").sum()),
            "draw_recall": float(group.loc[draws, "prediction"].eq("draw").mean()),
            "pooled_metrics": outcome_metrics(group.outcome, group[[f"p_{label}" for label in CLASSES]].to_numpy()),
        }
    fold_winners = []
    for year, group in per_tournament.loc[per_tournament.model.isin(MODEL_NAMES)].groupby("year"):
        best = group.sort_values("log_loss", kind="stable").iloc[0]
        fold_winners.append({"year": int(year), "model": best.model, "log_loss": float(best.log_loss)})
    result = {
        "per_tournament": per_tournament.replace({np.nan: None}).to_dict("records"),
        "mean_metrics": mean_rows, "splits": origins,
        "selection": {
            "model": winner["model"], "name": winner["name"], "mean_log_loss": winner["log_loss"],
            "runner_up": runner_up["model"], "margin_log_loss": runner_up["log_loss"] - winner["log_loss"],
            "rule": "Lowest unweighted mean tournament log loss; candidate order breaks exact ties",
            "baseline_won": winner["model"] in ("B0", "B1"),
        },
        "test_tournaments": len(origins), "test_matches": sum(origin["test_matches"] for origin in origins),
        "diagnostics": diagnostics, "fold_winners": fold_winners,
    }
    return result, per_tournament, pooled


def reliability_table(predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Pool only tournament-test predictions; include bin counts and Wilson intervals."""
    rows: list[dict[str, Any]] = []
    for model, group in predictions.groupby("model"):
        for label in CLASSES:
            probabilities = group[f"p_{label}"].to_numpy()
            observed = group.outcome.eq(label).to_numpy().astype(float)
            assignments = np.minimum((probabilities * bins).astype(int), bins - 1)
            for index in range(bins):
                mask = assignments == index
                count = int(mask.sum())
                if not count:
                    continue
                frequency = float(observed[mask].mean())
                z = 1.96
                denominator = 1 + z * z / count
                centre = (frequency + z * z / (2 * count)) / denominator
                radius = z * np.sqrt(frequency * (1 - frequency) / count + z * z / (4 * count**2)) / denominator
                rows.append({"model": model, "outcome": label, "bin": index, "count": count,
                             "predicted": float(probabilities[mask].mean()), "observed": frequency,
                             "lower": float(max(0, centre - radius)), "upper": float(min(1, centre + radius))})
    return pd.DataFrame(rows)
