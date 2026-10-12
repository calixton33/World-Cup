"""Check that measured reports, predictions, selection, and README stay consistent."""

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.write_readme import render_readme
from src.backtest import outcome_metrics
from src.models import CLASSES

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "reports/metrics.json").exists(), reason="Run a real backtest first")


def test_report_scores_recompute_from_saved_test_predictions() -> None:
    metrics = json.loads((ROOT / "reports/metrics.json").read_text())
    predictions = pd.read_csv(ROOT / "reports/predictions.csv")
    for row in metrics["per_tournament"]:
        if row["model"] == "Elo pick":
            continue
        group = predictions.loc[predictions.model.eq(row["model"]) & predictions.year.eq(row["year"])]
        scores = outcome_metrics(group.outcome, group[[f"p_{label}" for label in CLASSES]].to_numpy())
        assert len(group) == row["test_matches"]
        for name, value in scores.items():
            assert row[name] == pytest.approx(value)
    means = {row["model"]: row["log_loss"] for row in metrics["mean_metrics"] if row["log_loss"] is not None}
    assert metrics["selection"]["model"] == min(means, key=means.get)


def test_readme_is_generated_only_from_the_actual_metrics() -> None:
    metrics = json.loads((ROOT / "reports/metrics.json").read_text())
    assert (ROOT / "README.md").read_text() == render_readme(metrics)


def test_metrics_csv_includes_tournament_scores_and_means() -> None:
    metrics = json.loads((ROOT / "reports/metrics.json").read_text())
    table = pd.read_csv(ROOT / "reports/metrics.csv")
    assert len(table.loc[table.aggregation.eq("tournament")]) == len(metrics["per_tournament"])
    averages = table.loc[table.aggregation.eq("mean")].set_index("model")
    for row in metrics["mean_metrics"]:
        assert averages.loc[row["model"], "accuracy"] == pytest.approx(row["accuracy"])
