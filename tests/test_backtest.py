"""Temporal split, probability ordering, and proper scoring-rule tests."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import outcome_metrics, reliability_table, world_cup_splits
from src.features import ALL_FEATURES
from src.models import CLASSES, calibration_split, fit_model


def training_rows() -> pd.DataFrame:
    """Two chronological periods, each containing all three classes."""
    frame = pd.DataFrame({"date": pd.to_datetime(["1999-01-01"] * 3 + ["2004-01-01"] * 3),
                          "outcome": list(CLASSES) * 2, "tournament": "Friendly"})
    for index, feature in enumerate(ALL_FEATURES):
        frame[feature] = np.arange(len(frame)) + index
    return frame


def test_every_backtest_split_is_strictly_temporal() -> None:
    training = training_rows()
    cups = pd.DataFrame({"date": pd.to_datetime(["2006-06-09", "2006-07-09", "2010-06-11"]),
                         "tournament": "FIFA World Cup", "outcome": "draw"})
    splits = world_cup_splits(pd.concat([training, cups], ignore_index=True))
    assert [split.year for split in splits] == [2006, 2010]
    for split in splits:
        assert split.training.date.max() < split.test.date.min()
        assert split.training.date.min().year >= 1990
    assert pd.Timestamp("2006-07-09") in set(splits[1].training.date)


def test_calibration_never_overlaps_model_fit_or_test() -> None:
    cutoff = pd.Timestamp("2006-06-09")
    fit, calibration = calibration_split(training_rows(), cutoff, 4)
    assert fit.date.max() < calibration.date.min()
    assert calibration.date.max() < cutoff
    contaminated = pd.concat([training_rows(), training_rows().iloc[[0]].assign(date=cutoff)])
    with pytest.raises(ValueError, match="before cutoff"):
        calibration_split(contaminated, cutoff)


def test_metrics_hand_worked_uniform_probabilities() -> None:
    metrics = outcome_metrics(pd.Series(CLASSES), np.full((3, 3), 1 / 3))
    assert metrics["log_loss"] == pytest.approx(np.log(3))
    assert metrics["brier_score"] == pytest.approx(2 / 3)
    assert metrics["accuracy"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("model_id", ["B0", "B1", "M1", "M2"])
def test_candidate_outputs_valid_probabilities(model_id: str) -> None:
    fitted = fit_model(model_id, training_rows(), pd.Timestamp("2006-06-09"))
    probabilities = fitted.predict_proba(training_rows().iloc[:2])
    assert probabilities.shape == (2, 3)
    assert (probabilities >= 0).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert np.array_equal(fitted.estimator.classes_, CLASSES)


def test_base_rates_use_entire_training_period() -> None:
    frame = training_rows()
    frame.loc[3:, "outcome"] = "win"
    fitted = fit_model("B0", frame, pd.Timestamp("2006-06-09"))
    np.testing.assert_allclose(fitted.predict_proba(frame)[0], [1 / 6, 1 / 6, 4 / 6])


def test_scaler_is_fit_only_on_earlier_model_training() -> None:
    frame = training_rows()
    frame.loc[3:, list(ALL_FEATURES)] += 10000
    fitted = fit_model("M1", frame, pd.Timestamp("2006-06-09"))
    frozen = fitted.estimator.calibrated_classifiers_[0].estimator
    means = frozen.estimator.named_steps["standardscaler"].mean_
    np.testing.assert_allclose(means, frame.loc[:2, list(ALL_FEATURES)].mean().to_numpy())


def test_reliability_counts_all_pooled_predictions_including_one() -> None:
    predictions = pd.DataFrame({"model": ["B1", "B1"], "outcome": ["win", "draw"],
                                "p_draw": [0, 1], "p_loss": [0, 0], "p_win": [1, 0]})
    table = reliability_table(predictions)
    assert table.groupby("outcome")["count"].sum().to_dict() == {label: 2 for label in CLASSES}
    assert table.observed.eq(table.predicted).all()


def test_real_data_splits_when_local_features_exist() -> None:
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "data/processed/features.csv"
    if not path.exists():
        pytest.skip("Optional full-source audit: first build local features")
    frame = pd.read_csv(path, parse_dates=["date"])
    for split in world_cup_splits(frame):
        assert split.training.date.max() < split.test.date.min()
