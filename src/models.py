"""Fixed candidate models with a strictly later, pre-test calibration period."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.features import ALL_FEATURES, ELO_FEATURES

# Lexicographic order matches sklearn's class ordering and log_loss convention.
CLASSES = ("draw", "loss", "win")
MODEL_NAMES = {
    "B0": "Training base rates", "B1": "Elo logistic baseline",
    "M1": "All-feature logistic", "M2": "Histogram gradient boosting",
}
MODEL_PARAMETERS: dict[str, dict[str, Any]] = {
    "B0": {"method": "empirical training frequencies"},
    "B1": {"C": 1.0, "solver": "lbfgs", "max_iter": 1000, "tol": 1e-6},
    "M1": {"C": 1.0, "solver": "lbfgs", "max_iter": 1000, "tol": 1e-6},
    "M2": {"max_iter": 150, "learning_rate": 0.07, "max_leaf_nodes": 15,
           "min_samples_leaf": 30, "l2_regularization": 1.0, "early_stopping": False,
           "random_state": 42},
}


@dataclass
class BaseRates:
    """A probability-only baseline estimated from all pre-cutoff training rows."""

    probabilities: np.ndarray

    @property
    def classes_(self) -> np.ndarray:
        """Expose class order consistently with sklearn estimators."""
        return np.asarray(CLASSES)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Repeat training frequencies irrespective of the chosen matchup."""
        return np.tile(self.probabilities, (len(features), 1))


@dataclass
class FittedModel:
    """Persist the estimator together with its exact input columns and audit."""

    model_id: str
    columns: tuple[str, ...]
    estimator: Any
    training_info: dict[str, Any]

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return explicitly ordered draw/loss/win probabilities."""
        with threadpool_limits(limits=2):
            probabilities = self.estimator.predict_proba(features.loc[:, list(self.columns)])
        order = [list(self.estimator.classes_).index(label) for label in CLASSES]
        return probabilities[:, order]


def calibration_split(
    training: pd.DataFrame, cutoff: pd.Timestamp, calibration_years: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition training by calendar date; no date can straddle the split."""
    if calibration_years < 1:
        raise ValueError("calibration_years must be positive")
    if training.empty or not training.date.max() < cutoff:
        raise ValueError("Training must be nonempty and strictly before cutoff")
    boundary = cutoff - pd.DateOffset(years=calibration_years)
    fit = training.loc[training.date.lt(boundary)]
    calibration = training.loc[training.date.ge(boundary)]
    for name, part in (("fit", fit), ("calibration", calibration)):
        if part.empty or set(part.outcome) != set(CLASSES):
            raise ValueError(f"{name} period must contain all outcome classes; use more history")
    assert fit.date.max() < calibration.date.min() < cutoff
    return fit, calibration


def create_estimator(model_id: str) -> Any:
    """Construct untuned candidates; prevent random early-stopping splits."""
    if model_id in ("B1", "M1"):
        return make_pipeline(StandardScaler(), LogisticRegression(**MODEL_PARAMETERS[model_id]))
    if model_id == "M2":
        return HistGradientBoostingClassifier(**MODEL_PARAMETERS[model_id])
    raise ValueError(f"Unknown learned model {model_id}")


def fit_model(
    model_id: str, training: pd.DataFrame, cutoff: pd.Timestamp, calibration_years: int = 4,
) -> FittedModel:
    """Fit without test rows, reserving the trailing years for sigmoid calibration.

    The frozen base estimator is never refitted on calibration observations.
    Base rates use the entire training period, as specified for that baseline.
    """
    if model_id not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model_id}")
    if training.empty or training.date.max() >= cutoff:
        raise ValueError("Training dates must strictly precede the forecast cutoff")
    info: dict[str, Any] = {
        "train_matches": len(training), "train_start": training.date.min().date().isoformat(),
        "train_end": training.date.max().date().isoformat(), "cutoff": cutoff.date().isoformat(),
    }
    if model_id == "B0":
        rates = training.outcome.value_counts(normalize=True).reindex(CLASSES, fill_value=0).to_numpy()
        return FittedModel(model_id, (), BaseRates(rates), {**info, "calibration": "none (base rates)"})
    fit, calibration = calibration_split(training, cutoff, calibration_years)
    columns = ELO_FEATURES if model_id == "B1" else ALL_FEATURES
    estimator = create_estimator(model_id)
    with threadpool_limits(limits=2):
        estimator.fit(fit.loc[:, list(columns)], fit.outcome)
        # One prediction pass over the entire holdout. The empty fit indices
        # are intentional: FrozenEstimator already owns the earlier fitted
        # model and its fit() is a no-op. No internal random folds are needed.
        holdout = [(np.empty(0, dtype=int), np.arange(len(calibration)))]
        calibrated = CalibratedClassifierCV(
            FrozenEstimator(estimator), method="sigmoid", ensemble=False, cv=holdout,
        )
        calibrated.fit(calibration.loc[:, list(columns)], calibration.outcome)
    info.update({
        "fit_matches": len(fit), "fit_end": fit.date.max().date().isoformat(),
        "calibration_matches": len(calibration),
        "calibration_start": calibration.date.min().date().isoformat(),
        "calibration_end": calibration.date.max().date().isoformat(),
        "calibration": "sigmoid on chronological holdout", "calibration_years": calibration_years,
    })
    return FittedModel(model_id, columns, calibrated, info)
