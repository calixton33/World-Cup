"""Evaluate every available men's World Cup final tournament at a rolling origin."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import altair
import numpy
import pandas as pd
import sklearn
import streamlit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import FIRST_TEST_YEAR, reliability_table, run_backtest
from src.elo import HOME_ADVANTAGE, INITIAL_RATING
from src.features import ALL_FEATURES, FORM_WINDOWS
from src.models import CLASSES, MODEL_NAMES, MODEL_PARAMETERS
from src.plots import calibration_chart, comparison_chart


def main() -> None:
    """Write measured metrics, source provenance, predictions, and reusable chart specs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-start-year", type=int, default=1990)
    parser.add_argument("--calibration-years", type=int, default=4)
    args = parser.parse_args()
    processed, reports = ROOT / "data/processed", ROOT / "reports"
    features = pd.read_csv(processed / "features.csv", parse_dates=["date"])
    result, table, predictions = run_backtest(features, args.train_start_year, args.calibration_years)
    audit = json.loads((reports / "data_audit.json").read_text())
    metadata = json.loads((processed / "feature_metadata.json").read_text())
    if audit["source_sha256"] != metadata["data"]["source_sha256"] or audit["as_of"] != metadata["data"]["as_of"]:
        raise ValueError("Features and audit come from different data snapshots; rebuild features")
    result.update({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "data": audit,
        "config": {"train_start_year": args.train_start_year, "calibration_years": args.calibration_years,
                   "first_test_year": FIRST_TEST_YEAR, "initial_elo": INITIAL_RATING,
                   "home_elo_bonus": HOME_ADVANTAGE, "elo_scale": 400, "form_windows": list(FORM_WINDOWS),
                   "cold_start_prior": 1.0, "k_tiers": [20, 30, 40, 50, 60],
                   "feature_count": len(ALL_FEATURES), "reliability_bins": 10,
                   "brier_range": [0, 2], "wilson_confidence": 0.95,
                   "classes": list(CLASSES), "features": list(ALL_FEATURES),
                   "models": MODEL_NAMES, "parameters": MODEL_PARAMETERS},
        "methodology": {
            "forecast": "before each match date; earlier tournament dates may update Elo and form",
            "same_day": "all features frozen before any results from that date",
            "calibration": "sigmoid; base fit before holdout, never refit on holdout; B0 uses full training rates",
            "brier": "mean sum of squared class-probability errors (no division by number of classes)",
            "selection_caveat": "Backtests choose the model; no independent post-selection holdout remains",
            "elo_reference": "higher raw Elo predicts win/loss; exact equal-rating ties predict draw",
            "judgement_k": "Historical continental predecessors, Confederations Cup, CONMEBOL–UEFA and "
                           "Pan American finals use 50. Nations Leagues, regional qualifiers, invitation, "
                           "bilateral and non-FIFA cups use 30; every exact source name is listed in tournament_k.",
        },
        "environment": {"python": platform.python_version(), "pandas": pd.__version__,
                        "numpy": numpy.__version__, "scikit-learn": sklearn.__version__,
                        "streamlit": streamlit.__version__, "altair": altair.__version__},
    })
    reports.mkdir(exist_ok=True)
    (reports / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    means = pd.DataFrame(result["mean_metrics"]).assign(
        aggregation="mean", test_matches=result["test_matches"], year=None,
    )
    combined = pd.concat([table.assign(aggregation="tournament"), means], ignore_index=True)
    combined.to_csv(reports / "metrics.csv", index=False)
    predictions.to_csv(reports / "predictions.csv", index=False)
    reliability = reliability_table(predictions)
    reliability.to_csv(reports / "calibration.csv", index=False)
    for name, chart in (("calibration", calibration_chart(reliability)),
                        ("model_comparison", comparison_chart(pd.DataFrame(result["mean_metrics"])))):
        chart.save(reports / f"{name}.vl.json")
        chart.save(reports / f"{name}.html")
    print(f"Selected {result['selection']['model']}: {result['selection']['mean_log_loss']:.4f} mean log loss")


if __name__ == "__main__":
    main()
