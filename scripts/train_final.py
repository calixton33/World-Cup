"""Train only the selected candidate and save a ready-to-load forecasting bundle."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import fit_model
from src.data import file_sha256


def main() -> None:
    """Use the report's selected model and settings; do not revisit test rankings."""
    metrics = json.loads((ROOT / "reports/metrics.json").read_text())
    feature_metadata = json.loads((ROOT / "data/processed/feature_metadata.json").read_text())
    if metrics["data"]["source_sha256"] != feature_metadata["data"]["source_sha256"]:
        raise ValueError("Metrics and features disagree; rerun the backtest")
    if metrics["data"]["as_of"] != feature_metadata["data"]["as_of"]:
        raise ValueError("Metrics and features use different date cutoffs")
    features = pd.read_csv(ROOT / "data/processed/features.csv", parse_dates=["date"])
    config = metrics["config"]
    training = features.loc[features.date.dt.year.ge(config["train_start_year"])]
    cutoff = features.date.max() + pd.Timedelta(days=1)
    fitted = fit_model(metrics["selection"]["model"], training, cutoff, config["calibration_years"])
    artifact = ROOT / "models/final_model.pkl"
    artifact.parent.mkdir(exist_ok=True)
    artifact.write_bytes(pickle.dumps(fitted, protocol=pickle.HIGHEST_PROTOCOL))
    metadata = {
        "model": fitted.model_id, "data_through": metrics["data"]["last_match"],
        "data_as_of": metrics["data"]["as_of"], "source_sha256": metrics["data"]["source_sha256"],
        "environment": metrics["environment"], "training": fitted.training_info,
        "columns": list(fitted.columns), "selected_mean_log_loss": metrics["selection"]["mean_log_loss"],
        "artifact_sha256": {name: file_sha256(ROOT / "models" / name)
                            for name in ("final_model.pkl", "ratings.csv", "elo_history.csv")},
    }
    (ROOT / "models/metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    metrics["final_training"] = metadata
    (ROOT / "reports/metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(f"Saved calibrated {fitted.model_id} model using data through {metadata['data_through']}.")


if __name__ == "__main__":
    main()
