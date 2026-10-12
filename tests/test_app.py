"""Smoke-test both pages and verify app inference never trains on startup."""

from pathlib import Path
import shutil

import pytest
from streamlit.testing.v1 import AppTest

from src.inference import REQUIRED_ARTIFACTS, load_assets, predict_match

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "models/final_model.pkl").exists(), reason="Build app artifacts first")


def test_saved_bundle_predictions_and_invalid_fixture() -> None:
    assets = load_assets(ROOT)
    probabilities = predict_match(assets, "Argentina", "France", True)
    assert set(probabilities) == {"win", "draw", "loss"}
    assert sum(probabilities.values()) == pytest.approx(1)
    with pytest.raises(ValueError, match="different"):
        predict_match(assets, "France", "France", True)


def test_deployment_bundle_needs_no_raw_data(tmp_path: Path) -> None:
    for name in REQUIRED_ARTIFACTS:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    assert not (tmp_path / "data").exists()
    probabilities = predict_match(load_assets(tmp_path), "Argentina", "France", True)
    assert sum(probabilities.values()) == pytest.approx(1)


def test_both_pages_load_without_training(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_fit(*args: object, **kwargs: object) -> None:
        raise AssertionError("The app must never train")

    monkeypatch.setattr("src.models.fit_model", unexpected_fit)
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    assert not app.exception
    assert len(app.metric) == 5
    app.toggle[0].set_value(False).run()
    assert not app.exception
    app.sidebar.radio[0].set_value("How good is it?").run()
    assert not app.exception
    assert len(app.dataframe) == 2
