"""Materialise pre-match features and app-ready current ratings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_results
from src.features import ALL_FEATURES, build_features


def main() -> None:
    """Build only from completed source matches through the requested cutoff."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", default=None)
    args = parser.parse_args()
    matches, audit = load_results(ROOT / "data/raw", args.as_of)
    features, states, history = build_features(matches)
    processed = ROOT / "data/processed"
    artifacts = ROOT / "models"
    processed.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(exist_ok=True)
    features.to_csv(processed / "features.csv", index=False)
    states.to_csv(artifacts / "ratings.csv", index=False)
    history.to_csv(artifacts / "elo_history.csv", index=False)
    metadata = {"data": audit, "feature_columns": list(ALL_FEATURES)}
    (processed / "feature_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Built {len(features):,} pre-match rows with {len(ALL_FEATURES)} features.")


if __name__ == "__main__":
    main()
