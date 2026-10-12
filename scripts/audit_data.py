"""Milestone one: validate real data and compute the full-history Elo table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_results
from src.elo import classification_audit, compute_elo, tournament_k


def main() -> None:
    """Save source provenance, cleaning decisions, and pre/post-match Elo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", default=None, help="Last permitted match date (inclusive)")
    args = parser.parse_args()
    matches, audit = load_results(ROOT / "data/raw", args.as_of)
    audit["tournament_k"] = {name: tournament_k(name) for name in sorted(matches.tournament.unique())}
    manifest = ROOT / "data/raw/source.json"
    audit["source"] = json.loads(manifest.read_text()) if manifest.exists() else {"mode": "manual"}
    audit["source_files_match_manifest"] = audit["source"].get("sha256") == audit["source_sha256"]
    audit["tournament_classifications"] = classification_audit(matches.tournament.unique())
    before, history, ratings = compute_elo(matches)
    processed = ROOT / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    before.to_csv(processed / "prematch_elo.csv", index=False)
    history.to_csv(processed / "elo_history.csv", index=False)
    pd_ratings = pd.DataFrame(
        [{"team": team, "elo": rating} for team, rating in sorted(ratings.items())]
    )
    pd_ratings.to_csv(processed / "ratings.csv", index=False)
    (ROOT / "reports/data_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    pd.DataFrame(audit["tournament_classifications"]).to_csv(ROOT / "reports/tournament_classifications.csv", index=False)
    print(f"Validated {len(matches):,} matches; Elo computed for {len(ratings)} teams.")


if __name__ == "__main__":
    main()
