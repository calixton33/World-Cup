"""Generate the hiring-manager README using only numbers from actual metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def results_table(metrics: dict[str, Any]) -> str:
    """Render the same equal-weight tournament means displayed in the app."""
    lines = ["| Model | Mean log loss ↓ | Mean Brier ↓ | Mean accuracy ↑ |",
             "|---|---:|---:|---:|"]
    for row in metrics["mean_metrics"]:
        loss = "—" if row["log_loss"] is None else f"{row['log_loss']:.4f}"
        brier = "—" if row["brier_score"] is None else f"{row['brier_score']:.4f}"
        label = row["name"] if row["model"] == "Elo pick" else f"{row['model']} · {row['name']}"
        lines.append(f"| {label} | {loss} | {brier} | {row['accuracy']:.2%} |")
    return "\n".join(lines)


def render_readme(metrics: dict[str, Any]) -> str:
    """Keep claims traceable to measured scores, data audit, and recorded config."""
    data, config, selection = metrics["data"], metrics["config"], metrics["selection"]
    years = ", ".join(str(split["year"]) for split in metrics["splits"])
    diagnostics = metrics["diagnostics"][selection["model"]]
    source_revision = data["source"].get("revision", "manual local files")
    latest = metrics["splits"][-1]
    baseline_folds = ", ".join(str(row["year"]) for row in metrics["fold_winners"] if row["model"] == "B0") or "none"
    outcome = ("A baseline won; complexity was unnecessary." if selection["baseline_won"] else
               "Compare this measured margin with the small test sets; it is not a significance claim.")
    alias_note = (f"Source aliases are applied within their historical date windows; this snapshot already uses "
                  f"canonical names ({data['team_name_cells_normalised']} replacements)."
                  if data["former_names_available"] else
                  "Source aliases are unavailable, so names remain as-is and historical changes may fragment ratings.")
    return f"""# International Football Predictor

**Can pre-match Elo and recent form forecast international win / draw / loss outcomes better than simple baselines?**

Real results, causal features, and calibrated probabilities. Tested on **{metrics['test_matches']} matches across {metrics['test_tournaments']} World Cups ({years})**. The Streamlit app shows forecasts, rating histories, and the evidence behind them.

## Results

{results_table(metrics)}

**Winner: {selection['model']} · {selection['name']}, ahead of {selection['runner_up']} by {selection['margin_log_loss']:.5f} mean log loss.** {outcome}

Means weight each tournament equally. Log loss penalises confident mistakes; Brier sums squared probability errors across classes (range {config['brier_range'][0]}–{config['brier_range'][1]}); accuracy counts correct most-likely outcomes. Lower is better for probability scores. The higher-Elo pick supplies only a label.

## Data

[martj42 international results](https://github.com/martj42/international_results), [CC0](https://github.com/martj42/international_results/blob/master/LICENSE): **{data['completed_matches']:,} completed matches**, {data['first_match']} through {data['last_match']}. Includes historical and non-FIFA internationals. {alias_note} Raw files are ignored by Git; source revision and hashes are in [metrics.json](reports/metrics.json).

## Method

- Elo uses the full history, initial rating {config['initial_elo']:.0f}, {config['home_elo_bonus']:.0f}-point nonneutral home advantage, importance tiers, and score-margin multipliers. [Tournament classifications](reports/tournament_classifications.csv) flag every judgement call.
- Features use pre-match ratings, their difference, venue, importance, and each team's prior {config['form_windows'][0]}/{config['form_windows'][1]}-match points and goals for/against per game. Short histories use available means and counts; cold starts use fixed {config['cold_start_prior']:g}-point/{config['cold_start_prior']:g}-goal priors.
- Each World Cup model trains on matches from {config['train_start_year']} strictly before opening day. Learned models reserve the final {config['calibration_years']} training years for sigmoid calibration. Hyperparameters are fixed; no test-set tuning.
- Elo/form update between match dates; models stay frozen during tournaments. Same-day results are withheld together. These evaluate forecasts before each match.

## Findings and limitations

- Boosting performed worse overall; base rates won the **{baseline_folds}** fold. Elo already captures most of the signal.
- The winner's most-likely label was draw **{diagnostics['predicted_draws']} times**, despite **{diagnostics['actual_draws']} actual draws**. Draw probabilities still affect its scores.
- Scores include extra time; penalty shootouts count as draws. Team-name/successor conventions may bridge distinct eras. No player, injury, or lineup data.
- Small test sets: {metrics['splits'][0]['test_matches']} matches in earlier cups, {latest['test_matches']} in {latest['year']}. [Pooled reliability](reports/calibration.html) shows counts and uncertainty; future calibration is not guaranteed.
- Backtests also selected the model: **no independent post-selection test remains**. The small winning margin is not a significance claim.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py --as-of {data['as_of']}
pytest -q
streamlit run app.py
```

Configure `--train-start-year` and `--calibration-years`. If offline, place `results.csv` and optional `former_names.csv`/`shootouts.csv` in `data/raw/`, then use `--skip-download`. Reproduce this snapshot with `python scripts/download_data.py --refresh --ref {source_revision}`.

For **Streamlit Community Cloud**, use `app.py`, pinned requirements, and include `src/`, `models/`, and `reports/`. Startup loads saved artifacts without training or downloads. Check with `python scripts/check_deployment.py`; existing Replit support is retained.

The pipeline is in `src/` and `scripts/`; guardrails are in `tests/`. Every rebuild generates this README's numbers from actual [metrics.json](reports/metrics.json). See [methodology](docs/methodology.md) for assumptions, files, and individual stages. No notebook required.
"""


def main() -> None:
    """Refuse to substitute estimates or placeholder scores for missing reports."""
    path = ROOT / "reports/metrics.json"
    if not path.exists():
        raise FileNotFoundError("Run the real-data backtest first; README results cannot be invented")
    metrics = json.loads(path.read_text())
    (ROOT / "README.md").write_text(render_readme(metrics))
    print("README generated from the actual reports/metrics.json.")


if __name__ == "__main__":
    main()
