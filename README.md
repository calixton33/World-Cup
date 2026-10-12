# International Football Predictor

**Can pre-match Elo and recent form forecast international win / draw / loss outcomes better than simple baselines?**

Real results, causal features, and calibrated probabilities. Tested on **424 matches across 6 World Cups (2006, 2010, 2014, 2018, 2022, 2026)**. The Streamlit app shows forecasts, rating histories, and the evidence behind them.

## Results

| Model | Mean log loss ↓ | Mean Brier ↓ | Mean accuracy ↑ |
|---|---:|---:|---:|
| B0 · Training base rates | 1.0719 | 0.6494 | 44.05% |
| B1 · Elo logistic baseline | 0.9466 | 0.5551 | 58.19% |
| M1 · All-feature logistic | 0.9441 | 0.5527 | 58.13% |
| M2 · Histogram gradient boosting | 0.9808 | 0.5743 | 55.11% |
| Pick higher Elo | — | — | 59.33% |

**Winner: M1 · All-feature logistic, ahead of B1 by 0.00245 mean log loss.** Compare this measured margin with the small test sets; it is not a significance claim.

Means weight each tournament equally. Log loss penalises confident mistakes; Brier sums squared probability errors across classes (range 0–2); accuracy counts correct most-likely outcomes. Lower is better for probability scores. The higher-Elo pick supplies only a label.

## Data

[martj42 international results](https://github.com/martj42/international_results), [CC0](https://github.com/martj42/international_results/blob/master/LICENSE): **49,945 completed matches**, 1872-11-30 through 2026-10-06. Includes historical and non-FIFA internationals. Source aliases are applied within their historical date windows; this snapshot already uses canonical names (0 replacements). Raw files are ignored by Git; source revision and hashes are in [metrics.json](reports/metrics.json).

## Method

- Elo uses the full history, initial rating 1500, 100-point nonneutral home advantage, importance tiers, and score-margin multipliers. [Tournament classifications](reports/tournament_classifications.csv) flag every judgement call.
- Features use pre-match ratings, their difference, venue, importance, and each team's prior 5/10-match points and goals for/against per game. Short histories use available means and counts; cold starts use fixed 1-point/1-goal priors.
- Each World Cup model trains on matches from 1990 strictly before opening day. Learned models reserve the final 4 training years for sigmoid calibration. Hyperparameters are fixed; no test-set tuning.
- Elo/form update between match dates; models stay frozen during tournaments. Same-day results are withheld together. These evaluate forecasts before each match.

## Findings and limitations

- Boosting performed worse overall; base rates won the **2022** fold. Elo already captures most of the signal.
- The winner's most-likely label was draw **0 times**, despite **96 actual draws**. Draw probabilities still affect its scores.
- Scores include extra time; penalty shootouts count as draws. Team-name/successor conventions may bridge distinct eras. No player, injury, or lineup data.
- Small test sets: 64 matches in earlier cups, 104 in 2026. [Pooled reliability](reports/calibration.html) shows counts and uncertainty; future calibration is not guaranteed.
- Backtests also selected the model: **no independent post-selection test remains**. The small winning margin is not a significance claim.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py --as-of 2026-10-12
pytest -q
streamlit run app.py
```

Configure `--train-start-year` and `--calibration-years`. If offline, place `results.csv` and optional `former_names.csv`/`shootouts.csv` in `data/raw/`, then use `--skip-download`. Reproduce this snapshot with `python scripts/download_data.py --refresh --ref d5bd6de012c87e2f325de602ae77fe2f9ab61ef3`.

For **Streamlit Community Cloud**, use `app.py`, pinned requirements, and include `src/`, `models/`, and `reports/`. Startup loads saved artifacts without training or downloads. Check with `python scripts/check_deployment.py`; existing Replit support is retained.

The pipeline is in `src/` and `scripts/`; guardrails are in `tests/`. Every rebuild generates this README's numbers from actual [metrics.json](reports/metrics.json). See [methodology](docs/methodology.md) for assumptions, files, and individual stages. No notebook required.
