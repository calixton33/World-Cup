# Methodology and audit trail

## Target and data contract

One row is one men's international fixture as supplied by martj42. The target is the first-listed team's score outcome, including extra time and excluding shootouts: `win`, `draw`, or `loss`. `shootouts.csv` is downloaded for provenance; its winner is deliberately excluded from both labels and Elo updates. A tied score stays a draw. These probabilities do not mean regulation-time results or qualification to the next round.

The source also includes historical, regional, and non-FIFA teams. No federation filter is imposed. Missing-score fixtures and matches beyond `--as-of` are excluded and counted; exact duplicate rows are removed and counted. Invalid dates, scores, venues, and teams fail validation. The country or city column is not used to infer home advantage: the source `neutral` flag decides it.

All source files are fetched at one resolved GitHub commit. `data/raw/source.json` records the revision, timestamp, and SHA-256 fingerprints. Raw data is ignored; `reports/metrics.json` retains the provenance needed to reproduce a run. If the files are manually replaced, the audit flags whether their fingerprints still match the download manifest.

## Historical team names

`former_names.csv` maps `former` to `current` only between inclusive `start_date` and `end_date`. Existing canonical names are left alone. This matters for ambiguous historical names such as Ireland. If the aliases file is unavailable, names remain unchanged and `former_names_available` is false in the audit.

The dataset itself already uses many successor-team conventions. This can connect different political or sporting eras; neither aliases nor equal Elo initialization establish historical continuity. There is no additional manual merging of dissolved teams.

## Elo

Each team starts at 1500 when first observed. There is no rating decay, externally supplied ranking, or strength-dependent initialization. Ratings are warmed up on the entire history even though statistical model training begins in a configurable later year.

For first-listed/home rating `Rh`, away rating `Ra`, and nonneutral home bonus `H=100` (otherwise zero):

```text
dr = Rh - Ra + H
We = 1 / (10 ** (-dr / 400) + 1)
W  = 1 (win), 0.5 (draw), or 0 (loss)
delta = K * G * (W - We)
Rh_new = Rh + delta; Ra_new = Ra - delta
G = 1 for margin 0 or 1; 1.5 for margin 2; (11 + margin) / 8 otherwise
```

`src/elo.py::TOURNAMENT_K` is the single exact-name mapping: 60 for World Cup finals, 50 for principal continental finals and major intercontinental tournaments, 40 for their qualifiers, 30 for other tournaments, 20 for friendlies. The complete **judgement-name list** is `reports/tournament_classifications.csv`, filtering `judgement=True`. In particular, historic CCCF/NAFC continental predecessors, Confederations Cup, CONMEBOL–UEFA Cup of Champions, and Pan American Championship use 50. Nations Leagues, subregional qualifiers, AFC secondary tournaments, bilateral cups, invitation events, and non-FIFA competitions use 30. Unknown names raise an error instead of silently acquiring a K value. This is a documented approximation, not a recreation of the official site's complete ratings or initialization conventions.

There are no kickoff timestamps. All ratings and recent form are frozen before an entire date. After that date's features are emitted, Elo changes are summed and form histories updated. A team playing twice on one date cannot use its first result for its second fixture. Stable source sorting supplies a deterministic order within a day's historical form records; actual kickoff order remains unknown.

## Features and missing history

The all-feature models use each side's Elo, raw Elo difference, a nonneutral home indicator, K tier, and each team's last-five/last-ten points, goals scored, goals conceded per game, and available-game counts. Raw Elo difference excludes the home bonus: the separate indicator allows the classifier to learn venue effects. K is known from tournament identity, not its match result.

Form is calculated from that team's own perspective, including both home and away appearances, with three points for a win and one for a draw. A short history uses the mean of available prior matches. A team with no prior games receives fixed means of one point and one goal for/against per match, and zero support counts. No population statistic from future matches is used for imputation. Historical and live predictions call the same `match_features` function.

## Models, calibration, and temporal evaluation

The candidates and fixed hyperparameters are recorded in `reports/metrics.json` and `src/models.py`. Both logistic models use multinomial loss with standardization fit only on their fitting period. Histogram gradient boosting has early stopping disabled to avoid an implicit random validation split. No class balancing, outcome oversampling, or hyperparameter search is applied.

At each available World Cup final tournament from the configured first test year, all training matches must precede its first match, and must satisfy the configurable model-training start year. The last four calendar years of that training period, by default, are reserved for calibration. A base classifier is fitted on the earlier period and frozen. Sigmoid calibrators are fitted on the later holdout; the base model and scaler are never refitted on it. One explicit holdout prediction pass avoids implicit random folds in the calibration library. Each period must contain every outcome class or training stops with an explanatory error.

The empirical base-rate candidate is different by design: it uses the frequencies over the entire pre-tournament training window and is not separately recalibrated. This follows the requested baseline definition and gives it more rows to estimate its unconditional probabilities.

Models stay frozen during a tournament. Elo and form may incorporate already completed matches from earlier tournament dates. This estimates performance of forecasts made before each match, not forecasts of a whole tournament on opening day. Later origins can include previous World Cup matches in training. The final app model repeats the selected candidate's chronological fit/calibration procedure through the latest available completed match; its recent holdout remains reserved for calibration.

## Scores, selection, and reliability

- **Log loss:** mean negative natural log probability assigned to the observed class. It is the primary selection score.
- **Multiclass Brier:** mean sum of the three squared probability errors, unscaled, ranging from zero to two. It is not divided by three or by two.
- **Accuracy:** fraction of most-probable labels that match outcomes. Exact probability ties use lexicographic class order (`draw`, `loss`, `win`).
- **Higher-Elo reference:** predicts whichever side has the greater raw pre-match rating, with an exact rating tie predicting draw. It has no log loss or Brier score because it supplies no probability model.

Selection uses the lowest **unweighted mean log loss across tournaments**. Each tournament receives equal weight despite different sample sizes. Pooled scores are also stored under diagnostics; they are not the selection rule. Exact mean-score ties use the fixed candidate order. Partial tournaments, if ever present, are included and flagged against known expected match counts.

Reliability uses only saved test predictions, pooled across tournaments and plotted separately for each outcome and model. Ten fixed equal-width probability bins show mean predicted probability against observed frequency. Counts are available on hover and in `calibration.csv`; vertical bars are nominal 95% Wilson binomial intervals. These are descriptive intervals, since matches and tournament environments are not independent. Calibration on a general-international holdout does not guarantee World Cup calibration.

The same backtests rank and select candidates. There is **no separate untouched final test after selection**, and a narrow average winning margin is not a significance claim. Small test sets, team-name conventions, absent lineup/injury data, inherently variable draws, and domain shift limit generalization.

## Reproduction and deployment

Run `scripts/run_all.py` for download, validation/Elo, features, backtesting, final artifact training, and README generation. `--through` supports cumulative milestone runs. `--skip-download`, `--as-of`, `--train-start-year`, and `--calibration-years` make source and training choices explicit. Individual stage scripts can also be run directly; re-run later stages after changing any earlier stage. Source hashes and date cutoffs detect stale feature/report combinations; model and app-table checksums detect partially rebuilt app bundles.

`models/` contains only the final model, current team states, Elo trajectories, and metadata. `reports/` contains audit data, tournament and mean scores, held-out probabilities, reliability bins, and reusable HTML/Vega-Lite plots. These artifacts and `src/` are sufficient for the hosted app: no raw data, notebooks, downloads, or fitting are needed at startup. Dependencies are pinned to the artifact's scikit-learn version. The deployment check makes a real prediction without training.

Tests cover hand-calculated Elo arithmetic, home advantage, name-date scope, future-prefix invariance, own/same-date result invariance, available-history means, exact train/test and fit/calibration ordering, pre-calibration scaler fitting, class ordering, proper-score arithmetic, live/historical parity, and both Streamlit pages.
