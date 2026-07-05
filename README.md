# FIFA World Cup ML Analytics: Match Prediction and Player Clustering

This project is an introductory machine learning analytics project built from a non-official Kaggle FIFA World Cup 2026 player performance dataset.

It has two main parts:

1. Supervised learning: a lineup-based match outcome predictor that estimates Team A win, draw, and Team B win probabilities.
2. Unsupervised learning: K-Means player clustering based on performance style and overperformance versus expected output.

## Dataset

Dataset page:

https://www.kaggle.com/datasets/rauffauzanrambe/fifa-world-cup-2026-player-performance-dataset

After downloading from Kaggle, place the raw CSV here:

`data/fifa_world_cup_2026_player_performance-selected-columns-2.csv`

Each row is a player performance in one match. The prediction target is match-level, so the workflow transforms player rows into team-level, lineup-level, and matchup-level features before training a classifier.

This dataset is used for learning and demonstration only. The data and model outputs should not be treated as official FIFA records, live results, betting odds, or verified forecasts.

The raw Kaggle CSV, processed CSVs, saved models, charts, and generated reports are intentionally not committed to this public repository. Before redistributing any of those artifacts, confirm that the Kaggle dataset license allows redistribution.

## Project Structure

```text
fifa-world-cup-ml-analytics/
├── data/
│   └── README.md
├── notebooks/
│   └── fifa_world_cup_ml_analytics.ipynb
├── app.py
├── main.py
├── EXPLANATION.md
├── README.md
└── requirements.txt
```

## Supervised Learning

The supervised model predicts:

- `W`: Team A wins
- `D`: draw
- `L`: Team A loses, meaning Team B wins

The workflow creates one row for each team perspective in each match:

- Team A vs Team B
- Team B vs Team A

For each row, it creates:

- Team A lineup/team feature values
- Team B lineup/team feature values
- Difference features, such as `diff_avg_player_rating` and `diff_total_expected_goals_xg`

The workflow trains and compares:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

In the generated run, Logistic Regression was selected as the final supervised model based on the best log loss and solid accuracy.

## Unsupervised Learning

The clustering workflow uses aggregated player profiles, not raw match rows. It engineers overperformance features such as:

- `goal_overperformance = goals - expected_goals_xg`
- `assist_overperformance = assists - expected_assists_xa`
- `goal_contribution_overperformance = goals + assists - expected_goals_xg - expected_assists_xa`
- `shot_accuracy = shots_on_target / shots`

It scales the features, tries K-Means values from 2 to 8, checks inertia and silhouette score, then saves the final clustering model. In the generated run, 5 clusters were selected and labeled by inspecting the cluster average statistics.

## Evaluation Metrics

The supervised models are evaluated with:

- Accuracy: how often the predicted class is correct.
- Classification report: precision, recall, and F1-score for each class.
- Confusion matrix: where the model confuses one class for another.
- Log loss: how good the predicted probabilities are.

Log loss matters here because the app returns probabilities, not only a single class prediction.

## How to Run the Workflow

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full workflow:

```bash
python main.py
```

This regenerates the processed CSV files, saved models, evaluation outputs, and charts.

Generated artifacts are written locally to:

- `data/player_profiles.csv`
- `data/player_profiles_with_clusters.csv`
- `data/matchup_training_data.csv`
- `models/*.pkl`
- `outputs/`

## How to Run the Streamlit App

After downloading the Kaggle CSV and running `main.py`, start the app:

```bash
streamlit run app.py
```

## Deployment

This is a Streamlit application, so it is not designed to deploy directly on Vercel's Python serverless runtime. Vercel expects a Python file to export a web callable named `app`, `application`, or `handler`, while Streamlit apps are launched with `streamlit run app.py`.

Recommended deployment options:

- Streamlit Community Cloud
- Hugging Face Spaces with the Streamlit SDK
- Render or Railway using `streamlit run app.py`

Because the Kaggle dataset and generated model artifacts are intentionally not committed to this public repository, a hosted deployment will also need a permitted way to provide the dataset and regenerate the local artifacts. For a public demo, confirm the Kaggle dataset license before uploading the data or generated model files.

The redesigned app supports:

- Selecting two teams
- Selecting exactly 11 players per team
- Predicting win/draw/loss probabilities
- Viewing selected players' cluster labels
- Exploring player clusters and average cluster statistics
- Exploring individual players with team, position, cluster, rating, and overperformance filters

## Limitations

- The dataset may be simulated or projected.
- This beginner version uses full-tournament player averages for custom lineup predictions.
- Some player performance stats are post-match stats, so a production pre-match model should use only information available before kickoff.
- Real match results depend on tactics, injuries, substitutions, formations, weather, and randomness.
- The model estimates probabilities, not guaranteed outcomes.
- Clusters are statistical groupings and must be interpreted carefully.

## Future Improvements

- Use only pre-match player statistics.
- Add team formations.
- Add injuries and substitutions.
- Add Elo ratings or FIFA rankings.
- Add recent team form.
- Improve probability calibration.
- Try other clustering methods such as hierarchical clustering or DBSCAN.
- Add SHAP values for model explainability.
