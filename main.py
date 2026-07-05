"""
Main workflow for the FIFA World Cup ML Analytics project.

Run from the project folder:

    python main.py

The script loads the player-level CSV, creates player profiles, builds a
matchup-level supervised learning dataset, trains supervised models, clusters
players, and saves all requested artifacts.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
(ROOT / ".matplotlib").mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import joblib
except ImportError:  # pragma: no cover - pickle fallback is here for portability.
    joblib = None


DATA_FILE = DATA_DIR / "fifa_world_cup_2026_player_performance-selected-columns-2.csv"

IDENTIFIER_COLUMNS = [
    "player_name",
    "nationality",
    "team",
    "position",
    "preferred_foot",
    "match_id",
    "match_date",
    "city",
    "opponent_team",
    "tournament_stage",
    "match_result",
]

REQUESTED_SUPERVISED_COLUMNS = [
    "minutes_played",
    "goals",
    "assists",
    "expected_goals_xg",
    "expected_assists_xa",
    "shots",
    "shots_on_target",
    "key_passes",
    "pass_accuracy",
    "tackles",
    "interceptions",
    "clearances",
    "saves",
    "distance_covered_km",
    "top_speed_kmh",
    "stamina_score",
    "offensive_contribution",
    "defensive_contribution",
    "possession_impact",
    "creativity_score",
    "player_rating",
]

LEAKAGE_COLUMNS = [
    "performance_score",
    "tournament_rating",
]

VOLUME_STATS = [
    "goals",
    "assists",
    "expected_goals_xg",
    "expected_assists_xa",
    "shots",
    "shots_on_target",
    "key_passes",
    "tackles",
    "interceptions",
    "clearances",
    "saves",
    "offensive_contribution",
    "defensive_contribution",
    "creativity_score",
]

MEAN_STATS = [
    "player_rating",
    "pass_accuracy",
    "distance_covered_km",
    "top_speed_kmh",
    "stamina_score",
    "possession_impact",
    "minutes_played",
]

CLUSTERING_BASE_COLUMNS = [
    "goal_overperformance",
    "assist_overperformance",
    "goal_contribution",
    "expected_goal_contribution",
    "goal_contribution_overperformance",
    "shot_accuracy",
    "player_rating",
    "minutes_played",
    "goals",
    "assists",
    "expected_goals_xg",
    "expected_assists_xa",
    "key_passes",
    "creativity_score",
    "offensive_contribution",
    "defensive_contribution",
    "possession_impact",
    "stamina_score",
    "tackles",
    "interceptions",
    "clearances",
    "saves",
]


def save_object(obj: Any, path: Path) -> None:
    """Save a Python object with joblib when available, otherwise pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if joblib is not None:
        joblib.dump(obj, path)
    else:
        with path.open("wb") as f:
            pickle.dump(obj, f)


def load_object(path: Path) -> Any:
    """Load a Python object saved by save_object."""
    if joblib is not None:
        return joblib.load(path)
    with path.open("rb") as f:
        return pickle.load(f)


def available_columns(df: pd.DataFrame, requested: list[str]) -> list[str]:
    """Return requested columns that are present in the dataframe."""
    return [col for col in requested if col in df.columns]


def load_dataset(path: Path = DATA_FILE) -> pd.DataFrame:
    """Load the raw CSV dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def inspect_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Print and return beginner-friendly dataset inspection details."""
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isna().sum().to_dict(),
        "number_of_teams": int(df["team"].nunique()) if "team" in df.columns else None,
        "number_of_players": int(df["player_name"].nunique())
        if "player_name" in df.columns
        else None,
        "number_of_matches": int(df["match_id"].nunique()) if "match_id" in df.columns else None,
    }

    print("\n=== Dataset inspection ===")
    print(f"Shape: {summary['shape']}")
    print(f"Columns: {summary['columns']}")
    print("\nData types:")
    print(pd.Series(summary["dtypes"]))
    print("\nMissing values:")
    print(pd.Series(summary["missing_values"]).sort_values(ascending=False).head(20))

    for col in ["team", "position", "match_result"]:
        if col in df.columns:
            print(f"\nValue counts for {col}:")
            print(df[col].value_counts().head(20))

    print(
        "\nCounts:",
        {
            "teams": summary["number_of_teams"],
            "players": summary["number_of_players"],
            "matches": summary["number_of_matches"],
        },
    )
    return summary


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean duplicate rows, numeric types, missing values, and infinite values."""
    cleaned = df.copy()

    before = len(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    print(f"\nRemoved {before - len(cleaned)} exact duplicate rows.")

    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)

    for col in cleaned.columns:
        if col in IDENTIFIER_COLUMNS:
            continue
        if cleaned[col].dtype == "object":
            numeric_version = pd.to_numeric(cleaned[col], errors="coerce")
            non_null_original = cleaned[col].notna().sum()
            non_null_numeric = numeric_version.notna().sum()
            if non_null_original > 0 and non_null_numeric / non_null_original >= 0.9:
                cleaned[col] = numeric_version

    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = [col for col in cleaned.columns if col not in numeric_cols]

    for col in numeric_cols:
        if cleaned[col].isna().any():
            median = cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(0 if pd.isna(median) else median)

    for col in text_cols:
        if cleaned[col].isna().any():
            modes = cleaned[col].mode(dropna=True)
            fill_value = modes.iloc[0] if len(modes) else "Unknown"
            cleaned[col] = cleaned[col].fillna(fill_value)

    print(f"Numeric columns after cleaning: {len(numeric_cols)}")
    print(f"Remaining missing values: {int(cleaned.isna().sum().sum())}")
    return cleaned


def select_supervised_columns(df: pd.DataFrame) -> list[str]:
    """Select available supervised features while excluding explicit leakage columns."""
    selected = available_columns(df, REQUESTED_SUPERVISED_COLUMNS)
    selected = [col for col in selected if col not in LEAKAGE_COLUMNS]
    excluded = [col for col in LEAKAGE_COLUMNS if col in df.columns]
    print("\nSelected supervised player performance columns:")
    print(selected)
    if excluded:
        print("\nExcluded leakage-risk columns:")
        print(excluded)
    return selected


def create_eda_charts(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> None:
    """Create exploratory analysis charts and save them as PNG files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    def save_current(name: str) -> None:
        plt.tight_layout()
        plt.savefig(output_dir / name, dpi=160, bbox_inches="tight")
        plt.close()

    if "match_result" in df.columns:
        result_df = (
            df[["match_id", "team", "match_result"]]
            .drop_duplicates()
            .query("match_result in ['W', 'D', 'L']")
        )
        plt.figure(figsize=(7, 4))
        sns.countplot(data=result_df, x="match_result", order=["W", "D", "L"])
        plt.title("Match Result Distribution")
        plt.xlabel("Result from team perspective")
        plt.ylabel("Team-match count")
        save_current("match_result_distribution.png")

    if {"position", "player_rating"}.issubset(df.columns):
        order = df.groupby("position")["player_rating"].mean().sort_values().index
        plt.figure(figsize=(8, 4))
        sns.barplot(data=df, x="position", y="player_rating", order=order, estimator="mean")
        plt.title("Average Player Rating by Position")
        plt.xlabel("Position")
        plt.ylabel("Average rating")
        save_current("avg_player_rating_by_position.png")

    if "position" in df.columns and {"goals", "expected_goals_xg"}.intersection(df.columns):
        metric_cols = available_columns(df, ["goals", "expected_goals_xg"])
        pos_goal = df.groupby("position")[metric_cols].mean().reset_index()
        melted = pos_goal.melt(id_vars="position", var_name="metric", value_name="average")
        plt.figure(figsize=(8, 4))
        sns.barplot(data=melted, x="position", y="average", hue="metric")
        plt.title("Average Goals and xG by Position")
        plt.xlabel("Position")
        plt.ylabel("Average per player-match")
        save_current("avg_goals_xg_by_position.png")

    if {"team", "player_rating"}.issubset(df.columns):
        team_rating = (
            df.groupby("team")["player_rating"].mean().sort_values(ascending=False).head(15)
        )
        plt.figure(figsize=(9, 6))
        sns.barplot(x=team_rating.values, y=team_rating.index)
        plt.title("Top Teams by Average Player Rating")
        plt.xlabel("Average player rating")
        plt.ylabel("Team")
        save_current("top_teams_avg_rating.png")

    attack_cols = available_columns(
        df, ["goals", "expected_goals_xg", "shots_on_target", "offensive_contribution"]
    )
    if "team" in df.columns and attack_cols:
        attack = df.groupby("team")[attack_cols].mean()
        attack["attacking_index"] = attack.rank(pct=True).mean(axis=1)
        attack = attack.sort_values("attacking_index", ascending=False).head(15)
        plt.figure(figsize=(9, 6))
        sns.barplot(data=attack.reset_index(), x="attacking_index", y="team")
        plt.title("Top Teams by Average Attacking Metrics")
        plt.xlabel("Attacking index")
        plt.ylabel("Team")
        save_current("top_teams_attacking_metrics.png")

    defensive_cols = available_columns(
        df, ["tackles", "interceptions", "clearances", "saves", "defensive_contribution"]
    )
    if "team" in df.columns and defensive_cols:
        defense = df.groupby("team")[defensive_cols].mean()
        defense["defensive_index"] = defense.rank(pct=True).mean(axis=1)
        defense = defense.sort_values("defensive_index", ascending=False).head(15)
        plt.figure(figsize=(9, 6))
        sns.barplot(data=defense.reset_index(), x="defensive_index", y="team")
        plt.title("Top Teams by Average Defensive Metrics")
        plt.xlabel("Defensive index")
        plt.ylabel("Team")
        save_current("top_teams_defensive_metrics.png")

    if {"expected_goals_xg", "goals"}.issubset(df.columns):
        plt.figure(figsize=(7, 5))
        sns.scatterplot(data=df.sample(min(len(df), 3000), random_state=42), x="expected_goals_xg", y="goals", alpha=0.35)
        plt.title("Expected Goals vs Actual Goals")
        plt.xlabel("Expected goals (xG)")
        plt.ylabel("Actual goals")
        save_current("expected_goals_vs_goals.png")

    if {"expected_assists_xa", "assists"}.issubset(df.columns):
        plt.figure(figsize=(7, 5))
        sns.scatterplot(data=df.sample(min(len(df), 3000), random_state=42), x="expected_assists_xa", y="assists", alpha=0.35)
        plt.title("Expected Assists vs Actual Assists")
        plt.xlabel("Expected assists (xA)")
        plt.ylabel("Actual assists")
        save_current("expected_assists_vs_assists.png")

    corr_cols = available_columns(
        df,
        [
            "minutes_played",
            "goals",
            "assists",
            "expected_goals_xg",
            "expected_assists_xa",
            "shots",
            "shots_on_target",
            "key_passes",
            "pass_accuracy",
            "tackles",
            "interceptions",
            "clearances",
            "saves",
            "player_rating",
            "offensive_contribution",
            "defensive_contribution",
            "possession_impact",
            "creativity_score",
        ],
    )
    if len(corr_cols) >= 2:
        plt.figure(figsize=(12, 9))
        sns.heatmap(df[corr_cols].corr(), cmap="vlag", center=0, linewidths=0.3)
        plt.title("Correlation Heatmap of Performance Metrics")
        save_current("correlation_heatmap.png")

    print(f"Saved EDA charts to {output_dir}")


def create_player_profiles(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Create one average performance profile per team/player/position."""
    group_cols = available_columns(df, ["team", "player_name", "position"])
    if len(group_cols) < 2:
        raise ValueError("Player profiles need at least team/player columns.")

    profile_feature_cols = available_columns(
        df,
        sorted(set(feature_cols + CLUSTERING_BASE_COLUMNS + VOLUME_STATS + MEAN_STATS)),
    )
    profiles = df.groupby(group_cols, as_index=False)[profile_feature_cols].mean()
    appearances = df.groupby(group_cols).size().reset_index(name="matches_in_dataset")
    profiles = profiles.merge(appearances, on=group_cols, how="left")
    return profiles


def create_lineup_features(
    lineup_df: pd.DataFrame,
    validate_11: bool = False,
) -> dict[str, float]:
    """
    Aggregate a lineup or team-match dataframe into lineup-level features.

    Set validate_11=True for custom predictions where the user must choose
    exactly 11 players.
    """
    if validate_11 and len(lineup_df) != 11:
        raise ValueError(f"Expected exactly 11 players, but received {len(lineup_df)}.")
    if lineup_df.empty:
        raise ValueError("Lineup dataframe is empty.")

    features: dict[str, float] = {}
    for col in available_columns(lineup_df, VOLUME_STATS):
        values = pd.to_numeric(lineup_df[col], errors="coerce").fillna(0)
        features[f"total_{col}"] = float(values.sum())
    for col in available_columns(lineup_df, MEAN_STATS):
        values = pd.to_numeric(lineup_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        features[f"avg_{col}"] = float(values.mean()) if values.notna().any() else 0.0
    return features


def create_matchup_features(
    team_a_features: dict[str, float],
    team_b_features: dict[str, float],
) -> dict[str, float]:
    """Create Team A, Team B, and Team A-minus-Team B difference features."""
    matchup: dict[str, float] = {}
    all_feature_names = sorted(set(team_a_features) | set(team_b_features))

    for feature in all_feature_names:
        a_value = float(team_a_features.get(feature, 0.0))
        b_value = float(team_b_features.get(feature, 0.0))
        matchup[f"team_a_{feature}"] = a_value
        matchup[f"team_b_{feature}"] = b_value
        matchup[f"diff_{feature}"] = a_value - b_value

    return matchup


def infer_team_result(team_df: pd.DataFrame) -> str | None:
    """Infer W/D/L for one team within one match."""
    if "match_result" in team_df.columns:
        result = str(team_df["match_result"].iloc[0]).strip().upper()
        if result in {"W", "D", "L"}:
            return result

    if {"goals_team", "goals_opponent"}.issubset(team_df.columns):
        goals_for = pd.to_numeric(team_df["goals_team"], errors="coerce").iloc[0]
        goals_against = pd.to_numeric(team_df["goals_opponent"], errors="coerce").iloc[0]
        if pd.isna(goals_for) or pd.isna(goals_against):
            return None
        if goals_for > goals_against:
            return "W"
        if goals_for < goals_against:
            return "L"
        return "D"

    return None


def build_matchup_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Build one Team A vs Team B row and one reverse row for each match."""
    if not {"match_id", "team"}.issubset(df.columns):
        raise ValueError("Training data requires match_id and team columns.")

    rows: list[dict[str, Any]] = []
    skipped_matches = 0

    for match_id, match_df in df.groupby("match_id", sort=False):
        teams = list(match_df["team"].dropna().unique())
        if len(teams) != 2:
            skipped_matches += 1
            continue

        team_data = {}
        for team in teams:
            team_df = match_df[match_df["team"] == team]
            result = infer_team_result(team_df)
            if result is None:
                continue
            team_data[team] = {
                "features": create_lineup_features(team_df, validate_11=False),
                "result": result,
            }

        if len(team_data) != 2:
            skipped_matches += 1
            continue

        team_a, team_b = teams
        forward = create_matchup_features(
            team_data[team_a]["features"], team_data[team_b]["features"]
        )
        forward.update({"match_id": match_id, "team_a": team_a, "team_b": team_b, "result": team_data[team_a]["result"]})
        rows.append(forward)

        reverse = create_matchup_features(
            team_data[team_b]["features"], team_data[team_a]["features"]
        )
        reverse.update({"match_id": match_id, "team_a": team_b, "team_b": team_a, "result": team_data[team_b]["result"]})
        rows.append(reverse)

    matchup_df = pd.DataFrame(rows)
    if matchup_df.empty:
        raise ValueError("No matchup training rows could be created.")

    metadata_cols = ["match_id", "team_a", "team_b", "result"]
    feature_cols = [col for col in matchup_df.columns if col not in metadata_cols]
    matchup_df[feature_cols] = matchup_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"\nCreated {len(matchup_df)} matchup training rows.")
    if skipped_matches:
        print(f"Skipped {skipped_matches} matches because they did not have exactly two teams or a result.")
    print("Result distribution:")
    print(matchup_df["result"].value_counts())
    return matchup_df[metadata_cols + feature_cols]


def split_supervised_data(matchup_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Split matchup data into train and test sets."""
    feature_cols = [
        col
        for col in matchup_df.columns
        if col not in ["match_id", "team_a", "team_b", "result"]
    ]
    X = matchup_df[feature_cols]
    y = matchup_df["result"]

    stratify = y if y.value_counts().min() >= 2 else None
    return (*train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify), feature_cols)


def train_supervised_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict[str, Any], pd.DataFrame, str]:
    """Train, evaluate, and choose supervised classification models."""
    models: dict[str, Any] = {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, solver="lbfgs")),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    fitted_models: dict[str, Any] = {}
    metric_rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        model_log_loss = log_loss(y_test, proba, labels=model.classes_) if proba is not None else np.nan

        print(f"{name} accuracy: {accuracy:.3f}")
        if not pd.isna(model_log_loss):
            print(f"{name} log loss: {model_log_loss:.3f}")
        print(classification_report(y_test, preds, zero_division=0))
        print("Confusion matrix:")
        print(pd.DataFrame(confusion_matrix(y_test, preds, labels=model.classes_), index=model.classes_, columns=model.classes_))

        fitted_models[name] = model
        reports[name] = {
            "classification_report": classification_report(y_test, preds, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, preds, labels=model.classes_).tolist(),
            "classes": list(model.classes_),
        }
        metric_rows.append(
            {
                "model": name,
                "accuracy": accuracy,
                "log_loss": model_log_loss,
            }
        )

    metrics = pd.DataFrame(metric_rows)
    metrics["selection_log_loss"] = metrics["log_loss"].fillna(np.inf)
    metrics = metrics.sort_values(["selection_log_loss", "accuracy"], ascending=[True, False])
    final_model_name = str(metrics.iloc[0]["model"])

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.drop(columns=["selection_log_loss"]).to_csv(OUTPUTS_DIR / "supervised_model_metrics.csv", index=False)
    with (OUTPUTS_DIR / "supervised_model_reports.json").open("w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)

    print(f"\nSelected final supervised model: {final_model_name}")
    return fitted_models, metrics.drop(columns=["selection_log_loss"]), final_model_name


def plot_feature_importance(
    models: dict[str, Any],
    final_model_name: str,
    feature_cols: list[str],
    output_path: Path = FIGURES_DIR / "top_feature_importances.png",
) -> pd.DataFrame:
    """Plot top feature importances for the final tree model, or the best available tree model."""
    candidates = [final_model_name] + [name for name in ["Gradient Boosting", "Random Forest"] if name != final_model_name]
    chosen_name = None
    importances = None

    for name in candidates:
        model = models.get(name)
        if model is not None and hasattr(model, "feature_importances_"):
            chosen_name = name
            importances = model.feature_importances_
            break

    if importances is None:
        print("No tree-based feature importance available.")
        return pd.DataFrame()

    importance_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    importance_df = importance_df.sort_values("importance", ascending=False)
    top = importance_df.head(15).sort_values("importance")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7))
    sns.barplot(data=top, x="importance", y="feature")
    plt.title(f"Top 15 Feature Importances ({chosen_name})")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()

    importance_df.to_csv(OUTPUTS_DIR / "feature_importances.csv", index=False)
    print(f"Saved feature importance chart to {output_path}")
    return importance_df


def predict_custom_match(
    team_a: str,
    team_b: str,
    team_a_players: list[str],
    team_b_players: list[str],
    player_profiles: pd.DataFrame,
    model: Any,
    training_columns: list[str],
) -> dict[str, Any]:
    """Predict Team A win, draw, and Team B win probabilities for selected lineups."""
    if len(team_a_players) != 11:
        raise ValueError(f"{team_a} must have exactly 11 selected players.")
    if len(team_b_players) != 11:
        raise ValueError(f"{team_b} must have exactly 11 selected players.")

    team_a_lineup = player_profiles[
        (player_profiles["team"] == team_a) & (player_profiles["player_name"].isin(team_a_players))
    ].drop_duplicates(subset=["player_name"])
    team_b_lineup = player_profiles[
        (player_profiles["team"] == team_b) & (player_profiles["player_name"].isin(team_b_players))
    ].drop_duplicates(subset=["player_name"])

    if len(team_a_lineup) != 11:
        missing = sorted(set(team_a_players) - set(team_a_lineup["player_name"]))
        raise ValueError(f"Could not find 11 selected players for {team_a}. Missing: {missing}")
    if len(team_b_lineup) != 11:
        missing = sorted(set(team_b_players) - set(team_b_lineup["player_name"]))
        raise ValueError(f"Could not find 11 selected players for {team_b}. Missing: {missing}")

    team_a_features = create_lineup_features(team_a_lineup, validate_11=True)
    team_b_features = create_lineup_features(team_b_lineup, validate_11=True)
    matchup_features = create_matchup_features(team_a_features, team_b_features)

    X_custom = pd.DataFrame([matchup_features]).reindex(columns=training_columns, fill_value=0)
    probabilities = model.predict_proba(X_custom)[0]
    probability_map = {label: float(prob) for label, prob in zip(model.classes_, probabilities)}

    return {
        "team_a": team_a,
        "team_b": team_b,
        "team_a_win_probability": probability_map.get("W", 0.0),
        "draw_probability": probability_map.get("D", 0.0),
        "team_b_win_probability": probability_map.get("L", 0.0),
        "matchup_features": matchup_features,
    }


def add_overperformance_features(profiles: pd.DataFrame) -> pd.DataFrame:
    """Add engineered overperformance columns to player profiles when inputs exist."""
    enriched = profiles.copy()

    if {"goals", "expected_goals_xg"}.issubset(enriched.columns):
        enriched["goal_overperformance"] = enriched["goals"] - enriched["expected_goals_xg"]
    if {"assists", "expected_assists_xa"}.issubset(enriched.columns):
        enriched["assist_overperformance"] = enriched["assists"] - enriched["expected_assists_xa"]
    if {"goals", "assists"}.issubset(enriched.columns):
        enriched["goal_contribution"] = enriched["goals"] + enriched["assists"]
    if {"expected_goals_xg", "expected_assists_xa"}.issubset(enriched.columns):
        enriched["expected_goal_contribution"] = enriched["expected_goals_xg"] + enriched["expected_assists_xa"]
    if {"goal_contribution", "expected_goal_contribution"}.issubset(enriched.columns):
        enriched["goal_contribution_overperformance"] = (
            enriched["goal_contribution"] - enriched["expected_goal_contribution"]
        )
    if {"shots_on_target", "shots"}.issubset(enriched.columns):
        shots = enriched["shots"].replace(0, np.nan)
        enriched["shot_accuracy"] = (enriched["shots_on_target"] / shots).replace([np.inf, -np.inf], np.nan).fillna(0)

    return enriched


def choose_k(k_results: pd.DataFrame) -> int:
    """Choose an interpretable k, preferring 4 or 5 if their silhouette is close."""
    valid = k_results.dropna(subset=["silhouette_score"]).copy()
    if valid.empty:
        return int(k_results.iloc[0]["k"])

    best_score = valid["silhouette_score"].max()
    for preferred_k in [5, 4]:
        preferred = valid[valid["k"] == preferred_k]
        if not preferred.empty and float(preferred["silhouette_score"].iloc[0]) >= best_score - 0.03:
            return preferred_k

    return int(valid.sort_values("silhouette_score", ascending=False).iloc[0]["k"])


def assign_cluster_labels(clustered: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign readable labels after inspecting average cluster statistics."""
    profile = clustered.groupby("cluster")[feature_cols].mean()
    z_profile = (profile - profile.mean()) / profile.std(ddof=0).replace(0, np.nan)
    z_profile = z_profile.fillna(0)

    label_feature_map = {
        "Efficient Finishers": [
            "goal_overperformance",
            "goal_contribution_overperformance",
            "goals",
            "shot_accuracy",
        ],
        "Creative Playmakers": [
            "creativity_score",
            "key_passes",
            "assists",
            "expected_assists_xa",
            "possession_impact",
        ],
        "Defensive Specialists": [
            "defensive_contribution",
            "tackles",
            "interceptions",
            "clearances",
            "saves",
        ],
        "Underperforming Attackers": [
            "expected_goal_contribution",
            "expected_goals_xg",
            "expected_assists_xa",
        ],
        "Low-Impact / Low-Minute Players": [
            "minutes_played",
            "player_rating",
            "goal_contribution",
            "offensive_contribution",
        ],
    }

    scores: dict[int, dict[str, float]] = {}
    for cluster_id in z_profile.index:
        cluster_scores: dict[str, float] = {}
        for label, cols in label_feature_map.items():
            existing = [col for col in cols if col in z_profile.columns]
            if not existing:
                cluster_scores[label] = -999.0
                continue
            value = float(z_profile.loc[cluster_id, existing].mean())
            if label == "Underperforming Attackers" and "goal_contribution_overperformance" in z_profile.columns:
                value += float(-z_profile.loc[cluster_id, "goal_contribution_overperformance"])
            if label == "Low-Impact / Low-Minute Players":
                value = float(-z_profile.loc[cluster_id, existing].mean())
            cluster_scores[label] = value
        scores[int(cluster_id)] = cluster_scores

    used_labels: set[str] = set()
    cluster_labels: dict[int, str] = {}
    for cluster_id, cluster_scores in sorted(scores.items()):
        ranked = sorted(cluster_scores.items(), key=lambda item: item[1], reverse=True)
        chosen = next((label for label, _ in ranked if label not in used_labels), ranked[0][0])
        used_labels.add(chosen)
        cluster_labels[cluster_id] = chosen

    labeled = clustered.copy()
    labeled["cluster_label"] = labeled["cluster"].map(cluster_labels)
    summary = profile.reset_index()
    summary["cluster_label"] = summary["cluster"].map(cluster_labels)
    return labeled, summary


def train_player_clustering(
    player_profiles: pd.DataFrame,
) -> tuple[pd.DataFrame, KMeans, StandardScaler, list[str], pd.DataFrame, pd.DataFrame]:
    """Train K-Means clustering and return labeled player profiles."""
    profiles = add_overperformance_features(player_profiles)
    clustering_features = available_columns(profiles, CLUSTERING_BASE_COLUMNS)
    clustering_features = [
        col for col in clustering_features if pd.api.types.is_numeric_dtype(profiles[col])
    ]
    if len(clustering_features) < 2:
        raise ValueError("Need at least two numeric clustering features.")

    X_cluster = profiles[clustering_features].replace([np.inf, -np.inf], np.nan)
    X_cluster = X_cluster.fillna(X_cluster.median(numeric_only=True)).fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    max_k = min(8, len(profiles) - 1)
    k_values = list(range(2, max_k + 1))
    results = []
    fitted_by_k: dict[int, KMeans] = {}

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        fitted_by_k[k] = model
        sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else np.nan
        results.append({"k": k, "inertia": model.inertia_, "silhouette_score": sil})

    k_results = pd.DataFrame(results)
    final_k = choose_k(k_results)
    final_model = fitted_by_k[final_k]
    profiles["cluster"] = final_model.predict(X_scaled)
    labeled_profiles, cluster_summary = assign_cluster_labels(profiles, clustering_features)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    k_results.to_csv(OUTPUTS_DIR / "kmeans_k_selection.csv", index=False)
    cluster_summary.to_csv(OUTPUTS_DIR / "cluster_summary.csv", index=False)

    plot_clustering_outputs(X_scaled, labeled_profiles, k_results, cluster_summary, clustering_features)

    print(f"\nSelected k for K-Means: {final_k}")
    print("Cluster labels:")
    print(cluster_summary[["cluster", "cluster_label"]])
    return labeled_profiles, final_model, scaler, clustering_features, k_results, cluster_summary


def plot_clustering_outputs(
    X_scaled: np.ndarray,
    labeled_profiles: pd.DataFrame,
    k_results: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    clustering_features: list[str],
) -> None:
    """Save elbow, silhouette, PCA scatter, and cluster profile charts."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    sns.lineplot(data=k_results, x="k", y="inertia", marker="o")
    plt.title("K-Means Elbow Curve")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "kmeans_elbow_curve.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.lineplot(data=k_results, x="k", y="silhouette_score", marker="o")
    plt.title("K-Means Silhouette Scores")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "kmeans_silhouette_scores.png", dpi=160, bbox_inches="tight")
    plt.close()

    pca = PCA(n_components=2, random_state=42)
    pca_points = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(
        {
            "pca_1": pca_points[:, 0],
            "pca_2": pca_points[:, 1],
            "cluster": labeled_profiles["cluster"].astype(str),
            "cluster_label": labeled_profiles["cluster_label"],
        }
    )
    pca_df.to_csv(OUTPUTS_DIR / "player_cluster_pca.csv", index=False)

    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=pca_df, x="pca_1", y="pca_2", hue="cluster_label", alpha=0.75)
    plt.title("Player Clusters Visualized with PCA")
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.legend(title="Cluster label", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "player_clusters_pca.png", dpi=160, bbox_inches="tight")
    plt.close()

    heatmap_cols = available_columns(
        cluster_summary,
        [
            "goals",
            "assists",
            "expected_goals_xg",
            "expected_assists_xa",
            "goal_overperformance",
            "assist_overperformance",
            "goal_contribution_overperformance",
            "shot_accuracy",
            "player_rating",
            "creativity_score",
            "offensive_contribution",
            "defensive_contribution",
            "minutes_played",
        ],
    )
    if len(heatmap_cols) >= 2:
        summary_indexed = cluster_summary.set_index("cluster_label")[heatmap_cols]
        scaled_summary = (summary_indexed - summary_indexed.mean()) / summary_indexed.std(ddof=0).replace(0, np.nan)
        plt.figure(figsize=(12, 6))
        sns.heatmap(scaled_summary.fillna(0), cmap="vlag", center=0, annot=False)
        plt.title("Cluster Profile Heatmap (Standardized Cluster Averages)")
        plt.xlabel("Feature")
        plt.ylabel("Cluster label")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "cluster_profile_heatmap.png", dpi=160, bbox_inches="tight")
        plt.close()


def choose_example_lineups(player_profiles: pd.DataFrame) -> tuple[str, str, list[str], list[str]]:
    """Pick two teams and 11 players per team for a demo prediction."""
    team_counts = player_profiles.groupby("team")["player_name"].nunique().sort_values(ascending=False)
    eligible_teams = team_counts[team_counts >= 11].index.tolist()
    if len(eligible_teams) < 2:
        raise ValueError("Need at least two teams with 11 players.")

    team_a, team_b = eligible_teams[:2]

    def top_11(team: str) -> list[str]:
        team_players = player_profiles[player_profiles["team"] == team].copy()
        sort_cols = [col for col in ["matches_in_dataset", "minutes_played", "player_rating"] if col in team_players.columns]
        team_players = team_players.sort_values(sort_cols, ascending=[False] * len(sort_cols))
        return team_players["player_name"].drop_duplicates().head(11).tolist()

    return team_a, team_b, top_11(team_a), top_11(team_b)


def run_workflow() -> dict[str, Any]:
    """Run the full project workflow and save all requested artifacts."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    raw_df = load_dataset(DATA_FILE)
    inspection = inspect_dataset(raw_df)

    print("\nCleaning data...")
    df = clean_data(raw_df)

    print("\nCreating EDA charts...")
    create_eda_charts(df)

    supervised_columns = select_supervised_columns(df)

    print("\nCreating player profiles...")
    player_profiles = create_player_profiles(df, supervised_columns)
    player_profiles_path = DATA_DIR / "player_profiles.csv"
    player_profiles.to_csv(player_profiles_path, index=False)
    print(f"Saved player profiles to {player_profiles_path}")

    print("\nBuilding matchup training data...")
    matchup_training = build_matchup_training_data(df)
    matchup_path = DATA_DIR / "matchup_training_data.csv"
    matchup_training.to_csv(matchup_path, index=False)
    print(f"Saved matchup training data to {matchup_path}")

    print("\nTraining supervised models...")
    X_train, X_test, y_train, y_test, training_columns = split_supervised_data(matchup_training)
    models, metrics, final_model_name = train_supervised_models(X_train, X_test, y_train, y_test)
    final_model = models[final_model_name]

    save_object(final_model, MODELS_DIR / "final_match_model.pkl")
    save_object(training_columns, MODELS_DIR / "training_columns.pkl")
    plot_feature_importance(models, final_model_name, training_columns)

    print("\nRunning example custom match prediction...")
    team_a, team_b, team_a_players, team_b_players = choose_example_lineups(player_profiles)
    example_prediction = predict_custom_match(
        team_a,
        team_b,
        team_a_players,
        team_b_players,
        player_profiles,
        final_model,
        training_columns,
    )
    print(
        f"{team_a} win: {example_prediction['team_a_win_probability']:.1%}, "
        f"draw: {example_prediction['draw_probability']:.1%}, "
        f"{team_b} win: {example_prediction['team_b_win_probability']:.1%}"
    )

    print("\nTraining player clustering model...")
    clustered_profiles, kmeans_model, clustering_scaler, clustering_features, k_results, cluster_summary = train_player_clustering(player_profiles)
    clustered_profiles_path = DATA_DIR / "player_profiles_with_clusters.csv"
    clustered_profiles.to_csv(clustered_profiles_path, index=False)

    save_object(kmeans_model, MODELS_DIR / "kmeans_model.pkl")
    save_object(clustering_scaler, MODELS_DIR / "clustering_scaler.pkl")
    save_object(clustering_features, MODELS_DIR / "clustering_features.pkl")

    summary = {
        "inspection": inspection,
        "supervised_columns": supervised_columns,
        "training_columns_count": len(training_columns),
        "final_model_name": final_model_name,
        "supervised_metrics": metrics.to_dict(orient="records"),
        "example_prediction": {
            key: value
            for key, value in example_prediction.items()
            if key != "matchup_features"
        },
        "clustering_features": clustering_features,
        "k_selection": k_results.to_dict(orient="records"),
        "cluster_labels": cluster_summary[["cluster", "cluster_label"]].to_dict(orient="records"),
    }
    with (OUTPUTS_DIR / "workflow_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nWorkflow complete.")
    return summary


if __name__ == "__main__":
    run_workflow()
