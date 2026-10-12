"""International football forecasts backed by precomputed temporal evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.inference import PredictionAssets, load_assets, predict_match
from src.models import MODEL_NAMES

ROOT = Path(__file__).resolve().parent


@st.cache_resource
def assets() -> PredictionAssets:
    """Cache the saved bundle without training or downloading on startup."""
    return load_assets(ROOT)


@st.cache_data
def report_table() -> pd.DataFrame:
    """Read tournament metrics directly from the saved report."""
    table = pd.read_csv(ROOT / "reports/metrics.csv")
    return table.loc[table.aggregation.eq("tournament")]


@st.cache_data
def saved_chart(name: str) -> dict:
    """Read an immutable Vega-Lite figure produced by the offline pipeline."""
    return json.loads((ROOT / "reports" / f"{name}.vl.json").read_text())


def predict_page(bundle: PredictionAssets) -> None:
    """Let the user choose a fixture and inspect its probabilities and Elo history."""
    st.title("What could happen?")
    st.write("Choose a fixture to see three possible outcomes, with probabilities for each.")
    ratings = bundle.ratings
    with st.expander("Fixture settings"):
        historical = st.checkbox("Include teams inactive in recent years", value=False)
        importance = st.selectbox("Match type", ["World Cup finals", "Continental championship",
                                                "Qualifier", "Other tournament", "Friendly"])
    k = {"World Cup finals": 60, "Continental championship": 50, "Qualifier": 40,
         "Other tournament": 30, "Friendly": 20}[importance]
    active_since = pd.Timestamp(bundle.metadata["data_through"]) - pd.DateOffset(years=5)
    teams = sorted(ratings.index if historical else ratings.loc[ratings.last_match_date.ge(active_since)].index)
    first, second = st.columns(2)
    home = first.selectbox("First-listed / home team", teams, index=teams.index("Argentina") if "Argentina" in teams else 0)
    opponents = [team for team in teams if team != home]
    away = second.selectbox("Away team", opponents, index=opponents.index("France") if "France" in opponents else 0)
    neutral = st.toggle("Neutral venue", value=True)
    st.caption("At a nonneutral venue, the first-listed team receives home advantage.")
    probabilities = predict_match(bundle, home, away, neutral, k)
    win, draw, loss = st.columns(3)
    win.metric(f"{home} win", f"{probabilities['win']:.1%}")
    draw.metric("Draw", f"{probabilities['draw']:.1%}")
    loss.metric(f"{away} win", f"{probabilities['loss']:.1%}")
    st.caption("Scores include extra time. A penalty shootout is a draw here; these are not advancement probabilities.")
    st.divider()
    left, right = st.columns(2)
    for column, team in ((left, home), (right, away)):
        state = ratings.loc[team]
        column.metric(f"{team} · current Elo", f"{state.elo:,.0f}")
        column.caption(f"Last played: {state.last_match_date:%b %d, %Y} · "
                       f"Recent form: {state.ppg_5:.2f} points per match")
    st.subheader("How their ratings changed")
    full_history = st.checkbox("Show full match history", value=False)
    history = bundle.elo_history.loc[bundle.elo_history.team.isin([home, away])]
    if not full_history:
        history = history.loc[history.date.ge(pd.Timestamp(bundle.metadata["data_through"]) - pd.DateOffset(years=10))]
    chart = alt.Chart(history).mark_line(interpolate="step-after").encode(
        x=alt.X("date:T", title=None), y=alt.Y("elo:Q", title="Elo rating", scale=alt.Scale(zero=False)),
        color=alt.Color("team:N", title=None), tooltip=["team:N", "date:T", alt.Tooltip("elo:Q", format=".1f")],
    ).properties(height=300)
    st.altair_chart(chart, width="stretch")
    st.caption(f"Saved data through {bundle.metadata['data_through']}. Ratings are estimates built from "
               "match results. This model has no player, injury, or lineup information.")


def evaluation_page(bundle: PredictionAssets) -> None:
    """Expose measured results, baseline comparisons, and the saved calibration figure."""
    st.title("How good is it?")
    metrics, selection = bundle.metrics, bundle.metrics["selection"]
    st.write(f"Tested on {metrics['test_matches']} matches across {metrics['test_tournaments']} World Cups. "
             "Each model was trained before that tournament started.")
    st.info(f"Selected: {selection['model']} · {selection['name']}. Mean log loss "
            f"{selection['mean_log_loss']:.4f}; advantage over {selection['runner_up']}: "
            f"{selection['margin_log_loss']:.4f}. "
            + ("A baseline won." if selection["baseline_won"] else "The winning margin is small."))
    st.write("**Log loss:** rewards probabilities that put more weight on the outcome that happened; lower is better.")
    st.write("**Brier score:** squared error across all three probabilities; lower is better.")
    st.write("**Accuracy:** how often the most likely outcome was correct; higher is better.")
    means = pd.DataFrame(metrics["mean_metrics"]).rename(columns={
        "model": "Model", "name": "Description", "log_loss": "Log loss",
        "brier_score": "Brier score", "accuracy": "Accuracy",
    })
    st.dataframe(means, hide_index=True, column_config={
        "Log loss": st.column_config.NumberColumn(format="%.4f"),
        "Brier score": st.column_config.NumberColumn(format="%.4f"),
        "Accuracy": st.column_config.NumberColumn(format="percent"),
    }, width="stretch")
    st.caption("Each tournament has equal weight in the means. The higher-Elo pick only supplies a label, "
               "so it has no probability scores. Brier score is the unscaled sum over classes.")
    st.vega_lite_chart(saved_chart("model_comparison"), width="stretch")
    st.subheader("Tournament by tournament")
    table = report_table()[["year", "model", "test_matches", "log_loss", "brier_score", "accuracy"]]
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "year": st.column_config.NumberColumn("World Cup", format="%d"),
        "log_loss": st.column_config.NumberColumn(format="%.4f"),
        "brier_score": st.column_config.NumberColumn(format="%.4f"),
        "accuracy": st.column_config.NumberColumn(format="percent"),
    })
    st.subheader("Do the probabilities match reality?")
    st.write("Points near the diagonal mean predicted probabilities agreed with how often outcomes happened.")
    st.vega_lite_chart(saved_chart("calibration"), width="stretch")
    st.caption("All test tournaments pooled. Hover for bin counts. Vertical bars show sampling uncertainty; "
               "sparse bins can be noisy. Calibration was fitted only on earlier training matches.")
    diagnostics = metrics["diagnostics"][selection["model"]]
    st.warning(f"The selected model picked draw as its most likely outcome {diagnostics['predicted_draws']} times, "
               f"despite {diagnostics['actual_draws']} actual draws. Draw probabilities still contribute to its scores.")
    with st.expander("What this evaluation can and cannot tell us"):
        st.write("The model stays fixed during each tournament; Elo and recent form update between match dates. "
                 "This evaluates forecasts before each match. It does not simulate predicting every fixture on opening day.")
        st.write("These backtests also selected the model, so they are not an independent final test. "
                 "Small tournament samples and a narrow winning margin limit claims of superiority.")
        st.write("Team histories follow the source's naming and successor conventions. Penalty shootouts count as draws. "
                 "There is no information about players, injuries, lineups, or tactics.")
        st.write(f"Data source: martj42/international_results. Last match: {metrics['data']['last_match']}.")
    st.download_button("Download tournament results", (ROOT / "reports/metrics.csv").read_bytes(),
                       file_name="world_cup_backtest.csv", mime="text/csv")


def main() -> None:
    """Start the read-only application with useful missing-artifact feedback."""
    st.set_page_config(page_title="International Football Predictor", page_icon="⚽", layout="wide")
    st.sidebar.title("Football, with evidence")
    view = st.sidebar.radio("Page", ["Predict", "How good is it?"], label_visibility="collapsed")
    try:
        bundle = assets()
    except (OSError, ValueError, KeyError) as error:
        st.error(str(error))
        st.code("python scripts/run_all.py")
        st.stop()
    st.sidebar.caption(f"Saved model: {bundle.model.model_id} · {MODEL_NAMES[bundle.model.model_id]}")
    if view == "Predict":
        predict_page(bundle)
    else:
        evaluation_page(bundle)


if __name__ == "__main__":
    main()
