"""Small Altair figures shared by saved reports and the Streamlit UI."""

from __future__ import annotations

import altair as alt
import pandas as pd

MODEL_COLOURS = alt.Scale(domain=["B0", "B1", "M1", "M2"],
                         range=["#64748b", "#2563eb", "#059669", "#d97706"])


def calibration_chart(reliability: pd.DataFrame) -> alt.FacetChart:
    """Show class-wise pooled calibration, sample counts, and sampling uncertainty."""
    # Embed all rows rather than having Altair serialize a named runtime dataset.
    ideal = [{"outcome": outcome, "model": "Ideal", "predicted": probability, "observed": probability}
             for outcome in sorted(reliability.outcome.unique()) for probability in (0.0, 1.0)]
    base = alt.Chart(alt.Data(values=[*reliability.to_dict("records"), *ideal]))
    diagonal = base.transform_filter("datum.model === 'Ideal'").mark_line(
        color="#9ca3af", strokeDash=[5, 5]
    ).encode(x="predicted:Q", y="observed:Q")
    measured = base.transform_filter("datum.model !== 'Ideal'")
    curves = measured.mark_line(point=True).encode(
        x=alt.X("predicted:Q", title="Predicted probability", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("observed:Q", title="Observed frequency", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("model:N", title="Model", scale=MODEL_COLOURS),
        tooltip=["model:N", "outcome:N", "count:Q", alt.Tooltip("predicted:Q", format=".3f"),
                 alt.Tooltip("observed:Q", format=".3f")],
    )
    intervals = measured.mark_rule(opacity=0.25).encode(
        x="predicted:Q", y="lower:Q", y2="upper:Q", color=alt.Color("model:N", scale=MODEL_COLOURS),
    )
    return (diagonal + intervals + curves).properties(width=230, height=240).facet(
        column=alt.Column("outcome:N", title=None)
    ).properties(title="World Cup reliability: pooled test matches")


def comparison_chart(means: pd.DataFrame) -> alt.Chart:
    """Plot candidate log loss without turning the Elo accuracy reference into a model."""
    frame = means.dropna(subset=["log_loss"])
    return alt.Chart(alt.Data(values=frame.to_dict("records"))).mark_bar().encode(
        x=alt.X("log_loss:Q", title="Mean tournament log loss (lower is better)", scale=alt.Scale(zero=True)),
        y=alt.Y("name:N", sort="x", title=None), color=alt.Color("model:N", legend=None, scale=MODEL_COLOURS),
        tooltip=["model:N", alt.Tooltip("log_loss:Q", format=".4f")],
    ).properties(height=170, title="Comparison against the baselines")
