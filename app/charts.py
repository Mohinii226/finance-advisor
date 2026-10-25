"""Plotly figures for the Streamlit app. Pure functions: DataFrame / dict in, plotly Figure out."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

PALETTE = ["#2E86AB", "#F6AE2D", "#33658A", "#F26419", "#55A630", "#9B5DE5", "#8D99AE", "#E5383B"]
LAYOUT = dict(template="plotly_white", margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2))


def _fig(title: str, height: int = 340) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, height=height, **LAYOUT)
    return fig


def spend_trend(history: pd.DataFrame, forecast: dict | None = None) -> go.Figure:
    """Monthly spending with the customer's usual level and (optionally) next month's forecast and range."""
    fig = _fig("Monthly spending")
    fig.add_trace(go.Scatter(x=history["month"], y=history["expenses"], mode="lines+markers", name="Spending",
                             line=dict(color=PALETTE[0], width=3)))
    if len(history):
        usual = float(history["expenses"].median())
        fig.add_trace(go.Scatter(x=[history["month"].min(), history["month"].max()], y=[usual, usual], mode="lines",
                                 name="Usual (median)", line=dict(color=PALETTE[6], dash="dash")))
    if forecast and forecast.get("expected_spend") is not None:
        x = pd.Timestamp(history["month"].max()) + pd.DateOffset(months=1) if len(history) else None
        lo, hi, p = forecast.get("likely_range_low"), forecast.get("likely_range_high"), float(forecast["expected_spend"])
        err = dict(type="data", symmetric=False, array=[(hi - p) if hi is not None else 0], arrayminus=[(p - lo) if lo is not None else 0])
        fig.add_trace(go.Scatter(x=[x], y=[p], mode="markers", name="Forecast (80% range)", error_y=err,
                                 marker=dict(color=PALETTE[3], size=11, symbol="diamond")))
    fig.update_yaxes(title="Rs", rangemode="tozero")
    return fig


def category_donut(top: list[dict]) -> go.Figure:
    fig = _fig("Where the money went", 340)
    fig.add_trace(go.Pie(labels=[t["category"] for t in top], values=[t["spend"] for t in top], hole=0.55,
                         marker=dict(colors=PALETTE), sort=False))
    return fig


def category_stack(long_df: pd.DataFrame) -> go.Figure:
    fig = _fig("Spending by category, by month", 380)
    for i, (cat, g) in enumerate(long_df.groupby("category")):
        fig.add_trace(go.Bar(x=g["month"], y=g["spend"], name=cat, marker_color=PALETTE[i % len(PALETTE)]))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="Rs")
    return fig


def weekday_bar(df: pd.DataFrame) -> go.Figure:
    fig = _fig("Day-to-day spending by weekday (last 6 months)", 300)
    fig.add_trace(go.Bar(x=df["day"], y=df["spend"], marker_color=[PALETTE[3] if d in ("Sat", "Sun") else PALETTE[0] for d in df["day"]],
                         name="Spending"))
    fig.update_yaxes(title="Rs")
    return fig


def seasonal_line(df: pd.DataFrame) -> go.Figure:
    fig = _fig("Seasonal pattern (1.0 = your average month)", 300)
    fig.add_trace(go.Scatter(x=df["month_name"], y=df["discretionary_index"], mode="lines+markers", name="Discretionary spending",
                             line=dict(color=PALETTE[3], width=3)))
    fig.add_trace(go.Scatter(x=df["month_name"], y=df["expense_index"], mode="lines", name="All spending", line=dict(color=PALETTE[6], dash="dot")))
    fig.add_hline(y=1.0, line_dash="dash", line_color="#BBBBBB")
    return fig


def health_gauge(total: float, band: str) -> go.Figure:
    fig = _fig("Financial health score", 280)
    fig.add_trace(go.Indicator(mode="gauge+number", value=float(total), title={"text": band},
                               gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=PALETTE[0]),
                                          steps=[dict(range=[0, 50], color="#F9D9D6"), dict(range=[50, 65], color="#FCEBC5"),
                                                 dict(range=[65, 80], color="#DDEFD3"), dict(range=[80, 100], color="#C4E5C0")])))
    return fig


def components_bar(components: dict) -> go.Figure:
    names = list(components)
    fig = _fig("Score components (points earned of maximum)", 300)
    fig.add_trace(go.Bar(y=names, x=[components[n]["max_points"] for n in names], orientation="h", name="Maximum", marker_color="#E3E8EF"))
    fig.add_trace(go.Bar(y=names, x=[components[n]["points"] for n in names], orientation="h", name="Earned", marker_color=PALETTE[0]))
    fig.update_layout(barmode="overlay")
    fig.update_yaxes(autorange="reversed")
    return fig
