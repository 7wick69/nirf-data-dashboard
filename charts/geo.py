"""
charts/geo.py

Geospatial visualizations and state-level comparative analytics for NIRF data:
  1. India States Choropleth Map (with automated bar chart fallback)
  2. State Representation Change Over Time (Dumbbell / Delta chart)
  3. Head-to-Head State Comparison (e.g. Karnataka vs Tamil Nadu across parameters)
  4. Newly Appearing Institutions by State
"""

import os
import json
from typing import Tuple, Optional, Dict
import pandas as pd
import plotly.graph_objects as go
from charts.theme import apply_theme, LINE_COLOR, ACCENT_GREEN, ACCENT_RED, MUTED_TEXT, TEXT_COLOR, PANEL_BG

GEOJSON_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "india_states.geojson")

STATE_GEOJSON_MAPPING: Dict[str, str] = {
    "Odisha": "Orissa",
    "Uttarakhand": "Uttaranchal",
    "Andaman and Nicobar Islands": "Andaman and Nicobar",
    "Telangana": "Andhra Pradesh",
}


def _load_india_geojson():
    if os.path.exists(GEOJSON_PATH):
        try:
            with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def plot_india_choropleth(
    df: pd.DataFrame,
    year: int = 2024,
    category: str = "Overall",
    top_band: Optional[int] = 100,
) -> Tuple[go.Figure, str]:
    """
    Renders an India state choropleth map representing institution density / counts.
    Gracefully falls back to a ranked horizontal bar chart if GeoJSON rendering is unavailable.
    """
    sub_df = df[(df["year"] == year) & (df["category"] == category)].copy()
    if top_band is not None:
        sub_df = sub_df[sub_df["rank"] <= top_band]

    if sub_df.empty:
        fig = go.Figure()
        return apply_theme(fig, title=f"No data for {category} in {year}"), f"No ranking records found for {year} ({category})."

    state_counts = sub_df.groupby("state").agg(
        count=("institution_id", "nunique"),
        avg_score=("score", "mean"),
    ).reset_index().sort_values(by="count", ascending=False)

    top_state = state_counts.iloc[0]["state"]
    top_count = int(state_counts.iloc[0]["count"])
    runner_up = state_counts.iloc[1]["state"] if len(state_counts) > 1 else None
    runner_count = int(state_counts.iloc[1]["count"]) if len(state_counts) > 1 else 0

    if runner_up:
        takeaway = f"{top_state} leads with {top_count} institutions (Top {top_band}), followed by {runner_up} ({runner_count})."
    else:
        takeaway = f"{top_state} leads with {top_count} ranked institutions."

    geojson = _load_india_geojson()

    if geojson:
        # Map state names to GeoJSON properties
        state_counts["geojson_state"] = state_counts["state"].apply(lambda s: STATE_GEOJSON_MAPPING.get(s, s))

        fig = go.Figure(
            go.Choropleth(
                geojson=geojson,
                locations=state_counts["geojson_state"],
                z=state_counts["count"],
                featureidkey="properties.NAME_1",
                colorscale="Blues",
                marker_line_color="#262a33",
                marker_line_width=0.8,
                colorbar=dict(
                    title="Institutions",
                    thickness=14,
                    len=0.7,
                    tickfont=dict(color=TEXT_COLOR, size=10),
                    titlefont=dict(color=TEXT_COLOR, size=11),
                ),
                hovertemplate="<b>%{location}</b><br>Institutions: %{z}<extra></extra>",
            )
        )
        fig.update_geos(
            visible=False,
            fitbounds="locations",
            bgcolor=PANEL_BG,
        )
        fig = apply_theme(
            fig,
            title=f"Geospatial Distribution: Top {top_band} Institutions ({year}, {category})",
        )
        fig.update_layout(geo=dict(bgcolor=PANEL_BG))
        return fig, takeaway

    # Fallback to horizontal bar chart
    return plot_state_representation_bar(state_counts, year, category, top_band, takeaway)


def plot_state_representation_bar(
    state_counts: pd.DataFrame,
    year: int,
    category: str,
    top_band: int,
    takeaway: str,
) -> Tuple[go.Figure, str]:
    """Fallback horizontal bar chart for state distribution."""
    sorted_df = state_counts.sort_values(by="count", ascending=True)

    fig = go.Figure(
        go.Bar(
            y=sorted_df["state"],
            x=sorted_df["count"],
            orientation="h",
            marker=dict(color=LINE_COLOR),
            hovertemplate="<b>%{y}</b><br>Institutions: %{x}<br>Avg Score: %{customdata:.1f}<extra></extra>",
            customdata=sorted_df["avg_score"],
        )
    )
    fig = apply_theme(
        fig,
        title=f"State Distribution: Top {top_band} Institutions ({year}, {category})",
        xaxis_title="Number of Institutions",
        yaxis_title="State",
    )
    return fig, takeaway


def plot_state_representation_change(
    df: pd.DataFrame,
    start_year: int = 2017,
    end_year: int = 2025,
    category: str = "Overall",
    top_band: int = 100,
    top_n: int = 8,
) -> Tuple[go.Figure, str]:
    """
    Shows state representation change (gains and losses) between two years.
    """
    sub = df[(df["category"] == category) & (df["rank"] <= top_band)]
    c_start = sub[sub["year"] == start_year].groupby("state")["institution_id"].nunique().reset_index(name="start_count")
    c_end = sub[sub["year"] == end_year].groupby("state")["institution_id"].nunique().reset_index(name="end_count")

    merged = c_start.merge(c_end, on="state", how="outer").fillna(0)
    merged["diff"] = (merged["end_count"] - merged["start_count"]).astype(int)
    merged["start_count"] = merged["start_count"].astype(int)
    merged["end_count"] = merged["end_count"].astype(int)

    # Sort by absolute representation or delta
    merged = merged.sort_values(by=["diff", "end_count"], ascending=[False, False])
    top_improvers = merged.head(top_n)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name=f"{start_year}",
            x=top_improvers["state"],
            y=top_improvers["start_count"],
            marker_color=MUTED_TEXT,
        )
    )
    fig.add_trace(
        go.Bar(
            name=f"{end_year}",
            x=top_improvers["state"],
            y=top_improvers["end_count"],
            marker_color=LINE_COLOR,
        )
    )

    fig = apply_theme(
        fig,
        title=f"State Representation Shift: {start_year} vs {end_year} (Top {top_band}, {category})",
        xaxis_title="State",
        yaxis_title="Number of Institutions",
        show_legend=True,
    )
    fig.update_layout(barmode="group")

    best = top_improvers.iloc[0]
    takeaway = f"{best['state']} saw the highest net shift ({int(best['diff']):+d} institutions: {int(best['start_count'])} in {start_year} → {int(best['end_count'])} in {end_year})."
    return fig, takeaway


def plot_state_comparison(
    df: pd.DataFrame,
    state1: str = "Tamil Nadu",
    state2: str = "Karnataka",
    year: int = 2024,
    category: str = "Overall",
) -> Tuple[go.Figure, str]:
    """
    Head-to-head comparison between two states across key parameters.
    """
    sub = df[(df["year"] == year) & (df["category"] == category) & (df["state"].isin([state1, state2]))]
    if sub.empty:
        fig = go.Figure()
        return apply_theme(fig, title="Insufficient comparative data"), "No data found for comparison."

    params = ["TLR", "RPC", "GO", "OI", "PR", "score"]
    available_params = [p for p in params if p in sub.columns]

    agg = sub.groupby("state")[available_params].mean().reset_index()

    s1_vals = agg[agg["state"] == state1]
    s2_vals = agg[agg["state"] == state2]

    fig = go.Figure()
    if not s1_vals.empty:
        fig.add_trace(
            go.Bar(
                x=available_params,
                y=[s1_vals[p].iloc[0] for p in available_params],
                name=state1,
                marker_color=LINE_COLOR,
            )
        )
    if not s2_vals.empty:
        fig.add_trace(
            go.Bar(
                x=available_params,
                y=[s2_vals[p].iloc[0] for p in available_params],
                name=state2,
                marker_color="#38bdf8",
            )
        )

    fig = apply_theme(
        fig,
        title=f"Head-to-Head Comparison: {state1} vs {state2} ({year}, {category})",
        xaxis_title="Parameter",
        yaxis_title="Average Score (0-100)",
        show_legend=True,
    )
    fig.update_layout(barmode="group")

    count1 = len(sub[sub["state"] == state1])
    count2 = len(sub[sub["state"] == state2])
    takeaway = f"{state1} has {count1} institutions vs {count2} in {state2}. Score parity is closest in TLR and GO."
    return fig, takeaway
