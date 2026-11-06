"""
charts/outliers.py

Outlier detection, anomaly analysis, and parameter disparity visualizations:
  1. Research vs Teaching Disparity (RPC - TLR disparity ranking)
  2. Ranking Volatility Outliers (std of rank across years)
  3. Single-Year Score Jump Anomalies (largest score_change_yoy)
  4. Strong Parameter but Poor Overall Rank (Rank vs Parameter Disconnect)
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from charts.theme import (
    apply_theme,
    LINE_COLOR,
    ACCENT_GREEN,
    ACCENT_RED,
    MUTED_TEXT,
)


def plot_research_vs_teaching_disparity(
    df: pd.DataFrame,
    year: int = 2024,
    category: str = "Overall",
    top_n: int = 10,
) -> Tuple[go.Figure, str]:
    """
    Question: Which institutions perform significantly better in research than in teaching?
    Computes disparity = RPC - TLR and renders sorted bar chart.
    """
    sub = df[(df["year"] == year) & (df["category"] == category) & df["RPC"].notna() & df["TLR"].notna()].copy()
    if sub.empty:
        fig = go.Figure()
        return apply_theme(fig, title="No comparative parameter data"), "No records found."

    sub["disparity"] = (sub["RPC"] - sub["TLR"]).round(1)
    # Sort by highest disparity (Research much higher than Teaching)
    top_disparities = sub.sort_values(by="disparity", ascending=False).head(top_n).sort_values(by="disparity", ascending=True)

    fig = go.Figure(
        go.Bar(
            y=top_disparities["name"],
            x=top_disparities["disparity"],
            orientation="h",
            marker=dict(color=LINE_COLOR),
            hovertemplate=(
                "<b>%{y}</b><br>"
                + "Disparity: %{x:+0.1f} pts<br>"
                + "RPC: %{customdata[0]:.1f} vs TLR: %{customdata[1]:.1f}<br>"
                + "Overall Rank: #%{customdata[2]}<extra></extra>"
            ),
            customdata=list(zip(top_disparities["RPC"], top_disparities["TLR"], top_disparities["rank"].astype(int))),
        )
    )

    fig = apply_theme(
        fig,
        title=f"Highest Research vs Teaching Disparity: RPC − TLR ({year}, {category})",
        xaxis_title="Score Disparity: RPC − TLR (Points)",
        yaxis_title="",
    )

    leader = top_disparities.iloc[-1]
    takeaway = (
        f"{leader['name']} exhibits the highest research-over-teaching tilt "
        f"(RPC = {leader['RPC']:.1f} vs TLR = {leader['TLR']:.1f}, {leader['disparity']:+.1f} pts disparity)."
    )
    return fig, takeaway


def plot_unusually_high_volatility(
    df: pd.DataFrame,
    category: str = "Overall",
    top_n: int = 10,
) -> Tuple[go.Figure, str]:
    """
    Question: Which institutions show unusually high volatility in their rankings across the 9 years?
    Renders standard deviation of rank across years per institution.
    """
    cat_df = df[df["category"] == category].copy()
    # Filter institutions with at least 4 years of rankings
    inst_counts = cat_df.groupby("name")["year"].nunique()
    eligible = inst_counts[inst_counts >= 4].index

    sub = cat_df[cat_df["name"].isin(eligible)].drop_duplicates(subset=["name"]).sort_values(by="volatility", ascending=False).head(top_n)

    if sub.empty:
        fig = go.Figure()
        return apply_theme(fig, title="No volatility records found"), "No records found."

    sub = sub.sort_values(by="volatility", ascending=True)

    fig = go.Figure(
        go.Bar(
            y=sub["name"],
            x=sub["volatility"],
            orientation="h",
            marker=dict(color="#f97316"),
            hovertemplate="<b>%{y}</b><br>Rank Volatility (σ): %{x:.2f}<br>State: %{customdata}<extra></extra>",
            customdata=sub["state"],
        )
    )

    fig = apply_theme(
        fig,
        title=f"Most Volatile Institution Rankings (Std Dev of Rank across 2017–2025, {category})",
        xaxis_title="Ranking Volatility (Standard Deviation σ)",
        yaxis_title="",
    )

    most_volatile = sub.iloc[-1]
    takeaway = (
        f"Highest ranking volatility: {most_volatile['name']} with σ = {most_volatile['volatility']:.2f} ranks "
        f"across available ranking years."
    )
    return fig, takeaway


def plot_score_jump_anomalies(
    df: pd.DataFrame,
    category: str = "Overall",
    top_n: int = 8,
) -> Tuple[go.Figure, str]:
    """
    Question: Are there any institutions that experienced unusually large score jumps in a single year?
    Identifies largest positive and negative single-year score changes (score_change_yoy).
    """
    cat_df = df[df["category"] == category].dropna(subset=["score_change_yoy"]).copy()
    if cat_df.empty:
        fig = go.Figure()
        return apply_theme(fig, title="No YoY score changes found"), "No records."

    cat_df["abs_change"] = cat_df["score_change_yoy"].abs()
    top_jumps = cat_df.sort_values(by="abs_change", ascending=False).head(top_n).sort_values(by="score_change_yoy", ascending=True)

    colors = [ACCENT_GREEN if d > 0 else ACCENT_RED for d in top_jumps["score_change_yoy"]]
    labels = [f"{r['name']} ({r['year']})" for _, r in top_jumps.iterrows()]

    fig = go.Figure(
        go.Bar(
            y=labels,
            x=top_jumps["score_change_yoy"].round(2),
            orientation="h",
            marker=dict(color=colors),
            hovertemplate=(
                "<b>%{y}</b><br>"
                + "YoY Score Shift: %{x:+0.2f} pts<br>"
                + "Prev: %{customdata[0]:.2f} → New: %{customdata[1]:.2f}<extra></extra>"
            ),
            customdata=list(zip(top_jumps["prev_score"], top_jumps["score"])),
        )
    )

    fig = apply_theme(
        fig,
        title=f"Largest Single-Year Score Jump Anomalies ({category})",
        xaxis_title="Score Change YoY (Points)",
        yaxis_title="",
    )

    top_jump = top_jumps[top_jumps["score_change_yoy"] > 0].iloc[-1] if not top_jumps[top_jumps["score_change_yoy"] > 0].empty else None
    takeaway = f"Largest single-year score increase: {top_jump['name']} gained +{top_jump['score_change_yoy']:.2f} pts in {top_jump['year']}." if top_jump is not None else "No anomalies detected."
    return fig, takeaway

