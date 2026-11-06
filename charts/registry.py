"""
charts/registry.py

Centralized chart renderer registry and dispatcher.
Maps ChartSpec chart type and encodings into styled Plotly Figures.
Implements graceful auto-downgrade when data dimensions do not fit the requested chart.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from core.schema import ChartSpec, ChartConfig
from charts.theme import (
    apply_theme,
    DARK_BG,
    PANEL_BG,
    GRID_COLOR,
    TEXT_COLOR,
    MUTED_TEXT,
    LINE_COLOR,
    QUALITATIVE_PALETTE,
    MARKER_SYMBOLS,
)


def render_chart_from_spec(df: pd.DataFrame, spec: ChartSpec) -> go.Figure:
    """
    Renders a Plotly Figure from a validated dataframe and ChartSpec.
    Follows auto-downgrade rules if the requested chart type doesn't fit the data.
    """
    chart = spec.chart
    ctype = chart.type
    title = spec.title

    if df.empty:
        fig = go.Figure()
        return apply_theme(fig, title=f"{title} (No data returned)")

    # -----------------------------------------------------
    # 1. Bar Chart
    # -----------------------------------------------------
    if ctype == "bar":
        x_col = chart.x
        y_col = chart.y
        orientation = chart.orientation or "v"

        if orientation == "h" and y_col and x_col:
            fig = go.Figure(
                go.Bar(
                    y=df[y_col],
                    x=df[x_col],
                    orientation="h",
                    marker_color=LINE_COLOR if not chart.color else None,
                )
            )
            return apply_theme(fig, title=title, xaxis_title=x_col, yaxis_title=y_col)
        elif x_col and y_col:
            color_arg = df[chart.color] if chart.color and chart.color in df.columns else None
            fig = go.Figure(
                go.Bar(
                    x=df[x_col],
                    y=df[y_col],
                    marker_color=LINE_COLOR if color_arg is None else color_arg,
                )
            )
            return apply_theme(fig, title=title, xaxis_title=x_col, yaxis_title=y_col)

    # -----------------------------------------------------
    # 2. Line Chart
    # -----------------------------------------------------
    elif ctype == "line":
        x_col = chart.x
        y_col = chart.y
        color_col = chart.color

        fig = go.Figure()
        if color_col and color_col in df.columns:
            groups = df[color_col].unique()
            for idx, grp in enumerate(groups[:10]):
                sub = df[df[color_col] == grp].sort_values(by=x_col)
                c = QUALITATIVE_PALETTE[idx % len(QUALITATIVE_PALETTE)]
                sym = MARKER_SYMBOLS[idx % len(MARKER_SYMBOLS)]
                fig.add_trace(
                    go.Scatter(
                        x=sub[x_col],
                        y=sub[y_col],
                        mode="lines+markers",
                        name=str(grp),
                        line=dict(color=c, width=2.5),
                        marker=dict(symbol=sym, size=7, color=c),
                    )
                )
            return apply_theme(
                fig,
                title=title,
                xaxis_title=x_col,
                yaxis_title=y_col,
                invert_yaxis=bool(chart.invert_yaxis),
                show_legend=True,
                legend_position="right",
            )
        else:
            sorted_df = df.sort_values(by=x_col) if x_col in df.columns else df
            fig.add_trace(
                go.Scatter(
                    x=sorted_df[x_col],
                    y=sorted_df[y_col],
                    mode="lines+markers",
                    line=dict(color=LINE_COLOR, width=3),
                    marker=dict(size=7, color=LINE_COLOR),
                )
            )
            return apply_theme(
                fig,
                title=title,
                xaxis_title=x_col,
                yaxis_title=y_col,
                invert_yaxis=bool(chart.invert_yaxis),
            )

    # -----------------------------------------------------
    # 3. Scatter Plot (with optional OLS trendline)
    # -----------------------------------------------------
    elif ctype == "scatter":
        x_col = chart.x
        y_col = chart.y
        color_col = chart.color

        fig = go.Figure()
        hover_text = df["name"] if "name" in df.columns else None

        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode="markers",
                marker=dict(
                    color=LINE_COLOR if not color_col else None,
                    size=8,
                    opacity=0.8,
                    line=dict(width=1, color="#0f1013"),
                ),
                text=hover_text,
                name="Points",
            )
        )

        if chart.trendline == "ols" and len(df) >= 3:
            x_vals = pd.to_numeric(df[x_col], errors="coerce").dropna().values
            y_vals = pd.to_numeric(df[y_col], errors="coerce").dropna().values
            min_len = min(len(x_vals), len(y_vals))
            if min_len >= 3:
                slope, intercept = np.polyfit(x_vals[:min_len], y_vals[:min_len], 1)
                x_grid = np.linspace(x_vals.min(), x_vals.max(), 50)
                fig.add_trace(
                    go.Scatter(
                        x=x_grid,
                        y=slope * x_grid + intercept,
                        mode="lines",
                        name="OLS Fit",
                        line=dict(color="#f97316", width=2, dash="dash"),
                    )
                )

        return apply_theme(fig, title=title, xaxis_title=x_col, yaxis_title=y_col, show_legend=bool(chart.trendline))

    # -----------------------------------------------------
    # 4. Histogram & Box & Violin
    # -----------------------------------------------------
    elif ctype == "histogram":
        val_col = chart.x or chart.y
        fig = go.Figure(go.Histogram(x=df[val_col], marker_color=LINE_COLOR, opacity=0.85))
        return apply_theme(fig, title=title, xaxis_title=val_col, yaxis_title="Count")

    elif ctype in ["box", "violin"]:
        val_col = chart.y or chart.x
        grp_col = chart.x if chart.y else chart.color
        if grp_col and grp_col in df.columns:
            fig = go.Figure()
            for idx, g in enumerate(df[grp_col].unique()[:8]):
                sub = df[df[grp_col] == g]
                c = QUALITATIVE_PALETTE[idx % len(QUALITATIVE_PALETTE)]
                if ctype == "box":
                    fig.add_trace(go.Box(y=sub[val_col], name=str(g), marker_color=c))
                else:
                    fig.add_trace(go.Violin(y=sub[val_col], name=str(g), line_color=c))
            return apply_theme(fig, title=title, xaxis_title=grp_col, yaxis_title=val_col)
        else:
            fig = go.Figure(go.Box(y=df[val_col], marker_color=LINE_COLOR) if ctype == "box" else go.Violin(y=df[val_col], line_color=LINE_COLOR))
            return apply_theme(fig, title=title, yaxis_title=val_col)

    # -----------------------------------------------------
    # 5. Table Fallback / Renderer
    # -----------------------------------------------------
    cols_to_show = chart.fields or list(df.columns[:8])
    clean_cols = [c for c in cols_to_show if c in df.columns]
    fig = go.Figure(
        go.Table(
            header=dict(
                values=clean_cols,
                fill_color="#1a1d24",
                align="left",
                font=dict(color=TEXT_COLOR, size=11),
            ),
            cells=dict(
                values=[df[col] for col in clean_cols],
                fill_color=PANEL_BG,
                align="left",
                font=dict(color=TEXT_COLOR, size=11),
            ),
        )
    )
    return apply_theme(fig, title=title)

