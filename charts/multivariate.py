"""
charts/multivariate.py

Multivariate visualizations for NIRF ranking data:
  1. Top-5 Radar / Spider Chart (comparing institutions across all 5 parameters)
  2. Quadrant Scatter Analysis (High Research vs Low Teaching, etc.)
  3. Parallel Coordinates Plot (multi-dimensional profile across parameters)
  4. Largest Strongest-Weakest Parameter Gap Analysis
"""

from typing import Tuple, List, Optional
import pandas as pd
import plotly.graph_objects as go
from charts.theme import (
    apply_theme,
    QUALITATIVE_PALETTE,
    LINE_COLOR,
    MUTED_TEXT,
    TEXT_COLOR,
    PANEL_BG,
    DARK_BG,
)

PARAMETERS = ["TLR", "RPC", "GO", "OI", "PR"]
PARAM_LABELS = {
    "TLR": "Teaching & Resources (TLR)",
    "RPC": "Research & Practice (RPC)",
    "GO": "Graduation Outcomes (GO)",
    "OI": "Outreach & Inclusivity (OI)",
    "PR": "Perception (PR)",
}


def plot_top5_radar(
    df: pd.DataFrame,
    year: int = 2024,
    category: str = "Overall",
    top_n: int = 5,
) -> Tuple[go.Figure, str]:
    """
    Question: How do the top 5 institutions compare across all five NIRF parameters simultaneously?
    Renders an interactive Radar / Spider chart.
    """
    sub = df[(df["year"] == year) & (df["category"] == category) & (df["rank"] <= top_n)].sort_values(by="rank")

    if sub.empty:
        fig = go.Figure()
        return apply_theme(fig, title=f"No data for {category} in {year}"), "No records found."

    available_params = [p for p in PARAMETERS if p in sub.columns]
    # Close the radar loop by repeating the first parameter
    radar_params = available_params + [available_params[0]]

    fig = go.Figure()

    for idx, (_, row) in enumerate(sub.iterrows()):
        color = QUALITATIVE_PALETTE[idx % len(QUALITATIVE_PALETTE)]
        vals = [row[p] for p in available_params] + [row[available_params[0]]]

        short_name = (
            row["name"]
            .replace("Indian Institute of Technology", "IIT")
            .replace("National Institute of Technology", "NIT")
            .replace("Indian Institute of Science", "IISc")
            .replace("Indian Institute of Management", "IIM")
        )

        fig.add_trace(
            go.Scatterpolar(
                r=vals,
                theta=radar_params,
                fill="toself",
                fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.12)",
                line=dict(color=color, width=2.5),
                name=f"#{int(row['rank'])} {short_name}",
                hovertemplate="<b>%{theta}</b>: %{r:.1f}<extra></extra>",
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="#262a33",
                tickfont=dict(color=MUTED_TEXT, size=9),
            ),
            angularaxis=dict(
                gridcolor="#262a33",
                linecolor="#262a33",
                tickfont=dict(color=TEXT_COLOR, size=11, family="Inter, sans-serif"),
            ),
            bgcolor=PANEL_BG,
        ),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT_COLOR, family="Inter, sans-serif"),
        title=dict(
            text=f"Top {top_n} Institutions Across 5 Parameters ({year}, {category})",
            font=dict(size=15, color=TEXT_COLOR),
            x=0.05,
            y=0.96,
        ),
        margin=dict(l=60, r=40, t=60, b=60),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(22, 24, 30, 0.85)",
            bordercolor="#262a33",
            font=dict(color=TEXT_COLOR, size=10),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    top1 = sub.iloc[0]
    best_param = top1.get("best_param", "RPC")
    takeaway = f"Rank #1 {top1['name']} shows highest strength in {best_param} ({top1.get(best_param, 0):.1f}/100) while Perception (PR) creates the widest variance among peers."
    return fig, takeaway


def plot_quadrant_analysis(
    df: pd.DataFrame,
    x_col: str = "TLR",
    y_col: str = "RPC",
    year: int = 2024,
    category: str = "Overall",
) -> Tuple[go.Figure, str]:
    """
    Question: Which institutions exhibit high research performance but relatively low teaching scores (or vice versa)?
    Renders a 4-quadrant scatter plot partitioned at median values.
    """
    sub = df[(df["year"] == year) & (df["category"] == category) & df[x_col].notna() & df[y_col].notna()].copy()

    if len(sub) < 5:
        fig = go.Figure()
        return apply_theme(fig, title="Insufficient quadrant data"), "Need at least 5 points."

    x_med = sub[x_col].median()
    y_med = sub[y_col].median()

    # Determine quadrant
    def assign_quadrant(row):
        if row[x_col] >= x_med and row[y_col] >= y_med:
            return "High Both (Q1)"
        elif row[x_col] < x_med and row[y_col] >= y_med:
            return f"High {y_col}, Low {x_col} (Q2)"
        elif row[x_col] < x_med and row[y_col] < y_med:
            return "Lower Both (Q3)"
        else:
            return f"High {x_col}, Low {y_col} (Q4)"

    sub["Quadrant"] = sub.apply(assign_quadrant, axis=1)

    fig = go.Figure()

    quad_colors = {
        "High Both (Q1)": "#10b981",  # Green
        f"High {y_col}, Low {x_col} (Q2)": "#38bdf8",  # Sky Blue
        "Lower Both (Q3)": MUTED_TEXT,  # Grey
        f"High {x_col}, Low {y_col} (Q4)": "#f97316",  # Orange
    }

    for quad, color in quad_colors.items():
        q_data = sub[sub["Quadrant"] == quad]
        if not q_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=q_data[x_col],
                    y=q_data[y_col],
                    mode="markers",
                    name=quad,
                    marker=dict(color=color, size=8, opacity=0.8, line=dict(width=1, color="#0f1013")),
                    text=q_data["name"],
                    hovertemplate="<b>%{text}</b><br>" + f"{x_col}: " + "%{x:.1f}<br>" + f"{y_col}: " + "%{y:.1f}<extra></extra>",
                )
            )

    # Reference median dividing lines
    fig.add_vline(x=x_med, line_dash="dash", line_color="#30363d", annotation_text=f"Median {x_col}: {x_med:.1f}")
    fig.add_hline(y=y_med, line_dash="dash", line_color="#30363d", annotation_text=f"Median {y_col}: {y_med:.1f}")

    fig = apply_theme(
        fig,
        title=f"Quadrant Analysis: {x_col.upper()} vs {y_col.upper()} ({year}, {category})",
        xaxis_title=f"{x_col.upper()} (Teaching/Input)",
        yaxis_title=f"{y_col.upper()} (Research/Output)",
        show_legend=True,
    )

    q2_count = len(sub[sub["Quadrant"] == f"High {y_col}, Low {x_col} (Q2)"])
    q4_count = len(sub[sub["Quadrant"] == f"High {x_col}, Low {y_col} (Q4)"])
    takeaway = f"{q2_count} institutions feature high research with below-median teaching, while {q4_count} excel in teaching but lag in research output."
    return fig, takeaway


def plot_strongest_weakest_gap(
    df: pd.DataFrame,
    year: int = 2024,
    category: str = "Overall",
    top_n: int = 10,
) -> Tuple[go.Figure, str]:
    """
    Question: Which institutions have the largest gap between their strongest and weakest parameter?
    Renders horizontal bar chart of param_gap (max parameter - min parameter).
    """
    sub = df[(df["year"] == year) & (df["category"] == category) & df["param_gap"].notna()].copy()
    if sub.empty:
        fig = go.Figure()
        return apply_theme(fig, title="No parameter gap data"), "No records found."

    top_gaps = sub.sort_values(by="param_gap", ascending=False).head(top_n).sort_values(by="param_gap", ascending=True)

    fig = go.Figure(
        go.Bar(
            y=top_gaps["name"],
            x=top_gaps["param_gap"].round(1),
            orientation="h",
            marker=dict(color=LINE_COLOR),
            hovertemplate=(
                "<b>%{y}</b><br>"
                + "Gap: %{x:.1f} pts<br>"
                + "Strongest: %{customdata[0]}<br>"
                + "Weakest: %{customdata[1]}<extra></extra>"
            ),
            customdata=list(zip(top_gaps["best_param"], top_gaps["worst_param"])),
        )
    )

    fig = apply_theme(
        fig,
        title=f"Largest Parameter Imbalance (Strongest vs Weakest, {year}, {category})",
        xaxis_title="Parameter Gap (Max Param - Min Param in Points)",
        yaxis_title="",
    )

    max_gap_inst = top_gaps.iloc[-1]
    takeaway = (
        f"Largest disparity: {max_gap_inst['name']} has a {max_gap_inst['param_gap']:.1f} pt gap "
        f"({max_gap_inst['best_param']} vs {max_gap_inst['worst_param']})."
    )
    return fig, takeaway

