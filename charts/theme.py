"""
charts/theme.py

Unified minimal dark theme and Plotly styling for NIRF Interactive Dashboard.
Follows textbook data visualization rules (ColorBrewer, Munzner, Tufte):
  - Deep charcoal/black canvas (#0f1013)
  - Clean card surface (#16181e)
  - Subtle borders/grid (#262a33)
  - High-discrimination qualitative palette for multi-line nominal categories
  - Redundant encoding support (distinct hues + distinct marker symbols)
  - Bulletproof layout metrics: Dedicated legend margins to prevent title/legend collisions
"""

from typing import Literal
import plotly.graph_objects as go

# Base Canvas Palette
DARK_BG = "#0f1013"
PANEL_BG = "#16181e"
GRID_COLOR = "#262a33"
LINE_COLOR = "#38bdf8"
ACCENT_GREEN = "#10b981"
ACCENT_RED = "#f43f5e"
TEXT_COLOR = "#e6edf3"
MUTED_TEXT = "#8b949e"

# Textbook Qualitative Categorical Palette with high-contrast complementary pairs:
# Pair 1: Sky Blue (#38bdf8) & Vibrant Orange (#f97316)
# Pair 2: Emerald Green (#10b981) & Rose Red (#f43f5e)
# Triad/Accent: Purple (#a855f7), Amber (#f59e0b), Cyan (#06b6d4), Pink (#ec4899)
QUALITATIVE_PALETTE = [
    "#38bdf8",  # Sky Blue
    "#f97316",  # Vibrant Orange (complementary to Sky Blue)
    "#10b981",  # Emerald Green
    "#f43f5e",  # Rose Red (complementary to Emerald Green)
    "#a855f7",  # Purple
    "#f59e0b",  # Amber / Warm Gold
    "#06b6d4",  # Cyan
    "#ec4899",  # Pink
    "#84cc16",  # Lime Green
    "#6366f1",  # Indigo
]
PALETTE = QUALITATIVE_PALETTE

# Redundant encoding marker symbols (Textbook rule: do not rely on color alone)
MARKER_SYMBOLS = [
    "circle",
    "square",
    "diamond",
    "triangle-up",
    "star",
    "triangle-down",
    "hexagon",
    "cross",
    "hourglass",
    "pentagon",
]

PALETTE = QUALITATIVE_PALETTE


def apply_theme(
    fig: go.Figure,
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "",
    invert_yaxis: bool = False,
    show_legend: bool = False,
    legend_position: Literal["right", "bottom"] = "right",
) -> go.Figure:
    """
    Applies clean, minimal dark styling to any Plotly figure.
    Guarantees zero overlap between chart title and legend.
    """
    if show_legend and legend_position == "right":
        margin = dict(l=65, r=200, t=65, b=60)
        legend_layout = dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(22, 24, 30, 0.85)",
            bordercolor=GRID_COLOR,
            borderwidth=1,
            font=dict(color=TEXT_COLOR, size=11),
            itemwidth=30,
        )
    elif show_legend and legend_position == "bottom":
        margin = dict(l=65, r=40, t=65, b=95)
        legend_layout = dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(22, 24, 30, 0.85)",
            bordercolor=GRID_COLOR,
            borderwidth=1,
            font=dict(color=TEXT_COLOR, size=11),
        )
    else:
        margin = dict(l=65, r=40, t=65, b=60)
        legend_layout = dict(
            font=dict(color=TEXT_COLOR, size=11),
        )

    fig.update_layout(
        title=dict(
            text=title,
            y=0.96,
            x=0.02,
            xanchor="left",
            yanchor="top",
            font=dict(size=15, color=TEXT_COLOR, family="Inter, system-ui, sans-serif"),
        ),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT_COLOR, family="Inter, system-ui, sans-serif", size=12),
        margin=margin,
        showlegend=show_legend,
        legend=legend_layout,
        hoverlabel=dict(
            bgcolor=PANEL_BG,
            bordercolor=GRID_COLOR,
            font=dict(color=TEXT_COLOR, family="Inter, monospace", size=12),
        ),
    )

    fig.update_xaxes(
        title=dict(text=xaxis_title, font=dict(size=12, color=MUTED_TEXT)),
        gridcolor=GRID_COLOR,
        zerolinecolor=GRID_COLOR,
        showline=True,
        linecolor=GRID_COLOR,
        tickfont=dict(size=11, color=MUTED_TEXT),
    )

    fig.update_yaxes(
        title=dict(text=yaxis_title, font=dict(size=12, color=MUTED_TEXT)),
        gridcolor=GRID_COLOR,
        zerolinecolor=GRID_COLOR,
        showline=True,
        linecolor=GRID_COLOR,
        tickfont=dict(size=11, color=MUTED_TEXT),
        autorange="reversed" if invert_yaxis else True,
    )

    return fig
