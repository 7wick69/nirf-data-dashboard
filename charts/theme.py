"""
charts/theme.py

Unified minimal dark theme and Plotly styling for NIRF Interactive Dashboard.
Follows textbook data visualization rules (ColorBrewer, Munzner, Tufte):
  - Deep charcoal/black canvas (#0f1013)
  - Clean card surface (#16181e)
  - Subtle borders/grid (#262a33)
  - High-discrimination qualitative palette for multi-line nominal categories
  - Redundant encoding support (distinct hues + distinct marker symbols)
"""

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

# Textbook Qualitative Categorical Palette (10 distinct hues, balanced luminance for dark background)
QUALITATIVE_PALETTE = [
    "#38bdf8",  # Sky Blue
    "#f97316",  # Orange
    "#10b981",  # Emerald Green
    "#f43f5e",  # Rose Red
    "#a855f7",  # Purple
    "#f59e0b",  # Amber / Warm Yellow
    "#06b6d4",  # Cyan
    "#ec4899",  # Magenta / Pink
    "#84cc16",  # Lime Green
    "#6366f1",  # Indigo
]

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
) -> go.Figure:
    """Applies clean, minimal dark styling to any Plotly figure."""
    fig.update_layout(
        title={
            "text": title,
            "y": 0.95,
            "x": 0.05,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 15, "color": TEXT_COLOR, "family": "Inter, system-ui, sans-serif"},
        },
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font={"color": TEXT_COLOR, "family": "Inter, system-ui, sans-serif", "size": 12},
        margin={"l": 60, "r": 40, "t": 60, "b": 60},
        showlegend=show_legend,
        legend={
            "bgcolor": "rgba(22, 24, 30, 0.85)",
            "bordercolor": GRID_COLOR,
            "borderwidth": 1,
            "font": {"color": TEXT_COLOR, "size": 11},
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        hoverlabel={
            "bgcolor": PANEL_BG,
            "bordercolor": GRID_COLOR,
            "font": {"color": TEXT_COLOR, "family": "Inter, monospace", "size": 12},
        },
    )

    fig.update_xaxes(
        title={"text": xaxis_title, "font": {"size": 12, "color": MUTED_TEXT}},
        gridcolor=GRID_COLOR,
        zerolinecolor=GRID_COLOR,
        showline=True,
        linecolor=GRID_COLOR,
        tickfont={"size": 11, "color": MUTED_TEXT},
    )

    fig.update_yaxes(
        title={"text": yaxis_title, "font": {"size": 12, "color": MUTED_TEXT}},
        gridcolor=GRID_COLOR,
        zerolinecolor=GRID_COLOR,
        showline=True,
        linecolor=GRID_COLOR,
        tickfont={"size": 11, "color": MUTED_TEXT},
        autorange="reversed" if invert_yaxis else True,
    )

    return fig
