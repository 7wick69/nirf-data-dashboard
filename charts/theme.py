"""
charts/theme.py

Unified minimal dark theme and Plotly styling for NIRF Interactive Dashboard.
Matches the monochromatic dark UI aesthetic:
  - Deep charcoal/black canvas (#111215)
  - Clean card surface (#17191e)
  - Subtle borders/grid (#262a33)
  - Electric blue accent (#3b82f6 / #60a5fa)
  - Clean, readable typography
"""

import plotly.graph_objects as go

# Color Palette
DARK_BG = "#111215"
PANEL_BG = "#17191e"
GRID_COLOR = "#252830"
LINE_COLOR = "#3b82f6"
ACCENT_GREEN = "#22c55e"
ACCENT_RED = "#ef4444"
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
            "bgcolor": "rgba(23, 25, 30, 0.8)",
            "bordercolor": GRID_COLOR,
            "borderwidth": 1,
            "font": {"color": TEXT_COLOR, "size": 11},
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

