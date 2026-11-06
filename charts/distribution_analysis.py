"""
charts/distribution_analysis.py

Distribution & Statistical Comparison module — NIRF Interactive Dashboard.
Implements Annexure A, Section 6 — Questions 6.1 through 6.5.

Structure
---------
SECTION A : Pure data/computation helpers (no Streamlit).
SECTION B : Plotly figure functions  q6_<n>_...  each returning (fig, summary_df).
SECTION C : render_distribution_tab(df) — single entry-point for teammates.

Integration (one line in app.py)
---------------------------------
    from charts.distribution_analysis import render_distribution_tab
    with tabs[5]:          # "Distribution" placeholder tab
        render_distribution_tab(df_full)

Statistical glossary (for evaluation explanation)
--------------------------------------------------
std  : Standard deviation — average distance of each score from the mean.
       High std → scores are spread out; low std → scores are clustered.
CV   : Coefficient of Variation = std / mean × 100 (%).
       Dimensionless ratio, so parameters with different scales can be
       compared fairly. Higher CV means more relative variability.
IQR  : Interquartile Range = Q3 − Q1 (75th − 25th percentile).
       Measures spread of the middle 50 % of data, robust to outliers.

Author : Praneeth
Team   : NIRF Dashboard Project — Section 6
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Tuple, Dict

from charts.theme import (
    apply_theme,
    PALETTE,
    DARK_BG,
    PANEL_BG,
    GRID_COLOR,
    TEXT_COLOR,
    MUTED_TEXT,
    ACCENT_GREEN,
    ACCENT_RED,
    LINE_COLOR,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NIRF_PARAMS: List[str] = ["TLR", "RPC", "GO", "OI", "PR"]

PARAM_FULL: Dict[str, str] = {
    "TLR": "Teaching, Learning & Resources",
    "RPC": "Research & Professional Practice",
    "GO":  "Graduation Outcomes",
    "OI":  "Outreach & Inclusivity",
    "PR":  "Perception",
}

# Colour pair for Top-10 vs Rest
TOP10_COLOR = PALETTE[0]   # electric blue
REST_COLOR  = PALETTE[6]   # purple


# ===========================================================================
# SECTION A — DATA / COMPUTATION HELPERS
# ===========================================================================

def ds_filter(
    df: pd.DataFrame,
    years: Optional[List[int]] = None,
    category: Optional[str] = None,
    states: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Generic filter used by every Q6.x function.

    Parameters
    ----------
    df       : Full NIRF DataFrame (from load_nirf_data).
    years    : List of year integers to keep. None → all years.
    category : Category string. None → all categories.
    states   : List of state names. None → all states.

    Returns
    -------
    Filtered copy of df (never modifies in place).
    """
    mask = pd.Series(True, index=df.index)
    if years:
        mask &= df["year"].isin(years)
    if category:
        mask &= df["category"] == category
    if states:
        mask &= df["state"].isin(states)
    return df[mask].copy()


def ds_available_params(df: pd.DataFrame) -> List[str]:
    """
    Returns the subset of NIRF_PARAMS that exist in df AND have
    at least one non-NaN value. Guards against Research/rank-only categories.
    """
    return [p for p in NIRF_PARAMS if p in df.columns and df[p].notna().any()]


def ds_summary_stats(series: pd.Series, label: str = "") -> Dict:
    """
    Computes a standard statistical summary for a numeric series.

    Returns
    -------
    Dict with keys: label, n, mean, median, std, cv, q1, q3, iqr, min, max.
    CV = std/mean*100 (%). Returns NaN for CV if mean == 0.
    """
    s = series.dropna()
    if s.empty:
        return {k: np.nan for k in ["n", "mean", "median", "std", "cv", "q1", "q3", "iqr", "min", "max"]}
    mean = s.mean()
    std  = s.std()
    q1   = s.quantile(0.25)
    q3   = s.quantile(0.75)
    return {
        "label":  label,
        "n":      int(len(s)),
        "mean":   round(mean, 2),
        "median": round(s.median(), 2),
        "std":    round(std, 2),
        "cv":     round(std / mean * 100, 2) if mean != 0 else np.nan,
        "q1":     round(q1, 2),
        "q3":     round(q3, 2),
        "iqr":    round(q3 - q1, 2),
        "min":    round(s.min(), 2),
        "max":    round(s.max(), 2),
    }


def ds_note_missing(col: str, n_dropped: int) -> Optional[str]:
    """Returns a user-visible note string when rows were dropped for NaN."""
    if n_dropped > 0:
        return f"⚠️ {n_dropped} row(s) with missing **{col}** values excluded from this analysis."
    return None


# ===========================================================================
# SECTION B — PLOTTING FUNCTIONS
# ===========================================================================

# ---------------------------------------------------------------------------
# Q6.1 — Top-10 vs Rest score distribution
# ---------------------------------------------------------------------------

def q6_1_top10_vs_rest(
    df: pd.DataFrame,
    year: int,
    category: str,
    chart_type: str = "violin",
) -> Tuple[go.Figure, pd.DataFrame]:
    """
    Q6.1 — Compare score distributions of the Top-10 ranked institutions
    against the remaining institutions for a given year and category.

    Methodology
    -----------
    Institutions are split into two groups using the boolean column `in_top10`.
    Distribution is visualised as overlaid violin plots (default) or box plots.
    Summary stats (n, mean, median, std, IQR) are shown beside the chart.

    Parameters
    ----------
    df         : Full NIRF DataFrame.
    year       : Year to filter.
    category   : NIRF category.
    chart_type : "violin" | "box" | "histogram"

    Returns
    -------
    (fig, summary_df)
    summary_df : one row per group with summary statistics.
    """
    sub = ds_filter(df, years=[year], category=category)
    sub = sub.dropna(subset=["score"])
    n_dropped = len(ds_filter(df, years=[year], category=category)) - len(sub)

    if sub.empty:
        fig = go.Figure()
        apply_theme(fig, title=f"Top-10 vs Rest — {category} ({year})")
        return fig, pd.DataFrame()

    top10 = sub[sub["in_top10"] == True]["score"]
    rest  = sub[sub["in_top10"] == False]["score"]

    if top10.empty or rest.empty:
        fig = go.Figure()
        apply_theme(fig, title=f"Top-10 vs Rest — {category} ({year}): insufficient data")
        return fig, pd.DataFrame()

    fig = go.Figure()

    if chart_type == "histogram":
        # Overlaid normalised histograms
        for grp, color, name in [
            (top10, TOP10_COLOR, "Top 10"),
            (rest,  REST_COLOR,  "Rest"),
        ]:
            fig.add_trace(go.Histogram(
                x=grp,
                name=name,
                marker_color=color,
                opacity=0.7,
                histnorm="probability density",
                nbinsx=20,
                hovertemplate=f"<b>{name}</b><br>Score: %{{x:.1f}}<br>Density: %{{y:.4f}}<extra></extra>",
            ))
        fig.update_layout(barmode="overlay")
        apply_theme(
            fig,
            title=f"Score Distribution: Top-10 vs Rest | {category} · {year}",
            xaxis_title="Overall Score",
            yaxis_title="Probability Density",
            show_legend=True,
        )

    elif chart_type == "box":
        for grp, color, name in [
            (top10, TOP10_COLOR, "Top 10"),
            (rest,  REST_COLOR,  "Rest"),
        ]:
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            fig.add_trace(go.Box(
                y=grp,
                name=name,
                marker_color=color,
                line_color=color,
                fillcolor=f"rgba({r},{g},{b},0.25)",
                boxmean="sd",
                hovertemplate=f"<b>{name}</b><br>Score: %{{y:.2f}}<extra></extra>",
            ))
        apply_theme(
            fig,
            title=f"Score Distribution: Top-10 vs Rest | {category} · {year}",
            xaxis_title="Group",
            yaxis_title="Overall Score",
            show_legend=True,
        )

    else:  # violin (default)
        for grp, color, name in [
            (top10, TOP10_COLOR, "Top 10"),
            (rest,  REST_COLOR,  "Rest"),
        ]:
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            fig.add_trace(go.Violin(  # noqa
                y=grp,
                name=name,
                line_color=color,
                fillcolor=f"rgba({r},{g},{b},0.3)",
                box_visible=True,
                meanline_visible=True,
                points="outliers",
                hovertemplate=f"<b>{name}</b><br>Score: %{{y:.2f}}<extra></extra>",
            ))
        apply_theme(
            fig,
            title=f"Score Distribution: Top-10 vs Rest | {category} · {year}",
            xaxis_title="Group",
            yaxis_title="Overall Score",
            show_legend=True,
        )

    # Build summary stats dataframe
    rows = [
        ds_summary_stats(top10, "Top 10"),
        ds_summary_stats(rest,  "Rest"),
    ]
    summary_df = pd.DataFrame(rows)
    return fig, summary_df


# ---------------------------------------------------------------------------
# Q6.2 — Score dispersion across years
# ---------------------------------------------------------------------------

def q6_2_dispersion_over_years(
    df: pd.DataFrame,
    category: str,
    selected_years: Optional[List[int]] = None,
) -> Tuple[go.Figure, pd.DataFrame]:
    """
    Q6.2 — Are NIRF scores more dispersed in one year than another?

    Methodology
    -----------
    For each year, compute std, variance, and CV of overall scores.
    Display a box plot per year (showing full distribution shape) PLUS a
    dual-axis line/bar chart of std and CV.

    std     = standard deviation of score for that year.
    CV      = std / mean × 100 — normalised spread; useful for comparing
              years where the mean itself may shift.
    Variance = std² — emphasises outlier influence.

    Returns
    -------
    (fig, dispersion_df)
    dispersion_df : one row per year with n, mean, std, cv, variance, iqr.
    """
    sub = ds_filter(df, years=selected_years, category=category)
    sub = sub.dropna(subset=["score"])

    if sub.empty:
        fig = go.Figure()
        apply_theme(fig, title=f"Score Dispersion by Year — {category}")
        return fig, pd.DataFrame()

    years_in_data = sorted(sub["year"].unique())

    # ── subplot: top = box-per-year, bottom = std/CV line chart ───────────
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.6, 0.4],
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=["Score Distribution per Year", "Std Dev & CV (%) per Year"],
    )

    # Top panel — one Box trace per year
    for i, yr in enumerate(years_in_data):
        yr_data = sub[sub["year"] == yr]["score"]
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(
            go.Box(
                y=yr_data,
                name=str(yr),
                marker_color=color,
                line_color=color,
                boxmean=True,
                hovertemplate=f"<b>{yr}</b><br>Score: %{{y:.2f}}<extra></extra>",
                showlegend=False,
            ),
            row=1, col=1,
        )

    # Compute per-year stats
    stats_rows = []
    for yr in years_in_data:
        s = sub[sub["year"] == yr]["score"]
        stats_rows.append(ds_summary_stats(s, label=str(yr)))
    disp_df = pd.DataFrame(stats_rows).rename(columns={"label": "Year"})
    disp_df["variance"] = disp_df["std"] ** 2

    # Bottom panel — Std as bars, CV as line on secondary y
    fig.add_trace(
        go.Bar(
            x=disp_df["Year"],
            y=disp_df["std"],
            name="Std Dev",
            marker_color=LINE_COLOR,
            opacity=0.8,
            hovertemplate="Year: %{x}<br>Std Dev: %{y:.2f}<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=disp_df["Year"],
            y=disp_df["cv"],
            name="CV (%)",
            line=dict(color=ACCENT_GREEN, width=2.5),
            mode="lines+markers",
            marker=dict(size=7),
            yaxis="y3",
            hovertemplate="Year: %{x}<br>CV: %{y:.2f}%<extra></extra>",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT_COLOR, family="Inter, system-ui, sans-serif"),
        title=dict(
            text=f"Score Dispersion by Year | {category}",
            x=0.05, font=dict(size=15, color=TEXT_COLOR),
        ),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(23,25,30,0.8)", bordercolor=GRID_COLOR,
            borderwidth=1, font=dict(color=TEXT_COLOR, size=11),
        ),
        hoverlabel=dict(
            bgcolor=PANEL_BG, bordercolor=GRID_COLOR,
            font=dict(color=TEXT_COLOR, size=12),
        ),
        margin=dict(l=60, r=80, t=80, b=60),
        yaxis3=dict(
            title=dict(text="CV (%)", font=dict(size=11, color=ACCENT_GREEN)),
            overlaying="y2",
            side="right",
            tickfont=dict(color=ACCENT_GREEN, size=10),
            gridcolor=GRID_COLOR,
            showgrid=False,
        ),
    )
    fig.update_xaxes(tickfont=dict(color=MUTED_TEXT, size=11), gridcolor=GRID_COLOR)
    fig.update_yaxes(tickfont=dict(color=MUTED_TEXT, size=11), gridcolor=GRID_COLOR)
    fig.update_yaxes(title_text="Overall Score", row=1, col=1)
    fig.update_yaxes(title_text="Std Dev", row=2, col=1)

    return fig, disp_df


# ---------------------------------------------------------------------------
# Q6.3 — Parameter variability comparison
# ---------------------------------------------------------------------------

def q6_3_parameter_variability(
    df: pd.DataFrame,
    year: int,
    category: str,
    params: Optional[List[str]] = None,
) -> Tuple[go.Figure, pd.DataFrame]:
    """
    Q6.3 — Which NIRF parameter exhibits the greatest variability?

    Methodology
    -----------
    For each available parameter (TLR, RPC, GO, OI, PR):
      - Compute std and CV (= std/mean × 100).
      - CV is used as the fair comparison metric because all parameters are
        scored out of 100 but have very different mean levels (e.g. RPC mean
        ≈ 29 vs GO mean ≈ 69), so a raw std comparison would be misleading.
    Display: bar chart of CV (primary) and std (secondary), plus a combined
    box plot showing the full distribution shape for every parameter.

    Returns
    -------
    (fig, variability_df) — variability_df has one row per parameter.
    """
    sub = ds_filter(df, years=[year], category=category)
    available = params or [p for p in NIRF_PARAMS if p in sub.columns and sub[p].notna().any()]

    if not available:
        fig = go.Figure()
        apply_theme(fig, title=f"Parameter Variability — {category} ({year}): no parameter data")
        return fig, pd.DataFrame()

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.45, 0.55],
        subplot_titles=["CV & Std Dev per Parameter", "Score Distributions per Parameter"],
        horizontal_spacing=0.10,
    )

    # Compute stats
    rows = []
    for p in available:
        s = sub[p].dropna()
        stat = ds_summary_stats(s, label=p)
        rows.append(stat)
    var_df = pd.DataFrame(rows).rename(columns={"label": "Parameter"})

    # Left panel — CV bars + Std line
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(available))]
    fig.add_trace(
        go.Bar(
            x=var_df["Parameter"],
            y=var_df["cv"],
            name="CV (%)",
            marker_color=colors,
            hovertemplate="<b>%{x}</b><br>CV: %{y:.2f}%<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=var_df["Parameter"],
            y=var_df["std"],
            name="Std Dev",
            mode="lines+markers",
            line=dict(color=ACCENT_RED, width=2.5, dash="dot"),
            marker=dict(size=8, color=ACCENT_RED),
            yaxis="y2",
            hovertemplate="<b>%{x}</b><br>Std Dev: %{y:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Right panel — box per parameter
    for i, p in enumerate(available):
        s = sub[p].dropna()
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(
            go.Box(
                y=s,
                name=p,
                marker_color=color,
                line_color=color,
                boxmean=True,
                hovertemplate=f"<b>{p}</b><br>Score: %{{y:.2f}}<extra></extra>",
                showlegend=False,
            ),
            row=1, col=2,
        )

    fig.update_layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT_COLOR, family="Inter, system-ui, sans-serif"),
        title=dict(
            text=f"Parameter Variability | {category} · {year}",
            x=0.05, font=dict(size=15, color=TEXT_COLOR),
        ),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(23,25,30,0.8)", bordercolor=GRID_COLOR,
            borderwidth=1, font=dict(color=TEXT_COLOR, size=11),
        ),
        hoverlabel=dict(
            bgcolor=PANEL_BG, bordercolor=GRID_COLOR,
            font=dict(color=TEXT_COLOR, size=12),
        ),
        yaxis2=dict(
            title=dict(text="Std Dev", font=dict(size=11, color=ACCENT_RED)),
            overlaying="y",
            side="right",
            tickfont=dict(color=ACCENT_RED, size=10),
            showgrid=False,
        ),
        margin=dict(l=60, r=80, t=80, b=60),
    )
    fig.update_xaxes(tickfont=dict(color=MUTED_TEXT, size=11), gridcolor=GRID_COLOR)
    fig.update_yaxes(tickfont=dict(color=MUTED_TEXT, size=11), gridcolor=GRID_COLOR)
    fig.update_yaxes(title_text="CV (%)", row=1, col=1)
    fig.update_yaxes(title_text="Score (0–100)", row=1, col=2)

    return fig, var_df


# ---------------------------------------------------------------------------
# Q6.4 — Median and IQR over time
# ---------------------------------------------------------------------------

def q6_4_median_iqr_trend(
    df: pd.DataFrame,
    category: str,
    states: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
) -> Tuple[go.Figure, pd.DataFrame]:
    """
    Q6.4 — How do median and IQR of scores change over time?

    Methodology
    -----------
    For each year (optionally filtered by state), compute:
      Q1 (25th pct), median (50th pct), Q3 (75th pct), IQR = Q3 − Q1.
    Plot the median as a line and shade the Q1–Q3 band (IQR band).
    The shaded band shows how "wide" the middle 50 % of scores are:
      - Narrowing band → institutions converging in score.
      - Widening band → growing inequality between mid-tier institutions.

    Returns
    -------
    (fig, trend_df) — trend_df has columns [Year, q1, median, q3, iqr, n].
    """
    sub = ds_filter(df, years=years, category=category, states=states)
    sub = sub.dropna(subset=["score"])

    if sub.empty:
        fig = go.Figure()
        apply_theme(fig, title=f"Median & IQR Trend — {category}")
        return fig, pd.DataFrame()

    # Aggregate by year
    agg = (
        sub.groupby("year")["score"]
        .agg(
            q1=lambda x: x.quantile(0.25),
            median="median",
            q3=lambda x: x.quantile(0.75),
            n="count",
        )
        .reset_index()
        .rename(columns={"year": "Year"})
    )
    agg["iqr"] = agg["q3"] - agg["q1"]
    agg = agg.round(2)

    fig = go.Figure()

    # Shaded IQR band (Q1 → Q3)
    fig.add_trace(go.Scatter(
        x=pd.concat([agg["Year"], agg["Year"][::-1]]),
        y=pd.concat([agg["q3"], agg["q1"][::-1]]),
        fill="toself",
        fillcolor=f"rgba({int(LINE_COLOR[1:3],16)},{int(LINE_COLOR[3:5],16)},{int(LINE_COLOR[5:7],16)},0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="IQR Band (Q1–Q3)",
        hoverinfo="skip",
    ))

    # Q1 and Q3 boundary lines (dashed)
    for col, label, dash in [("q1", "Q1 (25th pct)", "dot"), ("q3", "Q3 (75th pct)", "dot")]:
        fig.add_trace(go.Scatter(
            x=agg["Year"], y=agg[col],
            mode="lines",
            name=label,
            line=dict(color=MUTED_TEXT, width=1, dash=dash),
            hovertemplate=f"Year: %{{x}}<br>{label}: %{{y:.2f}}<extra></extra>",
        ))

    # Median line (solid, prominent)
    fig.add_trace(go.Scatter(
        x=agg["Year"], y=agg["median"],
        mode="lines+markers",
        name="Median",
        line=dict(color=LINE_COLOR, width=3),
        marker=dict(size=8, color=LINE_COLOR),
        hovertemplate="Year: %{x}<br>Median: %{y:.2f}<br>IQR: %{customdata:.2f}<extra></extra>",
        customdata=agg["iqr"],
    ))

    apply_theme(
        fig,
        title=(
            f"Median Score & IQR Band Over Time | {category}"
            + (f" · {', '.join(map(str, states))}" if states else "")
        ),
        xaxis_title="Year",
        yaxis_title="Overall Score",
        show_legend=True,
    )
    fig.update_xaxes(tickmode="linear", dtick=1)

    return fig, agg


# ---------------------------------------------------------------------------
# Q6.5 — Score distribution across states
# ---------------------------------------------------------------------------

def q6_5_score_by_state(
    df: pd.DataFrame,
    year: int,
    category: str,
    selected_states: Optional[List[str]] = None,
    top_n: int = 10,
    min_institutions: int = 3,
    chart_type: str = "violin",
) -> Tuple[go.Figure, pd.DataFrame]:
    """
    Q6.5 — How does score distribution differ across states?

    Methodology
    -----------
    Groups institutions by state for a given year/category and compares
    score distributions via violin or box plot.

    - `min_institutions` : States with fewer than this many institutions are
      excluded to avoid misleading statistics from tiny samples.
    - `top_n` : When no specific states are selected, automatically picks the
      top-N states by institution count (after applying min_institutions filter).
    - States are sorted by median score (descending) so the chart reads
      "best → worst" top to bottom.

    Returns
    -------
    (fig, state_summary_df) — one row per state with summary stats.
    """
    sub = ds_filter(df, years=[year], category=category)
    sub = sub.dropna(subset=["score"])

    if sub.empty:
        fig = go.Figure()
        apply_theme(fig, title=f"Score by State — {category} ({year})")
        return fig, pd.DataFrame()

    # Count institutions per state; apply minimum threshold
    state_counts = sub.groupby("state")["name"].nunique().reset_index(name="n_inst")
    valid_states = state_counts[state_counts["n_inst"] >= min_institutions]["state"].tolist()

    if not valid_states:
        fig = go.Figure()
        apply_theme(fig, title=f"Score by State — {category} ({year}): no state meets minimum count")
        return fig, pd.DataFrame()

    # Determine which states to show
    if selected_states:
        show_states = [s for s in selected_states if s in valid_states]
        if not show_states:
            show_states = valid_states
    else:
        # Top-N by institution count
        top_states = (
            state_counts[state_counts["state"].isin(valid_states)]
            .nlargest(top_n, "n_inst")["state"]
            .tolist()
        )
        show_states = top_states

    sub = sub[sub["state"].isin(show_states)]

    # Sort states by median score descending
    state_median = sub.groupby("state")["score"].median().sort_values(ascending=False)
    ordered_states = state_median.index.tolist()

    fig = go.Figure()

    for i, state in enumerate(ordered_states):
        s = sub[sub["state"] == state]["score"]
        color = PALETTE[i % len(PALETTE)]
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

        if chart_type == "violin":
            fig.add_trace(go.Violin(
                y=s,
                name=state,
                line_color=color,
                fillcolor=f"rgba({r},{g},{b},0.3)",
                box_visible=True,
                meanline_visible=True,
                points="outliers",
                hovertemplate=f"<b>{state}</b><br>Score: %{{y:.2f}}<extra></extra>",
            ))
        else:  # box
            fig.add_trace(go.Box(
                y=s,
                name=state,
                marker_color=color,
                line_color=color,
                boxmean=True,
                hovertemplate=f"<b>{state}</b><br>Score: %{{y:.2f}}<extra></extra>",
            ))

    apply_theme(
        fig,
        title=(
            f"Score Distribution by State | {category} · {year}<br>"
            f"<sub>States sorted by median score · min {min_institutions} institutions per state</sub>"
        ),
        xaxis_title="State",
        yaxis_title="Overall Score",
        show_legend=False,
    )
    fig.update_layout(margin=dict(l=60, r=40, t=90, b=120))
    fig.update_xaxes(tickangle=-35, tickfont=dict(size=10, color=MUTED_TEXT))

    # Build summary table
    summary_rows = []
    for state in ordered_states:
        s = sub[sub["state"] == state]["score"]
        row = ds_summary_stats(s, label=state)
        row["n_institutions"] = int(state_counts[state_counts["state"] == state]["n_inst"].values[0])
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).rename(columns={"label": "State"})

    return fig, summary_df


# ===========================================================================
# SECTION C — STREAMLIT TAB RENDERER
# ===========================================================================

def _takeaway_card(text: str) -> None:
    """Renders the standard dark takeaway card without the side heading."""
    import streamlit as st
    st.markdown(
        f"""<div class="takeaway-card">{text}</div>""",
        unsafe_allow_html=True,
    )


def _warn(note: Optional[str]) -> None:
    """Renders an st.warning if note is not None."""
    if note:
        import streamlit as st
        st.warning(note)


def render_distribution_tab(df: pd.DataFrame) -> None:
    """
    Main entry point. Teammates import and call this inside tabs[5].

    Usage in app.py
    ---------------
        from charts.distribution_analysis import render_distribution_tab
        with tabs[5]:
            render_distribution_tab(df_full)

    Parameters
    ----------
    df : Full NIRF DataFrame returned by load_nirf_data().
    """
    import streamlit as st

    # ── Build filter option lists ─────────────────────────────────────────
    all_categories = sorted(df["category"].dropna().unique().tolist())
    if "Overall" in all_categories:
        all_categories.remove("Overall")
        all_categories.insert(0, "Overall")

    all_years  = sorted(df["year"].dropna().unique().tolist())
    all_states = sorted(df["state"].dropna().unique().tolist())

    # ── Trends-style two-column layout ────────────────────────────────────
    col_q, col_p = st.columns([1.5, 1.2])

    QUESTIONS = [
        "How does score distribution differ between Top-10 and remaining institutions?",
        "Are NIRF scores more dispersed in one year than another?",
        "Which NIRF parameter has the greatest variability across institutions?",
        "How do median and IQR of scores change over time?",
        "How does score distribution differ across states?",
    ]

    with col_q:
        question = st.selectbox("Question", QUESTIONS, index=0)

    # ================================================================
    # Q 6.1 — Top-10 vs Rest
    # ================================================================
    if question == QUESTIONS[0]:
        with col_p:
            year      = st.selectbox("Year", all_years, index=len(all_years)-1, key="ds_q1_year")
            category  = st.selectbox("Category", all_categories, key="ds_q1_cat")
            chart_type = st.radio("Chart type", ["Violin", "Box", "Histogram"],
                                   horizontal=True, key="ds_q1_chart")

        fig, summary_df = q6_1_top10_vs_rest(
            df, year=year, category=category,
            chart_type=chart_type.lower(),
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        if not summary_df.empty:
            top10_row = summary_df[summary_df["label"] == "Top 10"].iloc[0]
            rest_row  = summary_df[summary_df["label"] == "Rest"].iloc[0]
            gap = round(top10_row["mean"] - rest_row["mean"], 2)
            takeaway = (
                f"Top-10 mean score: <b>{top10_row['mean']}</b> vs Rest: "
                f"<b>{rest_row['mean']}</b> — gap of <b>{gap} pts</b>. "
                f"Top-10 std: {top10_row['std']} (IQR {top10_row['iqr']}); "
                f"Rest std: {rest_row['std']} (IQR {rest_row['iqr']})."
            )
            _takeaway_card(takeaway)
            with st.expander("📊 Summary statistics", expanded=True):
                display_cols = ["label", "n", "mean", "median", "std", "cv", "q1", "q3", "iqr"]
                st.dataframe(
                    summary_df[display_cols].rename(columns={
                        "label": "Group", "n": "N", "mean": "Mean",
                        "median": "Median", "std": "Std Dev", "cv": "CV (%)",
                        "q1": "Q1", "q3": "Q3", "iqr": "IQR",
                    }),
                    width="stretch",
                )

    # ================================================================
    # Q 6.2 — Score dispersion across years
    # ================================================================
    elif question == QUESTIONS[1]:
        with col_p:
            category       = st.selectbox("Category", all_categories, key="ds_q2_cat")
            selected_years = st.multiselect("Years to compare", all_years,
                                            default=all_years, key="ds_q2_years")

        if not selected_years:
            st.info("Select at least one year to compare.")
        else:
            fig, disp_df = q6_2_dispersion_over_years(
                df, category=category, selected_years=selected_years,
            )
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

            if not disp_df.empty:
                max_cv_row = disp_df.loc[disp_df["cv"].idxmax()]
                min_cv_row = disp_df.loc[disp_df["cv"].idxmin()]
                takeaway = (
                    f"Most dispersed year: <b>{max_cv_row['Year']}</b> "
                    f"(CV = {max_cv_row['cv']:.1f}%, std = {max_cv_row['std']:.2f}). "
                    f"Least dispersed year: <b>{min_cv_row['Year']}</b> "
                    f"(CV = {min_cv_row['cv']:.1f}%, std = {min_cv_row['std']:.2f})."
                )
                _takeaway_card(takeaway)
                with st.expander("📊 Dispersion table", expanded=False):
                    st.dataframe(
                        disp_df[["Year", "n", "mean", "std", "cv", "variance", "iqr"]].rename(columns={
                            "n": "N", "mean": "Mean", "std": "Std Dev",
                            "cv": "CV (%)", "variance": "Variance", "iqr": "IQR",
                        }),
                        width="stretch",
                    )

    # ================================================================
    # Q 6.3 — Parameter variability
    # ================================================================
    elif question == QUESTIONS[2]:
        with col_p:
            year     = st.selectbox("Year", all_years, index=len(all_years)-1, key="ds_q3_year")
            category = st.selectbox("Category", all_categories, key="ds_q3_cat")

        fig, var_df = q6_3_parameter_variability(df, year=year, category=category)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        if not var_df.empty:
            most_variable = var_df.loc[var_df["cv"].idxmax()]
            least_variable = var_df.loc[var_df["cv"].idxmin()]
            takeaway = (
                f"Most variable parameter: <b>{most_variable['Parameter']}</b> "
                f"(CV = {most_variable['cv']:.1f}%, std = {most_variable['std']:.2f}). "
                f"Least variable: <b>{least_variable['Parameter']}</b> "
                f"(CV = {least_variable['cv']:.1f}%). "
                "CV (Coefficient of Variation) normalises by mean, enabling fair cross-parameter comparison."
            )
            _takeaway_card(takeaway)
            with st.expander("📊 Variability table", expanded=False):
                st.dataframe(
                    var_df[["Parameter", "n", "mean", "std", "cv", "iqr"]].rename(columns={
                        "n": "N", "mean": "Mean", "std": "Std Dev", "cv": "CV (%)", "iqr": "IQR",
                    }),
                    width="stretch",
                )

    # ================================================================
    # Q 6.4 — Median and IQR over time
    # ================================================================
    elif question == QUESTIONS[3]:
        with col_p:
            category      = st.selectbox("Category", all_categories, key="ds_q4_cat")
            states_filter = st.multiselect("Filter by State (optional)",
                                           all_states, key="ds_q4_states")
            years_filter  = st.multiselect("Filter Years (optional, default = all)",
                                           all_years, default=[], key="ds_q4_years")

        fig, trend_df = q6_4_median_iqr_trend(
            df,
            category=category,
            states=states_filter if states_filter else None,
            years=years_filter if years_filter else None,
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        if not trend_df.empty:
            first = trend_df.iloc[0]
            last  = trend_df.iloc[-1]
            med_change = round(last["median"] - first["median"], 2)
            iqr_change = round(last["iqr"] - first["iqr"], 2)
            takeaway = (
                f"Median score <b>{'rose' if med_change >= 0 else 'fell'}</b> by "
                f"<b>{abs(med_change):.2f} pts</b> ({first['Year']} → {last['Year']}). "
                f"IQR <b>{'widened' if iqr_change >= 0 else 'narrowed'}</b> by "
                f"<b>{abs(iqr_change):.2f} pts</b> — indicating "
                f"{'growing spread' if iqr_change >= 0 else 'convergence'} in the middle 50% of institutions."
            )
            _takeaway_card(takeaway)
            with st.expander("📊 Median / IQR table per year", expanded=False):
                st.dataframe(
                    trend_df[["Year", "n", "q1", "median", "q3", "iqr"]].rename(columns={
                        "n": "N", "q1": "Q1 (25th)", "median": "Median",
                        "q3": "Q3 (75th)", "iqr": "IQR",
                    }),
                    width="stretch",
                )

    # ================================================================
    # Q 6.5 — Score distribution by state
    # ================================================================
    elif question == QUESTIONS[4]:
        with col_p:
            year           = st.selectbox("Year", all_years, index=len(all_years)-1, key="ds_q5_year")
            category       = st.selectbox("Category", all_categories, key="ds_q5_cat")
            chart_type5    = st.radio("Chart type", ["Violin", "Box"],
                                      horizontal=True, key="ds_q5_chart")
            top_n          = st.slider("Top N states (by institution count)", 3, 20, 10, key="ds_q5_topn")
            min_inst       = st.slider("Min institutions per state", 1, 10, 3, key="ds_q5_min")
            states_pick    = st.multiselect("Or pick specific states", all_states, key="ds_q5_states")

        fig, state_df = q6_5_score_by_state(
            df,
            year=year,
            category=category,
            selected_states=states_pick if states_pick else None,
            top_n=top_n,
            min_institutions=min_inst,
            chart_type=chart_type5.lower(),
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        if not state_df.empty:
            best_state  = state_df.iloc[0]   # sorted by median desc already
            worst_state = state_df.iloc[-1]
            takeaway = (
                f"Highest median: <b>{best_state['State']}</b> "
                f"({best_state['median']:.1f} pts, n={best_state['n_institutions']}). "
                f"Lowest median: <b>{worst_state['State']}</b> "
                f"({worst_state['median']:.1f} pts). "
                f"Spread (IQR) is widest in <b>"
                f"{state_df.loc[state_df['iqr'].idxmax(), 'State']}</b> "
                f"(IQR = {state_df['iqr'].max():.1f} pts)."
            )
            _takeaway_card(takeaway)
            with st.expander("📊 State summary table", expanded=False):
                st.dataframe(
                    state_df[["State", "n_institutions", "mean", "median", "std", "cv", "iqr"]].rename(columns={
                        "n_institutions": "# Institutions", "mean": "Mean",
                        "median": "Median", "std": "Std Dev",
                        "cv": "CV (%)", "iqr": "IQR",
                    }),
                    width="stretch",
                )
