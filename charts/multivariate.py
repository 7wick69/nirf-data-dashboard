"""
charts/multivariate.py

Multivariate visualizations for NIRF ranking data -- combined module.

Teammate functions (registry.py-compatible, return (fig, takeaway_str)):
  - plot_top5_radar            : Top-5 radar/spider chart
  - plot_quadrant_analysis     : 4-quadrant scatter (High X vs Low Y)
  - plot_strongest_weakest_gap : Largest parameter gap horizontal bar

Praneeth's Section 5 -- Annexure A, Q5.1-Q5.5 (return (fig, df)):
  - q5_1_top5_radar            : Q5.1 radar + grouped bar
  - q5_2_research_vs_teaching  : Q5.2 scatter quadrant
  - q5_3_teaching_vs_perception: Q5.3 scatter quadrant
  - q5_4_improvement_heatmap   : Q5.4 score improvement heatmap/bar
  - q5_5_param_spread          : Q5.5 largest param gap spread
  - render_multivariate_tab    : Main Streamlit entry point (import this)

Author : Praneeth + Teammates
Team   : NIRF Dashboard Project
"""

from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from charts.theme import (
    apply_theme,
    LINE_COLOR,
    MUTED_TEXT,
    TEXT_COLOR,
    PANEL_BG,
    DARK_BG,
    GRID_COLOR,
    ACCENT_GREEN,
    ACCENT_RED,
)
try:
    from charts.theme import QUALITATIVE_PALETTE, PALETTE
except ImportError:
    from charts.theme import PALETTE
    QUALITATIVE_PALETTE = PALETTE

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

    param_complementary_colors = {
        "TLR": "#38bdf8",  # Sky Blue (Teaching)
        "RPC": "#f97316",  # Vibrant Orange (Research) - complementary to Blue
        "GO":  "#10b981",  # Emerald Green (Graduation Outcomes)
        "OI":  "#f43f5e",  # Rose Red (Outreach & Inclusivity) - complementary to Green
        "PR":  "#a855f7",  # Purple (Perception)
    }
    bar_colors = [param_complementary_colors.get(p, LINE_COLOR) for p in top_gaps["best_param"]]

    fig = go.Figure(
        go.Bar(
            y=top_gaps["name"],
            x=top_gaps["param_gap"].round(1),
            orientation="h",
            marker=dict(color=bar_colors),
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


# ===========================================================================
# SECTION 5 -- Praneeth's Multivariate Analysis (Annexure A, Q5.1-Q5.5)
# ===========================================================================


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Standard NIRF sub-parameters in display order.
NIRF_PARAMS: List[str] = ["TLR", "RPC", "GO", "OI", "PR"]

#: Long-form labels used in axis titles / tooltips.
NIRF_PARAM_LABELS: dict[str, str] = {
    "TLR": "Teaching, Learning & Resources",
    "RPC": "Research & Professional Practice",
    "GO":  "Graduation Outcomes",
    "OI":  "Outreach & Inclusivity",
    "PR":  "Perception",
}

#: Maximum possible marks per parameter (all 100 per NIRF methodology).
#: Used in Q5.5 normalisation. Update here if NIRF changes the scale.
PARAM_MAX: dict[str, float] = {p: 100.0 for p in NIRF_PARAMS}


# ===========================================================================
# SECTION A -- DATA / COMPUTATION HELPERS
# (Pure functions -- no Streamlit imports. Safe to unit-test independently.)
# ===========================================================================

def mv_filter(
    df: pd.DataFrame,
    year: Optional[int] = None,
    category: Optional[str] = None,
    states: Optional[List[str]] = None,
    institutions: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Generic filter gate used by every Q5.x function.

    Parameters
    ----------
    df           : Full NIRF DataFrame (output of load_nirf_data()).
    year         : Single year integer to keep. None -> all years.
    category     : Category string (e.g. "Overall", "Engineering"). None -> all.
    states       : List of state names. None -> all states.
    institutions : List of institution names. None -> all institutions.

    Returns
    -------
    Filtered copy of df (never modifies in place).
    """
    mask = pd.Series(True, index=df.index)
    if year is not None:
        mask &= df["year"] == year
    if category is not None:
        mask &= df["category"] == category
    if states:
        mask &= df["state"].isin(states)
    if institutions:
        mask &= df["name"].isin(institutions)
    return df[mask].copy()


def mv_available_params(df: pd.DataFrame) -> List[str]:
    """
    Returns the subset of NIRF_PARAMS that actually exist in df AND have at
    least one non-NaN value.

    Why needed: categories that use the 'Research' or 'Rank-only' param sets
    do not publish TLR / RPC / GO, so those columns are entirely NaN for
    those rows. This guard prevents downstream errors.
    """
    return [
        p for p in NIRF_PARAMS
        if p in df.columns and df[p].notna().any()
    ]


def mv_note_missing_params(available: List[str]) -> Optional[str]:
    """
    Builds a human-readable warning string when some standard params are absent.
    Returns None if all five standard params are available (no note needed).
    """
    missing = [p for p in NIRF_PARAMS if p not in available]
    if not missing:
        return None
    return (
        f"\u26a0\ufe0f Parameters **{', '.join(missing)}** are not available for the "
        "selected category / year (Research or rank-only categories do not "
        "publish these scores). Analysis uses only the available parameters."
    )


# ---------------------------------------------------------------------------
# Q5.1 helper
# ---------------------------------------------------------------------------

def mv_1_top5_data(
    df: pd.DataFrame,
    year: int,
    category: str,
    params: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Extracts parameter scores for the top-5 ranked institutions.

    Methodology
    -----------
    1. Filter to the given year and category.
    2. Drop rows where rank_num is NaN (rank-band entries without a numeric rank).
    3. Sort ascending by rank_num and take the first 5.

    Returns
    -------
    top5_df : columns [name, rank_num, <available params>], sorted by rank.
    note    : warning string if any params are missing, else None.
    """
    sub = mv_filter(df, year=year, category=category)
    available = params or mv_available_params(sub)
    note = mv_note_missing_params(available)

    sub = sub.dropna(subset=["rank_num"]).sort_values("rank_num").head(5)

    if sub.empty:
        return pd.DataFrame(), "No data found for the selected Year / Category."

    cols = ["name", "rank_num"] + [p for p in available if p in sub.columns]
    return sub[cols].reset_index(drop=True), note


# ---------------------------------------------------------------------------
# Q5.2 / Q5.3 shared helper
# ---------------------------------------------------------------------------

def mv_2_quadrant_data(
    df: pd.DataFrame,
    year: int,
    category: str,
    x_param: str = "RPC",
    y_param: str = "TLR",
    x_threshold: Optional[float] = None,
    y_threshold: Optional[float] = None,
) -> Tuple[pd.DataFrame, float, float, Optional[str]]:
    """
    Prepares scatter data for a two-parameter quadrant analysis.

    Methodology
    -----------
    - Filters to year + category.
    - Drops rows missing either axis parameter.
    - Computes median thresholds when user does not supply them.

    Parameters
    ----------
    x_param / y_param  : Column names for the two axes.
    x_threshold        : Vertical divider position. None -> median of x_param.
    y_threshold        : Horizontal divider position. None -> median of y_param.

    Returns
    -------
    (scatter_df, x_thresh, y_thresh, note)
    scatter_df has columns: [name, state, rank_num, score, x_param, y_param]
    """
    sub = mv_filter(df, year=year, category=category)
    available = mv_available_params(sub)
    note = mv_note_missing_params(available)

    if x_param not in available or y_param not in available:
        return pd.DataFrame(), 50.0, 50.0, (
            f"\u26a0\ufe0f Parameters **{x_param}** and/or **{y_param}** are not available "
            "for the selected category / year."
        )

    keep = ["name", "state", "rank_num", "score", x_param, y_param]
    sub = sub.dropna(subset=[x_param, y_param])[keep].copy()

    x_thresh = x_threshold if x_threshold is not None else float(sub[x_param].median())
    y_thresh = y_threshold if y_threshold is not None else float(sub[y_param].median())

    return sub.reset_index(drop=True), x_thresh, y_thresh, note


# ---------------------------------------------------------------------------
# Q5.4 helper
# ---------------------------------------------------------------------------

def mv_4_improvement_data(
    df: pd.DataFrame,
    category: str,
    year1: int,
    year2: int,
    params: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Identifies institutions whose overall score INCREASED from year1 -> year2
    but had at least one sub-parameter DECREASE.

    Methodology
    -----------
    1. Subset data for year1 and year2 separately.
    2. Inner-merge on institution name (only institutions present in both years).
    3. Compute ?score = score_y2 - score_y1  and  ?param = param_y2 - param_y1.
    4. Keep rows where ?score > 0 AND any(?param < 0).
    5. Annotate with n_declined (count of declining params) and
       strongest_decline (the param with the largest negative change).

    Returns
    -------
    result_df : columns [name, state, score_change, dTLR, dRPC, dGO, dOI, dPR,
                          n_declined, strongest_decline]
    note      : missing-param warning or None.
    """
    sub1 = mv_filter(df, year=year1, category=category)
    sub2 = mv_filter(df, year=year2, category=category)

    # Determine params available in both year slices
    available = params or [
        p for p in NIRF_PARAMS
        if (p in sub1.columns and sub1[p].notna().any())
        and (p in sub2.columns and sub2[p].notna().any())
    ]
    note = mv_note_missing_params(available)

    if sub1.empty or sub2.empty:
        return pd.DataFrame(), (
            f"No data for category '{category}' in year {year1} or {year2}."
        )

    keep = ["name", "state", "score"] + available

    df1 = (
        sub1[keep]
        .dropna(subset=["score"])
        .rename(columns={c: f"{c}_y1" for c in ["score"] + available})
    )
    df2 = (
        sub2[keep]
        .dropna(subset=["score"])
        .rename(columns={c: f"{c}_y2" for c in ["score"] + available})
    )

    # Inner join: only institutions that appear in BOTH years
    merged = df1.merge(
        df2[["name"] + [f"{c}_y2" for c in ["score"] + available]],
        on="name",
        how="inner",
    )

    if merged.empty:
        return pd.DataFrame(), (
            "No institutions found in both selected years for this category."
        )

    # Compute changes
    merged["score_change"] = merged["score_y2"] - merged["score_y1"]
    for p in available:
        merged[f"d{p}"] = merged[f"{p}_y2"] - merged[f"{p}_y1"]

    dcols = [f"d{p}" for p in available]

    # Apply the dual criterion
    improved_mask    = merged["score_change"] > 0
    any_decline_mask = (merged[dcols] < 0).any(axis=1)
    result = merged[improved_mask & any_decline_mask].copy()

    if result.empty:
        return pd.DataFrame(), (
            "No institutions satisfy the criteria (score up + >=1 param down) "
            "for the selected years and category."
        )

    result["n_declined"]        = (result[dcols] < 0).sum(axis=1)
    result["strongest_decline"] = (
        result[dcols].idxmin(axis=1).str.lstrip("d")
    )

    out_cols = (
        ["name", "state", "score_change"]
        + dcols
        + ["n_declined", "strongest_decline"]
    )
    return (
        result[out_cols]
        .sort_values("score_change", ascending=False)
        .reset_index(drop=True),
        note,
    )


# ---------------------------------------------------------------------------
# Q5.5 helper
# ---------------------------------------------------------------------------

def mv_5_spread_data(
    df: pd.DataFrame,
    year: int,
    category: str,
    top_n: int = 15,
    params: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Computes per-institution parameter spread = max(norm_param) - min(norm_param).

    Methodology
    -----------
    Normalisation: each raw parameter score is divided by PARAM_MAX[param] * 100.
    Since all NIRF standard parameters are scored out of 100, this is a no-op for
    the standard set, but it makes the function correct if NIRF ever changes
    the scale or if a custom param list is passed.

    Returns the top-N institutions ranked by spread (descending).

    Returns
    -------
    result_df : columns [name, state, rank_num, spread,
                          strongest_param, weakest_param,
                          TLR_norm, RPC_norm, GO_norm, OI_norm, PR_norm]
    note      : missing-param warning or None.
    """
    sub = mv_filter(df, year=year, category=category)
    available = params or mv_available_params(sub)
    note = mv_note_missing_params(available)

    if len(available) < 2:
        return pd.DataFrame(), (
            "\u26a0\ufe0f At least 2 parameters are required to compute a spread. "
            "The selected category / year has fewer than 2 available parameters."
        )

    keep = ["name", "state", "rank_num"] + available
    sub = sub.dropna(subset=available)[keep].copy()

    # Normalise each parameter to a 0-100 scale
    norm_cols: List[str] = []
    for p in available:
        col = f"{p}_norm"
        sub[col] = sub[p] / PARAM_MAX.get(p, 100.0) * 100.0
        norm_cols.append(col)

    sub["spread"]          = sub[norm_cols].max(axis=1) - sub[norm_cols].min(axis=1)
    sub["strongest_param"] = (
        sub[norm_cols].idxmax(axis=1).str.replace("_norm", "", regex=False)
    )
    sub["weakest_param"] = (
        sub[norm_cols].idxmin(axis=1).str.replace("_norm", "", regex=False)
    )

    result = sub.nlargest(top_n, "spread")[
        ["name", "state", "rank_num", "spread",
         "strongest_param", "weakest_param"] + norm_cols
    ].reset_index(drop=True)

    return result, note


# ===========================================================================
# SECTION B -- PLOTTING FUNCTIONS
# (Each returns (plotly_figure, dataframe). No Streamlit calls here.)
# ===========================================================================

def q5_1_top5_radar(
    df: pd.DataFrame,
    year: int,
    category: str,
    show_bar: bool = False,
    params: Optional[List[str]] = None,
) -> Tuple[go.Figure, pd.DataFrame]:
    """
    Q5.1 -- Radar (spider) chart comparing the top-5 institutions across all
    available NIRF parameters for a chosen year and category.

    Analytical rationale
    --------------------
    A radar chart lets evaluators simultaneously see the shape of each
    institution's strengths and weaknesses across all five dimensions.  An
    institution with a large polygon dominates most dimensions; a narrow or
    uneven polygon reveals specific weaknesses.

    Parameters
    ----------
    df       : Full NIRF DataFrame.
    year     : Year to filter on.
    category : NIRF category string.
    show_bar : If True, return a grouped bar chart instead of radar.
    params   : Override which parameters to use. Defaults to all available.

    Returns
    -------
    (fig, top5_df)
    """
    top5_df, note = mv_1_top5_data(df, year, category, params=params)
    fig = go.Figure()

    if top5_df.empty:
        apply_theme(fig, title=f"Top-5 Radar -- {category} ({year})")
        return fig, top5_df

    available = [p for p in (params or NIRF_PARAMS) if p in top5_df.columns]

    if show_bar:
        # Grouped bar chart (alternative view)
        for i, (_, row) in enumerate(top5_df.iterrows()):
            short = row["name"][:40] + ("..." if len(row["name"]) > 40 else "")
            color = PALETTE[i % len(PALETTE)]
            fig.add_trace(go.Bar(
                name=short,
                x=available,
                y=[row[p] for p in available],
                marker_color=color,
                hovertemplate=(
                    f"<b>{row['name']}</b><br>"
                    "Parameter: %{x}<br>"
                    "Score: %{y:.2f}<extra></extra>"
                ),
            ))
        apply_theme(
            fig,
            title=f"Top-5 Institutions -- Parameter Scores | {category} · {year}",
            xaxis_title="NIRF Parameter",
            yaxis_title="Score (out of 100)",
            show_legend=True,
        )
        fig.update_layout(barmode="group")

    else:
        # Radar / spider chart with complementary colors
        # Close the polygon by repeating the first param at the end
        radar_params = available + [available[0]]

        for i, (_, row) in enumerate(top5_df.iterrows()):
            values    = [row[p] for p in available] + [row[available[0]]]
            short     = row["name"][:35] + ("..." if len(row["name"]) > 35 else "")
            hex_col   = PALETTE[i % len(PALETTE)]
            r_val, g_val, b_val = (
                int(hex_col[1:3], 16),
                int(hex_col[3:5], 16),
                int(hex_col[5:7], 16),
            )
            fill_col = f"rgba({r_val},{g_val},{b_val},0.15)"

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=radar_params,
                fill="toself",
                name=short,
                line=dict(color=hex_col, width=2.5),
                fillcolor=fill_col,
                hovertemplate=(
                    f"<b>{row['name']}</b><br>"
                    "Parameter: %{theta}<br>"
                    "Score: %{r:.2f}<extra></extra>"
                ),
            ))

        fig.update_layout(
            polar=dict(
                bgcolor=PANEL_BG,
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(size=10, color=MUTED_TEXT),
                    gridcolor=GRID_COLOR,
                    linecolor=GRID_COLOR,
                ),
                angularaxis=dict(
                    tickfont=dict(size=11, color=TEXT_COLOR),
                    linecolor=GRID_COLOR,
                    gridcolor=GRID_COLOR,
                ),
            ),
            paper_bgcolor=DARK_BG,
            font=dict(color=TEXT_COLOR, family="Inter, system-ui, sans-serif"),
            title=dict(
                text=(
                    f"Top-5 Institutions -- NIRF Parameter Radar "
                    f"| {category} · {year}"
                ),
                x=0.05, y=0.97, xanchor="left",
                font=dict(size=15, color=TEXT_COLOR),
            ),
            showlegend=True,
            legend=dict(
                bgcolor="rgba(23,25,30,0.8)",
                bordercolor=GRID_COLOR,
                borderwidth=1,
                font=dict(color=TEXT_COLOR, size=11),
            ),
            margin=dict(l=80, r=80, t=80, b=80),
        )

    return fig, top5_df


def q5_2_scatter_rpc_tlr(
    df: pd.DataFrame,
    year: int,
    category: str,
    x_threshold: Optional[float] = None,
    y_threshold: Optional[float] = None,
    highlight_states: Optional[List[str]] = None,
) -> Tuple[go.Figure, pd.DataFrame]:
    """
    Q5.2 -- Scatter plot: RPC (x-axis) vs TLR (y-axis) with four-quadrant
    dividers, coloured by quadrant.

    Analytical rationale
    --------------------
    Institutions in the high-RPC / low-TLR quadrant (upper-left) have strong
    research output but weaker teaching and learning environments -- a pattern
    of interest for policy interventions to improve classroom infrastructure.

    Default thresholds are the median of each axis (50th percentile),
    making the quadrant split symmetric. Users can override with sliders.

    Parameters
    ----------
    x_threshold     : RPC divider (vertical line). None -> median RPC.
    y_threshold     : TLR divider (horizontal line). None -> median TLR.
    highlight_states: Optional; not currently used by the renderer but
                      available for future programmatic use.

    Returns
    -------
    (fig, quadrant_df)
    quadrant_df : institutions in the High-RPC / Low-TLR quadrant only.
    """
    scatter_df, x_thresh, y_thresh, note = mv_2_quadrant_data(
        df, year, category,
        x_param="RPC", y_param="TLR",
        x_threshold=x_threshold, y_threshold=y_threshold,
    )

    fig = go.Figure()
    if scatter_df.empty:
        apply_theme(fig, title=f"RPC vs TLR -- {category} ({year})")
        return fig, pd.DataFrame()

    # Assign each point to a quadrant label
    scatter_df["quadrant"] = "Other"
    scatter_df.loc[
        (scatter_df["RPC"] >= x_thresh) & (scatter_df["TLR"] <  y_thresh),
        "quadrant"
    ] = "High RPC · Low TLR"
    scatter_df.loc[
        (scatter_df["RPC"] >= x_thresh) & (scatter_df["TLR"] >= y_thresh),
        "quadrant"
    ] = "High RPC · High TLR"
    scatter_df.loc[
        (scatter_df["RPC"] <  x_thresh) & (scatter_df["TLR"] >= y_thresh),
        "quadrant"
    ] = "Low RPC · High TLR"
    scatter_df.loc[
        (scatter_df["RPC"] <  x_thresh) & (scatter_df["TLR"] <  y_thresh),
        "quadrant"
    ] = "Low RPC · Low TLR"

    color_map = {
        "High RPC · Low TLR":  ACCENT_RED,     # target quadrant
        "High RPC · High TLR": ACCENT_GREEN,
        "Low RPC · High TLR":  "#f59e0b",       # amber
        "Low RPC · Low TLR":   "#64748b",        # slate-grey
    }

    for quadrant, color in color_map.items():
        qdf = scatter_df[scatter_df["quadrant"] == quadrant]
        if qdf.empty:
            continue
        fig.add_trace(go.Scatter(
            x=qdf["RPC"],
            y=qdf["TLR"],
            mode="markers",
            name=quadrant,
            marker=dict(
                color=color, size=8, opacity=0.85,
                line=dict(width=0.5, color=DARK_BG),
            ),
            customdata=qdf[["name", "state", "score", "rank_num"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "State: %{customdata[1]}<br>"
                "RPC: %{x:.2f} | TLR: %{y:.2f}<br>"
                "Overall Score: %{customdata[2]:.2f}<br>"
                "Rank: %{customdata[3]:.0f}"
                "<extra></extra>"
            ),
        ))

    # Quadrant divider lines with threshold labels
    fig.add_hline(
        y=y_thresh, line_dash="dot", line_color=MUTED_TEXT, line_width=1.2,
        annotation_text=f"TLR median = {y_thresh:.1f}",
        annotation_font=dict(size=10, color=MUTED_TEXT),
        annotation_position="bottom right",
    )
    fig.add_vline(
        x=x_thresh, line_dash="dot", line_color=MUTED_TEXT, line_width=1.2,
        annotation_text=f"RPC median = {x_thresh:.1f}",
        annotation_font=dict(size=10, color=MUTED_TEXT),
        annotation_position="top left",
    )

    apply_theme(
        fig,
        title=f"Research vs Teaching Scores | {category} · {year}",
        xaxis_title="RPC -- Research & Professional Practice (score out of 100)",
        yaxis_title="TLR -- Teaching, Learning & Resources (score out of 100)",
        show_legend=True,
    )

    quadrant_df = (
        scatter_df[scatter_df["quadrant"] == "High RPC · Low TLR"][
            ["name", "state", "RPC", "TLR", "score", "rank_num"]
        ]
        .sort_values("RPC", ascending=False)
        .reset_index(drop=True)
    )
    return fig, quadrant_df


def q5_3_scatter_tlr_pr(
    df: pd.DataFrame,
    year: int,
    category: str,
    x_threshold: Optional[float] = None,
    y_threshold: Optional[float] = None,
) -> Tuple[go.Figure, pd.DataFrame]:
    """
    Q5.3 -- Scatter plot: TLR (x-axis) vs PR (y-axis) with four-quadrant
    dividers.

    Analytical rationale
    --------------------
    Institutions in the high-TLR / low-PR quadrant (top-left) excel at
    teaching but suffer from weak public perception -- indicating possible
    gaps in outreach, branding, or graduate placement visibility.

    Returns
    -------
    (fig, quadrant_df)
    quadrant_df : institutions in the High-TLR / Low-PR quadrant only.
    """
    scatter_df, x_thresh, y_thresh, note = mv_2_quadrant_data(
        df, year, category,
        x_param="TLR", y_param="PR",
        x_threshold=x_threshold, y_threshold=y_threshold,
    )

    fig = go.Figure()
    if scatter_df.empty:
        apply_theme(fig, title=f"TLR vs PR -- {category} ({year})")
        return fig, pd.DataFrame()

    scatter_df["quadrant"] = "Other"
    scatter_df.loc[
        (scatter_df["TLR"] >= x_thresh) & (scatter_df["PR"] <  y_thresh),
        "quadrant"
    ] = "High TLR · Low PR"
    scatter_df.loc[
        (scatter_df["TLR"] >= x_thresh) & (scatter_df["PR"] >= y_thresh),
        "quadrant"
    ] = "High TLR · High PR"
    scatter_df.loc[
        (scatter_df["TLR"] <  x_thresh) & (scatter_df["PR"] >= y_thresh),
        "quadrant"
    ] = "Low TLR · High PR"
    scatter_df.loc[
        (scatter_df["TLR"] <  x_thresh) & (scatter_df["PR"] <  y_thresh),
        "quadrant"
    ] = "Low TLR · Low PR"

    color_map = {
        "High TLR · Low PR":  ACCENT_RED,
        "High TLR · High PR": ACCENT_GREEN,
        "Low TLR · High PR":  "#f59e0b",
        "Low TLR · Low PR":   "#64748b",
    }

    for quadrant, color in color_map.items():
        qdf = scatter_df[scatter_df["quadrant"] == quadrant]
        if qdf.empty:
            continue
        fig.add_trace(go.Scatter(
            x=qdf["TLR"],
            y=qdf["PR"],
            mode="markers",
            name=quadrant,
            marker=dict(
                color=color, size=8, opacity=0.85,
                line=dict(width=0.5, color=DARK_BG),
            ),
            customdata=qdf[["name", "state", "score", "rank_num"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "State: %{customdata[1]}<br>"
                "TLR: %{x:.2f} | PR: %{y:.2f}<br>"
                "Overall Score: %{customdata[2]:.2f}<br>"
                "Rank: %{customdata[3]:.0f}"
                "<extra></extra>"
            ),
        ))

    fig.add_hline(
        y=y_thresh, line_dash="dot", line_color=MUTED_TEXT, line_width=1.2,
        annotation_text=f"PR median = {y_thresh:.1f}",
        annotation_font=dict(size=10, color=MUTED_TEXT),
        annotation_position="bottom right",
    )
    fig.add_vline(
        x=x_thresh, line_dash="dot", line_color=MUTED_TEXT, line_width=1.2,
        annotation_text=f"TLR median = {x_thresh:.1f}",
        annotation_font=dict(size=10, color=MUTED_TEXT),
        annotation_position="top left",
    )

    apply_theme(
        fig,
        title=f"Teaching vs Perception Scores | {category} · {year}",
        xaxis_title="TLR -- Teaching, Learning & Resources (score out of 100)",
        yaxis_title="PR -- Perception (score out of 100)",
        show_legend=True,
    )

    quadrant_df = (
        scatter_df[scatter_df["quadrant"] == "High TLR · Low PR"][
            ["name", "state", "TLR", "PR", "score", "rank_num"]
        ]
        .sort_values("TLR", ascending=False)
        .reset_index(drop=True)
    )
    return fig, quadrant_df


def q5_4_improvement_chart(
    df: pd.DataFrame,
    category: str,
    year1: int,
    year2: int,
    view: str = "heatmap",
    params: Optional[List[str]] = None,
) -> Tuple[go.Figure, pd.DataFrame]:
    """
    Q5.4 -- Institutions that improved overall score despite declining in
    at least one sub-parameter between year1 and year2.

    Analytical rationale
    --------------------
    A rising overall score with a falling sub-parameter suggests the institution
    compensated through other dimensions -- or that the weighting formula masked
    a real weakness. The heatmap view makes the trade-off pattern instantly
    visible: red cells = declined, green cells = improved.

    Parameters
    ----------
    view : "heatmap" (default) -- colour grid of ?param per institution.
           "bar"              -- grouped diverging horizontal bar chart.

    Returns
    -------
    (fig, result_df)
    result_df columns: [name, state, score_change, dTLR, dRPC, dGO, dOI, dPR,
                         n_declined, strongest_decline]
    """
    result_df, note = mv_4_improvement_data(
        df, category, year1, year2, params=params
    )
    fig = go.Figure()

    if result_df.empty:
        apply_theme(
            fig,
            title=(
                f"Score Improvement with Partial Decline "
                f"| {category} · {year1} -> {year2}"
            ),
        )
        return fig, result_df

    available_dcols  = [
        c for c in result_df.columns
        if c.startswith("d") and c[1:] in NIRF_PARAMS
    ]
    available_params = [c[1:] for c in available_dcols]
    result_df        = result_df.copy()
    result_df["short_name"] = result_df["name"].str[:40].str.strip()

    if view == "heatmap":
        # ?? Heatmap view ??????????????????????????????????????????????????
        z_data   = result_df[available_dcols].values
        y_labels = result_df["short_name"].tolist()

        fig = go.Figure(go.Heatmap(
            z=z_data,
            x=available_params,
            y=y_labels,
            colorscale=[
                [0.0, ACCENT_RED],
                [0.5, PANEL_BG],
                [1.0, ACCENT_GREEN],
            ],
            zmid=0,
            text=[[f"{v:+.2f}" for v in row] for row in z_data],
            texttemplate="%{text}",
            hovertemplate=(
                "Institution: %{y}<br>"
                "Parameter:   %{x}<br>"
                "Change:      %{z:+.2f}"
                "<extra></extra>"
            ),
            showscale=True,
            colorbar=dict(
                title=dict(text="Score Change (\u0394)", font=dict(color=MUTED_TEXT, size=11)),
                tickfont=dict(color=TEXT_COLOR, size=10),
                bgcolor=DARK_BG,
                bordercolor=GRID_COLOR,
            ),
        ))
        apply_theme(
            fig,
            title=(
                f"Parameter Changes for Institutions with Rising Overall Score"
                f" ({year1} \u2192 {year2})<br>"
                f"<sub>{category}  \u00b7  {len(result_df)} institutions qualify"
                f"  \u00b7  green = improved, red = declined</sub>"
            ),
            xaxis_title="NIRF Parameter",
            yaxis_title="Institution",
        )
        fig.update_layout(margin=dict(l=260, r=80, t=110, b=60))

    else:
        # Diverging bar chart view
        for param in available_params:
            dcol   = f"d{param}"
            colors = [
                ACCENT_GREEN if v >= 0 else ACCENT_RED
                for v in result_df[dcol]
            ]
            fig.add_trace(go.Bar(
                name=param,
                x=result_df[dcol],
                y=result_df["short_name"],
                orientation="h",
                marker_color=colors,
                hovertemplate=(
                    f"<b>%{{y}}</b><br>{param} change: %{{x:+.2f}}"
                    "<extra></extra>"
                ),
            ))
        apply_theme(
            fig,
            title=(
                f"Parameter-wise Changes ({year1} \u2192 {year2}) | {category}<br>"
                f"<sub>Institutions with rising overall score"
                f" but \u2265 1 parameter decline  \u00b7  {len(result_df)} institutions</sub>"
            ),
            xaxis_title="Score Change (\u0394)",
            yaxis_title="Institution",
            show_legend=True,
        )
        fig.update_layout(barmode="group", margin=dict(l=260, r=60, t=100, b=60))

    return fig, result_df.drop(columns=["short_name"], errors="ignore")


def q5_5_spread_bar(
    df: pd.DataFrame,
    year: int,
    category: str,
    top_n: int = 15,
    params: Optional[List[str]] = None,
) -> Tuple[go.Figure, pd.DataFrame]:
    """
    Q5.5 -- Horizontal bar chart of institutions with the largest spread between
    their strongest and weakest normalised NIRF parameter.

    Analytical rationale
    --------------------
    A large spread indicates an institution is highly unbalanced -- exceptional
    in one dimension but weak in another. This metric highlights structural
    imbalances that an aggregate score alone would hide.

    Normalisation ensures a fair comparison: all parameters are expressed on
    a 0-100 scale before computing max − min.

    Parameters
    ----------
    top_n : Number of institutions to display (driven by slider in UI).

    Returns
    -------
    (fig, spread_df)
    """
    spread_df, note = mv_5_spread_data(
        df, year, category, top_n=top_n, params=params
    )
    fig = go.Figure()

    if spread_df.empty:
        apply_theme(fig, title=f"Parameter Spread -- {category} ({year})")
        return fig, spread_df

    # Sort ascending so largest spread renders at the top of the chart
    spread_df = spread_df.sort_values("spread", ascending=True).copy()
    spread_df["short_name"] = spread_df["name"].str[:45].str.strip()

    # Bar colour encodes the strongest parameter using high-contrast complementary hues:
    # TLR: Sky Blue (#38bdf8) & RPC: Vibrant Orange (#f97316) -> Complementary pair
    # GO: Emerald Green (#10b981) & OI: Rose Red (#f43f5e) -> Complementary pair
    # PR: Purple (#a855f7)
    param_complementary_colors = {
        "TLR": "#38bdf8",  # Sky Blue (Teaching)
        "RPC": "#f97316",  # Vibrant Orange (Research) - complement to Sky Blue
        "GO":  "#10b981",  # Emerald Green (Graduation Outcomes)
        "OI":  "#f43f5e",  # Rose Red (Outreach & Inclusivity) - complement to Green
        "PR":  "#a855f7",  # Purple (Perception)
    }
    param_color = {p: param_complementary_colors.get(p, PALETTE[idx % len(PALETTE)]) for idx, p in enumerate(NIRF_PARAMS)}
    bar_colors  = [
        param_color.get(p, LINE_COLOR) for p in spread_df["strongest_param"]
    ]

    fig.add_trace(go.Bar(
        x=spread_df["spread"],
        y=spread_df["short_name"],
        orientation="h",
        marker_color=bar_colors,
        customdata=spread_df[
            ["name", "strongest_param", "weakest_param", "rank_num", "state"]
        ].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Spread (max − min): %{x:.2f} pts<br>"
            "Strongest parameter: <b>%{customdata[1]}</b><br>"
            "Weakest parameter:   <b>%{customdata[2]}</b><br>"
            "Rank: %{customdata[3]:.0f} | State: %{customdata[4]}"
            "<extra></extra>"
        ),
        text=[
            f"↑ {r.strongest_param}   ↓ {r.weakest_param}"
            for r in spread_df.itertuples()
        ],
        textposition="outside",
        textfont=dict(size=10, color=MUTED_TEXT),
    ))

    # Manual colour-legend annotations (one per param colour used)
    annotations = [
        dict(
            xref="paper", yref="paper",
            x=1.01,
            y=0.95 - idx * 0.08,
            text=f"<span style='color:{color}'>■</span> {p}",
            showarrow=False,
            font=dict(size=11, color=TEXT_COLOR),
            xanchor="left",
        )
        for idx, (p, color) in enumerate(param_color.items())
    ]

    apply_theme(
        fig,
        title=(
            f"Top {top_n} Institutions by Parameter Spread "
            f"| {category} · {year}<br>"
            "<sub>Spread = normalised max(param) − min(param) | "
            "bar colour = strongest parameter</sub>"
        ),
        xaxis_title="Spread (percentage points, normalised to 100-pt scale)",
        yaxis_title="Institution",
    )
    fig.update_layout(
        annotations=annotations,
        margin=dict(l=290, r=170, t=110, b=60),
        xaxis=dict(range=[0, 110]),
    )

    return fig, spread_df.drop(columns=["short_name"], errors="ignore")


# ===========================================================================
# SECTION C -- STREAMLIT TAB RENDERER
# ===========================================================================

def _render_note(note: Optional[str]) -> None:
    """Renders a Streamlit st.warning() if note is not None."""
    if note:
        import streamlit as st
        st.warning(note)


def _takeaway_card(text: str) -> None:
    """Renders the standard dark takeaway card without the side heading."""
    import streamlit as st
    st.markdown(
        f"""<div class="takeaway-card">{text}</div>""",
        unsafe_allow_html=True,
    )


def render_multivariate_tab(df: pd.DataFrame) -> None:
    """
    Main entry point for the Multivariate tab.  Teammates import and call this.

    Usage in app.py
    ---------------
        from charts.multivariate import render_multivariate_tab

        with tabs[4]:           # the existing "Multivariate" placeholder tab
            render_multivariate_tab(df_full)

    Parameters
    ----------
    df : Full NIRF DataFrame returned by load_nirf_data().
         Expected columns: year, category, name, state, rank_num, score,
                           TLR, RPC, GO, OI, PR  (plus others that are ignored).
    """
    import streamlit as st

    # ?? Build filter option lists ?????????????????????????????????????????
    all_categories = sorted(df["category"].dropna().unique().tolist())
    if "Overall" in all_categories:
        all_categories.remove("Overall")
        all_categories.insert(0, "Overall")

    all_years  = sorted(df["year"].dropna().unique().tolist())
    all_states = sorted(df["state"].dropna().unique().tolist())

    # ?? Trends-style two-column header row ???????????????????????????????
    # col_q  : question selector (left, wider)
    # col_p  : context-sensitive parameter controls (right)
    col_q, col_p = st.columns([1.5, 1.2])

    QUESTIONS = [
        "How do the top 5 institutions compare across all NIRF parameters?",
        "Which institutions have high research scores but relatively low teaching scores?",
        "Which institutions have high teaching scores but relatively low perception scores?",
        "Which institutions improved overall score despite declining in some parameters?",
        "Which institutions have the largest gap between their strongest and weakest parameter?",
    ]

    with col_q:
        question = st.selectbox("Question", QUESTIONS, index=0)

    # ================================================================
    # Q 5.1 -- Top-5 Radar / Grouped Bar
    # ================================================================
    if question == QUESTIONS[0]:
        with col_p:
            year     = st.selectbox("Year", all_years, index=len(all_years) - 1, key="mv_q1_year")
            category = st.selectbox("Category", all_categories, key="mv_q1_cat")
            view     = st.radio("Chart type", ["Radar", "Grouped Bar"], horizontal=True, key="mv_q1_view")

        fig, top5_df = q5_1_top5_radar(
            df, year=year, category=category, show_bar=(view == "Grouped Bar"),
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        # Takeaway card -- matches trends.py style exactly
        if not top5_df.empty:
            avail = mv_available_params(mv_filter(df, year=year, category=category))
            top1  = top5_df.iloc[0]
            best_param = avail[top5_df.iloc[0][[p for p in avail]].values.argmax()] if avail else "N/A"
            takeaway = (
                f"<b>{top1['name']}</b> leads the {category} category in {year}; "
                f"strongest collective parameter is <b>{best_param}</b>. "
                f"Radar shape reveals relative strengths across all {len(avail)} published parameters."
            )
            _takeaway_card(takeaway)
            _render_note(mv_note_missing_params(avail))
            with st.expander("📋 Top-5 raw data table", expanded=False):
                st.dataframe(top5_df.rename(columns={"rank_num": "Rank"}), width="stretch")

    # ================================================================
    # Q 5.2 -- RPC vs TLR scatter
    # ================================================================
    elif question == QUESTIONS[1]:
        sub_tmp     = mv_filter(df, year=all_years[-1], category=all_categories[0])
        default_rpc = float(sub_tmp["RPC"].median()) if "RPC" in sub_tmp.columns and sub_tmp["RPC"].notna().any() else 50.0
        default_tlr = float(sub_tmp["TLR"].median()) if "TLR" in sub_tmp.columns and sub_tmp["TLR"].notna().any() else 50.0

        with col_p:
            year       = st.selectbox("Year", all_years, index=len(all_years) - 1, key="mv_q2_year")
            category   = st.selectbox("Category", all_categories, key="mv_q2_cat")
            rpc_thresh = st.slider("RPC threshold", 0.0, 100.0, default_rpc, 0.5, key="mv_q2_rpc")
            tlr_thresh = st.slider("TLR threshold", 0.0, 100.0, default_tlr, 0.5, key="mv_q2_tlr")

        fig, quad_df = q5_2_scatter_rpc_tlr(
            df, year=year, category=category,
            x_threshold=rpc_thresh, y_threshold=tlr_thresh,
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        # Takeaway card
        if not quad_df.empty:
            top_inst = quad_df.iloc[0]
            takeaway = (
                f"<b>{len(quad_df)}</b> institution(s) have high research (RPC >= {rpc_thresh:.1f}) "
                f"but weaker teaching (TLR &lt; {tlr_thresh:.1f}) in {category} {year}. "
                f"Top by RPC: <b>{top_inst['name']}</b> "
                f"(RPC {top_inst['RPC']:.1f}, TLR {top_inst['TLR']:.1f})."
            )
        else:
            takeaway = f"No institutions fall in the High-RPC / Low-TLR quadrant at the current thresholds -- try lowering TLR or raising RPC threshold."
        _takeaway_card(takeaway)
        if not quad_df.empty:
            st.dataframe(
                quad_df.rename(columns={"rank_num": "Rank", "score": "Overall Score"}),
                width="stretch",
            )

    # ================================================================
    # Q 5.3 -- TLR vs PR scatter
    # ================================================================
    elif question == QUESTIONS[2]:
        sub_tmp     = mv_filter(df, year=all_years[-1], category=all_categories[0])
        default_tlr = float(sub_tmp["TLR"].median()) if "TLR" in sub_tmp.columns and sub_tmp["TLR"].notna().any() else 50.0
        default_pr  = float(sub_tmp["PR"].median())  if "PR"  in sub_tmp.columns and sub_tmp["PR"].notna().any()  else 50.0

        with col_p:
            year       = st.selectbox("Year", all_years, index=len(all_years) - 1, key="mv_q3_year")
            category   = st.selectbox("Category", all_categories, key="mv_q3_cat")
            tlr_thresh = st.slider("TLR threshold", 0.0, 100.0, default_tlr, 0.5, key="mv_q3_tlr")
            pr_thresh  = st.slider("PR threshold",  0.0, 100.0, default_pr,  0.5, key="mv_q3_pr")

        fig, quad_df = q5_3_scatter_tlr_pr(
            df, year=year, category=category,
            x_threshold=tlr_thresh, y_threshold=pr_thresh,
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        # Takeaway card
        if not quad_df.empty:
            top_inst = quad_df.iloc[0]
            takeaway = (
                f"<b>{len(quad_df)}</b> institution(s) have strong teaching (TLR >= {tlr_thresh:.1f}) "
                f"but weak public perception (PR &lt; {pr_thresh:.1f}) in {category} {year}. "
                f"Top by TLR: <b>{top_inst['name']}</b> "
                f"(TLR {top_inst['TLR']:.1f}, PR {top_inst['PR']:.1f})."
            )
        else:
            takeaway = "No institutions fall in the High-TLR / Low-PR quadrant at the current thresholds -- try adjusting the sliders."
        _takeaway_card(takeaway)
        if not quad_df.empty:
            st.dataframe(
                quad_df.rename(columns={"rank_num": "Rank", "score": "Overall Score"}),
                width="stretch",
            )

    # ================================================================
    # Q 5.4 -- Score improved, parameter(s) declined
    # ================================================================
    elif question == QUESTIONS[3]:
        with col_p:
            category = st.selectbox("Category", all_categories, key="mv_q4_cat")
            year1    = st.selectbox("From Year", all_years, index=0, key="mv_q4_y1")
            valid_y2 = [y for y in all_years if y > year1]
            year2    = st.selectbox(
                "To Year",
                valid_y2 if valid_y2 else all_years,
                index=len(valid_y2) - 1 if valid_y2 else 0,
                key="mv_q4_y2",
            )
            view4 = st.radio("Chart type", ["Heatmap", "Diverging Bar"], horizontal=True, key="mv_q4_view")

        fig, result_df = q5_4_improvement_chart(
            df, category=category, year1=year1, year2=year2,
            view="heatmap" if view4 == "Heatmap" else "bar",
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        # Takeaway card
        if not result_df.empty:
            top_inst    = result_df.iloc[0]
            avg_improve = result_df["score_change"].mean()
            takeaway = (
                f"<b>{len(result_df)}</b> institution(s) in {category} raised their overall score "
                f"from {year1} \u2192 {year2} despite \u2265 1 parameter declining. "
                f"Average score gain: <b>+{avg_improve:.2f} pts</b>. "
                f"Most improved: <b>{top_inst['name']}</b> (+{top_inst['score_change']:.2f} pts, "
                f"biggest decline in {top_inst['strongest_decline']})."
            )
            _takeaway_card(takeaway)
            display_cols = (
                ["name", "state", "score_change"]
                + [c for c in result_df.columns if c.startswith("d") and c[1:] in NIRF_PARAMS]
                + ["n_declined", "strongest_decline"]
            )
            st.dataframe(
                result_df[display_cols].rename(columns={
                    "score_change": "\u0394 Score", "n_declined": "# Params Declined",
                    "strongest_decline": "Biggest Decline In",
                    **{f"d{p}": f"\u0394 {p}" for p in NIRF_PARAMS},
                }),
                width="stretch",
            )
        else:
            _takeaway_card("No institutions satisfy the criteria for the selected years and category.")

    # ================================================================
    # Q 5.5 -- Largest parameter spread
    # ================================================================
    elif question == QUESTIONS[4]:
        with col_p:
            year          = st.selectbox("Year", all_years, index=len(all_years) - 1, key="mv_q5_year")
            category      = st.selectbox("Category", all_categories, key="mv_q5_cat")
            top_n         = st.slider("Top N institutions", 5, 30, 15, key="mv_q5_n")
            states_filter = st.multiselect("Filter by State (optional)", all_states, key="mv_q5_states")

        df_state = df[df["state"].isin(states_filter)] if states_filter else df

        fig, spread_df = q5_5_spread_bar(df_state, year=year, category=category, top_n=top_n)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        # Takeaway card
        if not spread_df.empty:
            top_inst = spread_df.iloc[-1]   # last row = largest spread (sorted ascending)
            takeaway = (
                f"Most imbalanced institution in {category} {year}: "
                f"<b>{top_inst['name']}</b> -- spread of <b>{top_inst['spread']:.1f} pts</b> "
                f"(strongest: <b>{top_inst['strongest_param']}</b>, "
                f"weakest: <b>{top_inst['weakest_param']}</b>)."
            )
            _takeaway_card(takeaway)
            _render_note(mv_note_missing_params(mv_available_params(mv_filter(df, year=year, category=category))))
            st.dataframe(
                spread_df[["name", "state", "rank_num", "spread", "strongest_param", "weakest_param"]].rename(columns={
                    "rank_num": "Rank", "spread": "Spread (pts)",
                    "strongest_param": "Strongest", "weakest_param": "Weakest",
                }),
                width="stretch",
            )