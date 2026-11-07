"""
charts/univariate.py
====================
Univariate Analysis module for the NIRF Interactive Dashboard.

TWO PUBLIC LAYERS:
──────────────────
LAYER 1  Pure analysis functions (testable, chatbot-callable):
  score_distribution(df, year, category, bins, kde)        → go.Figure
  rank_distribution(df, year, category)                    → (go.Figure, go.Figure)
  score_summary_stats(df, year, category)                  → dict
  score_summary_stats_all_years(df, category, year_range)  → pd.DataFrame
  score_quartile_share(df, year, category)                 → (pd.DataFrame, go.Figure)
  category_spread_comparison(df, years, categories)        → (pd.DataFrame, go.Figure)

LAYER 2  app.py-compatible wrappers (keep exact signatures — called by app.py):
  plot_score_distribution(df, year, category, bins)        → (go.Figure, str)
  plot_rank_distribution(df, year, category, bins)         → (go.Figure, str)
  plot_summary_statistics(df, category, year_range)        → (go.Figure, str)
  plot_rank_band_breakdown(df, category, year_range)       → (go.Figure, str)
  plot_score_spread_by_category(df, year, categories)      → (go.Figure, str)
  plot_parameter_distribution(df, year, category, params,  → (go.Figure, str)
                              chart_style)

SCHEMA DUALITY:
  app.py passes a df with LOWER-CASE columns: year, category, name, state,
  rank, score, TLR, RPC, GO, OI, PR.
  univariate_app.py / data_prep.py produce CAPITALISED columns: Year,
  Category, Institute_Name, State, Rank_Numeric, Score.
  The _col() helper resolves both transparently.

CHART RATIONALE:
  Q1 Histogram + KDE : shows full distribution shape, skew, and outlier density.
  Q2 Histogram + Band-bar: raw histogram reveals clustering; band-bar quantifies
     membership in NIRF's own tier groupings for a viva-ready summary.
  Q3 Plotly Table: year-by-year stats side-by-side — the most exam-friendly form.
  Q4 Donut: immediately communicates relative proportions; Q4 (top 25 %) stands
     out visually against the other three slices.
  Q5 Box plots: Q1/median/Q3/whiskers show spread and outliers per category;
     boxmean="sd" overlays the mean+SD indicator for a richer view.
  Q6 Violin + Box: density shape (violin) AND quartiles (box) in one chart;
     ideal for comparing multi-modal parameter distributions.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

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
PARAMETERS = ["TLR", "RPC", "GO", "OI", "PR"]

# NIRF rank bands (6 tiers — finer than the 4-band legacy set)
RANK_BANDS = [
    ("1 – 10",    1,   10),
    ("11 – 50",   11,  50),
    ("51 – 100",  51,  100),
    ("101 – 150", 101, 150),
    ("151 – 200", 151, 200),
    ("201+",      201, 9_999),
]

# Green-shade palette for rank-band charts (dark = elite tier, light = lower)
_GREEN_Q = [
    "#1a5c2a",  # 1-10   (deepest)
    "#2e8b4a",  # 11-50
    "#6abf7a",  # 51-100
    "#a3d9a5",  # 101-150
    "#c8ebca",  # 151-200
    "#e0f5e1",  # 201+   (lightest)
]

# 4-band legacy set (used by plot_rank_band_breakdown for app.py compatibility)
_LEGACY_BANDS = ["Q1  (1-10)", "Q2  (11-50)", "Q3  (51-100)", "Q4  (101+)"]
_LEGACY_GREEN = ["#1a5c2a", "#2e8b4a", "#6abf7a", "#b7e4c0"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _col(df: pd.DataFrame, *candidates: str) -> str:
    """
    Return the first column name from candidates that exists in df.
    Enables transparent handling of both lower-case (app.py) and
    capitalised (data_prep.py / univariate_app.py) schemas.

    Example
    -------
    score_col = _col(df, "score", "Score")
    year_col  = _col(df, "year", "Year")
    """
    for c in candidates:
        if c in df.columns:
            return c
    return candidates[0]  # caller must handle KeyError gracefully


def _filter(
    df: pd.DataFrame,
    year=None,
    category=None,
    state=None,
) -> pd.DataFrame:
    """
    Apply Year / Category / State filters, schema-agnostically.
    Any argument of None means 'no filter on that dimension'.
    Accepts scalars or lists for year/category/state.
    """
    sub = df.copy()
    yr_col  = _col(df, "year",     "Year")
    cat_col = _col(df, "category", "Category")
    st_col  = _col(df, "state",    "State")

    if year is not None:
        if isinstance(year, (list, tuple)):
            sub = sub[sub[yr_col].isin(year)]
        else:
            sub = sub[sub[yr_col] == year]

    if category is not None:
        if isinstance(category, (list, tuple)):
            sub = sub[sub[cat_col].isin(category)]
        else:
            sub = sub[sub[cat_col] == category]

    if state is not None:
        if isinstance(state, (list, tuple)):
            sub = sub[sub[st_col].isin(state)]
        else:
            sub = sub[sub[st_col] == state]

    return sub


def _score_series(df: pd.DataFrame) -> pd.Series:
    """Return the score column as a Series, handling both schemas."""
    return df[_col(df, "score", "Score")]


def _rank_series(df: pd.DataFrame) -> pd.Series:
    """Return numeric rank column, handling both schemas."""
    return df[_col(df, "rank", "rank_num", "Rank_Numeric")]


def _name_col(df: pd.DataFrame) -> str:
    """Return institution name column key."""
    return _col(df, "name", "display_name", "Institute_Name")


def _band_assign_6(rank_val: float) -> str:
    """Assign a rank to one of the 6 NIRF tiers."""
    if pd.isna(rank_val):
        return "Unknown"
    for label, lo, hi in RANK_BANDS:
        if lo <= rank_val <= hi:
            return label
    return "201+"


def _band_assign_legacy(rank_val: float) -> str:
    """Assign rank to legacy 4-band set (for plot_rank_band_breakdown)."""
    if pd.isna(rank_val):
        return "Unknown"
    r = float(rank_val)
    if r <= 10:
        return "Q1  (1-10)"
    if r <= 50:
        return "Q2  (11-50)"
    if r <= 100:
        return "Q3  (51-100)"
    return "Q4  (101+)"


def _empty_fig(title: str = "No data for selected filters") -> go.Figure:
    """Return a themed empty figure with a centred 'no data' annotation."""
    fig = go.Figure()
    fig.add_annotation(
        text=f"<b>{title}</b>",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color=MUTED_TEXT),
    )
    return apply_theme(fig, title=title)


# ===========================================================================
# LAYER 1 — Pure Analysis Functions
# ===========================================================================

# ─────────────────────────────────────────────────────────────────────────────
# Q1  Score Distribution
# ─────────────────────────────────────────────────────────────────────────────

def score_distribution(
    df: pd.DataFrame,
    year: int,
    category: str,
    bins: int = 20,
    kde: bool = True,
) -> go.Figure:
    """
    Histogram of NIRF scores for every ranked institution in year/category.

    Visualization choice: histogram shows the full distribution shape, skew,
    and presence of outliers — better than a simple bar of summary stats.
    The rolling-window density curve (pseudo-KDE) reveals the overall trend
    without requiring scipy.

    Parameters
    ----------
    df       : Cleaned NIRF DataFrame (lower-case or capitalised columns).
    year     : Ranking year (int).
    category : NIRF category string.
    bins     : Number of histogram bins (5–50 recommended).
    kde      : If True, overlay a smoothed density trend line.

    Returns
    -------
    go.Figure
    """
    sub = _filter(df, year=year, category=category)
    scores = _score_series(sub).dropna()

    if scores.empty:
        return _empty_fig(f"No score data — {category}, {year}")

    fig = go.Figure()

    # ── Histogram bars ──────────────────────────────────────────────────────
    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=bins,
        name="Institutions",
        marker=dict(
            color=LINE_COLOR,
            opacity=0.75,
            line=dict(color=PANEL_BG, width=0.6),
        ),
        hovertemplate="Score range: %{x}<br>Count: %{y}<extra></extra>",
    ))

    # ── Pseudo-KDE density overlay (rolling-window smooth) ──────────────────
    if kde and len(scores) > 5:
        counts, edges = np.histogram(scores, bins=bins)
        centres = 0.5 * (edges[:-1] + edges[1:])
        # Window-3 rolling mean for a gentle smooth — no extra library needed
        smooth = (
            pd.Series(counts.astype(float))
            .rolling(3, center=True, min_periods=1)
            .mean()
            .values
        )
        fig.add_trace(go.Scatter(
            x=centres, y=smooth,
            mode="lines",
            name="Density trend",
            line=dict(color="#f59e0b", width=2.5, dash="solid"),
            hoverinfo="skip",
        ))

    # ── Mean & Median vertical reference lines ───────────────────────────────
    mean_v  = float(scores.mean())
    med_v   = float(scores.median())
    fig.add_vline(
        x=mean_v,
        line=dict(color=ACCENT_GREEN, width=1.8, dash="dash"),
        annotation_text=f"Mean {mean_v:.1f}",
        annotation_font=dict(color=ACCENT_GREEN, size=11),
        annotation_position="top right",
    )
    fig.add_vline(
        x=med_v,
        line=dict(color="#f59e0b", width=1.8, dash="dot"),
        annotation_text=f"Median {med_v:.1f}",
        annotation_font=dict(color="#f59e0b", size=11),
        annotation_position="top left",
    )

    skew = float(scores.skew())
    skew_label = (
        "right-skewed" if skew > 0.3
        else "left-skewed" if skew < -0.3
        else "symmetric"
    )

    return apply_theme(
        fig,
        title=(
            f"Score Distribution — {category}, {year} "
            f"| n={len(scores)}, skew={skew:.2f} ({skew_label})"
        ),
        xaxis_title="NIRF Score (0 – 100)",
        yaxis_title="Number of Institutions",
        show_legend=kde,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Q2  Rank Distribution
# ─────────────────────────────────────────────────────────────────────────────

def rank_distribution(
    df: pd.DataFrame,
    year,
    category: str,
) -> Tuple[go.Figure, go.Figure]:
    """
    Two-chart rank distribution view:

    Chart 1 – Histogram of raw numeric ranks (shows density / clustering).
    Chart 2 – 100 % stacked bar of NIRF tier membership per year (green monochrome).
              When year="All years", shows all years side by side.

    Visualization choice: the histogram reveals raw density patterns (e.g.
    many institutions tied at the 101 lower-bound of a band), while the
    stacked bar gives the viva-ready "what fraction is elite?" answer.

    Parameters
    ----------
    df       : Cleaned NIRF DataFrame.
    year     : int or the string "All years".
    category : NIRF category string.

    Returns
    -------
    (histogram_fig, band_bar_fig)
    """
    if year == "All years":
        sub = _filter(df, category=category)
    else:
        sub = _filter(df, year=year, category=category)

    ranks = _rank_series(sub).dropna()
    yr_col = _col(df, "year", "Year")

    if ranks.empty:
        empty = _empty_fig(
            f"No rank data — {category} "
            f"{'(all years)' if year == 'All years' else year}"
        )
        return empty, empty

    # ── Chart 1: Rank histogram ──────────────────────────────────────────────
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(
        x=ranks,
        nbinsx=20,
        name="Rank frequency",
        marker=dict(
            color=PALETTE[3],
            opacity=0.80,
            line=dict(color=PANEL_BG, width=0.5),
        ),
        hovertemplate="Rank bucket: %{x}<br>Count: %{y}<extra></extra>",
    ))
    yr_label = "All Years" if year == "All years" else str(year)
    hist_fig = apply_theme(
        hist_fig,
        title=f"Rank Distribution — {category}, {yr_label}",
        xaxis_title="NIRF Rank",
        yaxis_title="Number of Institutions",
    )

    # ── Chart 2: Rank-band bar chart ─────────────────────────────────────────
    sub_ranked = sub.copy()
    sub_ranked["_band"] = _rank_series(sub_ranked).apply(_band_assign_6)
    band_order = [b[0] for b in RANK_BANDS]
    band_fig = go.Figure()

    if year == "All years":
        # 100% stacked bar grouped by year
        all_yrs = sorted(sub_ranked[yr_col].dropna().unique().tolist())
        counts = (
            sub_ranked.groupby([yr_col, "_band"])
            .size()
            .reset_index(name="cnt")
        )
        totals = counts.groupby(yr_col)["cnt"].transform("sum")
        counts["pct"] = (counts["cnt"] / totals * 100).round(1)

        for idx, (band, *_) in enumerate(RANK_BANDS):
            bdf = counts[counts["_band"] == band].set_index(yr_col)
            pct_vals = [float(bdf.loc[y, "pct"]) if y in bdf.index else 0.0 for y in all_yrs]
            cnt_vals = [int(bdf.loc[y, "cnt"]) if y in bdf.index else 0 for y in all_yrs]
            labels = [f"{p:.0f}%" if p >= 4 else "" for p in pct_vals]

            band_fig.add_trace(go.Bar(
                x=all_yrs, y=pct_vals, name=band,
                marker=dict(
                    color=_GREEN_Q[idx % len(_GREEN_Q)],
                    line=dict(color=DARK_BG, width=0.8),
                ),
                text=labels, textposition="inside",
                textfont=dict(color="#ffffff", size=11),
                insidetextanchor="middle",
                customdata=cnt_vals,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    f"<b>{band}</b><br>"
                    "Share: %{y:.1f}%<br>"
                    "Count: %{customdata}<extra></extra>"
                ),
            ))
        band_fig.update_layout(barmode="stack", bargap=0.25)
        band_fig = apply_theme(
            band_fig,
            title=f"Rank-Band Composition Over Time — {category}",
            xaxis_title="Year",
            yaxis_title="Share of Institutions (%)",
            show_legend=True,
        )
        band_fig.update_xaxes(tickmode="linear", dtick=1)
        band_fig.update_yaxes(range=[0, 101], ticksuffix="%")
        band_fig.update_layout(
            legend=dict(
                orientation="h", x=1, xanchor="right",
                y=1.08, yanchor="bottom",
                bgcolor="rgba(0,0,0,0)",
            )
        )

    else:
        # Single year — simple vertical bar
        counts = (
            sub_ranked.groupby("_band")
            .size()
            .reset_index(name="cnt")
            .set_index("_band")
            .reindex(band_order, fill_value=0)
            .reset_index()
        )
        counts["pct"] = (counts["cnt"] / counts["cnt"].sum() * 100).round(1)
        counts = counts[counts["cnt"] > 0]

        band_fig.add_trace(go.Bar(
            x=counts["_band"],
            y=counts["pct"],
            marker=dict(
                color=[_GREEN_Q[i] for i in range(len(counts))],
                line=dict(color=DARK_BG, width=0.8),
            ),
            text=[f"{p:.1f}%<br>({c})" for p, c in zip(counts["pct"], counts["cnt"])],
            textposition="inside",
            textfont=dict(color="#ffffff", size=11),
            insidetextanchor="middle",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Share: %{y:.1f}%<extra></extra>"
            ),
        ))
        band_fig = apply_theme(
            band_fig,
            title=f"Rank-Band Counts — {category}, {year}",
            xaxis_title="Rank Band",
            yaxis_title="Share (%)",
        )
        band_fig.update_yaxes(ticksuffix="%")
        band_fig.update_layout(bargap=0.3)

    return hist_fig, band_fig


# ─────────────────────────────────────────────────────────────────────────────
# Q3  Score Summary Statistics
# ─────────────────────────────────────────────────────────────────────────────

def score_summary_stats(
    df: pd.DataFrame,
    year: int,
    category: str,
) -> Dict:
    """
    Compute scalar summary statistics for a given year/category slice.

    Returns
    -------
    dict with keys:
      mean, median, min, max, std, count (all float/int),
      min_inst (name of institution with lowest score),
      max_inst (name of institution with highest score).
    All floats rounded to 2 d.p. Returns None values on empty slice.
    """
    sub = _filter(df, year=year, category=category)
    scores = _score_series(sub).dropna()

    if scores.empty:
        return {k: None for k in
                ["mean", "median", "min", "max", "std", "count",
                 "min_inst", "max_inst"]}

    score_col = _col(df, "score", "Score")
    name_c    = _name_col(df)

    sub_valid = sub.dropna(subset=[score_col])
    min_idx   = sub_valid[score_col].idxmin()
    max_idx   = sub_valid[score_col].idxmax()

    return {
        "mean"    : round(float(scores.mean()), 2),
        "median"  : round(float(scores.median()), 2),
        "min"     : round(float(scores.min()), 2),
        "max"     : round(float(scores.max()), 2),
        "std"     : round(float(scores.std()), 2),
        "count"   : int(scores.count()),
        "min_inst": sub_valid.loc[min_idx, name_c] if name_c in sub_valid.columns else "—",
        "max_inst": sub_valid.loc[max_idx, name_c] if name_c in sub_valid.columns else "—",
    }


def score_summary_stats_all_years(
    df: pd.DataFrame,
    category: str,
    year_range: Optional[Tuple[int, int]] = None,
) -> pd.DataFrame:
    """
    Year-by-year summary statistics table for the selected category.

    Parameters
    ----------
    df         : Cleaned NIRF DataFrame.
    category   : NIRF category string.
    year_range : Optional (start_year, end_year) filter tuple.

    Returns
    -------
    pd.DataFrame with columns: Year, Mean, Median, Min, Max, Std_Dev, Count.
    Sorted ascending by Year.
    """
    sub = _filter(df, category=category)
    yr_col    = _col(df, "year", "Year")
    score_col = _col(df, "score", "Score")

    if year_range:
        sub = sub[(sub[yr_col] >= year_range[0]) & (sub[yr_col] <= year_range[1])]

    if sub.empty:
        return pd.DataFrame()

    agg = (
        sub.groupby(yr_col)[score_col]
        .agg(Mean="mean", Median="median", Min="min",
             Max="max", Std_Dev="std", Count="count")
        .round(2)
        .reset_index()
        .rename(columns={yr_col: "Year"})
        .sort_values("Year")
    )
    agg["Year"] = agg["Year"].astype(int)
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Q4  Score Quartile Share
# ─────────────────────────────────────────────────────────────────────────────

def score_quartile_share(
    df: pd.DataFrame,
    year: int,
    category: str,
) -> Tuple[pd.DataFrame, go.Figure]:
    """
    Split institutions into 4 score quartiles and visualize as a donut chart.

    Quartile boundaries are computed from the SELECTED DATA SLICE, not from
    the global distribution, so comparisons within a year/category are
    internally consistent.

    Visualization choice: donut chart gives an immediate visual answer to
    "what fraction is in the top 25%?" — far more intuitive than a bar chart
    for proportion-of-whole questions. The donut center shows the exact cut-offs.

    Parameters
    ----------
    df       : Cleaned NIRF DataFrame.
    year     : Ranking year (int).
    category : NIRF category string.

    Returns
    -------
    df_quartile  : Per-institution DataFrame with Quartile column.
    fig          : Donut chart go.Figure.
    """
    sub = _filter(df, year=year, category=category)
    score_col = _col(df, "score", "Score")
    name_c    = _name_col(df)
    sub = sub.dropna(subset=[score_col]).copy()

    if sub.empty:
        return pd.DataFrame(), _empty_fig(f"No data — {category}, {year}")

    # Compute quartile boundaries from this slice
    q25, q50, q75 = sub[score_col].quantile([0.25, 0.50, 0.75])

    def _label(s):
        if s <= q25: return "Q1 (bottom 25 %)"
        if s <= q50: return "Q2 (25 – 50 %)"
        if s <= q75: return "Q3 (50 – 75 %)"
        return                "Q4 (top 25 %)"

    sub["Quartile"] = sub[score_col].apply(_label)

    q_order = {
        "Q1 (bottom 25 %)": 1,
        "Q2 (25 – 50 %)":   2,
        "Q3 (50 – 75 %)":   3,
        "Q4 (top 25 %)":    4,
    }
    sub["_q_order"] = sub["Quartile"].map(q_order)

    # Counts + percentages
    counts = (
        sub.groupby("Quartile")
        .size()
        .reset_index(name="Count")
    )
    counts["Pct"] = (counts["Count"] / counts["Count"].sum() * 100).round(1)
    counts["_ord"] = counts["Quartile"].map(q_order)
    counts = counts.sort_values("_ord").drop(columns="_ord")

    # Green shades Q1 dark → Q4 light
    q_colors = ["#1a5c2a", "#2e8b4a", "#6abf7a", "#c8ebca"]

    fig = go.Figure(go.Pie(
        labels=counts["Quartile"],
        values=counts["Count"],
        hole=0.52,
        textinfo="label+percent",
        textposition="outside",
        marker=dict(
            colors=q_colors,
            line=dict(color=DARK_BG, width=2),
        ),
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Count: %{value}<br>"
            "Share: %{percent}<extra></extra>"
        ),
        sort=False,
    ))

    # Donut center annotation — score cut-off values
    fig.add_annotation(
        text=(
            f"<b>Score cut-offs</b><br>"
            f"Q1 ≤ {q25:.1f}<br>"
            f"Q2 ≤ {q50:.1f}<br>"
            f"Q3 ≤ {q75:.1f}"
        ),
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=10, color=MUTED_TEXT,
                  family="Inter, monospace"),
        align="center",
    )

    fig = apply_theme(
        fig,
        title=f"Score Quartile Share — {category}, {year}",
        show_legend=True,
    )
    fig.update_layout(
        legend=dict(
            orientation="h", x=0.5, xanchor="center",
            y=-0.08, yanchor="top",
        )
    )

    # Return per-institution quartile table
    out_col_candidates = [
        name_c,
        _col(df, "state", "State"),
        _col(df, "rank", "rank_num", "Rank_Numeric"),
        score_col,
        "Quartile",
    ]
    out_cols = [c for c in out_col_candidates if c in sub.columns]
    return (
        sub[out_cols].sort_values(score_col, ascending=False).reset_index(drop=True),
        fig,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Q5  Category Spread Comparison
# ─────────────────────────────────────────────────────────────────────────────

def category_spread_comparison(
    df: pd.DataFrame,
    years: List[int],
    categories: List[str],
) -> Tuple[pd.DataFrame, go.Figure]:
    """
    Violin plot (box inside) of score spread per NIRF category.

    Visualization choice: violin shows the density SHAPE (revealing bimodal
    or skewed distributions that a plain box hides), while the embedded box
    still shows Q1/Q3/median concisely. This is richer than a box plot alone.

    Parameters
    ----------
    df         : Cleaned NIRF DataFrame.
    years      : List of years to include. Empty list = all years.
    categories : List of category strings to compare.

    Returns
    -------
    stats_df : DataFrame with Std Dev, IQR, Range, CV per category.
    fig      : Violin go.Figure.
    """
    score_col = _col(df, "score", "Score")
    cat_col   = _col(df, "category", "Category")

    sub = _filter(
        df,
        year=years if years else None,
        category=categories if categories else None,
    )
    sub = sub.dropna(subset=[score_col])

    if sub.empty:
        return pd.DataFrame(), _empty_fig("No data for selected filters")

    fig = go.Figure()
    cat_list = sorted(sub[cat_col].unique())

    for idx, cat in enumerate(cat_list):
        cat_scores = sub[sub[cat_col] == cat][score_col]
        color = PALETTE[idx % len(PALETTE)]
        fig.add_trace(go.Violin(
            y=cat_scores,
            name=cat,
            box_visible=True,
            meanline_visible=True,
            fillcolor=color,
            opacity=0.60,
            line_color=color,
            hovertemplate=f"<b>{cat}</b><br>Score: %{{y:.2f}}<extra></extra>",
        ))

    yr_label = (
        f"{min(years)} – {max(years)}" if years else "All Years"
    )
    fig = apply_theme(
        fig,
        title=f"Score Spread by Category — {yr_label}",
        xaxis_title="NIRF Category",
        yaxis_title="Score (0 – 100)",
        show_legend=False,
    )
    fig.update_xaxes(tickangle=-30)
    fig.update_layout(violingap=0.15, violinmode="overlay")

    # ── Summary statistics table ──────────────────────────────────────────────
    rows = []
    for cat in cat_list:
        s = sub[sub[cat_col] == cat][score_col]
        q1_v = float(s.quantile(0.25))
        q3_v = float(s.quantile(0.75))
        mean_v = float(s.mean())
        rows.append({
            "Category" : cat,
            "N"        : len(s),
            "Mean"     : round(mean_v, 2),
            "Std Dev"  : round(float(s.std()), 2),
            "IQR"      : round(q3_v - q1_v, 2),
            "Range"    : round(float(s.max() - s.min()), 2),
            "CV (%)"   : round(s.std() / mean_v * 100, 1) if mean_v > 0 else None,
        })
    stats_df = pd.DataFrame(rows).sort_values("Std Dev", ascending=False)
    return stats_df, fig


# ===========================================================================
# LAYER 2 — app.py-Compatible Wrappers
# (Keep exact function signatures — called directly from app.py)
# ===========================================================================

# ─────────────────────────────────────────────────────────────────────────────
# Q1  plot_score_distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_score_distribution(
    df: pd.DataFrame,
    year: int,
    category: str = "Overall",
    bins: int = 20,
) -> Tuple[go.Figure, str]:
    """
    Histogram of overall scores for every ranked institution in year/category.
    A smoothed frequency curve is overlaid using a rolling-window density estimate.

    Args:
        df       : Full NIRF enriched DataFrame (output of load_nirf_data).
        year     : The ranking year to analyse.
        category : NIRF category filter.
        bins     : Number of histogram bins (default 20).

    Returns:
        (go.Figure, takeaway_str)
    """
    fig = score_distribution(df, year, category, bins=bins, kde=True)

    # Build takeaway from the data slice
    sub = _filter(df, year=year, category=category)
    scores = _score_series(sub).dropna()

    if scores.empty:
        return fig, f"No score records found for {category} in {year}."

    skew = float(scores.skew())
    skew_desc = (
        "right-skewed (long tail toward higher scores)" if skew > 0.3
        else "left-skewed (most institutions score high, few score low)" if skew < -0.3
        else "roughly symmetric"
    )
    takeaway = (
        f"In {year} ({category}), scores range "
        f"{scores.min():.1f} – {scores.max():.1f}. "
        f"Mean = {scores.mean():.1f}, Median = {scores.median():.1f}, "
        f"SD = {scores.std():.2f}. "
        f"Distribution is {skew_desc} (skew = {skew:.2f})."
    )
    return fig, takeaway


# ─────────────────────────────────────────────────────────────────────────────
# Q2  plot_rank_distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_rank_distribution(
    df: pd.DataFrame,
    year: int,
    category: str = "Overall",
    bins: int = 20,
) -> Tuple[go.Figure, str]:
    """
    Histogram showing how institutions are distributed across rank positions.

    Returns the HISTOGRAM figure only (for app.py compatibility).
    For both histogram + band chart, call rank_distribution() directly.

    Args:
        df       : Full NIRF enriched DataFrame.
        year     : Ranking year.
        category : NIRF category filter.
        bins     : Number of bins for the histogram.

    Returns:
        (go.Figure, takeaway_str)
    """
    sub    = _filter(df, year=year, category=category)
    ranks  = _rank_series(sub).dropna()

    fig = go.Figure()

    if ranks.empty:
        fig = _empty_fig(f"No rank data — {category}, {year}")
        return fig, f"No rank records found for {category} in {year}."

    fig.add_trace(go.Histogram(
        x=ranks,
        nbinsx=bins,
        name="Rank frequency",
        marker=dict(
            color=PALETTE[3],
            opacity=0.80,
            line=dict(color=PANEL_BG, width=0.5),
        ),
        hovertemplate="Rank bucket: %{x}<br>Count: %{y}<extra></extra>",
    ))

    fig = apply_theme(
        fig,
        title=f"Rank Distribution — {category}, {year}",
        xaxis_title="NIRF Rank",
        yaxis_title="Number of Institutions",
    )

    takeaway = (
        f"{len(ranks)} institutions ranked in {category}, {year}, "
        f"with ranks spanning 1 – {int(ranks.max())}. "
        f"Median rank = {ranks.median():.0f}."
    )
    return fig, takeaway


# ─────────────────────────────────────────────────────────────────────────────
# Q3  plot_summary_statistics
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary_statistics(
    df: pd.DataFrame,
    category: str = "Overall",
    year_range: Optional[Tuple[int, int]] = None,
) -> Tuple[go.Figure, str]:
    """
    Year-by-year table of score summary stats (mean, median, min, max, SD, N)
    presented as a Plotly Table with alternating dark-row coloring.

    Visualization choice: a table is the clearest way to compare summary
    statistics across all years simultaneously — better than sparklines when
    exact numbers matter.

    Args:
        df         : Full NIRF enriched DataFrame.
        category   : NIRF category filter.
        year_range : (start, end) year filter tuple.

    Returns:
        (go.Figure, takeaway_str)
    """
    agg = score_summary_stats_all_years(df, category, year_range)

    if agg.empty:
        fig = _empty_fig(f"No data — {category}")
        return fig, f"No score data for {category}."

    n = len(agg)
    # Alternating row fill for readability
    row_fills = [[PANEL_BG if i % 2 == 0 else "#1a1d24" for i in range(n)]] * len(agg.columns)

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["<b>Year</b>", "<b>Mean</b>", "<b>Median</b>",
                    "<b>Min</b>", "<b>Max</b>", "<b>SD</b>", "<b>N</b>"],
            fill_color="#1f2937",
            align="center",
            font=dict(color=TEXT_COLOR, size=12),
            height=36,
            line_color=GRID_COLOR,
        ),
        cells=dict(
            values=[
                agg["Year"],
                agg["Mean"],
                agg["Median"],
                agg["Min"],
                agg["Max"],
                agg["Std_Dev"],
                agg["Count"].astype(int),
            ],
            fill_color=row_fills,
            align="center",
            font=dict(color=TEXT_COLOR, size=11),
            height=30,
            line_color=GRID_COLOR,
        ),
    )])

    fig = apply_theme(
        fig, title=f"Score Summary Statistics by Year — {category}"
    )

    best_year = int(agg.loc[agg["Mean"].idxmax(), "Year"])
    best_mean = float(agg["Mean"].max())
    takeaway = (
        f"Highest average score for {category} was {best_mean:.2f} in {best_year}. "
        f"Mean ranged from {agg['Mean'].min():.2f} to {best_mean:.2f} "
        f"across the selected period."
    )
    return fig, takeaway


# ─────────────────────────────────────────────────────────────────────────────
# Q4  plot_rank_band_breakdown
# ─────────────────────────────────────────────────────────────────────────────

def plot_rank_band_breakdown(
    df: pd.DataFrame,
    category: str = "Overall",
    year_range: Optional[Tuple[int, int]] = None,
) -> Tuple[go.Figure, str]:
    """
    100 % stacked bar chart showing the % of institutions in each rank-band
    per year, using 4 legacy bands (Q1=1-10, Q2=11-50, Q3=51-100, Q4=101+).

    Visual style: 4 monochromatic green shades, darkest at bottom (Q1=elite)
    to lightest at top (Q4=lower tier); percentage labels inside each segment.

    Args:
        df         : Full NIRF enriched DataFrame.
        category   : NIRF category filter.
        year_range : (start, end) year filter.

    Returns:
        (go.Figure, takeaway_str)
    """
    yr_col  = _col(df, "year", "Year")
    filtered = _filter(df, category=category)

    if year_range:
        filtered = filtered[
            (filtered[yr_col] >= year_range[0]) &
            (filtered[yr_col] <= year_range[1])
        ]
    filtered = filtered.dropna(subset=[_col(df, "rank", "rank_num", "Rank_Numeric")])

    if filtered.empty:
        fig = _empty_fig(f"No rank data — {category}")
        return fig, "No rank data available for the selected filters."

    rk_col = _col(df, "rank", "rank_num", "Rank_Numeric")
    filtered = filtered.copy()
    filtered["_band"] = filtered[rk_col].apply(_band_assign_legacy)

    # Aggregate counts → percentages per year
    counts = (
        filtered.groupby([yr_col, "_band"])
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby(yr_col)["count"].transform("sum")
    counts["pct"] = (counts["count"] / totals * 100).round(1)

    years = sorted(counts[yr_col].unique())
    fig = go.Figure()

    for band, color in zip(_LEGACY_BANDS, _LEGACY_GREEN):
        band_df = counts[counts["_band"] == band].set_index(yr_col)
        pct_vals = [float(band_df.loc[y, "pct"]) if y in band_df.index else 0.0 for y in years]
        cnt_vals = [int(band_df.loc[y, "count"]) if y in band_df.index else 0 for y in years]
        text_labels = [f"{p:.0f}%" if p >= 4 else "" for p in pct_vals]

        fig.add_trace(go.Bar(
            x=years, y=pct_vals, name=band,
            marker=dict(
                color=color,
                line=dict(color=DARK_BG, width=0.8),
            ),
            text=text_labels,
            textposition="inside",
            textfont=dict(color="#ffffff", size=12,
                          family="Inter, system-ui, sans-serif"),
            insidetextanchor="middle",
            hovertemplate=(
                "<b>%{x}</b><br>"
                f"<b>{band}</b><br>"
                "Share: %{y:.1f}%<br>"
                "Count: %{customdata}<extra></extra>"
            ),
            customdata=cnt_vals,
        ))

    fig.update_layout(barmode="stack")
    fig = apply_theme(
        fig,
        title=f"Rank-Band Distribution by Year — {category}",
        xaxis_title="Year",
        yaxis_title="Share of Institutions (%)",
        show_legend=True,
    )
    fig.update_xaxes(tickmode="linear", dtick=1)
    fig.update_yaxes(range=[0, 101], ticksuffix="%")
    fig.update_layout(
        legend=dict(
            orientation="h", x=1.0, xanchor="right",
            y=1.08, yanchor="bottom",
            bgcolor="rgba(0,0,0,0)", font=dict(size=11),
        ),
        bargap=0.25,
    )

    # Auto takeaway
    q1_data = counts[counts["_band"] == "Q1  (1-10)"].sort_values(yr_col)
    q4_data = counts[counts["_band"] == "Q4  (101+)"].sort_values(yr_col)

    if len(q1_data) >= 2:
        first_yr = int(q1_data.iloc[0][yr_col])
        last_yr  = int(q1_data.iloc[-1][yr_col])
        q1_start = float(q1_data.iloc[0]["pct"])
        q1_end   = float(q1_data.iloc[-1]["pct"])
        q4_end   = float(q4_data.iloc[-1]["pct"]) if not q4_data.empty else 0
        takeaway = (
            f"Top-10 (Q1) share: {q1_start:.1f}% in {first_yr} → "
            f"{q1_end:.1f}% in {last_yr}. "
            f"In {last_yr}, the 101+ tier (Q4) accounts for "
            f"{q4_end:.1f}% of all ranked institutions."
        )
    else:
        takeaway = "Rank-band composition shown for the selected years."

    return fig, takeaway


# ─────────────────────────────────────────────────────────────────────────────
# Q5  plot_score_spread_by_category
# ─────────────────────────────────────────────────────────────────────────────

def plot_score_spread_by_category(
    df: pd.DataFrame,
    year: int,
    categories: Optional[List[str]] = None,
) -> Tuple[go.Figure, str]:
    """
    Box plots showing the spread of NIRF scores across selected categories.
    Each box shows Q1, median, Q3, whiskers, and outlier points.

    Visualization choice: box plots provide a compact multi-category comparison
    of spread. boxmean="sd" adds a mean diamond and SD indicator, so viewers
    can distinguish skewed from symmetric distributions at a glance.

    Args:
        df         : Full NIRF enriched DataFrame.
        year       : Ranking year to compare.
        categories : List of NIRF categories to include. None = all.

    Returns:
        (go.Figure, takeaway_str)
    """
    score_col = _col(df, "score", "Score")
    cat_col   = _col(df, "category", "Category")

    sub = _filter(df, year=year, category=categories if categories else None)
    sub = sub.dropna(subset=[score_col])

    if sub.empty:
        fig = _empty_fig(f"No data — {year}")
        return fig, f"No score data found for {year}."

    cat_list = sorted(sub[cat_col].unique())
    fig = go.Figure()

    for idx, cat in enumerate(cat_list):
        cat_scores = sub[sub[cat_col] == cat][score_col]
        color = PALETTE[idx % len(PALETTE)]
        fig.add_trace(go.Box(
            y=cat_scores,
            name=cat,
            marker=dict(color=color, size=4, opacity=0.7),
            line=dict(color=color),
            boxmean="sd",          # show mean diamond + SD indicator
            hovertemplate=(
                f"<b>{cat}</b><br>"
                "Score: %{y:.2f}<extra></extra>"
            ),
        ))

    fig = apply_theme(
        fig,
        title=f"Score Spread Across Categories — {year}",
        xaxis_title="NIRF Category",
        yaxis_title="Score (0 – 100)",
        show_legend=False,
    )
    fig.update_xaxes(tickangle=-30)

    medians = (
        sub.groupby(cat_col)[score_col]
        .median()
        .sort_values(ascending=False)
    )
    top_cat = medians.index[0]
    bot_cat = medians.index[-1]
    takeaway = (
        f"In {year}, '{top_cat}' has the highest median score "
        f"({medians.iloc[0]:.1f}) and '{bot_cat}' the lowest "
        f"({medians.iloc[-1]:.1f}). "
        f"Box width reflects intra-category variability; "
        f"the diamond marker shows the mean."
    )
    return fig, takeaway


# ─────────────────────────────────────────────────────────────────────────────
# Q6  plot_parameter_distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_parameter_distribution(
    df: pd.DataFrame,
    year: int,
    category: str = "Overall",
    params: Optional[List[str]] = None,
    chart_style: str = "violin",   # "violin" | "box"
) -> Tuple[go.Figure, str]:
    """
    Distribution of NIRF sub-parameters (TLR, RPC, GO, OI, PR) for a year/category.

    Violin : full density shape; best for revealing multi-modality.
    Box    : classic quartile view; faster to explain in a viva.

    Args:
        df          : Full NIRF enriched DataFrame.
        year        : Ranking year.
        category    : NIRF category filter.
        params      : Subset of parameters to show. None = all five.
        chart_style : "violin" or "box".

    Returns:
        (go.Figure, takeaway_str)
    """
    params = params or PARAMETERS
    # Keep only params that exist in the dataframe
    params = [p for p in params if p in df.columns]

    sub = _filter(df, year=year, category=category)

    if sub.empty or not params:
        fig = _empty_fig(f"No parameter data — {category}, {year}")
        return fig, f"No parameter data for {category} in {year}."

    fig = go.Figure()

    for idx, param in enumerate(params):
        vals = sub[param].dropna()
        color = PALETTE[idx % len(PALETTE)]

        if chart_style == "violin":
            fig.add_trace(go.Violin(
                y=vals,
                name=param,
                box_visible=True,          # embed mini box inside violin
                meanline_visible=True,
                fillcolor=color,
                opacity=0.65,
                line_color=color,
                hovertemplate=(
                    f"<b>{param}</b><br>"
                    "Value: %{y:.2f}<extra></extra>"
                ),
            ))
        else:
            fig.add_trace(go.Box(
                y=vals,
                name=param,
                marker=dict(color=color, size=4, opacity=0.6),
                line=dict(color=color),
                boxmean="sd",
                hovertemplate=(
                    f"<b>{param}</b><br>"
                    "Value: %{y:.2f}<extra></extra>"
                ),
            ))

    chart_label = "Violin" if chart_style == "violin" else "Box"
    fig = apply_theme(
        fig,
        title=f"Parameter Distribution ({chart_label}) — {category}, {year}",
        xaxis_title="Sub-Parameter",
        yaxis_title="Parameter Score (0 – 100)",
        show_legend=False,
    )

    # Takeaway: widest and narrowest IQR
    iqrs = {}
    for p in params:
        col_data = sub[p].dropna()
        if len(col_data) > 0:
            iqrs[p] = float(col_data.quantile(0.75) - col_data.quantile(0.25))

    if iqrs:
        widest   = max(iqrs, key=iqrs.get)
        narrowest = min(iqrs, key=iqrs.get)
        takeaway = (
            f"In {year} ({category}), '{widest}' shows the widest spread "
            f"(IQR = {iqrs[widest]:.1f}) and '{narrowest}' the narrowest "
            f"(IQR = {iqrs[narrowest]:.1f}). "
            f"A wider IQR indicates more inequality among institutions on that parameter."
        )
    else:
        takeaway = "Parameter distributions shown for the selected filters."

    return fig, takeaway
