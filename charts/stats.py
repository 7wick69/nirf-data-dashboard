"""
charts/stats.py

Statistical visualizations for Univariate, Bivariate, and Distribution analysis:
  - Score & Rank Histograms with KDE
  - Box and Violin plots across categories/tiers
  - Two-variable Scatter with Pearson r and OLS trendline
  - Parameter correlation matrix / heatmap
  - Quartile & Tier distribution bar charts
  - IQR and Dispersion over time
"""

from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from charts.theme import (
    apply_theme,
    LINE_COLOR,
    ACCENT_GREEN,
    ACCENT_RED,
    MUTED_TEXT,
    TEXT_COLOR,
    PANEL_BG,
    QUALITATIVE_PALETTE,
)

PARAMETERS = ["TLR", "RPC", "GO", "OI", "PR"]


# ---------------------------------------------------------
# Univariate Visualizations
# ---------------------------------------------------------
def plot_score_distribution(
    df: pd.DataFrame,
    year: int = 2024,
    category: str = "Overall",
    metric: str = "score",
    bins: int = 20,
) -> Tuple[go.Figure, str]:
    """
    Plots the histogram and distribution curve of a score or parameter for a given year.
    """
    sub = df[(df["year"] == year) & (df["category"] == category) & df[metric].notna()]
    if sub.empty:
        fig = go.Figure()
        return apply_theme(fig, title=f"No data for {category} in {year}"), "No records found."

    vals = sub[metric]
    mean_val = vals.mean()
    median_val = vals.median()
    std_val = vals.std()

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=vals,
            nbinsx=bins,
            marker_color=LINE_COLOR,
            opacity=0.8,
            name="Count",
            hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
        )
    )

    # Add vertical mean and median reference lines
    fig.add_vline(x=mean_val, line_dash="dash", line_color="#f97316", annotation_text=f"Mean: {mean_val:.1f}")
    fig.add_vline(x=median_val, line_dash="dot", line_color="#10b981", annotation_text=f"Median: {median_val:.1f}")

    fig = apply_theme(
        fig,
        title=f"Distribution of {metric.upper()} ({year}, {category})",
        xaxis_title=f"{metric.upper()} Value",
        yaxis_title="Number of Institutions",
    )

    takeaway = (
        f"{metric.upper()} in {year} ({category}): Mean = {mean_val:.1f}, Median = {median_val:.1f}, "
        f"Std Dev = {std_val:.1f} across {len(vals)} institutions."
    )
    return fig, takeaway


def plot_category_spread(
    df: pd.DataFrame,
    year: int = 2024,
    metric: str = "score",
    categories: Optional[List[str]] = None,
) -> Tuple[go.Figure, str]:
    """
    Compares the score or parameter spread across ranking categories using box plots.
    """
    sub = df[(df["year"] == year) & df[metric].notna()].copy()
    if categories:
        sub = sub[sub["category"].isin(categories)]
    else:
        # Pick top 5 categories by count
        top_cats = sub["category"].value_counts().head(5).index.tolist()
        sub = sub[sub["category"].isin(top_cats)]

    if sub.empty:
        fig = go.Figure()
        return apply_theme(fig, title="No data available"), "No comparative data."

    fig = go.Figure()
    cats = sorted(sub["category"].unique())

    for idx, cat in enumerate(cats):
        cat_vals = sub[sub["category"] == cat][metric]
        color = QUALITATIVE_PALETTE[idx % len(QUALITATIVE_PALETTE)]
        fig.add_trace(
            go.Box(
                y=cat_vals,
                name=cat,
                marker_color=color,
                boxmean=True,
                hovertemplate="<b>%{x}</b><br>Score: %{y:.1f}<extra></extra>",
            )
        )

    fig = apply_theme(
        fig,
        title=f"{metric.upper()} Spread Across Categories ({year})",
        xaxis_title="Category",
        yaxis_title=f"{metric.upper()} (0–100)",
        show_legend=False,
    )

    medians = sub.groupby("category")[metric].median().sort_values(ascending=False)
    highest_cat = medians.index[0]
    takeaway = f"Highest median {metric.upper()} in {year}: {highest_cat} ({medians.iloc[0]:.1f}), with highest variance in {sub.groupby('category')[metric].std().idxmax()}."
    return fig, takeaway


# ---------------------------------------------------------
# Bivariate Visualizations
# ---------------------------------------------------------
def plot_bivariate_scatter(
    df: pd.DataFrame,
    x_col: str = "RPC",
    y_col: str = "score",
    year: int = 2024,
    category: str = "Overall",
    add_trendline: bool = True,
) -> Tuple[go.Figure, str]:
    """
    Two-variable scatter plot with Pearson correlation r and OLS linear trendline.
    """
    sub = df[(df["year"] == year) & (df["category"] == category) & df[x_col].notna() & df[y_col].notna()].copy()

    if len(sub) < 3:
        fig = go.Figure()
        return apply_theme(fig, title="Insufficient data for correlation"), "Need at least 3 points."

    # Compute Pearson correlation r
    r = sub[x_col].corr(sub[y_col])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sub[x_col],
            y=sub[y_col],
            mode="markers",
            marker=dict(color=LINE_COLOR, size=8, opacity=0.75, line=dict(width=1, color="#0f1013")),
            text=sub["name"],
            hovertemplate="<b>%{text}</b><br>" + f"{x_col}: " + "%{x:.1f}<br>" + f"{y_col}: " + "%{y:.1f}<extra></extra>",
            name="Institutions",
        )
    )

    # OLS Trendline
    if add_trendline:
        x_vals = sub[x_col].values
        y_vals = sub[y_col].values
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 50)
        y_line = slope * x_line + intercept

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name=f"OLS Fit (r = {r:.2f})",
                line=dict(color="#f97316", width=2, dash="dash"),
            )
        )

    r_strength = "strong positive" if r >= 0.7 else ("moderate positive" if r >= 0.4 else ("weak positive" if r > 0 else "negative"))
    fig = apply_theme(
        fig,
        title=f"{x_col.upper()} vs {y_col.upper()} ({year}, {category}) — Pearson r = {r:.2f}",
        xaxis_title=f"{x_col.upper()} Score",
        yaxis_title=f"{y_col.upper()} Score",
        show_legend=True,
    )

    takeaway = f"Pearson correlation r = {r:.2f} ({r_strength} correlation between {x_col.upper()} and {y_col.upper()} in {year})."
    return fig, takeaway


def plot_parameter_correlations(
    df: pd.DataFrame,
    year: int = 2024,
    category: str = "Overall",
    target_col: str = "score",
) -> Tuple[go.Figure, str]:
    """
    Renders correlation of all 5 parameters (TLR, RPC, GO, OI, PR) against target (score or rank).
    """
    sub = df[(df["year"] == year) & (df["category"] == category)]
    cols = [p for p in PARAMETERS if p in sub.columns]

    if not cols or sub.empty:
        fig = go.Figure()
        return apply_theme(fig, title="No data available"), "No correlation data."

    corrs = {col: sub[col].corr(sub[target_col]) for col in cols}
    corr_df = pd.DataFrame(list(corrs.items()), columns=["Parameter", "Correlation"]).sort_values(by="Correlation", ascending=True)

    fig = go.Figure(
        go.Bar(
            y=corr_df["Parameter"],
            x=corr_df["Correlation"],
            orientation="h",
            marker=dict(color=LINE_COLOR),
            hovertemplate="<b>%{y}</b><br>Correlation with " + target_col.upper() + ": %{x:.3f}<extra></extra>",
        )
    )

    fig = apply_theme(
        fig,
        title=f"Parameter Correlation with {target_col.upper()} ({year}, {category})",
        xaxis_title="Pearson Correlation (r)",
        yaxis_title="Parameter",
    )
    fig.update_xaxes(range=[0, 1])

    strongest = corr_df.iloc[-1]
    takeaway = f"{strongest['Parameter']} has the strongest correlation with {target_col.upper()} (r = {strongest['Correlation']:.2f}) in {year}."
    return fig, takeaway


def plot_correlation_evolution(
    df: pd.DataFrame,
    param_col: str = "RPC",
    target_col: str = "score",
    category: str = "Overall",
) -> Tuple[go.Figure, str]:
    """
    Tracks how the correlation between a parameter (e.g. Research/RPC) and overall score evolved 2017–2025.
    """
    sub = df[(df["category"] == category) & df[param_col].notna() & df[target_col].notna()]
    years = sorted(sub["year"].unique())

    records = []
    for yr in years:
        yr_df = sub[sub["year"] == yr]
        if len(yr_df) >= 5:
            r = yr_df[param_col].corr(yr_df[target_col])
            records.append({"year": yr, "r": r})

    res_df = pd.DataFrame(records)
    if res_df.empty:
        fig = go.Figure()
        return apply_theme(fig, title="Insufficient data"), "No multi-year correlation data."

    fig = go.Figure(
        go.Scatter(
            x=res_df["year"],
            y=res_df["r"].round(3),
            mode="lines+markers",
            line=dict(color=LINE_COLOR, width=3),
            marker=dict(size=8, color=LINE_COLOR),
            hovertemplate="Year %{x}: r = %{y:.3f}<extra></extra>",
        )
    )

    fig = apply_theme(
        fig,
        title=f"Evolution of {param_col} vs {target_col.upper()} Correlation (2017–2025, {category})",
        xaxis_title="Year",
        yaxis_title="Pearson Correlation Coefficient (r)",
    )
    fig.update_xaxes(tickmode="linear", dtick=1)

    first_r = res_df["r"].iloc[0]
    last_r = res_df["r"].iloc[-1]
    takeaway = f"Correlation between {param_col} and {target_col.upper()} moved from r = {first_r:.2f} (in {res_df['year'].iloc[0]}) to r = {last_r:.2f} (in {res_df['year'].iloc[-1]})."
    return fig, takeaway


# ---------------------------------------------------------
# Distribution & Dispersion Comparison
# ---------------------------------------------------------
def plot_top10_vs_rest(
    df: pd.DataFrame,
    year: int = 2024,
    category: str = "Overall",
) -> Tuple[go.Figure, str]:
    """
    Compares the parameter scores of Top 10 institutions against the rest (Rank 11-100).
    """
    sub = df[(df["year"] == year) & (df["category"] == category) & (df["rank"] <= 100)].copy()
    if sub.empty:
        fig = go.Figure()
        return apply_theme(fig, title="No data available"), "No data."

    sub["Group"] = np.where(sub["rank"] <= 10, "Top 10", "Rank 11–100")

    available_params = [p for p in PARAMETERS if p in sub.columns]
    agg = sub.groupby("Group")[available_params].mean().reset_index()

    fig = go.Figure()
    for grp, color in [("Top 10", LINE_COLOR), ("Rank 11–100", MUTED_TEXT)]:
        grp_data = agg[agg["Group"] == grp]
        if not grp_data.empty:
            fig.add_trace(
                go.Bar(
                    name=grp,
                    x=available_params,
                    y=[grp_data[p].iloc[0] for p in available_params],
                    marker_color=color,
                )
            )

    fig = apply_theme(
        fig,
        title=f"Top 10 vs Rest (Rank 11–100) Parameter Breakdown ({year}, {category})",
        xaxis_title="Parameter",
        yaxis_title="Average Score (0–100)",
        show_legend=True,
    )
    fig.update_layout(barmode="group")

    top10_rpc = sub[sub["Group"] == "Top 10"]["RPC"].mean()
    rest_rpc = sub[sub["Group"] == "Rank 11–100"]["RPC"].mean()
    rpc_gap = top10_rpc - rest_rpc
    takeaway = f"Largest disparity between Top 10 and rest is in Research & Professional Practice (RPC) with a {rpc_gap:.1f} point gap."
    return fig, takeaway


def plot_dispersion_over_time(
    df: pd.DataFrame,
    category: str = "Overall",
) -> Tuple[go.Figure, str]:
    """
    Plots Interquartile Range (IQR) and Standard Deviation of scores over time.
    """
    cat_df = df[df["category"] == category].copy()
    stats = cat_df.groupby("year")["score"].agg(
        std="std",
        q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75),
    ).reset_index()

    stats["iqr"] = stats["q75"] - stats["q25"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=stats["year"],
            y=stats["std"].round(2),
            mode="lines+markers",
            name="Standard Deviation",
            line=dict(color=LINE_COLOR, width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=stats["year"],
            y=stats["iqr"].round(2),
            mode="lines+markers",
            name="Interquartile Range (IQR)",
            line=dict(color="#f97316", width=2, dash="dash"),
        )
    )

    fig = apply_theme(
        fig,
        title=f"Score Dispersion Over Time (Std Dev & IQR, {category})",
        xaxis_title="Ranking Year",
        yaxis_title="Dispersion (Points)",
        show_legend=True,
    )
    fig.update_xaxes(tickmode="linear", dtick=1)

    takeaway = f"Score dispersion (IQR) moved from {stats['iqr'].iloc[0]:.1f} in {stats['year'].iloc[0]} to {stats['iqr'].iloc[-1]:.1f} in {stats['year'].iloc[-1]}."
    return fig, takeaway

