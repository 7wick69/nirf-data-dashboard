"""
charts/trends.py

Time-series and trend visualizations for NIRF ranking data:
  1. Institution Rank Over Time (Line chart with inverted y-axis & takeaway)
  2. Biggest Improvers & Decliners (Dumbbell / horizontal delta chart)
  3. Average Score & Parameter Progression (Multi-metric trendlines)
  4. Consistently Top-10 Institutions (Bump / multi-line trajectory)
"""

from typing import Tuple, Optional
import pandas as pd
import plotly.graph_objects as go
from charts.theme import apply_theme, LINE_COLOR, ACCENT_GREEN, ACCENT_RED, MUTED_TEXT, PALETTE


def plot_institution_rank_trend(
    df: pd.DataFrame,
    institution_name: str,
    category: str = "Overall",
    year_range: Optional[Tuple[int, int]] = None,
) -> Tuple[go.Figure, str]:
    """
    Plots the rank of a selected institution over time.
    Inverts the Y-axis so Rank 1 appears at the very top.
    Returns (go.Figure, takeaway_text).
    """
    filtered = df[(df["name"] == institution_name) & (df["category"] == category) & df["rank"].notna()].copy()

    if year_range:
        filtered = filtered[(filtered["year"] >= year_range[0]) & (filtered["year"] <= year_range[1])]

    filtered = filtered.sort_values(by="year")

    if filtered.empty:
        fig = go.Figure()
        fig = apply_theme(fig, title=f"No data for {institution_name} in {category}")
        return fig, f"No ranking records found for {institution_name} in category '{category}'."

    first_row = filtered.iloc[0]
    last_row = filtered.iloc[-1]
    first_year, first_rank = int(first_row["year"]), int(first_row["rank"])
    last_year, last_rank = int(last_row["year"]), int(last_row["rank"])
    years_span = last_year - first_year + 1

    # Automated takeaway calculation
    if first_rank > last_rank:
        diff = first_rank - last_rank
        takeaway = f"rank improved from {first_rank} to {last_rank} (+{diff} places) over {years_span} years"
    elif first_rank < last_rank:
        diff = last_rank - first_rank
        takeaway = f"rank declined from {first_rank} to {last_rank} (-{diff} places) over {years_span} years"
    else:
        takeaway = f"rank remained stable at {first_rank} across all {years_span} years"

    # Build Plotly Figure
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=filtered["year"],
            y=filtered["rank"],
            mode="lines+markers",
            line=dict(color=LINE_COLOR, width=3),
            marker=dict(size=8, color=LINE_COLOR, symbol="circle"),
            name=institution_name,
            hovertemplate=(
                "<b>%{x}</b><br>"
                + "Rank: #%{y}<br>"
                + "Score: %{customdata[0]:.2f}<br>"
                + "YoY Change: %{customdata[1]}<extra></extra>"
            ),
            customdata=list(
                zip(
                    filtered["score"],
                    filtered["rank_change"].apply(
                        lambda c: f"+{int(c)}" if pd.notna(c) and c > 0 else (str(int(c)) if pd.notna(c) else "N/A")
                    ),
                )
            ),
        )
    )

    # Calculate nice y-axis tick values
    min_rank = int(filtered["rank"].min())
    max_rank = int(filtered["rank"].max())
    y_min_bound = max(1, min_rank - 1)
    y_max_bound = max_rank + 1
    tick_vals = list(range(y_min_bound, y_max_bound + 1))

    fig = apply_theme(
        fig,
        title=f"Rank Trajectory: {institution_name} ({category})",
        xaxis_title="Year",
        yaxis_title="NIRF Rank",
        invert_yaxis=True,
    )

    fig.update_xaxes(
        tickmode="linear",
        tick0=filtered["year"].min(),
        dtick=1,
    )
    if len(tick_vals) <= 15:
        fig.update_yaxes(tickvals=tick_vals, ticktext=[str(v) for v in tick_vals])

    return fig, takeaway


def plot_biggest_improvers_decliners(
    df: pd.DataFrame,
    category: str = "Overall",
    start_year: int = 2017,
    end_year: int = 2025,
    top_n: int = 5,
) -> Tuple[go.Figure, str]:
    """
    Identifies institutions with largest rank climb and drop between two years.
    Returns horizontal bar delta chart and takeaway string.
    """
    df_cat = df[df["category"] == category]
    df_start = df_cat[df_cat["year"] == start_year][["name", "rank"]].rename(columns={"rank": "start_rank"})
    df_end = df_cat[df_cat["year"] == end_year][["name", "rank"]].rename(columns={"rank": "end_rank"})

    merged = df_start.merge(df_end, on="name", how="inner")
    if merged.empty:
        fig = go.Figure()
        return apply_theme(fig, title="Insufficient data between selected years"), "No comparative data available."

    # delta = start_rank - end_rank (positive = improved)
    merged["rank_delta"] = (merged["start_rank"] - merged["end_rank"]).astype(int)

    # Top improvers and decliners
    improvers = merged.sort_values(by="rank_delta", ascending=False).head(top_n)
    decliners = merged.sort_values(by="rank_delta", ascending=True).head(top_n)

    combined = pd.concat([improvers, decliners]).drop_duplicates().sort_values(by="rank_delta")

    colors = [ACCENT_GREEN if d > 0 else (ACCENT_RED if d < 0 else MUTED_TEXT) for d in combined["rank_delta"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=combined["name"],
            x=combined["rank_delta"],
            orientation="h",
            marker=dict(color=colors),
            hovertemplate="<b>%{y}</b><br>Rank Change: %{x:+d}<br>From #%{customdata[0]} to #%{customdata[1]}<extra></extra>",
            customdata=list(zip(combined["start_rank"], combined["end_rank"])),
        )
    )

    fig = apply_theme(
        fig,
        title=f"Rank Shift: {start_year} vs {end_year} ({category})",
        xaxis_title="Rank Change (positive = climbed, negative = dropped)",
        yaxis_title="",
    )

    best = improvers.iloc[0]
    worst = decliners.iloc[0]
    takeaway = (
        f"Top improver: {best['name']} ({int(best['rank_delta']):+d} ranks, #{int(best['start_rank'])} → #{int(best['end_rank'])}); "
        f"Steepest decliner: {worst['name']} ({int(worst['rank_delta']):+d} ranks, #{int(worst['start_rank'])} → #{int(worst['end_rank'])})."
    )

    return fig, takeaway


def plot_average_score_trend(
    df: pd.DataFrame,
    category: str = "Overall",
    year_range: Optional[Tuple[int, int]] = None,
) -> Tuple[go.Figure, str]:
    """
    Plots the mean and median overall score over time for a category.
    """
    filtered = df[df["category"] == category].copy()
    if year_range:
        filtered = filtered[(filtered["year"] >= year_range[0]) & (filtered["year"] <= year_range[1])]

    agg = filtered.groupby("year")["score"].agg(["mean", "median"]).reset_index()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=agg["year"],
            y=agg["mean"],
            mode="lines+markers",
            name="Mean Score",
            line=dict(color=LINE_COLOR, width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=agg["year"],
            y=agg["median"],
            mode="lines+markers",
            name="Median Score",
            line=dict(color="#38bdf8", width=2, dash="dash"),
        )
    )

    fig = apply_theme(
        fig,
        title=f"Average & Median Score Trajectory ({category})",
        xaxis_title="Year",
        yaxis_title="Score (0-100)",
        show_legend=True,
    )

    first_val = agg["mean"].iloc[0]
    last_val = agg["mean"].iloc[-1]
    delta = last_val - first_val
    takeaway = f"Average score {'increased' if delta >= 0 else 'decreased'} by {abs(delta):.2f} pts ({first_val:.1f} → {last_val:.1f}) across {len(agg)} years."

    return fig, takeaway


def plot_consistently_top10(
    df: pd.DataFrame,
    category: str = "Overall",
    min_years: int = 5,
) -> Tuple[go.Figure, str]:
    """
    Shows multi-year trajectories of institutions appearing consistently in the Top 10.
    """
    cat_df = df[df["category"] == category]
    top_insts = (
        cat_df[cat_df["rank"] <= 10]
        .groupby("name")["year"]
        .nunique()
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
    )

    qualifying = top_insts[top_insts["count"] >= min_years]["name"].tolist()

    if not qualifying:
        qualifying = top_insts.head(5)["name"].tolist()

    sub_df = cat_df[cat_df["name"].isin(qualifying)].sort_values(by="year")

    fig = go.Figure()
    for idx, inst in enumerate(qualifying):
        inst_data = sub_df[sub_df["name"] == inst]
        color = PALETTE[idx % len(PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=inst_data["year"],
                y=inst_data["rank"],
                mode="lines+markers",
                name=inst.replace("Indian Institute of Technology", "IIT").replace("National Institute of Technology", "NIT"),
                line=dict(color=color, width=2),
                marker=dict(size=6),
            )
        )

    fig = apply_theme(
        fig,
        title=f"Consistently Top Ranked Institutions ({category})",
        xaxis_title="Year",
        yaxis_title="NIRF Rank",
        invert_yaxis=True,
        show_legend=True,
    )

    takeaway = f"{len(qualifying)} institutions maintained Top 10 positions for {min_years}+ years: {', '.join([q.split()[0] + ' ' + q.split()[-1] for q in qualifying[:3]])}..."
    return fig, takeaway
