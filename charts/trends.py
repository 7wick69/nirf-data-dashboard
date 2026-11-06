"""
charts/trends.py

Time-series and trend visualizations for NIRF ranking data:
  1. Rank Trajectory: How has the overall NIRF ranking of a selected institution changed from 2017 to 2025?
  2. Greatest Improvers: Which institutions have shown the greatest improvement in rank over the nine years?
  3. Largest Decliners: Which institutions have experienced the largest decline in ranking?
  4. Average Score: How has the average score of institutions changed over time?
  5. Consistently Top-10: Which institutions have remained consistently in the top 10 across multiple years?

Follows textbook data visualization guidelines:
  - Inverted Y-axis for ordinal ranks (Rank 1 at the top)
  - ColorBrewer / Okabe-Ito compliant qualitative palette with balanced luminance
  - Redundant encoding (distinct hues + distinct geometric marker symbols)
  - Full support for state-level filtering across all charts
"""

from typing import Tuple, Optional
import pandas as pd
import plotly.graph_objects as go
from charts.theme import (
    apply_theme,
    LINE_COLOR,
    ACCENT_GREEN,
    ACCENT_RED,
    MUTED_TEXT,
    QUALITATIVE_PALETTE,
    MARKER_SYMBOLS,
)


def get_clean_short_name(inst: str) -> str:
    """Produces clean, standardized short labels for institutions in legends and takeaways."""
    name = inst.strip()
    replacements = [
        ("Indian Institute of Technology", "IIT"),
        ("National Institute of Technology", "NIT"),
        ("Indian Institute of Management", "IIM"),
        ("Indian Institute of Information Technology", "IIIT"),
        ("Indian Institute of Science", "IISc"),
        ("All India Institute of Medical Sciences", "AIIMS"),
        ("University", "Univ"),
        ("College", "Coll"),
    ]
    for old, new in replacements:
        name = name.replace(old, new)
    name = name.replace(", ", " ").replace(",", "").strip()
    return name


def plot_institution_rank_trend(
    df: pd.DataFrame,
    institution_name: str,
    category: str = "Overall",
    year_range: Optional[Tuple[int, int]] = None,
    state: Optional[str] = None,
) -> Tuple[go.Figure, str]:
    """
    Question 1: How has the overall NIRF ranking of a selected institution changed from 2017 to 2025?
    Plots the rank trajectory with an inverted Y-axis so Rank 1 is at the top.
    """
    filtered = df[(df["name"] == institution_name) & (df["category"] == category)].copy()

    if state and state != "All states":
        filtered = filtered[filtered["state"] == state]

    if year_range:
        filtered = filtered[(filtered["year"] >= year_range[0]) & (filtered["year"] <= year_range[1])]

    filtered = filtered.sort_values(by="year")

    if filtered.empty:
        fig = go.Figure()
        fig = apply_theme(fig, title=f"No data for {institution_name} in {category}")
        return fig, f"No ranking records found for '{institution_name}' in category '{category}'" + (f" for state {state}." if state and state != "All states" else ".")

    first_row = filtered.iloc[0]
    last_row = filtered.iloc[-1]
    first_year, first_rank = int(first_row["year"]), int(first_row["rank"])
    last_year, last_rank = int(last_row["year"]), int(last_row["rank"])
    years_span = last_year - first_year + 1

    # Automated takeaway calculation
    if first_rank > last_rank:
        diff = first_rank - last_rank
        takeaway = f"rank improved from {first_rank} to {last_rank} (+{diff} places) over {years_span} years ({first_year}–{last_year})"
    elif first_rank < last_rank:
        diff = last_rank - first_rank
        takeaway = f"rank declined from {first_rank} to {last_rank} (-{diff} places) over {years_span} years ({first_year}–{last_year})"
    else:
        takeaway = f"rank remained stable at #{first_rank} across all {years_span} recorded years ({first_year}–{last_year})"

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
                    filtered["score"].fillna(0),
                    filtered["rank_change"].apply(
                        lambda c: f"+{int(c)}" if pd.notna(c) and c > 0 else (str(int(c)) if pd.notna(c) else "N/A")
                    ),
                )
            ),
        )
    )

    min_rank = int(filtered["rank"].min())
    max_rank = int(filtered["rank"].max())
    y_min_bound = max(1, min_rank - 1)
    y_max_bound = max_rank + 1
    tick_vals = list(range(y_min_bound, y_max_bound + 1))

    scope_suffix = f" — {state}" if state and state != "All states" else ""
    fig = apply_theme(
        fig,
        title=f"Rank Trajectory: {institution_name} ({category}{scope_suffix})",
        xaxis_title="Ranking Year",
        yaxis_title="NIRF Rank (Inverted: 1 is best)",
        invert_yaxis=True,
    )

    fig.update_xaxes(tickmode="linear", tick0=filtered["year"].min(), dtick=1)
    if len(tick_vals) <= 15:
        fig.update_yaxes(tickvals=tick_vals, ticktext=[str(v) for v in tick_vals])

    return fig, takeaway


def plot_greatest_improvers(
    df: pd.DataFrame,
    category: str = "Overall",
    start_year: int = 2017,
    end_year: int = 2025,
    top_n: int = 5,
    state: Optional[str] = None,
) -> Tuple[go.Figure, str]:
    """
    Question 2: Which institutions have shown the greatest improvement in rank over the nine years?
    Renders horizontal bar chart for institutions with the largest positive rank gains.
    """
    df_cat = df[df["category"] == category].copy()
    if state and state != "All states":
        df_cat = df_cat[df_cat["state"] == state]

    df_start = df_cat[df_cat["year"] == start_year][["name", "rank", "state"]].rename(columns={"rank": "start_rank"})
    df_end = df_cat[df_cat["year"] == end_year][["name", "rank"]].rename(columns={"rank": "end_rank"})

    merged = df_start.merge(df_end, on="name", how="inner")
    if merged.empty:
        fig = go.Figure()
        scope_msg = f" for state '{state}'" if state and state != "All states" else ""
        return apply_theme(fig, title=f"Insufficient comparative data between {start_year} and {end_year}"), f"No institutions found with rankings in both {start_year} and {end_year}{scope_msg}."

    # rank_delta = start_rank - end_rank (positive = improved, climbed higher)
    merged["rank_delta"] = (merged["start_rank"] - merged["end_rank"]).astype(int)
    improvers = merged[merged["rank_delta"] > 0].sort_values(by="rank_delta", ascending=False).head(top_n)

    if improvers.empty:
        fig = go.Figure()
        return apply_theme(fig, title="No rank improvements recorded"), f"No institutions showed a net rank improvement between {start_year} and {end_year}."

    improvers = improvers.sort_values(by="rank_delta", ascending=True)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=improvers["name"],
            x=improvers["rank_delta"],
            orientation="h",
            marker=dict(color=ACCENT_GREEN),
            hovertemplate=(
                "<b>%{y}</b><br>"
                + "Climbed: +%{x} ranks<br>"
                + "From #%{customdata[0]} ("
                + str(start_year)
                + ") → #%{customdata[1]} ("
                + str(end_year)
                + ")<extra></extra>"
            ),
            customdata=list(zip(improvers["start_rank"].astype(int), improvers["end_rank"].astype(int))),
        )
    )

    scope_title = f" in {state}" if state and state != "All states" else ""
    fig = apply_theme(
        fig,
        title=f"Greatest Rank Improvers ({start_year} → {end_year}, {category}{scope_title})",
        xaxis_title="Places Climbed (Positive = Improved Rank)",
        yaxis_title="",
    )

    top_inst = improvers.iloc[-1]
    takeaway = (
        f"Top improver: {top_inst['name']} climbed +{int(top_inst['rank_delta'])} ranks "
        f"(#{int(top_inst['start_rank'])} in {start_year} → #{int(top_inst['end_rank'])} in {end_year})."
    )
    return fig, takeaway


def plot_largest_decliners(
    df: pd.DataFrame,
    category: str = "Overall",
    start_year: int = 2017,
    end_year: int = 2025,
    top_n: int = 5,
    state: Optional[str] = None,
) -> Tuple[go.Figure, str]:
    """
    Question 3: Which institutions have experienced the largest decline in ranking?
    Renders horizontal bar chart for institutions with the steepest rank drops.
    """
    df_cat = df[df["category"] == category].copy()
    if state and state != "All states":
        df_cat = df_cat[df_cat["state"] == state]

    df_start = df_cat[df_cat["year"] == start_year][["name", "rank", "state"]].rename(columns={"rank": "start_rank"})
    df_end = df_cat[df_cat["year"] == end_year][["name", "rank"]].rename(columns={"rank": "end_rank"})

    merged = df_start.merge(df_end, on="name", how="inner")
    if merged.empty:
        fig = go.Figure()
        scope_msg = f" for state '{state}'" if state and state != "All states" else ""
        return apply_theme(fig, title=f"Insufficient comparative data between {start_year} and {end_year}"), f"No institutions found with rankings in both {start_year} and {end_year}{scope_msg}."

    # rank_delta = start_rank - end_rank (negative = declined)
    merged["rank_delta"] = (merged["start_rank"] - merged["end_rank"]).astype(int)
    decliners = merged[merged["rank_delta"] < 0].sort_values(by="rank_delta", ascending=True).head(top_n)

    if decliners.empty:
        fig = go.Figure()
        return apply_theme(fig, title="No rank declines recorded"), f"No institutions showed a net rank decline between {start_year} and {end_year}."

    # Sort so steepest decline is at the top
    decliners = decliners.sort_values(by="rank_delta", ascending=False)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=decliners["name"],
            x=decliners["rank_delta"],
            orientation="h",
            marker=dict(color=ACCENT_RED),
            hovertemplate=(
                "<b>%{y}</b><br>"
                + "Dropped: %{x} ranks<br>"
                + "From #%{customdata[0]} ("
                + str(start_year)
                + ") → #%{customdata[1]} ("
                + str(end_year)
                + ")<extra></extra>"
            ),
            customdata=list(zip(decliners["start_rank"].astype(int), decliners["end_rank"].astype(int))),
        )
    )

    scope_title = f" in {state}" if state and state != "All states" else ""
    fig = apply_theme(
        fig,
        title=f"Largest Rank Decliners ({start_year} → {end_year}, {category}{scope_title})",
        xaxis_title="Places Dropped (Negative = Declined Rank)",
        yaxis_title="",
    )

    steepest = decliners.iloc[0]
    takeaway = (
        f"Largest decline: {steepest['name']} dropped {int(steepest['rank_delta'])} ranks "
        f"(#{int(steepest['start_rank'])} in {start_year} → #{int(steepest['end_rank'])} in {end_year})."
    )
    return fig, takeaway


def plot_average_score_trend(
    df: pd.DataFrame,
    category: str = "Overall",
    year_range: Optional[Tuple[int, int]] = None,
    state: Optional[str] = None,
) -> Tuple[go.Figure, str]:
    """
    Question 4: How has the average score of institutions changed over time?
    Plots Mean and Median score progression over time, with benchmark comparison if state is filtered.
    """
    cat_df = df[df["category"] == category].copy()
    if year_range:
        cat_df = cat_df[(cat_df["year"] >= year_range[0]) & (cat_df["year"] <= year_range[1])]

    if cat_df.empty:
        fig = go.Figure()
        return apply_theme(fig, title=f"No data for {category}"), "No score records available."

    # Overall benchmark
    national_agg = cat_df.groupby("year")["score"].agg(["mean", "median"]).reset_index()

    fig = go.Figure()

    if state and state != "All states":
        state_df = cat_df[cat_df["state"] == state]
        if not state_df.empty:
            state_agg = state_df.groupby("year")["score"].agg(["mean", "median"]).reset_index()
            fig.add_trace(
                go.Scatter(
                    x=state_agg["year"],
                    y=state_agg["mean"].round(2),
                    mode="lines+markers",
                    name=f"{state} Mean",
                    line=dict(color=LINE_COLOR, width=3),
                    marker=dict(size=7),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=state_agg["year"],
                    y=state_agg["median"].round(2),
                    mode="lines+markers",
                    name=f"{state} Median",
                    line=dict(color="#38bdf8", width=2, dash="dot"),
                    marker=dict(size=6),
                )
            )

        # Add national benchmark for context
        fig.add_trace(
            go.Scatter(
                x=national_agg["year"],
                y=national_agg["mean"].round(2),
                mode="lines",
                name="All-India Mean",
                line=dict(color=MUTED_TEXT, width=1.5, dash="dash"),
            )
        )
        scope_title = f"{state} vs All-India ({category})"
        first_val = state_agg["mean"].iloc[0] if not state_df.empty else 0
        last_val = state_agg["mean"].iloc[-1] if not state_df.empty else 0
        delta = last_val - first_val
        takeaway = (
            f"Average score for {state} {'increased' if delta >= 0 else 'decreased'} by {abs(delta):.2f} pts "
            f"({first_val:.1f} → {last_val:.1f}) between {year_range[0] if year_range else 2017} and {year_range[1] if year_range else 2025}."
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=national_agg["year"],
                y=national_agg["mean"].round(2),
                mode="lines+markers",
                name="Mean Score",
                line=dict(color=LINE_COLOR, width=3),
                marker=dict(size=7),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=national_agg["year"],
                y=national_agg["median"].round(2),
                mode="lines+markers",
                name="Median Score",
                line=dict(color="#f97316", width=2, dash="dash"),
                marker=dict(size=6),
            )
        )
        scope_title = f"All-India ({category})"
        first_val = national_agg["mean"].iloc[0]
        last_val = national_agg["mean"].iloc[-1]
        delta = last_val - first_val
        takeaway = (
            f"Average score {'increased' if delta >= 0 else 'decreased'} by {abs(delta):.2f} pts "
            f"({first_val:.1f} → {last_val:.1f}) across {len(national_agg)} years."
        )

    fig = apply_theme(
        fig,
        title=f"Average & Median Score Trajectory: {scope_title}",
        xaxis_title="Ranking Year",
        yaxis_title="Score (0–100)",
        show_legend=True,
    )
    fig.update_xaxes(tickmode="linear", tick0=national_agg["year"].min(), dtick=1)

    return fig, takeaway


def plot_consistently_top10(
    df: pd.DataFrame,
    category: str = "Overall",
    min_years: int = 5,
    state: Optional[str] = None,
) -> Tuple[go.Figure, str]:
    """
    Question 5: Which institutions have remained consistently in the top 10 across multiple years?
    Multi-line trajectory using textbook visualization rules:
      - 10 distinct, non-confusing qualitative hues (ColorBrewer/Okabe-Ito based)
      - Redundant geometric marker symbols (circle, square, diamond, triangle, etc.)
      - Inverted Y-axis
      - State-aware filtering
    """
    cat_df = df[df["category"] == category].copy()
    if state and state != "All states":
        cat_df = cat_df[cat_df["state"] == state]

    if cat_df.empty:
        fig = go.Figure()
        scope_msg = f" in {state}" if state and state != "All states" else ""
        return apply_theme(fig, title=f"No Top 10 data{scope_msg}"), f"No ranking records found for category '{category}'{scope_msg}."

    # Identify institutions ranked <= 10
    top10_insts = (
        cat_df[cat_df["rank"] <= 10]
        .groupby("name")["year"]
        .nunique()
        .reset_index(name="appearances")
        .sort_values(by="appearances", ascending=False)
    )

    qualifying = top10_insts[top10_insts["appearances"] >= min_years]["name"].tolist()

    # If fewer qualify under strict min_years, pick top available (up to 8)
    if not qualifying:
        qualifying = top10_insts.head(8)["name"].tolist()
    else:
        qualifying = qualifying[:10]  # Cap at 10 for visual clarity per textbook rules

    if not qualifying:
        fig = go.Figure()
        return apply_theme(fig, title="No institutions meet the criteria"), f"No institutions appeared in the Top 10 for {min_years}+ years."

    sub_df = cat_df[cat_df["name"].isin(qualifying)].sort_values(by="year")

    fig = go.Figure()

    for idx, inst in enumerate(qualifying):
        inst_data = sub_df[sub_df["name"] == inst]
        color = QUALITATIVE_PALETTE[idx % len(QUALITATIVE_PALETTE)]
        symbol = MARKER_SYMBOLS[idx % len(MARKER_SYMBOLS)]

        short_label = get_clean_short_name(inst)

        fig.add_trace(
            go.Scatter(
                x=inst_data["year"],
                y=inst_data["rank"],
                mode="lines+markers",
                name=short_label,
                line=dict(color=color, width=2.5),
                marker=dict(symbol=symbol, size=8, color=color, line=dict(width=1, color="#0f1013")),
                hovertemplate="<b>%{data.name}</b><br>Year: %{x}<br>Rank: #%{y}<extra></extra>",
            )
        )

    scope_title = f" in {state}" if state and state != "All states" else ""
    fig = apply_theme(
        fig,
        title=f"Consistently Top 10 Ranked Institutions ({category}{scope_title})",
        xaxis_title="Ranking Year",
        yaxis_title="NIRF Rank (Inverted: 1 is best)",
        invert_yaxis=True,
        show_legend=True,
        legend_position="right",
    )

    fig.update_xaxes(tickmode="linear", tick0=sub_df["year"].min(), dtick=1)

    # Calculate nice y-axis tick bounds
    all_ranks = sub_df["rank"].dropna().astype(int)
    if not all_ranks.empty:
        min_r = max(1, all_ranks.min() - 1)
        max_r = min(15, all_ranks.max() + 1)
        ticks = list(range(min_r, max_r + 1))
        fig.update_yaxes(tickvals=ticks, ticktext=[str(t) for t in ticks])

    names_summary = ", ".join([get_clean_short_name(q) for q in qualifying[:4]])
    takeaway = f"{len(qualifying)} institutions maintained Top 10 positions for {min_years}+ years ({names_summary}...)."
    return fig, takeaway
