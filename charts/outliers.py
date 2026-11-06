"""
charts/outliers.py

Outlier Detection & Performance Gaps visualizations for NIRF Annexure A — Section 8.
Covers all 5 core questions:

  Q1. Connected Scatter Plot: High Research Scores Relative to Overall Scores
      - Maps Overall Score to X-axis, Research Score (RPC) to Y-axis.
      - Trajectory lines grouped by institution across 2017-2025 with yearly markers.
  Q2. Research vs. Teaching Gap:
      - Scatter plot: Teaching (TLR) on X-axis, Research (RPC) on Y-axis.
      - Diagonal y=x reference line with automatic outlier callouts.
  Q3. Score Volatility (Line Chart):
      - Multi-line chart of absolute scores across 2017-2025.
      - Highlights steep year-to-year vertical drops or climbs.
  Q4. High Parameter, Low Rank:
      - Scatter plot: Selected parameter on X-axis vs. Overall Rank on Y-axis.
      - Inverted Y-axis (Rank 1 at top) with anomaly quadrant callouts.
  Q5. Ranking Volatility (Dual Options):
      - Option A: Box Plot showing rank distribution/spread over 2017-2025.
      - Option B: Line Chart comparing jagged fluctuating trajectories vs. flat stable lines.
"""

from typing import Dict, List, Optional, Tuple, Literal
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from charts.theme import (
    apply_theme,
    DARK_BG,
    PANEL_BG,
    GRID_COLOR,
    QUALITATIVE_PALETTE,
    MARKER_SYMBOLS,
    TEXT_COLOR,
    MUTED_TEXT,
    ACCENT_GREEN,
    ACCENT_RED,
)

PARAM_LABELS: Dict[str, str] = {
    "TLR": "Teaching, Learning & Resources (TLR)",
    "RPC": "Research & Professional Practice (RPC)",
    "GO":  "Graduation Outcomes (GO)",
    "OI":  "Outreach & Inclusivity (OI)",
    "PR":  "Perception (PR)",
}


def _shorten(name: str) -> str:
    """Abbreviates institution names for clean labeling."""
    replacements = [
        ("Indian Institute of Technology", "IIT"),
        ("Indian Institute of Science", "IISc"),
        ("National Institute of Technology", "NIT"),
        ("Indian Institute of Management", "IIM"),
        ("Indian Institute of Information Technology", "IIIT"),
        ("All India Institute of Medical Sciences", "AIIMS"),
        ("Jawaharlal Nehru University", "JNU"),
        ("University of Delhi", "DU"),
        ("Banaras Hindu University", "BHU"),
        ("Jamia Millia Islamia", "JMI"),
        ("University", "Univ"),
        ("Technology", "Tech"),
    ]
    for long_n, short_n in replacements:
        name = name.replace(long_n, short_n)
    return name


# ── Q1: Connected Scatter Plot (Research vs Overall Score Trajectories) ────────

def plot_connected_scatter_research_outliers(
    df: pd.DataFrame,
    category: str = "Overall",
    state: Optional[str] = None,
    top_n: int = 5,
    selected_institutions: Optional[List[str]] = None,
) -> Tuple[go.Figure, str]:
    """
    Connected scatter plot: Overall Score (X) vs RPC Score (Y) over 2017-2025.
    Connects each institution's yearly coordinates with a line to show trajectory.
    Highlights institutions with unusually high research relative to overall score.
    """
    sub = df[df["category"] == category].copy()
    if state and state != "All states":
        sub = sub[sub["state"] == state]

    sub = sub.dropna(subset=["score", "RPC", "year"])
    if sub.empty:
        fig = go.Figure()
        return apply_theme(fig, title="Insufficient data"), "No data available."

    # Identify research-heavy outliers: average ratio or residual of RPC relative to score
    # Fit simple linear model to find expected RPC given score
    scores = sub["score"].to_numpy(dtype=float)
    rpcs = sub["RPC"].to_numpy(dtype=float)
    if len(sub) >= 5 and np.std(scores) > 0:
        slope, intercept = np.polyfit(scores, rpcs, 1)
        sub["rpc_residual"] = sub["RPC"] - (slope * sub["score"] + intercept)
    else:
        sub["rpc_residual"] = sub["RPC"] - sub["score"]

    avg_outliers = (
        sub.groupby("name")["rpc_residual"]
        .mean()
        .reset_index()
        .sort_values(by="rpc_residual", ascending=False)
    )

    if selected_institutions:
        chosen_insts = [inst for inst in selected_institutions if inst in sub["name"].values]
    else:
        chosen_insts = avg_outliers.head(top_n)["name"].tolist()

    if not chosen_insts:
        chosen_insts = avg_outliers.head(3)["name"].tolist()

    fig = go.Figure()

    # Background cloud of all points for visual context (muted)
    fig.add_trace(
        go.Scatter(
            x=sub["score"],
            y=sub["RPC"],
            mode="markers",
            marker=dict(size=4.5, color="rgba(148, 163, 184, 0.18)", symbol="circle"),
            name="All Institutions",
            hoverinfo="skip",
            showlegend=True,
        )
    )

    # Connected trajectories for selected outlier institutions
    for idx, inst in enumerate(chosen_insts):
        inst_df = sub[sub["name"] == inst].sort_values(by="year")
        color = QUALITATIVE_PALETTE[idx % len(QUALITATIVE_PALETTE)]
        symbol = MARKER_SYMBOLS[idx % len(MARKER_SYMBOLS)]
        short_label = _shorten(inst)

        # Line + Markers
        fig.add_trace(
            go.Scatter(
                x=inst_df["score"],
                y=inst_df["RPC"],
                mode="lines+markers",
                name=short_label,
                line=dict(color=color, width=2.5),
                marker=dict(
                    size=8,
                    color=color,
                    symbol=symbol,
                    line=dict(color="#0f1013", width=1),
                ),
                customdata=list(zip(inst_df["year"].astype(int), inst_df["rank"].fillna(-1).astype(int))),
                hovertemplate=(
                    f"<b>{short_label}</b><br>"
                    + "Year: %{customdata[0]}<br>"
                    + "Overall Score: %{x:.2f}<br>"
                    + "Research (RPC): %{y:.2f}<br>"
                    + "Rank: #%{customdata[1]}<extra></extra>"
                ),
            )
        )

        # Annotate endpoint (latest available year)
        if not inst_df.empty:
            last_row = inst_df.iloc[-1]
            fig.add_annotation(
                x=float(last_row["score"]),
                y=float(last_row["RPC"]),
                text=f"{short_label} ('{str(int(last_row['year']))[-2:]})",
                showarrow=False,
                xshift=10,
                yshift=8,
                font=dict(size=10, color=color, family="Inter, sans-serif"),
                align="left",
            )

    fig = apply_theme(
        fig,
        title=f"Research Outliers: 9-Year Trajectory ({category} 2017–2025)",
        xaxis_title="Overall Score",
        yaxis_title="Research Score (RPC)",
        show_legend=True,
        legend_position="right",
    )
    fig.update_layout(height=650)

    top_outlier_names = ", ".join([_shorten(n) for n in chosen_insts[:3]])
    takeaway = (
        f"Institutions like {top_outlier_names} sustain research scores (RPC) significantly "
        f"above what their overall scores predict across the 2017–2025 timeline."
    )
    return fig, takeaway


# ── Q2: Research vs. Teaching Gap ─────────────────────────────────────────────

def plot_research_vs_teaching_gap(
    df: pd.DataFrame,
    year: int = 2025,
    category: str = "Overall",
    state: Optional[str] = None,
    top_n_outliers: int = 6,
) -> Tuple[go.Figure, str]:
    """
    Scatter plot: Teaching (TLR) on X-axis vs. Research (RPC) on Y-axis.
    Includes diagonal y=x reference line where RPC == TLR.
    Points above y=x excel in research relative to teaching.
    """
    sub = df[(df["year"] == year) & (df["category"] == category)].copy()
    if state and state != "All states":
        sub = sub[sub["state"] == state]

    sub = sub.dropna(subset=["TLR", "RPC", "score"]).reset_index(drop=True)
    if sub.empty:
        fig = go.Figure()
        return apply_theme(fig, title="Insufficient data"), "No data available."

    # Gap = RPC - TLR (positive means research exceeds teaching)
    sub["research_gap"] = sub["RPC"] - sub["TLR"]

    max_val = max(float(sub["TLR"].max()), float(sub["RPC"].max()), 95.0)
    min_val = min(float(sub["TLR"].min()), float(sub["RPC"].min()), 15.0)

    fig = go.Figure()

    # Diagonal parity line: y = x
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Parity (RPC = TLR)",
            line=dict(color=MUTED_TEXT, width=1.5, dash="dash"),
            hoverinfo="skip",
        )
    )

    # Shaded upper half annotation
    fig.add_annotation(
        x=min_val + (max_val - min_val) * 0.25,
        y=min_val + (max_val - min_val) * 0.75,
        text="<b>Research Dominant (RPC > TLR)</b>",
        showarrow=False,
        font=dict(color=ACCENT_GREEN, size=11),
        opacity=0.6,
    )
    fig.add_annotation(
        x=min_val + (max_val - min_val) * 0.75,
        y=min_val + (max_val - min_val) * 0.25,
        text="<b>Teaching Dominant (TLR > RPC)</b>",
        showarrow=False,
        font=dict(color="#f59e0b", size=11),
        opacity=0.6,
    )

    # Scatter points colored by research gap
    fig.add_trace(
        go.Scatter(
            x=sub["TLR"],
            y=sub["RPC"],
            mode="markers",
            name="Institutions",
            marker=dict(
                size=9,
                color=sub["research_gap"],
                colorscale=[
                    [0.0, "#f59e0b"],    # Teaching dominant (amber)
                    [0.5, "#94a3b8"],    # Balanced (slate)
                    [1.0, ACCENT_GREEN], # Research dominant (green)
                ],
                colorbar=dict(
                    title=dict(text="RPC − TLR Gap", font=dict(color=TEXT_COLOR, size=11)),
                    thickness=12,
                    tickfont=dict(color=MUTED_TEXT, size=10),
                ),
                line=dict(color="#0f1013", width=0.8),
            ),
            customdata=list(zip(sub["name"].apply(_shorten), sub["research_gap"].round(2), sub["rank"].fillna(-1).astype(int))),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                + "Teaching (TLR): %{x:.2f}<br>"
                + "Research (RPC): %{y:.2f}<br>"
                + "Gap (RPC − TLR): %{customdata[1]:+.2f}<br>"
                + "Rank: #%{customdata[2]}<extra></extra>"
            ),
        )
    )

    # Annotate top research-heavy outliers (highest positive gap)
    top_research = sub.sort_values(by="research_gap", ascending=False).head(top_n_outliers)
    for _, row in top_research.iterrows():
        fig.add_annotation(
            x=float(row["TLR"]),
            y=float(row["RPC"]),
            text=f"<b>{_shorten(str(row['name']))}</b> (+{row['research_gap']:.1f})",
            showarrow=True,
            arrowhead=2,
            arrowsize=0.8,
            arrowcolor=ACCENT_GREEN,
            ax=28,
            ay=-24,
            font=dict(size=9.5, color=TEXT_COLOR),
            bgcolor=PANEL_BG,
            bordercolor=GRID_COLOR,
            borderwidth=1,
            borderpad=3,
        )

    fig = apply_theme(
        fig,
        title=f"Research vs. Teaching Gap ({category} · {year})",
        xaxis_title="Teaching, Learning & Resources (TLR)",
        yaxis_title="Research and Professional Practice (RPC)",
        show_legend=True,
        legend_position="right",
    )
    fig.update_layout(height=650)
    fig.update_xaxes(range=[min_val - 2, max_val + 2])
    fig.update_yaxes(range=[min_val - 2, max_val + 2])

    top_names = ", ".join(top_research["name"].apply(_shorten).head(3).tolist())
    takeaway = (
        f"Institutions positioned above the diagonal (led by {top_names}) exhibit substantially "
        f"higher Research output (RPC) than Teaching resources (TLR), revealing specialized research-heavy missions."
    )
    return fig, takeaway


# ── Q3: Score Volatility (Line Chart with Vertical Shifts) ─────────────────────

def plot_score_volatility_trends(
    df: pd.DataFrame,
    category: str = "Overall",
    state: Optional[str] = None,
    top_n: int = 6,
    selected_institutions: Optional[List[str]] = None,
    year_range: Tuple[int, int] = (2017, 2025),
) -> Tuple[go.Figure, str]:
    """
    Multi-line chart of absolute scores across 2017-2025.
    Identifies and highlights institutions with steep vertical climbs or drops.
    """
    sub = df[
        (df["category"] == category)
        & (df["year"] >= year_range[0])
        & (df["year"] <= year_range[1])
    ].copy()
    if state and state != "All states":
        sub = sub[sub["state"] == state]

    sub = sub.dropna(subset=["score", "year"])
    if sub.empty:
        fig = go.Figure()
        return apply_theme(fig, title="Insufficient data"), "No data available."

    # Compute maximum absolute single-year score jump
    volatility_df = (
        sub.groupby("name")["score_change_yoy"]
        .apply(lambda s: s.abs().max())
        .reset_index(name="max_score_shift")
        .sort_values(by="max_score_shift", ascending=False)
    )

    if selected_institutions:
        chosen_insts = [inst for inst in selected_institutions if inst in sub["name"].values]
    else:
        chosen_insts = volatility_df.head(top_n)["name"].tolist()

    if not chosen_insts:
        chosen_insts = volatility_df.head(3)["name"].tolist()

    fig = go.Figure()

    # Draw trajectories
    max_jump_records = []
    for idx, inst in enumerate(chosen_insts):
        inst_df = sub[sub["name"] == inst].sort_values(by="year")
        color = QUALITATIVE_PALETTE[idx % len(QUALITATIVE_PALETTE)]
        symbol = MARKER_SYMBOLS[idx % len(MARKER_SYMBOLS)]
        short_label = _shorten(inst)

        fig.add_trace(
            go.Scatter(
                x=inst_df["year"],
                y=inst_df["score"],
                mode="lines+markers",
                name=short_label,
                line=dict(color=color, width=2.8),
                marker=dict(size=8, color=color, symbol=symbol),
                customdata=inst_df["score_change_yoy"].fillna(0).round(2),
                hovertemplate=(
                    f"<b>{short_label}</b><br>"
                    + "Year: %{x}<br>"
                    + "Score: %{y:.2f}<br>"
                    + "YoY Shift: %{customdata:+0.2f} pts<extra></extra>"
                ),
            )
        )

        # Detect the steepest single-year jump for an annotation badge
        if "score_change_yoy" in inst_df.columns:
            valid_shifts = inst_df.dropna(subset=["score_change_yoy"])
            if not valid_shifts.empty:
                biggest_jump_row = valid_shifts.loc[valid_shifts["score_change_yoy"].abs().idxmax()]
                max_jump_records.append((inst, biggest_jump_row))

    # Add callout on the single biggest shift
    if max_jump_records:
        max_jump_records.sort(key=lambda item: abs(item[1]["score_change_yoy"]), reverse=True)
        top_jump_inst, top_jump_row = max_jump_records[0]
        shift_val = float(top_jump_row["score_change_yoy"])
        fig.add_annotation(
            x=int(top_jump_row["year"]),
            y=float(top_jump_row["score"]),
            text=f"<b>{_shorten(top_jump_inst)}</b><br>{shift_val:+0.1f} pts in {int(top_jump_row['year'])}",
            showarrow=True,
            arrowhead=2,
            arrowsize=0.8,
            arrowcolor=ACCENT_GREEN if shift_val > 0 else ACCENT_RED,
            ax=35,
            ay=-35,
            font=dict(size=9.5, color=TEXT_COLOR),
            bgcolor=PANEL_BG,
            bordercolor=GRID_COLOR,
            borderwidth=1,
            borderpad=4,
        )

    fig = apply_theme(
        fig,
        title=f"Score Volatility & Steep YoY Shifts ({category} · {year_range[0]}–{year_range[1]})",
        xaxis_title="Year",
        yaxis_title="Overall Score (0–100)",
        show_legend=True,
        legend_position="right",
    )
    fig.update_layout(height=620)
    fig.update_xaxes(dtick=1)

    top_shift_name = _shorten(max_jump_records[0][0]) if max_jump_records else "Institutions"
    top_shift_val = max_jump_records[0][1]["score_change_yoy"] if max_jump_records else 0.0
    takeaway = (
        f"Largest single-year score swing was recorded by {top_shift_name} "
        f"({top_shift_val:+0.1f} pts), illustrating vertical volatility distinct from gradual score growth."
    )
    return fig, takeaway


# ── Q4: High Parameter, Low Rank Anomaly Scatter ──────────────────────────────

def plot_high_param_low_rank(
    df: pd.DataFrame,
    year: int = 2025,
    category: str = "Overall",
    state: Optional[str] = None,
    param: str = "RPC",
    top_n_outliers: int = 6,
) -> Tuple[go.Figure, str]:
    """
    Scatter plot: Parameter Score on X-axis vs. Overall Rank on Y-axis.
    Y-axis is inverted so Rank 1 is at the top.
    Highlights anomalies: High parameter score (right) but poor rank (lower on screen).
    """
    sub = df[(df["year"] == year) & (df["category"] == category)].copy()
    if state and state != "All states":
        sub = sub[sub["state"] == state]

    sub = sub.dropna(subset=[param, "rank", "score"]).reset_index(drop=True)
    if sub.empty:
        fig = go.Figure()
        return apply_theme(fig, title="Insufficient data"), "No data available."

    param_label = PARAM_LABELS.get(param, param)
    med_param = float(sub[param].median())
    med_rank = float(sub["rank"].median())

    # Anomaly metric: high parameter score but high numeric rank (poor position)
    # Scaled rank (0 to 1) vs scaled param (0 to 1)
    param_norm = (sub[param] - sub[param].min()) / (sub[param].max() - sub[param].min() + 1e-6)
    rank_norm = (sub["rank"] - sub["rank"].min()) / (sub["rank"].max() - sub["rank"].min() + 1e-6)
    # The anomaly is high param_norm and high rank_norm (high numeric rank = poor rank)
    sub["anomaly_score"] = param_norm * rank_norm

    fig = go.Figure()

    # Median benchmark lines (dividing into 4 quadrants)
    fig.add_shape(
        type="line",
        x0=med_param, x1=med_param,
        y0=sub["rank"].min(), y1=sub["rank"].max(),
        line=dict(color=GRID_COLOR, width=1.5, dash="dot"),
    )
    fig.add_shape(
        type="line",
        x0=sub[param].min(), x1=sub[param].max(),
        y0=med_rank, y1=med_rank,
        line=dict(color=GRID_COLOR, width=1.5, dash="dot"),
    )

    # Quadrant annotations
    fig.add_annotation(
        x=float(sub[param].max()) - 2,
        y=float(sub["rank"].max()) - 4,
        text="<b>⚠️ Anomaly Zone:<br>High Parameter, Poor Rank</b>",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(color=ACCENT_RED, size=10.5),
        bgcolor=PANEL_BG,
        opacity=0.85,
    )
    fig.add_annotation(
        x=float(sub[param].max()) - 2,
        y=float(sub["rank"].min()) + 2,
        text="<b>Elite Tier:<br>High Parameter, Top Rank</b>",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        font=dict(color=ACCENT_GREEN, size=10.5),
        bgcolor=PANEL_BG,
        opacity=0.85,
    )

    # Scatter points
    fig.add_trace(
        go.Scatter(
            x=sub[param],
            y=sub["rank"],
            mode="markers",
            name="Institutions",
            marker=dict(
                size=8.5,
                color=sub["score"],
                colorscale="Viridis",
                colorbar=dict(
                    title=dict(text="Overall Score", font=dict(color=TEXT_COLOR, size=11)),
                    thickness=12,
                    tickfont=dict(color=MUTED_TEXT, size=10),
                ),
                line=dict(color="#0f1013", width=0.8),
            ),
            customdata=list(zip(sub["name"].apply(_shorten), sub["score"].round(2))),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                + f"{param}: %{{x:.2f}}<br>"
                + "NIRF Rank: #%{y}<br>"
                + "Overall Score: %{customdata[1]:.2f}<extra></extra>"
            ),
        )
    )

    # Highlight top anomalies (high parameter, poor overall rank)
    anomalies = (
        sub[(sub[param] >= med_param) & (sub["rank"] >= med_rank)]
        .sort_values(by="anomaly_score", ascending=False)
        .head(top_n_outliers)
    )
    if anomalies.empty:
        anomalies = sub.sort_values(by="anomaly_score", ascending=False).head(top_n_outliers)

    for _, row in anomalies.iterrows():
        fig.add_annotation(
            x=float(row[param]),
            y=float(row["rank"]),
            text=f"<b>{_shorten(str(row['name']))}</b><br>{param} {row[param]:.1f} | #{int(row['rank'])}",
            showarrow=True,
            arrowhead=2,
            arrowsize=0.8,
            arrowcolor=ACCENT_RED,
            ax=30,
            ay=25,
            font=dict(size=9, color=TEXT_COLOR),
            bgcolor=PANEL_BG,
            bordercolor=ACCENT_RED,
            borderwidth=1,
            borderpad=3,
        )

    fig = apply_theme(
        fig,
        title=f"High {param} Score vs. Overall Rank ({category} · {year})",
        xaxis_title=param_label,
        yaxis_title="Overall NIRF Rank (Rank 1 at Top)",
        invert_yaxis=True,
        show_legend=False,
    )
    fig.update_layout(height=640)

    top_names = ", ".join(anomalies["name"].apply(_shorten).head(3).tolist())
    takeaway = (
        f"Institutions in the anomaly zone (such as {top_names}) score above median in {param} "
        f"yet rank in the bottom half overall, highlighting bottlenecks in other NIRF pillars."
    )
    return fig, takeaway


# ── Q5: Ranking Volatility (Box Plot & Sequence Line Chart) ───────────────────

def plot_ranking_volatility(
    df: pd.DataFrame,
    category: str = "Overall",
    state: Optional[str] = None,
    top_n: int = 8,
    mode: Literal["boxplot", "line"] = "boxplot",
    selected_institutions: Optional[List[str]] = None,
) -> Tuple[go.Figure, str]:
    """
    Analyzes institutions with unusually high ranking volatility over 2017-2025.
      - mode='boxplot': Shows the distribution / spread of each institution's rank.
      - mode='line': Plots rank trajectory sequence comparing jagged vs. stable lines.
    Inverts Y-axis so Rank 1 appears at the top.
    """
    sub = df[df["category"] == category].copy()
    if state and state != "All states":
        sub = sub[sub["state"] == state]

    sub = sub.dropna(subset=["rank", "year"])
    if sub.empty:
        fig = go.Figure()
        return apply_theme(fig, title="Insufficient data"), "No data available."

    # Compute ranking standard deviation across years
    vol_stats = (
        sub.groupby("name")["rank"]
        .agg(["std", "min", "max", "count"])
        .reset_index()
        .rename(columns={"std": "rank_std"})
        .sort_values(by="rank_std", ascending=False)
    )
    vol_stats = vol_stats[vol_stats["count"] >= 3]  # at least 3 years to measure volatility

    if selected_institutions:
        chosen_insts = [inst for inst in selected_institutions if inst in sub["name"].values]
    else:
        chosen_insts = vol_stats.head(top_n)["name"].tolist()

    if not chosen_insts:
        chosen_insts = vol_stats.head(3)["name"].tolist()

    sub_inst = sub[sub["name"].isin(chosen_insts)].copy()

    fig = go.Figure()

    if mode == "boxplot":
        # Box plot for each institution
        for idx, inst in enumerate(chosen_insts):
            inst_data = sub_inst[sub_inst["name"] == inst]
            color = QUALITATIVE_PALETTE[idx % len(QUALITATIVE_PALETTE)]
            short_label = _shorten(inst)

            fig.add_trace(
                go.Box(
                    y=inst_data["rank"],
                    name=short_label,
                    marker=dict(color=color, size=6),
                    boxpoints="all",
                    jitter=0.25,
                    pointpos=-1.5,
                    line=dict(width=1.8),
                    customdata=inst_data["year"].astype(int),
                    hovertemplate=(
                        f"<b>{short_label}</b><br>"
                        + "Year: %{customdata}<br>"
                        + "Rank: #%{y}<extra></extra>"
                    ),
                )
            )

        fig = apply_theme(
            fig,
            title=f"Rank Spread & Volatility Distribution (Top {len(chosen_insts)} · {category})",
            xaxis_title="Institution",
            yaxis_title="NIRF Rank (Rank 1 at Top)",
            invert_yaxis=True,
            show_legend=False,
        )
        fig.update_layout(height=620)
        takeaway = (
            f"Box widths indicate rank volatility: {', '.join([_shorten(i) for i in chosen_insts[:3]])} "
            f"display the widest rank dispersion across 2017–2025."
        )

    else:
        # Line chart comparing jagged vs flat trajectories
        for idx, inst in enumerate(chosen_insts):
            inst_data = sub_inst[sub_inst["name"] == inst].sort_values(by="year")
            color = QUALITATIVE_PALETTE[idx % len(QUALITATIVE_PALETTE)]
            symbol = MARKER_SYMBOLS[idx % len(MARKER_SYMBOLS)]
            short_label = _shorten(inst)

            fig.add_trace(
                go.Scatter(
                    x=inst_data["year"],
                    y=inst_data["rank"],
                    mode="lines+markers",
                    name=short_label,
                    line=dict(color=color, width=2.4),
                    marker=dict(size=7.5, symbol=symbol),
                    hovertemplate=f"<b>{short_label}</b><br>Year: %{{x}}<br>Rank: #%{{y}}<extra></extra>",
                )
            )

        fig = apply_theme(
            fig,
            title=f"Rank Trajectory Volatility Sequence ({category} · 2017–2025)",
            xaxis_title="Year",
            yaxis_title="NIRF Rank (Rank 1 at Top)",
            invert_yaxis=True,
            show_legend=True,
            legend_position="right",
        )
        fig.update_layout(height=620)
        fig.update_xaxes(dtick=1)
        takeaway = (
            f"Highly jagged trajectory lines reveal volatile institutional mobility compared "
            f"to the flat, consistent paths of perennial top-ranked institutions."
        )

    return fig, takeaway

