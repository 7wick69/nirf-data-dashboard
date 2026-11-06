"""
charts/transitions.py

Ranking transitions, mobility, and retention visualizations for NIRF ranking data:
  1. Sankey Diagram: Rank band flow between two years (1-10, 11-50, 51-100, 101+)
  2. Movement Breakdown: Up, Down, or Same rank transitions YoY (Stacked Bar)
  3. Band Retention & Mobility: % of institutions staying in the same tier
  4. Largest Single-Year Rank Shifts: Historical single-year climbs and drops
"""

from typing import Tuple, List, Optional
import pandas as pd
import plotly.graph_objects as go
from charts.theme import (
    apply_theme,
    ACCENT_GREEN,
    ACCENT_RED,
    MUTED_TEXT,
    LINE_COLOR,
    TEXT_COLOR,
    PANEL_BG,
    DARK_BG,
    QUALITATIVE_PALETTE,
)

BANDS = ["1-10", "11-50", "51-100", "101+"]


def plot_band_transitions_sankey(
    df: pd.DataFrame,
    start_year: int = 2017,
    end_year: int = 2025,
    category: str = "Overall",
) -> Tuple[go.Figure, str]:
    """
    Question: How do institutions transition across ranking bands between two years?
    Renders an interactive Sankey diagram connecting source bands to target bands.
    """
    cat_df = df[df["category"] == category].copy()

    df_start = cat_df[cat_df["year"] == start_year][["institution_id", "name", "rank_band"]].rename(
        columns={"rank_band": "start_band"}
    )
    df_end = cat_df[cat_df["year"] == end_year][["institution_id", "name", "rank_band"]].rename(
        columns={"rank_band": "end_band"}
    )

    merged = df_start.merge(df_end, on=["institution_id", "name"], how="inner")

    if merged.empty:
        fig = go.Figure()
        return apply_theme(fig, title="Insufficient transition data"), "No comparative transition records."

    # Build Sankey nodes
    # Source nodes (left): 0 to 3
    # Target nodes (right): 4 to 7
    source_labels = [f"{b} ({start_year})" for b in BANDS]
    target_labels = [f"{b} ({end_year})" for b in BANDS]
    all_labels = source_labels + target_labels

    band_to_src_idx = {b: idx for idx, b in enumerate(BANDS)}
    band_to_tgt_idx = {b: idx + len(BANDS) for idx, b in enumerate(BANDS)}

    transition_counts = (
        merged.groupby(["start_band", "end_band"])
        .size()
        .reset_index(name="count")
    )

    sources = []
    targets = []
    values = []
    link_colors = []

    for _, row in transition_counts.iterrows():
        s_band = row["start_band"]
        e_band = row["end_band"]
        if s_band in band_to_src_idx and e_band in band_to_tgt_idx:
            s_idx = band_to_src_idx[s_band]
            t_idx = band_to_tgt_idx[e_band]
            sources.append(s_idx)
            targets.append(t_idx)
            val = int(row["count"])
            values.append(val)

            # Link color: Green if improved, Red if dropped, Slate if retained
            s_rank_tier = BANDS.index(s_band)
            e_rank_tier = BANDS.index(e_band)
            if e_rank_tier < s_rank_tier:  # Lower index = better rank
                link_colors.append("rgba(16, 185, 129, 0.45)")  # Green
            elif e_rank_tier > s_rank_tier:
                link_colors.append("rgba(244, 63, 94, 0.45)")  # Red
            else:
                link_colors.append("rgba(56, 189, 248, 0.35)")  # Blue/neutral

    node_colors = [
        "#38bdf8", "#f97316", "#10b981", "#8b949e",  # Left
        "#38bdf8", "#f97316", "#10b981", "#8b949e",  # Right
    ]

    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=15,
                thickness=18,
                line=dict(color="#0f1013", width=1),
                label=all_labels,
                color=node_colors,
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Rank Band Transitions ({start_year} → {end_year}, {category})",
            font=dict(size=15, color=TEXT_COLOR),
            x=0.02,
            y=0.96,
        ),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT_COLOR, family="Inter, sans-serif", size=12),
        margin=dict(l=60, r=40, t=65, b=50),
    )

    # Calculate key takeaway: 51-100 into top 50, and retention rate
    same_band_count = merged[merged["start_band"] == merged["end_band"]].shape[0]
    retention_pct = (same_band_count / len(merged)) * 100 if len(merged) > 0 else 0
    climbers_51_50 = merged[(merged["start_band"] == "51-100") & (merged["end_band"].isin(["1-10", "11-50"]))].shape[0]

    takeaway = (
        f"{retention_pct:.1f}% of institutions stayed in their original band between {start_year} and {end_year}; "
        f"{climbers_51_50} institutions climbed from the 51–100 tier into the Top 50."
    )
    return fig, takeaway


def plot_up_down_same_counts(
    df: pd.DataFrame,
    category: str = "Overall",
) -> Tuple[go.Figure, str]:
    """
    Question: How does the proportion of institutions moving up, down, or staying the same change each year?
    Renders stacked bar chart of YoY movement classifications.
    """
    cat_df = df[df["category"] == category].copy()
    valid = cat_df[cat_df["rank_change"].notna()].copy()

    if valid.empty:
        fig = go.Figure()
        return apply_theme(fig, title="Insufficient YoY data"), "No YoY movement records."

    def classify_movement(delta):
        if delta > 0:
            return "Rank Climbed (Up)"
        elif delta < 0:
            return "Rank Dropped (Down)"
        else:
            return "Unchanged (Same)"

    valid["Movement"] = valid["rank_change"].apply(classify_movement)

    counts = (
        valid.groupby(["year", "Movement"])
        .size()
        .reset_index(name="count")
    )

    fig = go.Figure()
    palette = {
        "Rank Climbed (Up)": ACCENT_GREEN,
        "Unchanged (Same)": MUTED_TEXT,
        "Rank Dropped (Down)": ACCENT_RED,
    }

    years = sorted(valid["year"].unique())

    for m_type, color in palette.items():
        m_data = counts[counts["Movement"] == m_type]
        fig.add_trace(
            go.Bar(
                name=m_type,
                x=m_data["year"],
                y=m_data["count"],
                marker_color=color,
                hovertemplate="<b>%{x}</b><br>" + f"{m_type}: " + "%{y}<extra></extra>",
            )
        )

    fig = apply_theme(
        fig,
        title=f"Year-over-Year Ranking Movement Counts ({category})",
        xaxis_title="Ranking Year",
        yaxis_title="Number of Institutions",
        show_legend=True,
    )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(tickmode="linear", dtick=1)

    latest_yr = years[-1]
    latest_counts = counts[counts["year"] == latest_yr].set_index("Movement")["count"].to_dict()
    up = latest_counts.get("Rank Climbed (Up)", 0)
    down = latest_counts.get("Rank Dropped (Down)", 0)
    same = latest_counts.get("Unchanged (Same)", 0)

    takeaway = f"In {latest_yr}: {up} institutions climbed, {down} dropped, and {same} held their exact rank."
    return fig, takeaway


def plot_largest_single_year_shifts(
    df: pd.DataFrame,
    category: str = "Overall",
    top_n: int = 8,
) -> Tuple[go.Figure, str]:
    """
    Question: What were the largest single-year upward and downward movements across the dataset?
    """
    cat_df = df[df["category"] == category].dropna(subset=["rank_change"]).copy()

    if cat_df.empty:
        fig = go.Figure()
        return apply_theme(fig, title="No movement data"), "No rank change records."

    cat_df["shift_abs"] = cat_df["rank_change"].abs()
    top_shifts = cat_df.sort_values(by="shift_abs", ascending=False).head(top_n).sort_values(by="rank_change", ascending=True)

    colors = [ACCENT_GREEN if d > 0 else ACCENT_RED for d in top_shifts["rank_change"]]

    labels = [
        f"{row['name']} ({row['year']})" for _, row in top_shifts.iterrows()
    ]

    fig = go.Figure(
        go.Bar(
            y=labels,
            x=top_shifts["rank_change"],
            orientation="h",
            marker=dict(color=colors),
            hovertemplate=(
                "<b>%{y}</b><br>"
                + "Shift: %{x:+d} ranks<br>"
                + "Prev Rank: #%{customdata[0]} → New Rank: #%{customdata[1]}<extra></extra>"
            ),
            customdata=list(zip(top_shifts["prev_rank"].astype(int), top_shifts["rank"].astype(int))),
        )
    )

    fig = apply_theme(
        fig,
        title=f"Largest Single-Year Ranking Shifts ({category})",
        xaxis_title="Rank Change in a Single Year",
        yaxis_title="",
    )

    best_shift = top_shifts[top_shifts["rank_change"] > 0].iloc[-1] if not top_shifts[top_shifts["rank_change"] > 0].empty else None
    worst_shift = top_shifts[top_shifts["rank_change"] < 0].iloc[0] if not top_shifts[top_shifts["rank_change"] < 0].empty else None

    takeaway_parts = []
    if best_shift is not None:
        takeaway_parts.append(f"Biggest 1-yr jump: {best_shift['name']} (+{int(best_shift['rank_change'])} ranks in {best_shift['year']})")
    if worst_shift is not None:
        takeaway_parts.append(f"biggest 1-yr drop: {worst_shift['name']} ({int(worst_shift['rank_change'])} ranks in {worst_shift['year']})")

    takeaway = "; ".join(takeaway_parts) + "."
    return fig, takeaway

