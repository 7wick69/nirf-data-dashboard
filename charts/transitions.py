"""
charts/transitions.py

Plotly visualization suite for Section 7:
"Ranking Transitions and Institutional Mobility" (NIRF 2017-2025).

Adheres strictly to the project's minimal dark design language (charts/theme.py):
  - Deep charcoal canvas (#0f1013) with panel card surface (#16181e)
  - Inverted Y-axis for ordinal rank visualizations (Rank 1 at the top)
  - Distinct semantic color encodings:
      * Emerald Green (#10b981): Upward mobility / Climbers / Promoted / Top 10 Entrants / Target Destination
      * Rose Red (#f43f5e): Downward mobility / Decliners / Demoted / Top 10 Exits
      * Sky Blue (#38bdf8): Retained / Stable positions / Stayed in band
      * Amber (#f59e0b): New Entrants
      * Slate Gray (#64748b): Dropped Out / Exited cohort
  - High-information hover tooltips, active selection subtitles, and automated takeaways.
"""

from typing import Tuple, Optional, List, Any, Dict
import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    go = None

from charts.theme import (
    apply_theme,
    DARK_BG,
    PANEL_BG,
    GRID_COLOR,
    LINE_COLOR,
    ACCENT_GREEN,
    ACCENT_RED,
    TEXT_COLOR,
    MUTED_TEXT,
    QUALITATIVE_PALETTE,
    MARKER_SYMBOLS,
)
from core.transitions import (
    DEFAULT_RANK_BANDS,
    compare_two_years,
    q1_band_transition,
    q2_top10_mobility,
    q3_band_retention,
    q4_biggest_movers,
    q5_annual_mobility_summary,
    get_category_consecutive_pairs,
)

ACCENT_AMBER = "#f59e0b"
ACCENT_SLATE = "#64748b"

# Fixed semantic colors for rank bands
BAND_COLOR_MAP: Dict[str, str] = {
    "Top 10": "#10b981",    # Emerald
    "11-50": "#38bdf8",     # Sky Blue
    "51-100": "#818cf8",    # Indigo
    "101-150": "#c084fc",   # Purple
    "151-200": "#f472b6",   # Pink
    "201+": "#f59e0b",      # Amber
    "Unranked": "#64748b",  # Slate
}


def _empty_figure_with_message(message: str) -> Tuple[Any, str]:
    """Helper returning an empty figure when data is unavailable."""
    if not HAS_PLOTLY:
        return None, message
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color=MUTED_TEXT)
    )
    fig = apply_theme(fig, title="No Data Available")
    return fig, message


# ---------------------------------------------------------
# Question 1: Visualizations
# ---------------------------------------------------------

def plot_band_destination_bar(
    df: pd.DataFrame,
    category: str = "Overall",
    y1: int = 2023,
    y2: int = 2024,
    from_band: str = "51-100",
    to_band: str = "Top 50",
    state: Optional[str] = None,
    basis: str = "Within selected state",
    bands: Optional[List[Tuple[str, float, float]]] = None,
    band_rank_method: str = "midpoint",
) -> Tuple[Any, str]:
    """
    Q1 Primary Visualization:
    Single 100% stacked horizontal bar answering:
    "Where did the institutions from the origin band go?"
    
    Segments:
      1. Moved to Target (Highlighted Emerald Green #10b981)
      2. Stayed in Origin Band (Sky Blue #38bdf8)
      3. Moved to Other Bands (Indigo #818cf8)
      4. Dropped Out (Slate Gray #64748b)
    """
    if not HAS_PLOTLY:
        return None, "Plotly is required for visualization."

    q1_res = q1_band_transition(
        df, category, y1, y2, from_band=from_band, to_band=to_band,
        state=state, basis=basis, bands=bands, band_rank_method=band_rank_method
    )
    dest = q1_res.get("dest_counts", {})
    total = dest.get("Total Origin", 0)

    if total == 0:
        return _empty_figure_with_message(
            f"No institutions found in '{from_band}' in {y1} for {category} ({state or 'All India'})."
        )

    fig = go.Figure()

    segments = [
        ("Moved to Target", dest.get("Moved to Target", 0), ACCENT_GREEN, f"Moved to {to_band}"),
        ("Stayed in Origin Band", dest.get("Stayed in Origin Band", 0), LINE_COLOR, f"Stayed in {from_band}"),
        ("Moved to Other Bands", dest.get("Moved to Other Bands", 0), "#818cf8", "Moved to Other Bands"),
        ("Dropped Out", dest.get("Dropped Out", 0), ACCENT_SLATE, "Dropped Out of Rankings"),
    ]

    for label, count, color, desc in segments:
        pct = (count / total * 100.0) if total > 0 else 0.0
        text_label = f"<b>{count}</b> ({pct:.1f}%)" if count > 0 else ""
        
        fig.add_trace(
            go.Bar(
                y=[f"Cohort in {y1}<br>({total} inst.)"],
                x=[pct],
                name=label,
                orientation="h",
                marker=dict(color=color, line=dict(color=GRID_COLOR, width=1)),
                text=text_label,
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(color=TEXT_COLOR, size=12),
                hovertemplate=(
                    f"<b>{label}</b> ({desc})<br>"
                    f"Count: {count} of {total} institutions<br>"
                    f"Share: {pct:.1f}%<extra></extra>"
                ),
            )
        )

    title_text = (
        f"Where Did Institutions from {from_band} Go? ({y1} → {y2})<br>"
        f"<sup style='color:{MUTED_TEXT}; font-size:11px;'>{q1_res['subtitle']}</sup>"
    )

    fig = apply_theme(
        fig,
        title=title_text,
        xaxis_title="Proportion of Origin Cohort (%)",
        yaxis_title="",
        show_legend=True,
        legend_position="top",
    )
    fig.update_layout(
        barmode="stack",
        height=280,
        margin=dict(l=110, r=40, t=80, b=50),
    )
    fig.update_xaxes(range=[0, 100.5], ticksuffix="%")

    takeaway = (
        f"In '{category}' ({y1}→{y2}, {q1_res['basis_label']}): Out of {total} institutions in {from_band}, "
        f"{dest.get('Moved to Target', 0)} ({q1_res['kpi_rate']}%) reached {to_band}, "
        f"{dest.get('Stayed in Origin Band', 0)} remained in {from_band}, and "
        f"{dest.get('Dropped Out', 0)} dropped out of rankings."
    )
    return fig, takeaway


def plot_mover_dumbbell(
    df: pd.DataFrame,
    category: str = "Overall",
    y1: int = 2023,
    y2: int = 2024,
    from_band: str = "51-100",
    to_band: str = "Top 50",
    state: Optional[str] = None,
    basis: str = "Within selected state",
    bands: Optional[List[Tuple[str, float, float]]] = None,
    band_rank_method: str = "midpoint",
) -> Tuple[Any, str]:
    """
    Q1 Secondary Visualization:
    Dumbbell chart of movers only:
    Dot for rank in Year 1, dot for rank in Year 2, connected by a line,
    sorted by rank gain (biggest climbers at the top).
    """
    if not HAS_PLOTLY:
        return None, "Plotly is required for visualization."

    q1_res = q1_band_transition(
        df, category, y1, y2, from_band=from_band, to_band=to_band,
        state=state, basis=basis, bands=bands, band_rank_method=band_rank_method
    )
    movers = q1_res["movers_df"].copy()

    if movers.empty:
        return _empty_figure_with_message(
            f"No institutions moved from {from_band} into {to_band} in {category} ({y1}→{y2})."
        )

    # Sort ascending so largest positive climbers appear at top of vertical axis
    movers = movers.sort_values(by="rank_change", ascending=True).reset_index(drop=True)

    fig = go.Figure()

    # 1. Connecting horizontal lines
    for _, row in movers.iterrows():
        r1, r2 = row["rank_y1"], row["rank_y2"]
        fig.add_trace(
            go.Scatter(
                x=[r1, r2],
                y=[row["name"], row["name"]],
                mode="lines",
                line=dict(color="#484f58", width=2.5),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # 2. Starting rank dots (Year 1)
    fig.add_trace(
        go.Scatter(
            x=movers["rank_y1"],
            y=movers["name"],
            mode="markers+text",
            name=f"Rank in {y1}",
            marker=dict(size=11, color=ACCENT_SLATE, symbol="circle", line=dict(color=TEXT_COLOR, width=1)),
            text=[f"#{int(r)}" for r in movers["rank_y1"]],
            textposition="middle left",
            textfont=dict(color=MUTED_TEXT, size=10),
            hovertemplate=(
                f"<b>%{{y}}</b><br>"
                f"{y1} Rank: #%{{x}}<extra></extra>"
            ),
        )
    )

    # 3. Ending rank dots (Year 2)
    fig.add_trace(
        go.Scatter(
            x=movers["rank_y2"],
            y=movers["name"],
            mode="markers+text",
            name=f"Rank in {y2} (Target)",
            marker=dict(size=13, color=ACCENT_GREEN, symbol="diamond", line=dict(color="#ffffff", width=1.5)),
            text=[f"#{int(r2)} (+{int(rc)})" for r2, rc in zip(movers["rank_y2"], movers["rank_change"])],
            textposition="middle right",
            textfont=dict(color=TEXT_COLOR, size=11),
            hovertemplate=(
                f"<b>%{{y}}</b><br>"
                f"{y2} Rank: #%{{x}}<br>"
                "Climb: +%{customdata} places<extra></extra>"
            ),
            customdata=movers["rank_change"],
        )
    )

    title_text = (
        f"Institutional Rank Improvement ({from_band} → {to_band}) [{y1} → {y2}]<br>"
        f"<sup style='color:{MUTED_TEXT}; font-size:11px;'>{q1_res['subtitle']} | Sorted by Rank Gain</sup>"
    )

    fig = apply_theme(
        fig,
        title=title_text,
        xaxis_title=f"Rank Position (Inverted: Left is Better)",
        yaxis_title="",
        show_legend=True,
        legend_position="top",
        invert_yaxis=False,
    )
    # Invert X axis so Rank 1 is on the left
    min_x = max(1, movers["rank_y2"].min() - 5)
    max_x = movers["rank_y1"].max() + 10
    fig.update_xaxes(autorange="reversed", range=[max_x, min_x])
    fig.update_layout(height=max(320, len(movers) * 36 + 140))

    best_mover = movers.iloc[-1]
    takeaway = (
        f"Top climber in {category} ({y1}→{y2}): {best_mover['name']} jumped +{int(best_mover['rank_change'])} places "
        f"from #{int(best_mover['rank_y1'])} to #{int(best_mover['rank_y2'])}."
    )
    return fig, takeaway


def plot_band_transition_heatmap(
    df: pd.DataFrame,
    category: str = "Overall",
    y1: int = 2023,
    y2: int = 2024,
    from_band: str = "51-100",
    to_band: str = "Top 50",
    state: Optional[str] = None,
    basis: str = "Within selected state",
    bands: Optional[List[Tuple[str, float, float]]] = None,
    band_rank_method: str = "midpoint",
) -> Tuple[Any, str]:
    """
    Q1 Optional View:
    Transition Matrix Heatmap with Top 10 at the top, zero cells blank,
    no repeated From/To prefixes, and target cell outlined.
    """
    if not HAS_PLOTLY:
        return None, "Plotly is required for visualization."

    q1_res = q1_band_transition(
        df, category, y1, y2, from_band=from_band, to_band=to_band,
        state=state, basis=basis, bands=bands, band_rank_method=band_rank_method
    )
    matrix = q1_res["transition_matrix"]
    
    if matrix.empty or matrix.values.sum() == 0:
        return _empty_figure_with_message(
            f"No transition records for '{category}' ({state or 'All India'}) between {y1} and {y2}."
        )

    x_labels = list(matrix.columns)
    y_labels = list(matrix.index)
    z_values = matrix.values

    target_cols = []
    if to_band == "Top 50":
        target_cols = [c for c in ["Top 10", "11-50"] if c in x_labels]
    elif to_band in x_labels:
        target_cols = [to_band]

    annotations = []
    for i, row_name in enumerate(y_labels):
        for j, col_name in enumerate(x_labels):
            val = int(z_values[i, j])
            is_target_cell = (row_name == from_band) and (col_name in target_cols)
            
            if val > 0:
                text_str = str(val)
                font_color = "#fbbf24" if is_target_cell else TEXT_COLOR
            else:
                text_str = ""  # Zero cells blank
                font_color = MUTED_TEXT
                
            annotations.append(dict(
                x=col_name,
                y=row_name,
                text=text_str,
                showarrow=False,
                font=dict(color=font_color, size=12, family="monospace"),
            ))

    dark_sequential_colorscale = [
        [0.0, PANEL_BG],
        [0.10, "#13233a"],
        [0.35, "#1d3d63"],
        [0.70, "#2563eb"],
        [1.0, "#38bdf8"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=x_labels,
            y=y_labels,
            colorscale=dark_sequential_colorscale,
            showscale=True,
            colorbar=dict(
                title=dict(text="Institutions", font=dict(color=TEXT_COLOR, size=11)),
                tickfont=dict(color=MUTED_TEXT, size=10),
            ),
            hoverongaps=False,
            hovertemplate=(
                f"<b>{y1} Band:</b> %{{y}}<br>"
                f"<b>{y2} Band:</b> %{{x}}<br>"
                "<b>Count:</b> %{z} institutions<extra></extra>"
            ),
        )
    )

    if from_band in y_labels and target_cols:
        r_idx = y_labels.index(from_band)
        for t_col in target_cols:
            c_idx = x_labels.index(t_col)
            fig.add_shape(
                type="rect",
                x0=c_idx - 0.48, x1=c_idx + 0.48,
                y0=r_idx - 0.48, y1=r_idx + 0.48,
                line=dict(color="#fbbf24", width=2.5),
                fillcolor="rgba(251, 191, 36, 0.08)",
            )

    title_text = (
        f"Rank Band Transition Matrix ({y1} → {y2})<br>"
        f"<sup style='color:{MUTED_TEXT}; font-size:11px;'>{q1_res['subtitle']}</sup>"
    )

    fig = apply_theme(
        fig,
        title=title_text,
        xaxis_title=f"Band in {y2}",
        yaxis_title=f"Band in {y1}",
    )
    
    # Invert Y-axis so Top 10 is at the top, Unranked at bottom
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(annotations=annotations)

    jump_str = f"+{q1_res['avg_rank_jump']} places" if q1_res['avg_rank_jump'] else "N/A"
    takeaway = (
        f"In '{category}' ({y1}→{y2}, {q1_res['basis_label']}): {q1_res['kpi_count']} of "
        f"{q1_res['kpi_total_from']} institution(s) ({q1_res['kpi_rate']}%) transitioned from "
        f"{from_band} into {to_band}, averaging a rank shift of {jump_str}."
    )
    return fig, takeaway


def plot_band_transition_sankey(
    df: pd.DataFrame,
    category: str = "Overall",
    y1: int = 2023,
    y2: int = 2024,
    from_band: str = "51-100",
    to_band: str = "Top 50",
    state: Optional[str] = None,
    basis: str = "Within selected state",
    bands: Optional[List[Tuple[str, float, float]]] = None,
    band_rank_method: str = "midpoint",
) -> Tuple[Any, str]:
    """
    Q1 Optional View:
    Sankey Flow diagram with consistent band colors and highlighted selected flow.
    """
    if not HAS_PLOTLY:
        return None, "Plotly is required for visualization."

    is_state_filtered = bool(state and state != "All states")
    effective_basis = basis if is_state_filtered else "National rank"

    merged = compare_two_years(
        df, category, y1, y2, state=state, basis=effective_basis, bands=bands, band_rank_method=band_rank_method
    )
    if merged.empty:
        return _empty_figure_with_message(f"No records found for '{category}' ({y1}→{y2}).")

    active_bands = bands or DEFAULT_RANK_BANDS
    band_names = [b[0] for b in active_bands] + ["Unranked"]

    counts_y1 = merged["band_y1"].value_counts().to_dict()
    counts_y2 = merged["band_y2"].value_counts().to_dict()

    source_labels = [f"{b} ({counts_y1.get(b, 0)})" for b in band_names]
    target_labels = [f"{b} ({counts_y2.get(b, 0)})" for b in band_names]
    all_node_labels = source_labels + target_labels

    node_colors = [BAND_COLOR_MAP.get(b, "#64748b") for b in band_names] * 2

    source_index_map = {b: i for i, b in enumerate(band_names)}
    target_index_map = {b: i + len(band_names) for i, b in enumerate(band_names)}

    flow_counts = (
        merged.groupby(["band_y1", "band_y2"])
        .size()
        .reset_index(name="count")
    )

    target_band_list = ["Top 10", "11-50"] if to_band == "Top 50" else [to_band]

    sources = []
    targets = []
    values = []
    link_colors = []

    for _, row in flow_counts.iterrows():
        b1, b2, cnt = row["band_y1"], row["band_y2"], int(row["count"])
        if cnt == 0:
            continue
        s_idx = source_index_map.get(b1)
        t_idx = target_index_map.get(b2)
        if s_idx is None or t_idx is None:
            continue
            
        sources.append(s_idx)
        targets.append(t_idx)
        values.append(cnt)

        is_selected_flow = (b1 == from_band) and (b2 in target_band_list)
        if is_selected_flow:
            link_colors.append("rgba(251, 191, 36, 0.85)")  # Vivid Gold highlight
        elif b1 == "Unranked":
            link_colors.append("rgba(245, 158, 11, 0.25)")
        elif b2 == "Unranked":
            link_colors.append("rgba(100, 116, 139, 0.20)")
        elif b1 == b2:
            link_colors.append("rgba(56, 189, 248, 0.30)")
        else:
            idx1 = band_names.index(b1) if b1 in band_names else 99
            idx2 = band_names.index(b2) if b2 in band_names else 99
            if idx2 < idx1:
                link_colors.append("rgba(16, 185, 129, 0.35)")
            else:
                link_colors.append("rgba(244, 63, 94, 0.30)")

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=16,
                    thickness=20,
                    line=dict(color=GRID_COLOR, width=1),
                    label=all_node_labels,
                    color=node_colors,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                    hovertemplate="%{source.label} → %{target.label}<br>Count: %{value} institutions<extra></extra>",
                ),
            )
        ]
    )

    basis_label = "Rank within state" if (is_state_filtered and effective_basis == "Within selected state") else "National rank"
    state_display = state if is_state_filtered else "All India"
    subtitle = f"{category} | {state_display} | {basis_label} | {y1} → {y2}"

    title_text = (
        f"Institutional Flow Across Rank Bands ({y1} → {y2})<br>"
        f"<sup style='color:{MUTED_TEXT}; font-size:11px;'>{subtitle}</sup>"
    )

    fig = apply_theme(fig, title=title_text)

    q1_info = q1_band_transition(
        df, category, y1, y2, from_band=from_band, to_band=to_band, state=state, basis=basis, bands=bands
    )
    takeaway = (
        f"Flow analysis for {category} ({y1}→{y2}, {basis_label}): {q1_info['kpi_count']} of {q1_info['kpi_total_from']} "
        f"institutions ({q1_info['kpi_rate']}%) moved from {from_band} to {to_band} (highlighted in gold)."
    )
    return fig, takeaway


# ---------------------------------------------------------
# Question 2: Visualizations (Top 10 Mobility)
# ---------------------------------------------------------

def plot_top10_mobility_slope(
    df: pd.DataFrame,
    category: str = "Overall",
    y1: int = 2023,
    y2: int = 2024,
    state: Optional[str] = None,
    basis: str = "Within selected state",
    band_rank_method: str = "midpoint",
) -> Tuple[Any, str]:
    """
    Q2 Primary Visualization:
    Slope chart tracking movements into, out of, and within the Top 10 tier.
    
    Key Features:
      - Clean outcome colors: Green = entered, Red = exited, Blue = retained
      - Dropped-out / unranked institutions are plotted in a labeled 'Dropped out'
        zone below the chart, avoiding fake numeric ranks.
      - Y-axis starts at Rank 1, reversed, with clean ticks.
    """
    if not HAS_PLOTLY:
        return None, "Plotly is required for visualization."

    q2_res = q2_top10_mobility(df, category, y1, y2, state=state, basis=basis, band_rank_method=band_rank_method)
    all_top10 = q2_res["all_top10_df"]

    if all_top10.empty:
        return _empty_figure_with_message(f"No Top-10 data found for '{category}' between {y1} and {y2}.")

    fig = go.Figure()
    is_state_focus = q2_res.get("is_state_active", False)
    is_state_basis = q2_res.get("basis") == "Within selected state"

    DROPOUT_Y = 12.5  # Distinct visual zone below rank 10 cutoff

    for _, row in all_top10.iterrows():
        inst_name = row["name"]
        r1, r2 = row["rank_y1"], row["rank_y2"]
        from_selected_state = row.get("is_selected_state", False)

        is_entered = pd.isna(r1) or r1 > 10.0
        is_exited = pd.isna(r2) or r2 > 10.0

        if is_entered:
            color = ACCENT_GREEN
            dash = "solid"
            group = "Entered Top 10"
        elif is_exited:
            color = ACCENT_RED
            dash = "dash"
            group = "Exited Top 10"
        else:
            color = LINE_COLOR
            dash = "solid"
            group = "Retained Top 10"

        # Highlight for target state when in National mode
        if not is_state_basis and is_state_focus and from_selected_state:
            color = "#fbbf24"
            width = 4.0
            marker_size = 11
            marker_sym = "star"
        else:
            width = 3.2 if (is_entered or is_exited) else 2.0
            marker_size = 8
            marker_sym = "circle"

        # Handle Y coordinates without fake ranks
        if pd.isna(r1):
            y_coord1 = DROPOUT_Y
            label_y1 = "Unranked"
        elif r1 > 10.0:
            y_coord1 = min(r1, 11.5)
            label_y1 = f"#{int(r1)}"
        else:
            y_coord1 = r1
            label_y1 = f"#{int(r1)}"

        if pd.isna(r2):
            y_coord2 = DROPOUT_Y
            label_y2 = "Dropped Out"
        elif r2 > 10.0:
            y_coord2 = min(r2, 11.5)
            label_y2 = f"#{int(r2)}"
        else:
            y_coord2 = r2
            label_y2 = f"#{int(r2)}"

        fig.add_trace(
            go.Scatter(
                x=[y1, y2],
                y=[y_coord1, y_coord2],
                mode="lines+markers+text",
                line=dict(color=color, width=width, dash=dash),
                marker=dict(size=marker_size, color=color, symbol=marker_sym),
                text=[f"{inst_name} ({label_y1})", f"{label_y2}"],
                textposition=["middle left", "middle right"],
                textfont=dict(size=10, color=TEXT_COLOR if (not is_state_focus or from_selected_state or is_state_basis) else MUTED_TEXT),
                name=inst_name,
                legendgroup=group,
                hovertemplate=(
                    f"<b>{inst_name}</b> ({row['state']})<br>"
                    f"{y1} Rank: {label_y1}<br>"
                    f"{y2} Rank: {label_y2}<br>"
                    f"Status: {group}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    # Reference cutoff line at Rank 10.5
    fig.add_hline(
        y=10.5,
        line=dict(color=MUTED_TEXT, width=1.2, dash="dot"),
        annotation_text="Top 10 Cutoff",
        annotation_position="bottom right",
        annotation_font=dict(color=MUTED_TEXT, size=10),
    )

    # Shaded zone boundary for Dropped Out / Unranked
    fig.add_hrect(
        y0=11.5, y1=13.5,
        fillcolor="rgba(100, 116, 139, 0.08)",
        line=dict(color="#30363d", width=1, dash="dash"),
        annotation_text="Dropped out / Unranked Zone",
        annotation_position="bottom left",
        annotation_font=dict(color=MUTED_TEXT, size=10),
    )

    basis_str = "Within State" if is_state_basis else "National Rank"
    state_display = state if (is_state_focus and state != "All states") else "All India"
    title_text = (
        f"Top 10 Roster Mobility ({y1} → {y2})<br>"
        f"<sup style='color:{MUTED_TEXT}; font-size:11px;'>{category} | {state_display} | {basis_str} | Green: Entered, Red: Exited, Blue: Retained</sup>"
    )

    fig = apply_theme(
        fig,
        title=title_text,
        xaxis_title="Year",
        yaxis_title="Rank Position (Rank 1 at Top)",
        invert_yaxis=False,
    )

    # Clean ticks starting from Rank 1 down to 10 and the Dropped-Out Zone
    fig.update_xaxes(tickvals=[y1, y2], ticktext=[str(y1), str(y2)], range=[y1 - 0.55, y2 + 0.35])
    fig.update_yaxes(
        autorange="reversed",
        range=[13.5, 0.5],
        tickvals=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, DROPOUT_Y],
        ticktext=["#1", "#2", "#3", "#4", "#5", "#6", "#7", "#8", "#9", "#10", "Dropped Out"],
    )

    return fig, q2_res["takeaway"]


def plot_top10_turnover_bars(
    df: pd.DataFrame,
    category: str = "Overall",
    state: Optional[str] = None,
    basis: str = "Within selected state",
) -> Tuple[Any, str]:
    """
    Q2 Consecutive Years Mode Secondary Visualization:
    Bar chart showing counts of Entered (green) and Exited (red) institutions
    per consecutive year cycle.
    """
    if not HAS_PLOTLY:
        return None, "Plotly is required for visualization."

    pairs = get_category_consecutive_pairs(df, category)
    if not pairs:
        return _empty_figure_with_message(f"No consecutive year cycles available for '{category}'.")

    records = []
    for y1, y2 in pairs:
        res = q2_top10_mobility(df, category, y1, y2, state=state, basis=basis)
        records.append({
            "pair": f"{y1}→{y2}",
            "entered": res["num_entered"],
            "exited": res["num_exited"],
            "retained": res["num_retained"],
        })

    turnover_df = pd.DataFrame(records)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=turnover_df["pair"],
            y=turnover_df["entered"],
            name="Entered Top 10",
            marker=dict(color=ACCENT_GREEN, line=dict(color=GRID_COLOR, width=0.5)),
            text=[str(v) if v > 0 else "" for v in turnover_df["entered"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Entered Top 10: %{y} institutions<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            x=turnover_df["pair"],
            y=turnover_df["exited"],
            name="Exited Top 10",
            marker=dict(color=ACCENT_RED, line=dict(color=GRID_COLOR, width=0.5)),
            text=[str(v) if v > 0 else "" for v in turnover_df["exited"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Exited Top 10: %{y} institutions<extra></extra>",
        )
    )

    is_state_filtered = bool(state and state != "All states")
    basis_str = "Within State" if (is_state_filtered and basis == "Within selected state") else "National"
    title_text = (
        f"Top 10 Roster Turnover Across Annual Cycles — {category}<br>"
        f"<sup style='color:{MUTED_TEXT}; font-size:11px;'>{state or 'All India'} | {basis_str} | Entered vs Exited Counts</sup>"
    )

    fig = apply_theme(
        fig,
        title=title_text,
        xaxis_title="Consecutive Year Pair",
        yaxis_title="Number of Institutions",
        show_legend=True,
        legend_position="top",
    )
    fig.update_layout(barmode="group")

    avg_turnover = (turnover_df["entered"].mean() + turnover_df["exited"].mean()) / 2.0
    takeaway = (
        f"Across all consecutive year cycles in '{category}', an average of {avg_turnover:.1f} institution(s) "
        f"entered and exited the Top 10 per cycle."
    )
    return fig, takeaway


# ---------------------------------------------------------
# Question 3: Visualizations (Band Retention)
# ---------------------------------------------------------

def plot_band_retention_trend(
    df: pd.DataFrame,
    category: str = "Overall",
    state: Optional[str] = None,
    basis: str = "Within selected state",
    bands: Optional[List[Tuple[str, float, float]]] = None,
    band_rank_method: str = "midpoint",
) -> Tuple[Any, str]:
    """
    Q3 Primary Visualization:
    Retention line chart:
      - Y-axis starts at 0 (range=[0, 105])
      - Shows pooled historical average reference line
      - Hover displays 'x of y' retention counts
    """
    if not HAS_PLOTLY:
        return None, "Plotly is required for visualization."

    ret_df = q3_band_retention(
        df, category, state=state, basis=basis, bands=bands, band_rank_method=band_rank_method
    )
    if ret_df.empty:
        return _empty_figure_with_message(f"Insufficient multi-year records for '{category}' ({state or 'All India'}).")

    fig = go.Figure()

    # Primary retention line
    fig.add_trace(
        go.Scatter(
            x=ret_df["pair"],
            y=ret_df["retention_rate_pct"],
            mode="lines+markers+text",
            line=dict(color=LINE_COLOR, width=3),
            marker=dict(size=9, color=LINE_COLOR, symbol="diamond"),
            text=[f"{v:.1f}%" for v in ret_df["retention_rate_pct"]],
            textposition="top center",
            textfont=dict(color=TEXT_COLOR, size=11),
            name="Retention Rate %",
            hovertemplate=(
                "<b>Period: %{x}</b><br>"
                "Retained: <b>%{customdata[1]} of %{customdata[0]}</b> institutions (%{y:.1f}%)<br>"
                "Promoted to Higher Band: %{customdata[2]}<br>"
                "Demoted to Lower Band: %{customdata[3]}<extra></extra>"
            ),
            customdata=np.stack((
                ret_df["continuing_total"],
                ret_df["same_band_count"],
                ret_df["promoted_count"],
                ret_df["demoted_count"],
            ), axis=-1),
        )
    )

    # Pooled historical average line
    mean_retention = ret_df["retention_rate_pct"].mean()
    fig.add_hline(
        y=mean_retention,
        line=dict(color=MUTED_TEXT, width=1.5, dash="dash"),
        annotation_text=f"Historical Mean: {mean_retention:.1f}%",
        annotation_position="bottom right",
        annotation_font=dict(color=MUTED_TEXT, size=10),
    )

    is_state_filtered = bool(state and state != "All states")
    basis_label = "Within State" if (is_state_filtered and basis == "Within selected state") else "National"
    title_text = (
        f"Rank Band Retention Rate Trend (2017–2025) — {category}<br>"
        f"<sup style='color:{MUTED_TEXT}; font-size:11px;'>{state or 'All India'} | {basis_label} | Baseline at 0%</sup>"
    )

    fig = apply_theme(
        fig,
        title=title_text,
        xaxis_title="Consecutive Year Pair",
        yaxis_title="Institutions Retained in Same Band (%)",
    )
    # Start y-axis strictly at 0
    fig.update_yaxes(range=[0, 105], ticksuffix="%")

    min_pair_row = ret_df.loc[ret_df["retention_rate_pct"].idxmin()]
    max_pair_row = ret_df.loc[ret_df["retention_rate_pct"].idxmax()]
    takeaway = (
        f"Across all consecutive years in '{category}' ({basis_label}), an average of {mean_retention:.1f}% "
        f"of institutions maintained their rank tier. Highest stability was in {max_pair_row['pair']} "
        f"({max_pair_row['retention_rate_pct']}%), while highest mobility occurred in {min_pair_row['pair']} "
        f"({min_pair_row['retention_rate_pct']}% retained)."
    )
    return fig, takeaway


def plot_band_retention_stacked_bar(
    df: pd.DataFrame,
    category: str = "Overall",
    state: Optional[str] = None,
    basis: str = "Within selected state",
    bands: Optional[List[Tuple[str, float, float]]] = None,
    band_rank_method: str = "midpoint",
) -> Tuple[Any, str]:
    """
    Q3 Secondary Visualization:
    Stacked bar chart per year pair showing what happened to continuing institutions:
      - Retained in Same Band (Sky Blue #38bdf8)
      - Promoted to Higher Band (Emerald Green #10b981)
      - Demoted to Lower Band (Rose Red #f43f5e)
    """
    if not HAS_PLOTLY:
        return None, "Plotly is required for visualization."

    ret_df = q3_band_retention(
        df, category, state=state, basis=basis, bands=bands, band_rank_method=band_rank_method
    )
    if ret_df.empty:
        return _empty_figure_with_message(f"No records available for '{category}'.")

    fig = go.Figure()

    series = [
        ("Retained in Same Band", ret_df["same_band_count"], LINE_COLOR),
        ("Promoted to Higher Band", ret_df["promoted_count"], ACCENT_GREEN),
        ("Demoted to Lower Band", ret_df["demoted_count"], ACCENT_RED),
    ]

    for label, vals, color in series:
        fig.add_trace(
            go.Bar(
                x=ret_df["pair"],
                y=vals,
                name=label,
                marker=dict(color=color, line=dict(color=GRID_COLOR, width=0.5)),
                hovertemplate=(
                    f"<b>{label}</b> (%{{x}})<br>"
                    f"Count: %{{y}} institutions<br>"
                    f"Total Continuing: %{{customdata}} institutions<extra></extra>"
                ),
                customdata=ret_df["continuing_total"],
            )
        )

    is_state_filtered = bool(state and state != "All states")
    basis_label = "Within State" if (is_state_filtered and basis == "Within selected state") else "National"
    title_text = (
        f"Annual Band Transition Breakdown — {category}<br>"
        f"<sup style='color:{MUTED_TEXT}; font-size:11px;'>{state or 'All India'} | {basis_label} | Continuing Institutional Cohort</sup>"
    )

    fig = apply_theme(
        fig,
        title=title_text,
        xaxis_title="Consecutive Year Pair",
        yaxis_title="Number of Continuing Institutions",
        show_legend=True,
        legend_position="top",
    )
    fig.update_layout(barmode="stack")

    takeaway = (
        f"Band mobility composition for '{category}' ({basis_label}): Most continuing institutions retain their "
        f"tier, while promotions (green) and demotions (red) capture inter-tier volatility."
    )
    return fig, takeaway


# ---------------------------------------------------------
# Question 4: Visualizations (Largest Movers)
# ---------------------------------------------------------

def plot_diverging_movers(
    df: pd.DataFrame,
    category: str = "Overall",
    y1: int = 2023,
    y2: int = 2024,
    top_n: int = 10,
    state: Optional[str] = None,
    basis: str = "Within selected state",
    band_rank_method: str = "midpoint",
) -> Tuple[Any, str]:
    """
    Q4 Primary Visualization:
    Diverging horizontal bar chart contrasting Top N Risers (green) against Top N Fallers (red).
    
    Features:
      - Badges: Green with up arrow (▲ +X) for risers, Red with down arrow (▼ -X) for fallers
      - Hover shows Rank Y1 -> Rank Y2, State, and Bands
      - Footnote explaining midpoint / lower bound convention for rank bands
    """
    if not HAS_PLOTLY:
        return None, "Plotly is required for visualization."

    q4_res = q4_biggest_movers(
        df, category, y1, y2, top_n=top_n, state=state, basis=basis, band_rank_method=band_rank_method
    )
    risers = q4_res["risers_df"]
    fallers = q4_res["fallers_df"]

    if risers.empty and fallers.empty:
        return _empty_figure_with_message(f"No rank mover records for '{category}' ({y1}→{y2}).")

    chart_df = pd.concat([fallers, risers], ignore_index=True)
    chart_df = chart_df.sort_values(by="rank_change", ascending=True).reset_index(drop=True)

    colors = [ACCENT_GREEN if val > 0 else ACCENT_RED for val in chart_df["rank_change"]]
    badge_texts = [
        f"▲ +{int(c)}" if c > 0 else f"▼ {int(c)}" for c in chart_df["rank_change"]
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=chart_df["name"],
            x=chart_df["rank_change"],
            orientation="h",
            marker=dict(color=colors, line=dict(color=GRID_COLOR, width=1)),
            text=badge_texts,
            textposition="outside",
            textfont=dict(color=TEXT_COLOR, size=12, family="monospace"),
            hovertemplate=(
                "<b>%{y}</b> (%{customdata[0]})<br>"
                f"Rank in {y1}: #%{{customdata[1]}}<br>"
                f"Rank in {y2}: #%{{customdata[2]}}<br>"
                "Rank Change: %{x:+d} places<extra></extra>"
            ),
            customdata=np.stack((chart_df["state"], chart_df["rank_y1"], chart_df["rank_y2"]), axis=-1),
        )
    )

    is_state_filtered = bool(state and state != "All states")
    basis_label = "Within State" if (is_state_filtered and basis == "Within selected state") else "National"
    title_text = (
        f"Largest Rank Climbers & Decliners ({y1} → {y2}) — {category}<br>"
        f"<sup style='color:{MUTED_TEXT}; font-size:11px;'>{state or 'All India'} | {basis_label} | Top {top_n} Risers (Green) vs Top {top_n} Fallers (Red)</sup>"
    )

    fig = apply_theme(
        fig,
        title=title_text,
        xaxis_title=f"Rank Shift (Rank_{y1} − Rank_{y2}, Positive = Improvement)",
        yaxis_title="",
    )

    max_abs = max(abs(chart_df["rank_change"].min()), abs(chart_df["rank_change"].max())) + 6
    fig.update_xaxes(range=[-max_abs, max_abs], zeroline=True, zerolinewidth=1.5, zerolinecolor="#8b949e")
    fig.update_layout(height=max(360, len(chart_df) * 26 + 120))

    # Footnote explaining the ranking method convention
    method_val = "125.5" if band_rank_method == "midpoint" else "101.0"
    fig.add_annotation(
        text=f"* Note: Ranks for banded cohorts (e.g. 101–150) are evaluated using the {band_rank_method} convention ({method_val}).",
        xref="paper", yref="paper",
        x=0.0, y=-0.14,
        showarrow=False,
        font=dict(size=10, color=MUTED_TEXT),
    )

    return fig, q4_res["takeaway"]


# ---------------------------------------------------------
# Question 5: Visualizations (Annual Mobility Breakdown)
# ---------------------------------------------------------

def plot_mobility_composition_bar(
    df: pd.DataFrame,
    category: str = "Overall",
    state: Optional[str] = None,
    basis: str = "Within selected state",
    mode: str = "count",
    bands: Optional[List[Tuple[str, float, float]]] = None,
    band_rank_method: str = "midpoint",
) -> Tuple[Any, str]:
    """
    Q5 Primary Visualization:
    Stacked bar chart showing annual mobility breakdown:
    Moved Up (green), Retained (blue), Moved Down (red), New Entry (amber), Dropped Out (grey).
    
    Supports:
      - mode='count': raw counts of institutions
      - mode='percent': 100% stacked percentage share
      - Denominator caption explaining cohort calculation
    """
    if not HAS_PLOTLY:
        return None, "Plotly is required for visualization."

    mob_df = q5_annual_mobility_summary(
        df, category, state=state, basis=basis, bands=bands, band_rank_method=band_rank_method
    )
    if mob_df.empty:
        return _empty_figure_with_message(f"No annual mobility records for '{category}'.")

    fig = go.Figure()

    series_config = [
        ("Moved Up", ACCENT_GREEN, "Moved Up (%)"),
        ("Retained", LINE_COLOR, "Retained (%)"),
        ("Moved Down", ACCENT_RED, "Moved Down (%)"),
        ("New Entry", ACCENT_AMBER, "New Entry (%)"),
        ("Dropped Out", ACCENT_SLATE, "Dropped Out (%)"),
    ]

    for label, color, pct_col in series_config:
        y_vals = mob_df[pct_col] if mode == "percent" else mob_df[label]
        text_vals = [f"{v:.1f}%" if mode == "percent" else str(int(v)) for v in y_vals]
        
        fig.add_trace(
            go.Bar(
                x=mob_df["pair"],
                y=y_vals,
                name=label,
                marker=dict(color=color, line=dict(color=GRID_COLOR, width=0.5)),
                text=[t if ((float(v) > 3.0 if mode == 'percent' else v > 0)) else "" for t, v in zip(text_vals, y_vals)],
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(color=TEXT_COLOR, size=10),
                hovertemplate=(
                    f"<b>{label}</b> (%{{x}})<br>"
                    f"Count: %{{customdata[0]}} institutions<br>"
                    f"Share: %{{customdata[1]:.1f}}%<br>"
                    f"Total Tracked Cohort: %{{customdata[2]}}<extra></extra>"
                ),
                customdata=np.stack((mob_df[label], mob_df[pct_col], mob_df["Total Tracked"]), axis=-1),
            )
        )

    is_state_filtered = bool(state and state != "All states")
    basis_label = "Within State" if (is_state_filtered and basis == "Within selected state") else "National"
    y_axis_label = "Percentage Share of Tracked Cohort (%)" if mode == "percent" else "Number of Institutions"

    title_text = (
        f"Annual Institutional Mobility Breakdown (2017–2025) — {category}<br>"
        f"<sup style='color:{MUTED_TEXT}; font-size:11px;'>{state or 'All India'} | {basis_label} | {mode.capitalize()} View</sup>"
    )

    fig = apply_theme(
        fig,
        title=title_text,
        xaxis_title="Consecutive Year Pair",
        yaxis_title=y_axis_label,
        show_legend=True,
        legend_position="top",
    )
    fig.update_layout(barmode="stack")
    if mode == "percent":
        fig.update_yaxes(range=[0, 100.5], ticksuffix="%")

    avg_up = mob_df["Moved Up (%)"].mean()
    avg_down = mob_df["Moved Down (%)"].mean()
    avg_retained = mob_df["Retained (%)"].mean()
    avg_turnover = (mob_df["New Entry"] + mob_df["Dropped Out"]).mean()

    takeaway = (
        f"Annual stability overview for '{category}' ({basis_label}): On average, {avg_up:.1f}% climbed, "
        f"{avg_down:.1f}% dropped, and {avg_retained:.1f}% held their exact position. "
        f"Average annual turnover (New Entries + Dropouts) was {avg_turnover:.1f} institutions per cycle."
    )
    return fig, takeaway
