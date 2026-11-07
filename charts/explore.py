"""
charts/explore.py

Tableau-Style Visual Analytics Workbench (Level 2 Explore Tab) — NIRF Interactive Dashboard.
Replicates Tableau's core visual architecture:
  - Shelves: Columns (X), Rows (Y), Marks Card (Chart Type & Color Encoding), Filters Shelf
  - Automatic Level-of-Detail: Aggregates measures across all shelf dimensions (X, Color)
  - Smart Marks Inference: Replicates Tableau's 'Show Me' logic
  - Automatic Layout: Horizontal bars for long labels (institutions, states, cities)
  - Statistical Analysis: Dynamic Pearson correlation (r), R², p-value, and OLS trendlines
  - Export: Accurate JSON Query Spec inspector and one-click CSV export
"""

import json
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from charts.theme import (
    apply_theme,
    DARK_BG,
    PANEL_BG,
    GRID_COLOR,
    TEXT_COLOR,
    MUTED_TEXT,
    LINE_COLOR,
    QUALITATIVE_PALETTE,
    MARKER_SYMBOLS,
)

# ---------------------------------------------------------------------------
# Dimension & Measure Definitions (Tableau Data Pane)
# ---------------------------------------------------------------------------

ALL_DIMENSIONS: Dict[str, str] = {
    "Institution Name": "name",
    "State": "state",
    "Ranking Year": "year",
    "Category": "category",
    "Rank Band": "rank_band",
    "City": "city",
}

ALL_MEASURES: Dict[str, str] = {
    "Total Score": "score",
    "Teaching & Resources (TLR)": "TLR",
    "Research Practice (RPC)": "RPC",
    "Graduation Outcomes (GO)": "GO",
    "Outreach & Inclusivity (OI)": "OI",
    "Perception (PR)": "PR",
    "NIRF Rank": "rank",
    "Rank Change YoY": "rank_change",
    "Score Change YoY": "score_change_yoy",
    "Parameter Disparity (Gap)": "param_gap",
    "Rank Volatility (σ)": "volatility",
}

AGG_FUNCS: Dict[str, str] = {
    "Average (Mean)": "mean",
    "Median": "median",
    "Sum": "sum",
    "Maximum": "max",
    "Minimum": "min",
    "Count of Records": "count",
    "Standard Deviation (std)": "std",
}

COLOR_OPTIONS: Dict[str, Optional[str]] = {
    "None": None,
    "Category": "category",
    "State": "state",
    "Rank Band": "rank_band",
    "Institution Name": "name",
}

# Analytical Presets (Unseen Viva Evaluation Helpers)
PRESETS: Dict[str, Dict[str, Any]] = {
    "1. Top 15 Institutions by Graduation Outcomes (GO) in Engineering": {
        "tab_col_type": "Dimension",
        "tab_col_dim": "Institution Name",
        "tab_row_meas": "Graduation Outcomes (GO)",
        "tab_agg_func": "Average (Mean)",
        "tab_mark_type": "Bar",
        "tab_color_pill": "None",
        "tab_filter_cat": "Engineering",
        "tab_top_n": 15,
        "tab_sort_order": "Descending (Highest First)",
        "tab_filter_states": [],
        "tab_filter_insts": [],
    },
    "2. Teaching (TLR) vs Perception (PR) Correlation across Universities": {
        "tab_col_type": "Measure",
        "tab_col_meas": "Teaching & Resources (TLR)",
        "tab_row_meas": "Perception (PR)",
        "tab_mark_type": "Scatter",
        "tab_color_pill": "None",
        "tab_filter_cat": "University",
        "tab_top_n": 15,
        "tab_filter_states": [],
        "tab_filter_insts": [],
    },
    "3. Average Research Productivity (RPC) by State (2024)": {
        "tab_col_type": "Dimension",
        "tab_col_dim": "State",
        "tab_row_meas": "Research Practice (RPC)",
        "tab_agg_func": "Average (Mean)",
        "tab_mark_type": "Bar",
        "tab_color_pill": "None",
        "tab_filter_cat": "Overall",
        "tab_filter_year": (2024, 2024),
        "tab_top_n": 15,
        "tab_sort_order": "Descending (Highest First)",
        "tab_filter_states": [],
        "tab_filter_insts": [],
    },
    "4. Score Distribution across Disciplines (Box Plot)": {
        "tab_col_type": "Dimension",
        "tab_col_dim": "Category",
        "tab_row_meas": "Total Score",
        "tab_mark_type": "Box Plot",
        "tab_color_pill": "None",
        "tab_filter_cat": "All Categories",
        "tab_top_n": 15,
        "tab_filter_states": [],
        "tab_filter_insts": [],
    },
    "5. Overall Score Trajectory over Time (2017–2025)": {
        "tab_col_type": "Dimension",
        "tab_col_dim": "Ranking Year",
        "tab_row_meas": "Total Score",
        "tab_agg_func": "Average (Mean)",
        "tab_mark_type": "Line",
        "tab_color_pill": "None",
        "tab_filter_cat": "Overall",
        "tab_top_n": 15,
        "tab_filter_states": [],
        "tab_filter_insts": [],
    },
}


def calculate_pearson(df: pd.DataFrame, x_col: str, y_col: str) -> Optional[Dict[str, Any]]:
    """Calculates Pearson r, R², p-value between two continuous measures using pairwise dropna."""
    if x_col not in df.columns or y_col not in df.columns or x_col == y_col:
        return None
    valid = df[[x_col, y_col]].dropna()
    if len(valid) < 3:
        return None
    try:
        x_vals = pd.to_numeric(valid[x_col], errors="coerce").dropna()
        y_vals = pd.to_numeric(valid[y_col], errors="coerce").dropna()
        common_idx = x_vals.index.intersection(y_vals.index)
        if len(common_idx) < 3:
            return None
        r, p = stats.pearsonr(x_vals.loc[common_idx], y_vals.loc[common_idx])
        return {
            "r": float(r),
            "r2": float(r ** 2),
            "p": float(p),
            "n": int(len(common_idx)),
        }
    except Exception:
        return None


# Backward-compatible alias
compute_pearson_r = calculate_pearson


def validate_shelves(x_is_dim: bool, x_field: str, y_field: str, mark: str) -> Tuple[bool, str]:
    """
    Validates whether the selected shelves (Columns, Rows) are compatible with the Mark type.
    Returns (is_valid, user_friendly_message).
    """
    if mark == "Scatter":
        if x_is_dim:
            return False, "Scatter plots require continuous measures on both X (Columns) and Y (Rows). Please switch Columns to a Measure."
        if x_field == y_field:
            return False, f"Scatter plot requires two different measures. Columns and Rows are both '{x_field}'. Please select different measures."
        return True, ""

    if mark in ("Bar", "Line", "Box Plot", "Box"):
        if not x_is_dim:
            return False, f"{mark} charts require a Dimension on X (Columns) and a Measure on Y (Rows). Please switch Columns to a Dimension, or switch Mark to Scatter."
        return True, ""

    if mark == "Histogram":
        # Only measure is required (uses Y; ignores X)
        return True, ""

    if mark == "Table":
        return True, ""

    return True, ""


def render_explore_tab(df_full: pd.DataFrame, df_filtered: pd.DataFrame):
    """
    Renders the Tableau-style visual analytics workbench.
    Follows Tableau's Level of Detail (LOD) aggregation model.
    """
    st.markdown("### Explore: Tableau-Style Visual Query Builder")
    st.caption("Drag-and-drop shelf logic: select Columns, Rows, and Aggregations to generate clean, publication-grade charts.")

    # -----------------------------------------------------------------------
    # Guard against missing columns (Fix 9)
    # -----------------------------------------------------------------------
    avail_cols = set(df_full.columns)
    dimensions = {lbl: col for lbl, col in ALL_DIMENSIONS.items() if col in avail_cols}
    measures = {lbl: col for lbl, col in ALL_MEASURES.items() if col in avail_cols}

    dropped_dims = [lbl for lbl, col in ALL_DIMENSIONS.items() if col not in avail_cols]
    dropped_meas = [lbl for lbl, col in ALL_MEASURES.items() if col not in avail_cols]
    if dropped_dims or dropped_meas:
        st.warning(f"Note: Some expected fields are not present in dataset: {dropped_dims + dropped_meas}")

    if not dimensions or not measures:
        st.error("Dataset lacks sufficient dimension or measure fields to render the visual workbench.")
        return

    # Category choices: Default to 'Overall' if present, never 'All Categories' (Fix 2)
    unique_cats = sorted([str(c) for c in df_full["category"].dropna().unique()])
    if "Overall" in unique_cats:
        cat_options = ["Overall"] + [c for c in unique_cats if c != "Overall"] + ["All Categories"]
        default_cat = "Overall"
    else:
        cat_options = unique_cats + ["All Categories"]
        default_cat = unique_cats[0] if unique_cats else "All Categories"

    min_y = int(df_full["year"].min()) if "year" in df_full.columns else 2017
    max_y = int(df_full["year"].max()) if "year" in df_full.columns else 2025

    # -----------------------------------------------------------------------
    # Preset Quick-Loads Callback (Fix 1: replaces override block)
    # -----------------------------------------------------------------------
    def apply_preset():
        choice = st.session_state.get("tab_preset_selector")
        p = PRESETS.get(choice)
        if p:
            for k, v in p.items():
                if k == "tab_filter_cat" and v not in cat_options:
                    continue
                if k == "tab_filter_year" and isinstance(v, tuple):
                    v = (max(min_y, v[0]), min(max_y, v[1]))
                st.session_state[k] = v

    # Initialize session state keys cleanly before widget instantiation (prevents Streamlit warnings)
    st.session_state.setdefault("tab_preset_selector", "None (Custom Query)")
    st.session_state.setdefault("tab_col_type", "Dimension")
    st.session_state.setdefault("tab_col_dim", "State" if "State" in dimensions else list(dimensions.keys())[0])
    st.session_state.setdefault("tab_col_meas", "Teaching & Resources (TLR)" if "Teaching & Resources (TLR)" in measures else list(measures.keys())[0])
    st.session_state.setdefault("tab_row_meas", "Research Practice (RPC)" if "Research Practice (RPC)" in measures else list(measures.keys())[0])
    st.session_state.setdefault("tab_agg_func", "Average (Mean)")
    st.session_state.setdefault("tab_mark_type", "Automatic (Show Me)")
    st.session_state.setdefault("tab_color_pill", "None")
    st.session_state.setdefault("tab_filter_cat", default_cat)
    st.session_state.setdefault("tab_filter_year", (min_y, max_y))
    st.session_state.setdefault("tab_filter_states", [])
    st.session_state.setdefault("tab_filter_insts", [])
    st.session_state.setdefault("tab_top_n", 15)
    st.session_state.setdefault("tab_sort_order", "Descending (Highest First)")

    # Safety checks against stale session state
    if st.session_state.get("tab_col_dim") not in dimensions:
        st.session_state["tab_col_dim"] = list(dimensions.keys())[0]
    if st.session_state.get("tab_col_meas") not in measures:
        st.session_state["tab_col_meas"] = list(measures.keys())[0]
    if st.session_state.get("tab_row_meas") not in measures:
        st.session_state["tab_row_meas"] = list(measures.keys())[0]
    if st.session_state.get("tab_agg_func") not in AGG_FUNCS:
        st.session_state["tab_agg_func"] = "Average (Mean)"
    if st.session_state.get("tab_filter_cat") not in cat_options:
        st.session_state["tab_filter_cat"] = default_cat

    preset_options = [
        "None (Custom Query)",
        "1. Top 15 Institutions by Graduation Outcomes (GO) in Engineering",
        "2. Teaching (TLR) vs Perception (PR) Correlation across Universities",
        "3. Average Research Productivity (RPC) by State (2024)",
        "4. Score Distribution across Disciplines (Box Plot)",
        "5. Overall Score Trajectory over Time (2017–2025)",
    ]

    c_preset1, _ = st.columns([2.5, 1])
    with c_preset1:
        st.selectbox(
            "⚡ Quick-Load Analytical Preset (Unseen Evaluation Questions)",
            options=preset_options,
            key="tab_preset_selector",
            on_change=apply_preset,
        )

    # -----------------------------------------------------------------------
    # Shelves Container (Columns, Rows, Marks) — Fix 8: native border container
    # -----------------------------------------------------------------------
    with st.container(border=True):
        col_c, col_r, col_m = st.columns([1.2, 1.2, 1.0], gap="medium")

        # 1. COLUMNS SHELF (X-Axis)
        with col_c:
            st.markdown("**Columns (X Shelf)**")
            col_shelf_type = st.radio("Field Type", ["Dimension", "Measure"], horizontal=True, key="tab_col_type")
            if col_shelf_type == "Dimension":
                col_display_name = st.selectbox("Dimension", list(dimensions.keys()), key="tab_col_dim")
                x_field = dimensions[col_display_name]
                x_is_dim = True
            else:
                col_display_name = st.selectbox("Measure", list(measures.keys()), key="tab_col_meas")
                x_field = measures[col_display_name]
                x_is_dim = False

        # 2. ROWS SHELF (Y-Axis)
        with col_r:
            st.markdown("**Rows (Y Shelf)**")
            row_display_name = st.selectbox("Measure", list(measures.keys()), key="tab_row_meas")
            y_field = measures[row_display_name]

            # Aggregation Selector
            agg_display = st.selectbox("Aggregation", list(AGG_FUNCS.keys()), key="tab_agg_func")
            agg_code = AGG_FUNCS[agg_display]

        # 3. MARKS CARD (Chart Family & Options)
        with col_m:
            st.markdown("**Marks Card (Show Me)**")
            mark_choice = st.selectbox(
                "Mark Type",
                ["Automatic (Show Me)", "Bar", "Line", "Scatter", "Box Plot", "Histogram", "Table"],
                key="tab_mark_type",
            )
            color_choice = st.selectbox(
                "Color / Group",
                list(COLOR_OPTIONS.keys()),
                key="tab_color_pill",
            )
            color_field = COLOR_OPTIONS[color_choice]

    # -----------------------------------------------------------------------
    # Filters Shelf (Fix 5: Independent tab filters, States & Institutions multiselect)
    # -----------------------------------------------------------------------
    with st.expander("🔍 Filters Shelf & Sorting Controls", expanded=False):
        st.caption("This tab uses its own filters (the sidebar filters don't apply here).")
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            f_cat = st.selectbox("Filter Category", cat_options, key="tab_filter_cat")
        with fc2:
            f_year_span = st.slider("Year Range", min_value=min_y, max_value=max_y, key="tab_filter_year")
        with fc3:
            f_top_n = st.slider("Top N Limit", min_value=5, max_value=50, step=5, key="tab_top_n")
        with fc4:
            f_sort = st.selectbox(
                "Sort Order",
                ["Descending (Highest First)", "Ascending (Lowest First)", "Alphabetical", "None"],
                key="tab_sort_order",
            )

        # Cascading options for States and Institutions based on selected category & years
        filter_slice = df_full.copy()
        if f_cat != "All Categories":
            filter_slice = filter_slice[filter_slice["category"] == f_cat]
        filter_slice = filter_slice[(filter_slice["year"] >= f_year_span[0]) & (filter_slice["year"] <= f_year_span[1])]

        available_states = sorted([str(s) for s in filter_slice["state"].dropna().unique()])
        available_insts = sorted([str(n) for n in filter_slice["name"].dropna().unique()])

        # Clean session state to only contain valid options
        st.session_state["tab_filter_states"] = [s for s in st.session_state.get("tab_filter_states", []) if s in available_states]
        st.session_state["tab_filter_insts"] = [i for i in st.session_state.get("tab_filter_insts", []) if i in available_insts]

        fc5, fc6 = st.columns(2)
        with fc5:
            f_states = st.multiselect("Filter States (Optional)", options=available_states, key="tab_filter_states")
        with fc6:
            f_insts = st.multiselect("Filter Institutions (Optional)", options=available_insts, key="tab_filter_insts")

    # -----------------------------------------------------------------------
    # Apply Filters to Dataset (Fix 5: Tab uses df_full + own filters)
    # -----------------------------------------------------------------------
    work_df = df_full.copy()
    if f_cat != "All Categories":
        work_df = work_df[work_df["category"] == f_cat]
    work_df = work_df[(work_df["year"] >= f_year_span[0]) & (work_df["year"] <= f_year_span[1])]
    if f_states:
        work_df = work_df[work_df["state"].isin(f_states)]
    if f_insts:
        work_df = work_df[work_df["name"].isin(f_insts)]

    # Fix 11: Filters returning no rows must warn cleanly, no traceback
    if work_df.empty:
        st.warning("No records match the selected filters. Try widening the year range, category, state, or institution filters.")
        return

    # -----------------------------------------------------------------------
    # Show Me (Automatic Chart Type Inference) & Validation (Fix 3)
    # -----------------------------------------------------------------------
    resolved_mark = mark_choice
    if mark_choice == "Automatic (Show Me)":
        if not x_is_dim:
            resolved_mark = "Scatter"
        elif x_field == "year":
            resolved_mark = "Line"
        elif x_field in ["name", "state", "category", "rank_band", "city"]:
            resolved_mark = "Bar"
        else:
            resolved_mark = "Table"

    # Shelf validation check
    ok, val_msg = validate_shelves(x_is_dim, x_field, y_field, resolved_mark)
    if not ok:
        st.info(f"💡 {val_msg}")
        return

    # Title & Caption pooling label (Fix 2 & Fix 10)
    is_pooled = (f_cat == "All Categories")
    cat_label = " (all categories pooled)" if is_pooled else f" ({f_cat})"

    fig: Optional[go.Figure] = None
    transformed_df = pd.DataFrame()
    invert_y = (y_field == "rank")

    # -----------------------------------------------------------------------
    # Chart Renderers (Fix 4: Plotly Express with Color/Group on every mark)
    # -----------------------------------------------------------------------

    # CASE 1: SCATTER PLOT (Two Continuous Measures) — Fix 7
    if resolved_mark == "Scatter":
        # Pairwise dropna (never global drop)
        scat_df = work_df.dropna(subset=[x_field, y_field]).copy()

        # Dedicated Year Selector for Scatter (Fix 7)
        scatter_years = sorted(scat_df["year"].dropna().unique().astype(int).tolist())
        if not scatter_years:
            st.warning("No data points available for scatter plot with non-null values in both measures.")
            return

        latest_y = scatter_years[-1]
        scat_year_opts = [f"{latest_y} (Latest Year)"] + [str(y) for y in scatter_years[:-1][::-1]] + ["All Years (Pooled)"]
        scat_year_choice = st.selectbox("Scatter Year Selection", scat_year_opts, index=0, key="tab_scatter_year_sel")

        if "All Years" in scat_year_choice:
            is_scatter_pooled = True
        else:
            is_scatter_pooled = False
            sel_y = int(scat_year_choice.split()[0])
            scat_df = scat_df[scat_df["year"] == sel_y]

        if scat_df.empty:
            st.warning("No records found for the chosen scatter year.")
            return

        # Statistical Calculations on FULL filtered data before sampling
        corr_stats = calculate_pearson(scat_df, x_field, y_field)
        if corr_stats:
            r, r2, n = corr_stats["r"], corr_stats["r2"], corr_stats["n"]
            strength = (
                "Strong Positive" if r >= 0.7
                else "Moderate Positive" if r >= 0.4
                else "Weak Positive" if r > 0.1
                else "Negligible" if abs(r) <= 0.1
                else "Weak Negative" if r >= -0.4
                else "Moderate Negative" if r >= -0.7
                else "Strong Negative"
            )
            k1, k2, k3 = st.columns(3)
            with k1:
                st.metric("Pearson Correlation (r)", f"{r:+.3f}")
            with k2:
                st.metric("R² (Variance Explained)", f"{r2 * 100:.1f}%")
            with k3:
                st.metric("Relationship (n = " + str(n) + ")", strength)

        if is_scatter_pooled:
            st.caption("ℹ️ Data pooled across years: Institutions repeat across multiple years, so sample size n is larger and correlation r may be inflated.")
        elif is_pooled:
            st.caption("ℹ️ Rankings pooled across all NIRF categories.")

        # Color grouping handling (Fix 4: cap at 8 groups by measure)
        scat_color = color_field if (color_field and color_field in scat_df.columns) else None
        if scat_color:
            if scat_df[scat_color].nunique() > 8:
                top_c = scat_df.groupby(scat_color)[y_field].mean().sort_values(ascending=invert_y).head(8).index.tolist()
                scat_df = scat_df[scat_df[scat_color].isin(top_c)]
                st.caption(f"Color legend limited to top 8 {color_choice} groups by {row_display_name}.")

        # Plotting sample (max 500 points for responsiveness)
        plot_scat_df = scat_df.sample(500, random_state=42) if len(scat_df) > 500 else scat_df

        # Deduplicate hover columns (Fix 3)
        hover_cols = list(dict.fromkeys([c for c in ["state", "year", "category"] if c in plot_scat_df.columns]))

        fig = px.scatter(
            plot_scat_df,
            x=x_field,
            y=y_field,
            color=scat_color,
            hover_name="name" if "name" in plot_scat_df.columns else None,
            hover_data=hover_cols,
            color_discrete_sequence=QUALITATIVE_PALETTE if scat_color else [LINE_COLOR],
            labels={x_field: col_display_name, y_field: row_display_name},
        )

        # OLS Linear Trendline computed on FULL filtered dataset
        xv = pd.to_numeric(scat_df[x_field], errors="coerce").values
        yv = pd.to_numeric(scat_df[y_field], errors="coerce").values
        valid_m = np.isfinite(xv) & np.isfinite(yv)
        if np.sum(valid_m) >= 3:
            slope, intercept = np.polyfit(xv[valid_m], yv[valid_m], 1)
            x_trend = np.linspace(xv[valid_m].min(), xv[valid_m].max(), 100)
            trend_label = f"OLS Trendline (r = {corr_stats['r']:.2f})" if corr_stats else "OLS Trendline"
            fig.add_trace(
                go.Scatter(
                    x=x_trend,
                    y=slope * x_trend + intercept,
                    mode="lines",
                    name=trend_label,
                    line=dict(color="#f97316", width=2.5, dash="dash"),
                    hoverinfo="skip",
                )
            )

        title = f"{row_display_name} vs {col_display_name}{cat_label}"
        fig = apply_theme(
            fig,
            title=title,
            xaxis_title=col_display_name,
            yaxis_title=row_display_name,
            invert_yaxis=invert_y,
            show_legend=bool(scat_color),
        )
        transformed_df = scat_df

    # CASE 2: BAR CHART (Dimension + Aggregated Measure) — Fix 4 & Fix 6
    elif resolved_mark == "Bar":
        work_bar = work_df.dropna(subset=[x_field, y_field]).copy()
        if work_bar.empty:
            st.warning("No records with valid data for the bar chart.")
            return

        # Fix 6: Step 1 — Compute aggregation order on the X dimension BEFORE color split
        order = work_bar.groupby(x_field)[y_field].agg(agg_code)
        if f_sort == "Descending (Highest First)":
            # For rank, lower numeric value is best (rank 1 is best)
            order = order.sort_values(ascending=invert_y)
        elif f_sort == "Ascending (Lowest First)":
            order = order.sort_values(ascending=not invert_y)
        elif f_sort == "Alphabetical":
            order = order.sort_index(ascending=True)

        # Fix 6: Step 2 — Top N applies to X categories (except year and rank_band)
        if x_field in ["year", "rank_band"]:
            top_x_values = order.index.tolist()
            if x_field == "year":
                top_x_values = sorted(top_x_values)
        else:
            top_x_values = order.head(f_top_n).index.tolist()

        work_bar = work_bar[work_bar[x_field].isin(top_x_values)]

        # Fix 4: Color grouping (cap at 8 groups if needed)
        bar_color = color_field if (color_field and color_field != x_field and color_field in work_bar.columns) else None
        if bar_color:
            if work_bar[bar_color].nunique() > 8:
                top_colors = work_bar.groupby(bar_color)[y_field].mean().sort_values(ascending=invert_y).head(8).index.tolist()
                work_bar = work_bar[work_bar[bar_color].isin(top_colors)]
                st.caption(f"Color legend limited to top 8 {color_choice} groups by {row_display_name}.")
            group_cols = [x_field, bar_color]
        else:
            group_cols = [x_field]

        # Fix 6: Step 3 — Group by [x, color] and aggregate
        agg_df = work_bar.groupby(group_cols, as_index=False)[y_field].agg(agg_code)
        metric_label = f"{agg_display} of {row_display_name}"

        # Horizontal layout for readable labels when names, states, or cities are long
        use_horizontal = x_field in ["name", "state", "city"]

        if use_horizontal:
            fig = px.bar(
                agg_df,
                x=y_field,
                y=x_field,
                color=bar_color,
                barmode="group",
                orientation="h",
                category_orders={x_field: top_x_values[::-1]},  # Top ranked appears at the top
                color_discrete_sequence=QUALITATIVE_PALETTE if bar_color else [LINE_COLOR],
                labels={y_field: metric_label, x_field: col_display_name},
            )
            title = f"{agg_display} of {row_display_name} by {col_display_name}{cat_label}"
            fig = apply_theme(
                fig,
                title=title,
                xaxis_title=metric_label,
                yaxis_title=col_display_name,
                show_legend=bool(bar_color),
            )
            if invert_y:
                fig.update_xaxes(autorange="reversed")
        else:
            fig = px.bar(
                agg_df,
                x=x_field,
                y=y_field,
                color=bar_color,
                barmode="group",
                category_orders={x_field: top_x_values},
                color_discrete_sequence=QUALITATIVE_PALETTE if bar_color else [LINE_COLOR],
                labels={y_field: metric_label, x_field: col_display_name},
            )
            if x_field == "year":
                fig.update_xaxes(type="category")
            title = f"{agg_display} of {row_display_name} by {col_display_name}{cat_label}"
            fig = apply_theme(
                fig,
                title=title,
                xaxis_title=col_display_name,
                yaxis_title=metric_label,
                invert_yaxis=invert_y,
                show_legend=bool(bar_color),
            )

        if is_pooled:
            st.caption("ℹ️ Rankings pooled across all NIRF categories.")
        transformed_df = agg_df

    # CASE 3: LINE CHART (Time-series / Trends) — Fix 4 & Fix 8
    elif resolved_mark == "Line":
        work_line = work_df.dropna(subset=[x_field, y_field]).copy()
        if work_line.empty:
            st.warning("No records with valid data for the line chart.")
            return

        line_color = color_field if (color_field and color_field != x_field and color_field in work_line.columns) else None
        if line_color:
            # Fix 8: Choose groups by the measure (top by mean), not by value_counts()
            if work_line[line_color].nunique() > 8:
                top_groups = work_line.groupby(line_color)[y_field].mean().sort_values(ascending=invert_y).head(8).index.tolist()
                work_line = work_line[work_line[line_color].isin(top_groups)]
                st.caption(f"Color legend limited to top 8 {color_choice} groups by {row_display_name}.")
            group_cols = [x_field, line_color]
        else:
            group_cols = [x_field]

        agg_df = work_line.groupby(group_cols, as_index=False)[y_field].agg(agg_code)
        agg_df = agg_df.sort_values(by=x_field)
        metric_label = f"{agg_display} of {row_display_name}"

        fig = px.line(
            agg_df,
            x=x_field,
            y=y_field,
            color=line_color,
            symbol=line_color,
            markers=True,
            color_discrete_sequence=QUALITATIVE_PALETTE if line_color else [LINE_COLOR],
            labels={y_field: metric_label, x_field: col_display_name},
        )
        if x_field == "year":
            # Fix 8: Discrete year axis without decimals
            fig.update_xaxes(type="category")

        title = f"{agg_display} of {row_display_name} over {col_display_name}{cat_label}"
        fig = apply_theme(
            fig,
            title=title,
            xaxis_title=col_display_name,
            yaxis_title=metric_label,
            invert_yaxis=invert_y,
            show_legend=bool(line_color),
        )
        if is_pooled:
            st.caption("ℹ️ Rankings pooled across all NIRF categories.")
        transformed_df = agg_df

    # CASE 4: BOX & WHISKER PLOT — Fix 4 & Fix 8
    elif resolved_mark == "Box Plot":
        work_box = work_df.dropna(subset=[x_field, y_field]).copy()
        if work_box.empty:
            st.warning("No records with valid data for box plot.")
            return

        # Fix 8: User controls number of categories via Top N slider (capped at 15)
        box_cap = min(f_top_n, 15)
        box_order = work_box.groupby(x_field)[y_field].mean().sort_values(ascending=invert_y)
        top_cats = box_order.head(box_cap).index.tolist()
        work_box = work_box[work_box[x_field].isin(top_cats)]

        box_color = color_field if (color_field and color_field != x_field and color_field in work_box.columns) else None
        if box_color and work_box[box_color].nunique() > 8:
            top_colors = work_box.groupby(box_color)[y_field].mean().sort_values(ascending=invert_y).head(8).index.tolist()
            work_box = work_box[work_box[box_color].isin(top_colors)]
            st.caption(f"Color legend limited to top 8 {color_choice} groups by {row_display_name}.")

        fig = px.box(
            work_box,
            x=x_field,
            y=y_field,
            color=box_color,
            category_orders={x_field: top_cats},
            color_discrete_sequence=QUALITATIVE_PALETTE if box_color else [LINE_COLOR],
            labels={y_field: row_display_name, x_field: col_display_name},
        )
        if x_field == "year":
            fig.update_xaxes(type="category")

        title = f"{row_display_name} Distribution across {col_display_name}{cat_label}"
        fig = apply_theme(
            fig,
            title=title,
            xaxis_title=col_display_name,
            yaxis_title=row_display_name,
            invert_yaxis=invert_y,
            show_legend=bool(box_color),
        )
        if is_pooled:
            st.caption("ℹ️ Rankings pooled across all NIRF categories.")
        transformed_df = work_box

    # CASE 5: HISTOGRAM — Fix 4 & Fix 8
    elif resolved_mark == "Histogram":
        work_hist = work_df.dropna(subset=[y_field]).copy()
        if work_hist.empty:
            st.warning("No records with valid data for histogram.")
            return

        # Fix 8: Bins slider (default 20)
        n_bins = st.slider("Number of Histogram Bins", min_value=5, max_value=100, value=20, step=5, key="tab_hist_bins")

        hist_color = color_field if (color_field and color_field in work_hist.columns) else None
        if hist_color and work_hist[hist_color].nunique() > 8:
            top_colors = work_hist.groupby(hist_color)[y_field].mean().sort_values(ascending=invert_y).head(8).index.tolist()
            work_hist = work_hist[work_hist[hist_color].isin(top_colors)]
            st.caption(f"Color legend limited to top 8 {color_choice} groups by {row_display_name}.")

        fig = px.histogram(
            work_hist,
            x=y_field,
            color=hist_color,
            nbins=n_bins,
            opacity=0.85,
            color_discrete_sequence=QUALITATIVE_PALETTE if hist_color else [LINE_COLOR],
            labels={y_field: row_display_name},
        )
        title = f"Distribution of {row_display_name}{cat_label}"
        fig = apply_theme(
            fig,
            title=title,
            xaxis_title=row_display_name,
            yaxis_title="Count of Records",
            show_legend=bool(hist_color),
        )
        if is_pooled:
            st.caption("ℹ️ Rankings pooled across all NIRF categories.")
        # Fix 3: Deduplicate columns
        hist_cols = list(dict.fromkeys([c for c in [y_field, hist_color] if c]))
        transformed_df = work_hist[hist_cols]

    # CASE 6: TABLE (Crosstab) — Fix 8: Aggregated result, no early return
    else:
        title = f"Data Table: {row_display_name} by {col_display_name}{cat_label}"
        if x_is_dim:
            group_cols = [x_field]
            if color_field and color_field != x_field and color_field in work_df.columns:
                group_cols.append(color_field)
            agg_df = work_df.groupby(group_cols, as_index=False)[y_field].agg(agg_code)
            transformed_df = agg_df
        else:
            cols_to_show = list(dict.fromkeys([c for c in [x_field, y_field, color_field, "name", "state", "year"] if c and c in work_df.columns]))
            transformed_df = work_df[cols_to_show].head(200)

        st.dataframe(transformed_df, use_container_width=True)
        if is_pooled:
            st.caption("ℹ️ Rankings pooled across all NIRF categories.")

    # Render Plotly Chart (for non-Table marks)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # -----------------------------------------------------------------------
    # Spec JSON & CSV Export — Fix 10: Accurate query spec representation
    # -----------------------------------------------------------------------
    spec_export = {
        "title": title,
        "engine": "Tableau-Style Workbench",
        "shelves": {
            "columns": x_field,
            "rows": y_field,
            "aggregation": agg_code if x_is_dim else None,
            "mark": resolved_mark,
            "color": color_field,
        },
        "filters": {
            "category": f_cat,
            "year_range": f_year_span,
            "states": f_states,
            "institutions": f_insts,
            "top_n": f_top_n,
            "sort": f_sort,
        },
    }

    with st.expander("📝 View Underlying Query Spec JSON", expanded=False):
        st.json(spec_export)

    with st.expander("📥 Inspect Result Data & Export CSV", expanded=False):
        st.dataframe(transformed_df, use_container_width=True)
        csv_bytes = transformed_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv_bytes,
            file_name="tableau_explore_export.csv",
            mime="text/csv",
            key="tab_download_csv_btn",
        )
