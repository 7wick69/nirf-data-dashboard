"""
charts/explore.py

Tableau-Style Visual Analytics Workbench (Level 2 Explore Tab) — NIRF Interactive Dashboard.

Replicates Tableau's core visual architecture:
  - Shelves: Columns (X), Rows (Y), Marks Card (chart type & color encoding), Filters Shelf
  - Aggregation "None (raw rows)": plot each data row; auto-fall back to Mean when multiple
    rows share the same X value and show a friendly banner.
  - Automatic Level-of-Detail: aggregates measures across all shelf dimensions (X, Color)
  - Smart Marks inference: replicates Tableau's 'Show Me' logic
  - Statistical Analysis: Pearson r, R², p-value, OLS trendline
  - P1: rank cutoff filter, measure range filter, multiple Y-measures, facet shelf,
    Area mark, Heatmap mark, bubble size shelf
  - Fail-safe wrapper: every code path ends in a chart or a friendly message
  - Export: JSON Query Spec inspector, summary statistics, one-click CSV export
"""

import json
import os
import traceback as _tb
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
    "Teaching & Resources (TLR)": "TLR",
    "Research Practice (RPC)": "RPC",
    "Graduation Outcomes (GO)": "GO",
    "Outreach & Inclusivity (OI)": "OI",
    "Perception (PR)": "PR",
    "Total Score": "score",
    "NIRF Rank": "rank",
    "Rank Change YoY": "rank_change",
    "Score Change YoY": "score_change_yoy",
    "Parameter Disparity (Gap)": "param_gap",
    "Rank Volatility (σ)": "volatility",
    # Count pseudo-measures (sentinel values handled specially in chart code)
    "Number of Institutions": "__n_inst__",
    "Number of Records": "__n_rows__",
}

# "None (raw rows)" is FIRST so it's the default.
AGG_FUNCS: Dict[str, Optional[str]] = {
    "None (raw rows)": None,
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

# ---------------------------------------------------------------------------
# Analytical Presets
# ---------------------------------------------------------------------------
PRESETS: Dict[str, Dict[str, Any]] = {
    "1. Top 15 Institutions by Graduation Outcomes (GO) in Engineering": {
        "tab_col_type": "Dimension",
        "tab_col_dim": "Institution Name",
        "tab_col_meas": "Teaching & Resources (TLR)",
        "tab_row_meas": "Graduation Outcomes (GO)",
        "tab_agg_func": "Average (Mean)",
        "tab_mark_type": "Bar",
        "tab_color_pill": "None",
        "tab_filter_cat": "Engineering",
        "tab_filter_year": (2017, 2025),
        "tab_top_n": 15,
        "tab_sort_order": "Descending (Highest First)",
        "tab_filter_states": [],
        "tab_filter_insts": [],
        "tab_rank_cutoff": 0,
        "tab_meas_range_field": "None",
        "tab_facet_col": "None",
        "tab_size_meas": "None",
    },
    "2. Teaching (TLR) vs Perception (PR) Correlation across Universities": {
        "tab_col_type": "Measure",
        "tab_col_meas": "Teaching & Resources (TLR)",
        "tab_col_dim": "State",
        "tab_row_meas": "Perception (PR)",
        "tab_mark_type": "Scatter",
        "tab_agg_func": "None (raw rows)",
        "tab_color_pill": "None",
        "tab_filter_cat": "University",
        "tab_filter_year": (2017, 2025),
        "tab_top_n": 15,
        "tab_sort_order": "Descending (Highest First)",
        "tab_filter_states": [],
        "tab_filter_insts": [],
        "tab_rank_cutoff": 0,
        "tab_meas_range_field": "None",
        "tab_facet_col": "None",
        "tab_size_meas": "None",
    },
    "3. Average Research Productivity (RPC) by State (2024)": {
        "tab_col_type": "Dimension",
        "tab_col_dim": "State",
        "tab_col_meas": "Teaching & Resources (TLR)",
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
        "tab_rank_cutoff": 0,
        "tab_meas_range_field": "None",
        "tab_facet_col": "None",
        "tab_size_meas": "None",
    },
    "4. Score Distribution across Disciplines (Box Plot)": {
        "tab_col_type": "Dimension",
        "tab_col_dim": "Category",
        "tab_col_meas": "Teaching & Resources (TLR)",
        "tab_row_meas": "Total Score",
        "tab_mark_type": "Box Plot",
        "tab_agg_func": "None (raw rows)",
        "tab_color_pill": "None",
        "tab_filter_cat": "All Categories",
        "tab_filter_year": (2017, 2025),
        "tab_top_n": 15,
        "tab_sort_order": "Descending (Highest First)",
        "tab_filter_states": [],
        "tab_filter_insts": [],
        "tab_rank_cutoff": 0,
        "tab_meas_range_field": "None",
        "tab_facet_col": "None",
        "tab_size_meas": "None",
    },
    "5. Overall Score Trajectory over Time (2017–2025)": {
        "tab_col_type": "Dimension",
        "tab_col_dim": "Ranking Year",
        "tab_col_meas": "Teaching & Resources (TLR)",
        "tab_row_meas": "Total Score",
        "tab_agg_func": "Average (Mean)",
        "tab_mark_type": "Line",
        "tab_color_pill": "None",
        "tab_filter_cat": "Overall",
        "tab_filter_year": (2017, 2025),
        "tab_top_n": 15,
        "tab_sort_order": "Descending (Highest First)",
        "tab_filter_states": [],
        "tab_filter_insts": [],
        "tab_rank_cutoff": 0,
        "tab_meas_range_field": "None",
        "tab_facet_col": "None",
        "tab_size_meas": "None",
    },
}

# Rank-band ordering for category_orders
RANK_BAND_ORDER = ["1-10", "11-50", "51-100", "101+"]


# ---------------------------------------------------------------------------
# compute_default_state: returns a fresh DEFAULT_STATE dict (data-driven)
# ---------------------------------------------------------------------------
def compute_default_state(min_y: int, max_y: int, default_cat: str) -> Dict[str, Any]:
    """
    Returns the canonical default state dict for all tab_* widget keys.
    Call this once per render (after data is loaded) to get the runtime defaults.
    """
    return {
        "tab_preset_selector": "None (Custom Query)",
        "tab_col_type": "Measure",
        "tab_col_dim": "State",
        "tab_col_meas": "Teaching & Resources (TLR)",
        "tab_row_meas": ["Research Practice (RPC)"],
        "tab_agg_func": "None (raw rows)",
        "tab_mark_type": "Automatic (Show Me)",
        "tab_color_pill": "None",
        "tab_filter_cat": default_cat,
        "tab_filter_year": (min_y, max_y),
        "tab_filter_states": [],
        "tab_filter_insts": [],
        "tab_top_n": 15,
        "tab_sort_order": "Descending (Highest First)",
        "tab_scatter_year_sel": f"{max_y} (Latest Year)",
        "tab_hist_bins": 20,
        "tab_rank_cutoff": 0,
        "tab_meas_range_field": "None",
        "tab_facet_col": "None",
        "tab_heatmap_row_dim": "State",
        "tab_size_meas": "None",
    }


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def calculate_pearson(df: pd.DataFrame, x_col: str, y_col: str) -> Optional[Dict[str, Any]]:
    """Calculates Pearson r, R², p-value between two continuous measures using pairwise dropna."""
    if x_col not in df.columns or y_col not in df.columns or x_col == y_col:
        return None
    valid = df[[x_col, y_col]].dropna()
    if len(valid) < 3:
        return None
    try:
        x_vals = pd.to_numeric(valid[x_col], errors="coerce")
        y_vals = pd.to_numeric(valid[y_col], errors="coerce")
        common_idx = x_vals.dropna().index.intersection(y_vals.dropna().index)
        if len(common_idx) < 3:
            return None
        r, p = stats.pearsonr(x_vals.loc[common_idx], y_vals.loc[common_idx])
        return {"r": float(r), "r2": float(r ** 2), "p": float(p), "n": int(len(common_idx))}
    except Exception:
        return None


# Backward-compatible alias
compute_pearson_r = calculate_pearson


def validate_shelves(
    x_is_dim: bool,
    x_field: str,
    y_field: str,
    mark: str,
    y_fields: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """
    Validates whether the selected shelves (Columns, Rows) are compatible with the Mark type.
    Returns (is_valid, user_friendly_message).
    """
    _COUNT_MEASURES = ("__n_inst__", "__n_rows__")

    # Count measures are invalid for Scatter/Box/Histogram
    eff_y = y_fields[0] if y_fields else y_field
    if eff_y in _COUNT_MEASURES and mark in ("Scatter", "Box Plot", "Histogram"):
        return False, "Count measures (Number of Institutions / Records) cannot be used with Scatter, Box Plot, or Histogram. Choose Bar or Table."

    if mark == "Scatter":
        if x_is_dim:
            return False, "Scatter plots require continuous measures on both X (Columns) and Y (Rows). Please switch Columns to a Measure."
        if x_field == y_field:
            return False, f"Scatter plot requires two different measures. Columns and Rows are both '{x_field}'. Please select different measures."
        return True, ""

    if mark in ("Bar", "Box Plot"):
        if not x_is_dim:
            return False, f"{mark} charts require a Dimension on X (Columns) and a Measure on Y (Rows). Please switch Columns to a Dimension, or switch Mark to Scatter."
        return True, ""

    if mark in ("Line", "Area"):
        if not x_is_dim:
            return False, f"{mark} charts require a Dimension on X (Columns). Please switch Columns to a Dimension."
        if x_field not in ("year", "rank_band"):
            return False, f"{mark} charts need an ordered X axis. '{x_field}' is not naturally ordered. Switch to Bar, or change X to 'Ranking Year' or 'Rank Band'."
        return True, ""

    if mark == "Histogram":
        return True, ""

    if mark in ("Table", "Heatmap"):
        return True, ""

    return True, ""


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_explore_tab(df_full: pd.DataFrame, df_filtered: pd.DataFrame):
    """
    Renders the Tableau-style visual analytics workbench.
    Follows Tableau's Level of Detail (LOD) aggregation model.
    Signature kept stable: render_explore_tab(df_full, df_filtered).
    df_full is used so the tab has its own independent filters.
    """
    st.markdown("### 🔭 Explore: Tableau-Style Visual Query Builder")
    st.caption("Build any chart: pick your X (Columns), Y (Rows), mark type, and filters — just like Tableau.")

    # -----------------------------------------------------------------------
    # Guard: missing columns
    # -----------------------------------------------------------------------
    avail_cols = set(df_full.columns)
    dimensions = {lbl: col for lbl, col in ALL_DIMENSIONS.items() if col in avail_cols}
    measures = {lbl: col for lbl, col in ALL_MEASURES.items()
                if col in avail_cols or col in ("__n_inst__", "__n_rows__")}

    dropped_dims = [lbl for lbl, col in ALL_DIMENSIONS.items() if col not in avail_cols]
    dropped_meas = [lbl for lbl, col in ALL_MEASURES.items()
                    if col not in avail_cols and col not in ("__n_inst__", "__n_rows__")]
    if dropped_dims or dropped_meas:
        st.warning(f"Note: Some expected fields are not in the dataset: {dropped_dims + dropped_meas}")

    if not dimensions or not measures:
        st.error("Dataset lacks sufficient dimension or measure fields to render the visual workbench.")
        return

    # -----------------------------------------------------------------------
    # Category & year defaults (data-driven)
    # -----------------------------------------------------------------------
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
    # Preset callback: resets ALL keys to defaults then applies preset
    # -----------------------------------------------------------------------
    def apply_preset():
        """on_change callback for the preset selectbox."""
        choice = st.session_state.get("tab_preset_selector", "None (Custom Query)")
        # Always reset to defaults first
        defaults = compute_default_state(min_y, max_y, default_cat)
        st.session_state.update(defaults)
        p = PRESETS.get(choice)
        if p:
            for k, v in p.items():
                # Guard: skip invalid category options
                if k == "tab_filter_cat" and v not in cat_options:
                    continue
                # Guard: clamp year range to data range
                if k == "tab_filter_year" and isinstance(v, tuple):
                    v = (max(min_y, v[0]), min(max_y, v[1]))
                st.session_state[k] = v

    # -----------------------------------------------------------------------
    # Session state initialisation (setdefault from compute_default_state)
    # -----------------------------------------------------------------------
    _defaults = compute_default_state(min_y, max_y, default_cat)
    for _k, _v in _defaults.items():
        st.session_state.setdefault(_k, _v)

    # Safety: fix stale/invalid values
    _dim_keys = list(dimensions.keys())
    _meas_keys = list(measures.keys())

    if st.session_state.get("tab_col_dim") not in dimensions:
        st.session_state["tab_col_dim"] = _dim_keys[0]
    if st.session_state.get("tab_col_meas") not in measures:
        st.session_state["tab_col_meas"] = _meas_keys[0]

    # tab_row_meas must be a list
    _row_val = st.session_state.get("tab_row_meas")
    if isinstance(_row_val, str):
        st.session_state["tab_row_meas"] = [_row_val] if _row_val in measures else [_meas_keys[0]]
    elif not isinstance(_row_val, list) or not _row_val:
        st.session_state["tab_row_meas"] = [_meas_keys[0]]
    else:
        st.session_state["tab_row_meas"] = [m for m in _row_val if m in measures] or [_meas_keys[0]]

    if st.session_state.get("tab_agg_func") not in AGG_FUNCS:
        st.session_state["tab_agg_func"] = "None (raw rows)"
    if st.session_state.get("tab_filter_cat") not in cat_options:
        st.session_state["tab_filter_cat"] = default_cat

    # -----------------------------------------------------------------------
    # Preset selector widget
    # -----------------------------------------------------------------------
    preset_options = ["None (Custom Query)"] + list(PRESETS.keys())
    c_preset1, _ = st.columns([2.5, 1])
    with c_preset1:
        st.selectbox(
            "⚡ Quick-Load Analytical Preset",
            options=preset_options,
            key="tab_preset_selector",
            on_change=apply_preset,
        )

    # -----------------------------------------------------------------------
    # Shelves Container (Columns, Rows, Marks)
    # -----------------------------------------------------------------------
    with st.container(border=True):
        col_c, col_r, col_m = st.columns([1.2, 1.2, 1.0], gap="medium")

        # 1. COLUMNS SHELF (X-Axis)
        with col_c:
            st.markdown("**Columns (X Shelf)**")
            col_shelf_type = st.radio(
                "Field Type", ["Dimension", "Measure"],
                horizontal=True, key="tab_col_type",
            )
            if col_shelf_type == "Dimension":
                col_display_name = st.selectbox("Dimension", _dim_keys, key="tab_col_dim")
                x_field = dimensions[col_display_name]
                x_is_dim = True
            else:
                col_display_name = st.selectbox("Measure", _meas_keys, key="tab_col_meas")
                x_field = measures[col_display_name]
                x_is_dim = False

        # 2. ROWS SHELF (Y-Axis) — multiselect for multiple measures
        with col_r:
            st.markdown("**Rows (Y Shelf)**")
            row_display_names = st.multiselect(
                "Measure(s) — up to 5",
                options=_meas_keys,
                key="tab_row_meas",
                max_selections=5,
            )
            if not row_display_names:
                row_display_names = [_meas_keys[0]]

            # Primary Y for single-measure charts
            row_display_name = row_display_names[0]
            y_field = measures[row_display_name]
            y_fields = [measures[n] for n in row_display_names]

            # Determine if mark type disables aggregation
            _current_mark = st.session_state.get("tab_mark_type", "Automatic (Show Me)")
            _agg_disabled = _current_mark in ("Scatter", "Box Plot", "Histogram")

            agg_display = st.selectbox(
                "Aggregation",
                list(AGG_FUNCS.keys()),
                key="tab_agg_func",
                disabled=_agg_disabled,
            )
            if _agg_disabled:
                st.caption("Aggregation isn't used by this mark type")
            agg_code = AGG_FUNCS[agg_display]  # may be None

        # 3. MARKS CARD
        with col_m:
            st.markdown("**Marks Card (Show Me)**")
            mark_choice = st.selectbox(
                "Mark Type",
                ["Automatic (Show Me)", "Bar", "Line", "Area", "Scatter",
                 "Box Plot", "Histogram", "Heatmap", "Table"],
                key="tab_mark_type",
            )
            color_choice = st.selectbox(
                "Color / Group",
                list(COLOR_OPTIONS.keys()),
                key="tab_color_pill",
            )
            color_field = COLOR_OPTIONS[color_choice]

            # Facet shelf
            _facet_dim_opts = ["None"] + _dim_keys
            facet_choice = st.selectbox(
                "Facet (small multiples)",
                _facet_dim_opts,
                key="tab_facet_col",
            )
            facet_field = dimensions.get(facet_choice) if facet_choice != "None" else None

            # Size shelf (Scatter only)
            if mark_choice in ("Scatter", "Automatic (Show Me)"):
                size_meas_choice = st.selectbox(
                    "Bubble Size (optional)",
                    ["None"] + _meas_keys,
                    key="tab_size_meas",
                )
                size_field = measures.get(size_meas_choice) if size_meas_choice != "None" else None
            else:
                size_field = None

            # Heatmap Y-dimension
            if mark_choice == "Heatmap":
                heatmap_row_dim_name = st.selectbox(
                    "Heatmap Y-Dimension",
                    _dim_keys,
                    key="tab_heatmap_row_dim",
                )
                heatmap_y_field = dimensions[heatmap_row_dim_name]
            else:
                heatmap_y_field = None

    # -----------------------------------------------------------------------
    # Filters Shelf
    # -----------------------------------------------------------------------
    with st.expander("🔍 Filters Shelf & Sorting Controls", expanded=False):
        st.caption("This tab uses its own filters — the sidebar filters don't apply here.")
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            f_cat = st.selectbox("Filter Category", cat_options, key="tab_filter_cat")
        with fc2:
            f_year_span = st.slider(
                "Year Range", min_value=min_y, max_value=max_y,
                key="tab_filter_year",
            )
        with fc3:
            f_top_n = st.slider("Top N Limit", min_value=1, max_value=50, step=1, key="tab_top_n")
        with fc4:
            f_sort = st.selectbox(
                "Sort Order",
                ["Descending (Highest First)", "Ascending (Lowest First)", "Alphabetical", "None"],
                key="tab_sort_order",
            )

        # Rank cutoff & measure range
        fc_r1, fc_r2 = st.columns(2)
        with fc_r1:
            rank_cutoff = st.number_input(
                "Only institutions ranked ≤ N (0 = off)",
                min_value=0, max_value=1000, step=10,
                key="tab_rank_cutoff",
            )
        with fc_r2:
            meas_range_field_name = st.selectbox(
                "Measure Range Filter",
                ["None"] + _meas_keys,
                key="tab_meas_range_field",
            )

        meas_range_col = measures.get(meas_range_field_name)
        if meas_range_field_name != "None" and meas_range_col and meas_range_col in df_full.columns:
            _mrc = df_full[meas_range_col].dropna()
            _mrmin = float(_mrc.min()) if not _mrc.empty else 0.0
            _mrmax = float(_mrc.max()) if not _mrc.empty else 100.0
            if _mrmin < _mrmax:
                meas_range = st.slider(
                    f"{meas_range_field_name} Range",
                    min_value=_mrmin, max_value=_mrmax,
                    value=(_mrmin, _mrmax),
                    key="tab_meas_range",
                )
            else:
                meas_range = (_mrmin, _mrmax)
        else:
            meas_range = None

        # Cascading states / institutions
        filter_slice = df_full.copy()
        if f_cat != "All Categories":
            filter_slice = filter_slice[filter_slice["category"] == f_cat]
        filter_slice = filter_slice[
            (filter_slice["year"] >= f_year_span[0]) & (filter_slice["year"] <= f_year_span[1])
        ]
        available_states = sorted([str(s) for s in filter_slice["state"].dropna().unique()])
        available_insts = sorted([str(n) for n in filter_slice["name"].dropna().unique()])

        # Sanitize session state to only valid options
        st.session_state["tab_filter_states"] = [
            s for s in st.session_state.get("tab_filter_states", []) if s in available_states
        ]
        st.session_state["tab_filter_insts"] = [
            i for i in st.session_state.get("tab_filter_insts", []) if i in available_insts
        ]

        fc5, fc6 = st.columns(2)
        with fc5:
            f_states = st.multiselect(
                "Filter States (Optional)", options=available_states, key="tab_filter_states"
            )
        with fc6:
            f_insts = st.multiselect(
                "Filter Institutions (Optional)", options=available_insts, key="tab_filter_insts"
            )

    # -----------------------------------------------------------------------
    # Apply filters to dataset
    # -----------------------------------------------------------------------
    work_df = df_full.copy()
    if f_cat != "All Categories":
        work_df = work_df[work_df["category"] == f_cat]
    work_df = work_df[
        (work_df["year"] >= f_year_span[0]) & (work_df["year"] <= f_year_span[1])
    ]
    if f_states:
        work_df = work_df[work_df["state"].isin(f_states)]
    if f_insts:
        work_df = work_df[work_df["name"].isin(f_insts)]

    # Apply rank cutoff
    if rank_cutoff > 0 and "rank" in work_df.columns:
        work_df = work_df[pd.to_numeric(work_df["rank"], errors="coerce") <= rank_cutoff]

    # Apply measure range
    if meas_range is not None and meas_range_col and meas_range_col in work_df.columns:
        work_df = work_df[
            (work_df[meas_range_col] >= meas_range[0]) &
            (work_df[meas_range_col] <= meas_range[1])
        ]

    # Add count pseudo-measure columns
    work_df = work_df.copy()
    work_df["__n_rows__"] = 1
    if "institution_id" in work_df.columns:
        work_df["__n_inst__"] = work_df["institution_id"]
    else:
        work_df["__n_inst__"] = work_df["name"]

    if work_df.empty:
        st.warning("No records match the selected filters. Try widening the year range, category, state, or institution filters.")
        return

    # -----------------------------------------------------------------------
    # Show Me inference
    # -----------------------------------------------------------------------
    resolved_mark = mark_choice
    if mark_choice == "Automatic (Show Me)":
        if not x_is_dim:
            resolved_mark = "Scatter"
        elif x_field == "year":
            resolved_mark = "Line"
        elif x_field in ("name", "state", "category", "rank_band", "city"):
            resolved_mark = "Bar"
        else:
            resolved_mark = "Table"

    # -----------------------------------------------------------------------
    # Shelf validation
    # -----------------------------------------------------------------------
    ok, val_msg = validate_shelves(x_is_dim, x_field, y_field, resolved_mark, y_fields)
    if not ok:
        st.info(f"💡 {val_msg}")
        return

    # -----------------------------------------------------------------------
    # Conflict hints
    # -----------------------------------------------------------------------
    if (x_field == "category" or color_field == "category") and f_cat != "All Categories":
        st.info("💡 'Category' is on a shelf but the Category filter is set to a single value. Consider setting it to 'All Categories' for a more informative view.")
    if x_field == "year" and f_year_span[0] == f_year_span[1]:
        st.info("💡 'Ranking Year' is on the X axis but the year filter spans a single year. Widen the year range to see trends.")
    if (x_field == "state" or color_field == "state") and len(f_states) == 1:
        st.info("💡 'State' is on a shelf but only one state is selected. Consider removing the state filter to compare across states.")

    # -----------------------------------------------------------------------
    # Title & pooling label
    # -----------------------------------------------------------------------
    is_pooled = (f_cat == "All Categories")
    cat_label = " (all categories pooled)" if is_pooled else f" ({f_cat})"

    # Initialise output variables (must be set before try block exits)
    fig: Optional[go.Figure] = None
    transformed_df = pd.DataFrame()
    invert_y = (y_field == "rank")
    title = f"{row_display_name} by {col_display_name}{cat_label}"
    agg_effective = agg_code  # may be overridden inside each chart case

    # Helper: determine effective aggregation for Bar/Line/Table with agg=None
    def _resolve_agg_none(work: pd.DataFrame, group_cols: List[str]) -> str:
        """
        When user chose 'None (raw rows)' for Bar/Line/Table:
        check if any group has >1 row. If yes, auto-fall back to Mean.
        Returns the effective agg string ('mean' or the original None placeholder sentinel 'none').
        Shows a banner if falling back.
        """
        if not group_cols:
            return "none"
        try:
            rpg = work.groupby(group_cols, observed=True).size().max()
        except Exception:
            rpg = 2
        if rpg > 1:
            n = int(rpg)
            st.info(
                f"ℹ️ Aggregation is 'None (raw rows)' but up to **{n} rows** share the same X "
                f"value, so values are averaged. Add 'Institution Name' or 'Ranking Year' to "
                f"Color, narrow the filters, or choose an explicit aggregation."
            )
            return "mean"
        return "none"

    # Helper: apply aggregation to a grouped dataframe
    def _aggregate(df: pd.DataFrame, group_cols: List[str], val_cols: List[str], agg: Optional[str]) -> pd.DataFrame:
        """Groups df by group_cols and aggregates val_cols."""
        # Count pseudo-measures need special handling
        result_parts = []
        for vc in val_cols:
            if vc == "__n_inst__":
                part = df.groupby(group_cols, observed=True)["__n_inst__"].nunique().reset_index()
                part.rename(columns={"__n_inst__": vc}, inplace=True)
            elif vc == "__n_rows__":
                part = df.groupby(group_cols, observed=True)["__n_rows__"].count().reset_index()
                part.rename(columns={"__n_rows__": vc}, inplace=True)
            else:
                part = df.groupby(group_cols, observed=True)[vc].agg(agg).reset_index()
            result_parts.append(part)
        if len(result_parts) == 1:
            return result_parts[0]
        # Merge all parts on group_cols
        merged = result_parts[0]
        for part in result_parts[1:]:
            merged = merged.merge(part, on=group_cols, how="outer")
        return merged

    def _safe_agg_series(df_in: pd.DataFrame, g_col: str, v_col: str, agg_name: Optional[str] = "mean") -> pd.Series:
        """Computes aggregation of v_col grouped by g_col safely (handling count measures and non-numeric types)."""
        if v_col == "__n_inst__":
            return df_in.groupby(g_col, observed=True)[v_col].nunique()
        elif v_col == "__n_rows__":
            return df_in.groupby(g_col, observed=True)[v_col].count()
        fn = agg_name if agg_name and agg_name != "none" else "mean"
        num_s = pd.to_numeric(df_in[v_col], errors="coerce")
        return num_s.groupby(df_in[g_col], observed=True).agg(fn)

    # -----------------------------------------------------------------------
    # Count measure caption
    # -----------------------------------------------------------------------
    if y_field == "__n_inst__":
        st.caption("📊 Count measure: aggregation fixed to Count Distinct Institutions")
    elif y_field == "__n_rows__":
        st.caption("📊 Count measure: aggregation fixed to Count of Records")

    # -----------------------------------------------------------------------
    # Rank-averaging warning
    # -----------------------------------------------------------------------
    if y_field == "rank" and agg_code in ("mean", "median", "sum"):
        st.caption("⚠️ Averaging ranks pools different categories/years — interpret with care.")

    # -----------------------------------------------------------------------
    # Multi-measure: melt helper
    # -----------------------------------------------------------------------
    _multi_measure = len(y_fields) > 1

    # -----------------------------------------------------------------------
    # CHART BUILDING — wrapped in fail-safe try/except
    # -----------------------------------------------------------------------
    try:
        # -------------------------------------------------------------------
        # CASE 1: SCATTER PLOT
        # -------------------------------------------------------------------
        if resolved_mark == "Scatter":
            scat_df = work_df.dropna(subset=[x_field, y_field]).copy()

            scatter_years = sorted(scat_df["year"].dropna().unique().astype(int).tolist())
            if not scatter_years:
                st.warning("No data points with non-null values in both measures.")
                return

            latest_y = scatter_years[-1]
            scat_year_opts = (
                [f"{latest_y} (Latest Year)"]
                + [str(y) for y in scatter_years[:-1][::-1]]
                + ["All Years (Pooled)"]
            )
            # Guard stale session state for year selector
            _ssy = st.session_state.get("tab_scatter_year_sel", f"{latest_y} (Latest Year)")
            if _ssy not in scat_year_opts:
                st.session_state["tab_scatter_year_sel"] = f"{latest_y} (Latest Year)"

            scat_year_choice = st.selectbox(
                "Scatter Year Selection", scat_year_opts, key="tab_scatter_year_sel"
            )

            is_scatter_pooled = "All Years" in scat_year_choice
            if not is_scatter_pooled:
                sel_y = int(scat_year_choice.split()[0])
                scat_df = scat_df[scat_df["year"] == sel_y]

            if scat_df.empty:
                st.warning("No records found for the chosen scatter year.")
                return

            # Pearson / stats on ALL filtered rows (before sampling)
            corr_stats = calculate_pearson(scat_df, x_field, y_field)
            if corr_stats:
                r, r2, n_pts = corr_stats["r"], corr_stats["r2"], corr_stats["n"]
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
                k1.metric("Pearson r", f"{r:+.3f}")
                k2.metric("R² (Variance Explained)", f"{r2 * 100:.1f}%")
                k3.metric(f"Relationship (n={n_pts})", strength)

            if is_scatter_pooled:
                st.caption("ℹ️ Data pooled across years — Pearson r may be inflated.")
            elif is_pooled:
                st.caption("ℹ️ Rankings pooled across all NIRF categories.")

            # Color: relabel >8 groups as "Other" (don't drop)
            scat_color = color_field if (color_field and color_field in scat_df.columns) else None
            if scat_color and scat_df[scat_color].nunique() > 8:
                top_c = scat_df[scat_color].value_counts().head(8).index.tolist()
                scat_df = scat_df.copy()
                scat_df[scat_color] = scat_df[scat_color].where(
                    scat_df[scat_color].isin(top_c), other="Other"
                )
                st.caption("Color groups beyond the top 8 have been relabeled as 'Other'.")

            # Sample up to 2000 for drawing
            plot_scat_df = scat_df.sample(2000, random_state=42) if len(scat_df) > 2000 else scat_df

            hover_cols = list(dict.fromkeys(
                [c for c in ["state", "year", "category"] if c in plot_scat_df.columns]
            ))

            # Size field (bubble chart)
            _size_col = size_field if (size_field and size_field in plot_scat_df.columns) else None

            fig = px.scatter(
                plot_scat_df,
                x=x_field,
                y=y_field,
                color=scat_color,
                size=_size_col,
                hover_name="name" if "name" in plot_scat_df.columns else None,
                hover_data=hover_cols,
                facet_col=facet_field if facet_field and facet_field in plot_scat_df.columns else None,
                facet_col_wrap=3,
                color_discrete_sequence=QUALITATIVE_PALETTE if scat_color else [LINE_COLOR],
                labels={x_field: col_display_name, y_field: row_display_name},
            )

            # OLS trendline (computed on full data before sampling)
            xv = pd.to_numeric(scat_df[x_field], errors="coerce").values
            yv = pd.to_numeric(scat_df[y_field], errors="coerce").values
            valid_m = np.isfinite(xv) & np.isfinite(yv)
            if np.sum(valid_m) >= 3:
                slope, intercept = np.polyfit(xv[valid_m], yv[valid_m], 1)
                x_trend = np.linspace(xv[valid_m].min(), xv[valid_m].max(), 100)
                trend_label = f"OLS Trendline (r={corr_stats['r']:.2f})" if corr_stats else "OLS Trendline"
                fig.add_trace(go.Scatter(
                    x=x_trend, y=slope * x_trend + intercept,
                    mode="lines", name=trend_label,
                    line=dict(color="#f97316", width=2.5, dash="dash"),
                    hoverinfo="skip",
                ))

            title = f"{row_display_name} vs {col_display_name}{cat_label}"
            fig = apply_theme(
                fig, title=title,
                xaxis_title=col_display_name, yaxis_title=row_display_name,
                invert_yaxis=invert_y, show_legend=bool(scat_color),
            )
            transformed_df = scat_df

        # -------------------------------------------------------------------
        # CASE 2: BAR CHART
        # -------------------------------------------------------------------
        elif resolved_mark == "Bar":
            work_bar = work_df.dropna(subset=[x_field, y_field]).copy()
            if work_bar.empty:
                st.warning("No records with valid data for the bar chart.")
                return

            # Color grouping
            bar_color = color_field if (
                color_field and color_field != x_field and color_field in work_bar.columns
            ) else None
            if bar_color and work_bar[bar_color].nunique() > 8:
                top_colors = (
                    _safe_agg_series(work_bar, bar_color, y_field, "mean")
                    .sort_values(ascending=invert_y).head(8).index.tolist()
                )
                work_bar = work_bar[work_bar[bar_color].isin(top_colors)]
                st.caption(f"Color legend limited to top 8 {color_choice} groups.")

            group_cols = [x_field] + ([bar_color] if bar_color else [])
            if facet_field and facet_field in work_bar.columns and facet_field not in group_cols:
                group_cols.append(facet_field)

            # Resolve aggregation
            if agg_code is None:
                agg_effective = _resolve_agg_none(work_bar, group_cols)
                if agg_effective == "none":
                    # True one-row-per-group: plot raw
                    agg_df = work_bar[list(dict.fromkeys(group_cols + y_fields))].copy()
                    metric_label = row_display_name
                else:
                    agg_df = _aggregate(work_bar, group_cols, y_fields, agg_effective)
                    metric_label = f"Average of {row_display_name}"
            else:
                agg_effective = agg_code
                # Count pseudo-measure
                if y_field in ("__n_inst__", "__n_rows__"):
                    agg_df = _aggregate(work_bar, group_cols, [y_field], None)
                    metric_label = row_display_name
                else:
                    agg_df = _aggregate(work_bar, group_cols, y_fields, agg_code)
                    metric_label = f"{agg_display} of {row_display_name}"

            # Sort order (on primary Y field)
            if x_field not in ("year", "rank_band"):
                _order_series = _safe_agg_series(agg_df, x_field, y_field, "mean") if y_field in agg_df.columns else agg_df.set_index(x_field).iloc[:, 0]
                if f_sort == "Descending (Highest First)":
                    order_idx = _order_series.sort_values(ascending=invert_y).index.tolist()
                elif f_sort == "Ascending (Lowest First)":
                    order_idx = _order_series.sort_values(ascending=not invert_y).index.tolist()
                elif f_sort == "Alphabetical":
                    order_idx = sorted(_order_series.index.tolist())
                else:
                    order_idx = _order_series.index.tolist()
                top_x_values = order_idx[:f_top_n]
            else:
                top_x_values = sorted(agg_df[x_field].unique().tolist())
                if x_field == "rank_band":
                    top_x_values = [v for v in RANK_BAND_ORDER if v in top_x_values]

            agg_df = agg_df[agg_df[x_field].isin(top_x_values)]

            # Multi-measure melt
            if _multi_measure and len(y_fields) > 1:
                _id_cols = list(dict.fromkeys([x_field] + ([bar_color] if bar_color else []) + ([facet_field] if facet_field else [])))
                _id_cols = [c for c in _id_cols if c in agg_df.columns]
                agg_df = pd.melt(agg_df, id_vars=_id_cols, value_vars=[c for c in y_fields if c in agg_df.columns],
                                 var_name="__measure__", value_name="__value__")
                _measure_label_map = {v: k for k, v in measures.items()}
                agg_df["__measure__"] = agg_df["__measure__"].map(_measure_label_map).fillna(agg_df["__measure__"])
                bar_color = "__measure__"
                y_field = "__value__"
                metric_label = "Value"

            use_horizontal = x_field in ("name", "state", "city")
            _facet_col_arg = facet_field if (facet_field and facet_field in agg_df.columns) else None

            if use_horizontal:
                fig = px.bar(
                    agg_df, x=y_field, y=x_field, color=bar_color,
                    barmode="group", orientation="h",
                    category_orders={x_field: top_x_values[::-1],
                                     "rank_band": RANK_BAND_ORDER},
                    facet_col=_facet_col_arg, facet_col_wrap=3,
                    color_discrete_sequence=QUALITATIVE_PALETTE if bar_color else [LINE_COLOR],
                    labels={y_field: metric_label, x_field: col_display_name},
                )
                title = f"{metric_label} by {col_display_name}{cat_label}"
                fig = apply_theme(fig, title=title, xaxis_title=metric_label,
                                  yaxis_title=col_display_name, show_legend=bool(bar_color))
                if invert_y and not _multi_measure:
                    fig.update_xaxes(autorange="reversed")
            else:
                fig = px.bar(
                    agg_df, x=x_field, y=y_field, color=bar_color,
                    barmode="group",
                    category_orders={x_field: top_x_values, "rank_band": RANK_BAND_ORDER},
                    facet_col=_facet_col_arg, facet_col_wrap=3,
                    color_discrete_sequence=QUALITATIVE_PALETTE if bar_color else [LINE_COLOR],
                    labels={y_field: metric_label, x_field: col_display_name},
                )
                if x_field == "year":
                    fig.update_xaxes(type="category")
                title = f"{metric_label} by {col_display_name}{cat_label}"
                fig = apply_theme(fig, title=title, xaxis_title=col_display_name,
                                  yaxis_title=metric_label,
                                  invert_yaxis=(invert_y and not _multi_measure),
                                  show_legend=bool(bar_color))

            if is_pooled:
                st.caption("ℹ️ Rankings pooled across all NIRF categories.")
            transformed_df = agg_df

        # -------------------------------------------------------------------
        # CASE 3: LINE CHART
        # -------------------------------------------------------------------
        elif resolved_mark == "Line":
            work_line = work_df.dropna(subset=[x_field, y_field]).copy()
            if work_line.empty:
                st.warning("No records with valid data for the line chart.")
                return

            line_color = color_field if (
                color_field and color_field != x_field and color_field in work_line.columns
            ) else None
            if line_color and work_line[line_color].nunique() > 8:
                top_groups = (
                    _safe_agg_series(work_line, line_color, y_field, "mean")
                    .sort_values(ascending=invert_y).head(8).index.tolist()
                )
                work_line = work_line[work_line[line_color].isin(top_groups)]
                st.caption(f"Color legend limited to top 8 {color_choice} groups.")

            group_cols = [x_field] + ([line_color] if line_color else [])
            if facet_field and facet_field in work_line.columns and facet_field not in group_cols:
                group_cols.append(facet_field)

            if agg_code is None:
                agg_effective = _resolve_agg_none(work_line, group_cols)
                if agg_effective == "none":
                    agg_df = work_line.copy()
                    metric_label = row_display_name
                else:
                    agg_df = _aggregate(work_line, group_cols, y_fields, agg_effective)
                    metric_label = f"Average of {row_display_name}"
            else:
                agg_effective = agg_code
                agg_df = _aggregate(work_line, group_cols, y_fields, agg_code)
                metric_label = f"{agg_display} of {row_display_name}"

            agg_df = agg_df.sort_values(by=x_field)

            # Multi-measure melt
            if _multi_measure and len(y_fields) > 1:
                _id_cols = list(dict.fromkeys([x_field] + ([line_color] if line_color else []) + ([facet_field] if facet_field else [])))
                _id_cols = [c for c in _id_cols if c in agg_df.columns]
                agg_df = pd.melt(agg_df, id_vars=_id_cols, value_vars=[c for c in y_fields if c in agg_df.columns],
                                 var_name="__measure__", value_name="__value__")
                _measure_label_map = {v: k for k, v in measures.items()}
                agg_df["__measure__"] = agg_df["__measure__"].map(_measure_label_map).fillna(agg_df["__measure__"])
                line_color = "__measure__"
                y_field = "__value__"
                metric_label = "Value"

            _facet_col_arg = facet_field if (facet_field and facet_field in agg_df.columns) else None

            fig = px.line(
                agg_df, x=x_field, y=y_field,
                color=line_color, symbol=line_color, markers=True,
                facet_col=_facet_col_arg, facet_col_wrap=3,
                color_discrete_sequence=QUALITATIVE_PALETTE if line_color else [LINE_COLOR],
                labels={y_field: metric_label, x_field: col_display_name},
                category_orders={"rank_band": RANK_BAND_ORDER},
            )
            if x_field == "year":
                fig.update_xaxes(type="category")

            title = f"{metric_label} over {col_display_name}{cat_label}"
            fig = apply_theme(fig, title=title, xaxis_title=col_display_name,
                              yaxis_title=metric_label,
                              invert_yaxis=(invert_y and not _multi_measure),
                              show_legend=bool(line_color))
            if is_pooled:
                st.caption("ℹ️ Rankings pooled across all NIRF categories.")
            transformed_df = agg_df

        # -------------------------------------------------------------------
        # CASE 4: AREA CHART
        # -------------------------------------------------------------------
        elif resolved_mark == "Area":
            work_area = work_df.dropna(subset=[x_field, y_field]).copy()
            if work_area.empty:
                st.warning("No records with valid data for the area chart.")
                return

            area_color = color_field if (
                color_field and color_field != x_field and color_field in work_area.columns
            ) else None
            if area_color and work_area[area_color].nunique() > 8:
                top_groups = (
                    _safe_agg_series(work_area, area_color, y_field, "mean")
                    .sort_values(ascending=invert_y).head(8).index.tolist()
                )
                work_area = work_area[work_area[area_color].isin(top_groups)]
                st.caption(f"Color legend limited to top 8 {color_choice} groups.")

            group_cols = [x_field] + ([area_color] if area_color else [])
            if agg_code is None:
                agg_effective = _resolve_agg_none(work_area, group_cols)
                agg_df = (work_area if agg_effective == "none"
                          else _aggregate(work_area, group_cols, y_fields, "mean"))
                metric_label = row_display_name if agg_effective == "none" else f"Average of {row_display_name}"
            else:
                agg_effective = agg_code
                agg_df = _aggregate(work_area, group_cols, y_fields, agg_code)
                metric_label = f"{agg_display} of {row_display_name}"

            agg_df = agg_df.sort_values(by=x_field)

            fig = px.area(
                agg_df, x=x_field, y=y_field, color=area_color,
                color_discrete_sequence=QUALITATIVE_PALETTE if area_color else [LINE_COLOR],
                labels={y_field: metric_label, x_field: col_display_name},
                category_orders={"rank_band": RANK_BAND_ORDER},
            )
            if x_field == "year":
                fig.update_xaxes(type="category")

            title = f"{metric_label} over {col_display_name}{cat_label}"
            fig = apply_theme(fig, title=title, xaxis_title=col_display_name,
                              yaxis_title=metric_label, invert_yaxis=invert_y,
                              show_legend=bool(area_color))
            transformed_df = agg_df

        # -------------------------------------------------------------------
        # CASE 5: BOX & WHISKER PLOT
        # -------------------------------------------------------------------
        elif resolved_mark == "Box Plot":
            work_box = work_df.dropna(subset=[x_field, y_field]).copy()
            if work_box.empty:
                st.warning("No records with valid data for box plot.")
                return

            box_cap = min(f_top_n, 20)
            box_order = (
                _safe_agg_series(work_box, x_field, y_field, "mean")
                .sort_values(ascending=invert_y)
            )
            top_cats = box_order.head(box_cap).index.tolist()
            if x_field == "rank_band":
                top_cats = [v for v in RANK_BAND_ORDER if v in top_cats]
            work_box = work_box[work_box[x_field].isin(top_cats)]

            box_color = color_field if (
                color_field and color_field != x_field and color_field in work_box.columns
            ) else None
            if box_color and work_box[box_color].nunique() > 8:
                top_colors = (
                    _safe_agg_series(work_box, box_color, y_field, "mean")
                    .sort_values(ascending=invert_y).head(8).index.tolist()
                )
                work_box = work_box[work_box[box_color].isin(top_colors)]
                st.caption(f"Color legend limited to top 8 {color_choice} groups.")

            _facet_col_arg = facet_field if (facet_field and facet_field in work_box.columns) else None
            if _facet_col_arg:
                st.info("ℹ️ Facet is not supported for Box Plot — ignoring facet shelf.")
                _facet_col_arg = None

            fig = px.box(
                work_box, x=x_field, y=y_field, color=box_color,
                category_orders={x_field: top_cats, "rank_band": RANK_BAND_ORDER},
                color_discrete_sequence=QUALITATIVE_PALETTE if box_color else [LINE_COLOR],
                labels={y_field: row_display_name, x_field: col_display_name},
            )
            if x_field == "year":
                fig.update_xaxes(type="category")

            title = f"{row_display_name} Distribution by {col_display_name}{cat_label}"
            fig = apply_theme(fig, title=title, xaxis_title=col_display_name,
                              yaxis_title=row_display_name, invert_yaxis=invert_y,
                              show_legend=bool(box_color))
            if is_pooled:
                st.caption("ℹ️ Rankings pooled across all NIRF categories.")
            transformed_df = work_box

        # -------------------------------------------------------------------
        # CASE 6: HISTOGRAM
        # -------------------------------------------------------------------
        elif resolved_mark == "Histogram":
            work_hist = work_df.dropna(subset=[y_field]).copy()
            if work_hist.empty:
                st.warning("No records with valid data for histogram.")
                return

            n_bins = st.slider(
                "Number of Histogram Bins", min_value=5, max_value=100,
                value=20, step=5, key="tab_hist_bins",
            )

            hist_color = color_field if (color_field and color_field in work_hist.columns) else None
            if hist_color and work_hist[hist_color].nunique() > 8:
                top_c = work_hist[hist_color].value_counts().head(8).index.tolist()
                work_hist = work_hist.copy()
                work_hist[hist_color] = work_hist[hist_color].where(
                    work_hist[hist_color].isin(top_c), other="Other"
                )
                st.caption("Color groups beyond the top 8 have been relabeled as 'Other'.")

            fig = px.histogram(
                work_hist, x=y_field, color=hist_color,
                nbins=n_bins, opacity=0.85,
                color_discrete_sequence=QUALITATIVE_PALETTE if hist_color else [LINE_COLOR],
                labels={y_field: row_display_name},
            )
            title = f"Distribution of {row_display_name}{cat_label}"
            fig = apply_theme(fig, title=title, xaxis_title=row_display_name,
                              yaxis_title="Count of Records", show_legend=bool(hist_color))
            if is_pooled:
                st.caption("ℹ️ Rankings pooled across all NIRF categories.")
            hist_cols = list(dict.fromkeys([c for c in [y_field, hist_color] if c]))
            transformed_df = work_hist[[c for c in hist_cols if c in work_hist.columns]]

        # -------------------------------------------------------------------
        # CASE 7: HEATMAP
        # -------------------------------------------------------------------
        elif resolved_mark == "Heatmap":
            if heatmap_y_field is None:
                st.info("Select a Y-Dimension in the Marks Card to generate a heatmap.")
                return

            work_heat = work_df.dropna(subset=[x_field, heatmap_y_field, y_field]).copy()
            if work_heat.empty:
                st.warning("No records with valid data for heatmap.")
                return

            _heat_agg = agg_code if agg_code is not None else "mean"
            pivot_df = (
                work_heat.groupby([heatmap_y_field, x_field], observed=True)[y_field]
                .agg(_heat_agg)
                .reset_index()
                .pivot(index=heatmap_y_field, columns=x_field, values=y_field)
            )

            _heat_scale = "RdYlGn_r" if invert_y else "Blues"
            fig = px.imshow(
                pivot_df,
                color_continuous_scale=_heat_scale,
                text_auto=".1f",
                aspect="auto",
                labels={"x": col_display_name, "y": heatmap_row_dim_name, "color": row_display_name},
            )
            title = f"{row_display_name} Heatmap: {col_display_name} × {heatmap_row_dim_name}{cat_label}"
            fig = apply_theme(fig, title=title, xaxis_title=col_display_name,
                              yaxis_title=heatmap_row_dim_name, show_legend=False)
            transformed_df = pivot_df.reset_index()

        # -------------------------------------------------------------------
        # CASE 8: TABLE
        # -------------------------------------------------------------------
        else:
            title = f"Data Table: {row_display_name} by {col_display_name}{cat_label}"
            if x_is_dim:
                group_cols = list(dict.fromkeys(
                    [x_field] + ([color_field] if color_field and color_field != x_field and color_field in work_df.columns else [])
                ))
                if agg_code is None:
                    agg_effective_tbl = _resolve_agg_none(work_df, group_cols)
                    if agg_effective_tbl == "none":
                        transformed_df = work_df[list(dict.fromkeys(
                            group_cols + [c for c in y_fields if c in work_df.columns]
                        ))].copy()
                    else:
                        transformed_df = _aggregate(work_df, group_cols, y_fields, "mean")
                else:
                    transformed_df = _aggregate(work_df, group_cols, y_fields, agg_code)
            else:
                display_cols = list(dict.fromkeys(
                    [c for c in [x_field] + y_fields + ([color_field] if color_field else []) + ["name", "state", "year"]
                     if c and c in work_df.columns]
                ))
                transformed_df = work_df[display_cols].head(200)

            st.dataframe(transformed_df, use_container_width=True)
            if is_pooled:
                st.caption("ℹ️ Rankings pooled across all NIRF categories.")

    except Exception as _chart_exc:
        st.error(f"⚠️ This combination couldn't be drawn: {str(_chart_exc)[:300]}")
        st.info("Here is a preview of the data that would have been plotted:")
        st.dataframe(work_df.head(50), use_container_width=True)
        # Log to file for debugging
        try:
            os.makedirs("logs", exist_ok=True)
            with open("logs/explore_failures.jsonl", "a") as _logf:
                _logf.write(json.dumps({
                    "error": str(_chart_exc),
                    "traceback": _tb.format_exc()[:1000],
                    "shelves": {"x": x_field, "y": y_field, "mark": resolved_mark, "agg": agg_display},
                }) + "\n")
        except Exception:
            pass
        fig = None
        transformed_df = pd.DataFrame()

    # -----------------------------------------------------------------------
    # Render chart
    # -----------------------------------------------------------------------
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # -----------------------------------------------------------------------
    # Summary statistics expander
    # -----------------------------------------------------------------------
    if not transformed_df.empty:
        with st.expander("📊 Summary Statistics", expanded=False):
            _stat_col = y_field if y_field in transformed_df.columns else None
            if _stat_col:
                _numeric = pd.to_numeric(transformed_df[_stat_col], errors="coerce").dropna()
                if not _numeric.empty:
                    _stats = _numeric.describe()
                    _stats["Q1"] = _numeric.quantile(0.25)
                    _stats["Q3"] = _numeric.quantile(0.75)
                    st.dataframe(_stats.to_frame(name=row_display_name), use_container_width=True)
            else:
                st.write(transformed_df.describe())

    # -----------------------------------------------------------------------
    # Query spec JSON + CSV export
    # -----------------------------------------------------------------------
    spec_export = {
        "title": title,
        "engine": "Tableau-Style Workbench v2",
        "shelves": {
            "columns": {"field": x_field, "display": col_display_name, "type": "Dimension" if x_is_dim else "Measure"},
            "rows": {"fields": y_fields, "display": row_display_names},
            "aggregation": agg_effective if agg_effective != "none" else "none (raw rows)",
            "mark": resolved_mark,
            "color": color_field,
            "facet": facet_field,
            "size": size_field,
        },
        "filters": {
            "category": f_cat,
            "year_range": list(f_year_span),
            "states": f_states,
            "institutions": f_insts,
            "top_n": f_top_n,
            "sort": f_sort,
            "rank_cutoff": rank_cutoff,
        },
    }

    with st.expander("📝 View Underlying Query Spec JSON", expanded=False):
        st.json(spec_export)

    with st.expander("📥 Inspect Result Data & Export CSV", expanded=False):
        if not transformed_df.empty:
            st.dataframe(transformed_df, use_container_width=True)
            csv_bytes = transformed_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download CSV",
                data=csv_bytes,
                file_name="tableau_explore_export.csv",
                mime="text/csv",
                key="tab_download_csv_btn",
            )
        else:
            st.caption("No result data to display.")
