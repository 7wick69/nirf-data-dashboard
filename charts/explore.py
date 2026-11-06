"""
charts/explore.py

Tableau-Style Visual Analytics Workbench (Level 2 Explore Tab) — NIRF Interactive Dashboard.
Replicates Tableau's core visual architecture:
  - Shelves: Columns (X), Rows (Y), Marks (Chart Type & Color Encoding), Filters
  - Automatic Aggregation: Automatically averages/sums measures across dimensions to eliminate overlapping/glitchy bars
  - Smart Marks Inference: Replicates Tableau's 'Show Me' logic
  - Automatic Layout: Switches to horizontal bars for long labels (institutions/states)
  - Statistical Analysis: Dynamic Pearson correlation (r), R², and OLS trendlines
  - Export: JSON Query Spec inspector and one-click CSV export
"""

import json
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
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
from core.executor import execute_spec

# ---------------------------------------------------------------------------
# Dimension & Measure Definitions (Tableau Data Pane)
# ---------------------------------------------------------------------------

DIMENSIONS = {
    "Institution Name": "name",
    "State": "state",
    "Ranking Year": "year",
    "Category": "category",
    "Rank Band": "rank_band",
    "City": "city",
}

MEASURES = {
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

AGG_FUNCS = {
    "Average (Mean)": "mean",
    "Median": "median",
    "Sum": "sum",
    "Maximum": "max",
    "Minimum": "min",
    "Count of Records": "count",
    "Standard Deviation (std)": "std",
}


def calculate_pearson(df: pd.DataFrame, x_col: str, y_col: str) -> Optional[Dict[str, Any]]:
    """Calculates Pearson r, R², p-value between two continuous measures."""
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


def render_explore_tab(df_full: pd.DataFrame, df_filtered: pd.DataFrame):
    """Renders the Tableau-style visual analytics interface."""
    st.markdown("### Explore: Tableau-Style Visual Query Builder")
    st.caption("Drag-and-drop shelf logic: select Columns, Rows, and Aggregations to generate clean, publication-grade charts.")

    # -----------------------------------------------------------------------
    # Preset Quick-Loads (Viva Evaluation Helpers)
    # -----------------------------------------------------------------------
    c_preset1, c_preset2 = st.columns([2.5, 1])
    with c_preset1:
        preset_choice = st.selectbox(
            "⚡ Quick-Load Analytical Preset (Unseen Evaluation Questions)",
            options=[
                "None (Custom Query)",
                "1. Top 15 Institutions by Graduation Outcomes (GO) in Engineering",
                "2. Teaching (TLR) vs Perception (PR) Correlation across Universities",
                "3. Average Research Productivity (RPC) by State (2024)",
                "4. Score Distribution across Disciplines (Box Plot)",
                "5. Overall Score Trajectory over Time (2017–2025)",
            ],
            index=0,
            key="tab_preset_selector",
        )

    # -----------------------------------------------------------------------
    # Shelves Container (Columns, Rows, Marks)
    # -----------------------------------------------------------------------
    st.markdown(
        """
        <style>
        .shelf-container {
            background-color: #14161a;
            border: 1px solid #23272f;
            border-radius: 8px;
            padding: 14px 18px;
            margin-bottom: 16px;
        }
        .shelf-header {
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #58a6ff;
            margin-bottom: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<div class="shelf-container">', unsafe_allow_html=True)
        col_c, col_r, col_m = st.columns([1.2, 1.2, 1.0], gap="medium")

        # 1. COLUMNS SHELF (X-Axis)
        with col_c:
            st.markdown('<div class="shelf-header">Columns (X Shelf)</div>', unsafe_allow_html=True)
            col_shelf_type = st.radio("Field Type", ["Dimension", "Measure"], horizontal=True, key="tab_col_type")
            if col_shelf_type == "Dimension":
                col_display_name = st.selectbox("Dimension", list(DIMENSIONS.keys()), index=1, key="tab_col_dim")
                x_field = DIMENSIONS[col_display_name]
                x_is_dim = True
            else:
                col_display_name = st.selectbox("Measure", list(MEASURES.keys()), index=1, key="tab_col_meas")
                x_field = MEASURES[col_display_name]
                x_is_dim = False

        # 2. ROWS SHELF (Y-Axis)
        with col_r:
            st.markdown('<div class="shelf-header">Rows (Y Shelf)</div>', unsafe_allow_html=True)
            row_display_name = st.selectbox("Measure", list(MEASURES.keys()), index=0, key="tab_row_meas")
            y_field = MEASURES[row_display_name]

            # Aggregation Selector (Tableau's core strength)
            agg_display = st.selectbox("Aggregation", list(AGG_FUNCS.keys()), index=0, key="tab_agg_func")
            agg_code = AGG_FUNCS[agg_display]

        # 3. MARKS CARD (Chart Family & Options)
        with col_m:
            st.markdown('<div class="shelf-header">Marks Card (Show Me)</div>', unsafe_allow_html=True)
            mark_choice = st.selectbox(
                "Mark Type",
                ["Automatic (Show Me)", "Bar", "Line", "Scatter", "Box Plot", "Histogram", "Table"],
                index=0,
                key="tab_mark_type",
            )
            color_choice = st.selectbox("Color / Group", ["None", "Category", "State", "Rank Band"], index=0, key="tab_color_pill")
            color_field = {"None": None, "Category": "category", "State": "state", "Rank Band": "rank_band"}[color_choice]

        st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Filters Shelf (Category, Year, State, Top N, Sort)
    # -----------------------------------------------------------------------
    with st.expander("🔍 Filters Shelf & Sorting Controls", expanded=False):
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            all_cats = ["All Categories"] + sorted([c for c in df_full["category"].dropna().unique()])
            f_cat = st.selectbox("Filter Category", all_cats, index=0, key="tab_filter_cat")
        with fc2:
            min_y, max_y = int(df_full["year"].min()), int(df_full["year"].max())
            f_year_span = st.slider("Year Range", min_value=min_y, max_value=max_y, value=(min_y, max_y), key="tab_filter_year")
        with fc3:
            f_top_n = st.slider("Top N Limit", min_value=5, max_value=50, value=15, step=5, key="tab_top_n")
        with fc4:
            f_sort = st.selectbox("Sort Order", ["Descending (Highest First)", "Ascending (Lowest First)", "Alphabetical", "None"], index=0, key="tab_sort_order")

    # Handle Presets Override if selected
    if preset_choice != "None (Custom Query)":
        if "1." in preset_choice:
            x_field, y_field, agg_code = "name", "GO", "mean"
            x_is_dim, mark_choice, f_cat, f_top_n = True, "Bar", "Engineering", 15
        elif "2." in preset_choice:
            x_field, y_field = "TLR", "PR"
            x_is_dim, mark_choice, f_cat = False, "Scatter", "University"
        elif "3." in preset_choice:
            x_field, y_field, agg_code = "state", "RPC", "mean"
            x_is_dim, mark_choice = True, "Bar"
        elif "4." in preset_choice:
            x_field, y_field = "category", "score"
            x_is_dim, mark_choice = True, "Box Plot"
        elif "5." in preset_choice:
            x_field, y_field, agg_code = "year", "score", "mean"
            x_is_dim, mark_choice = True, "Line"

    # -----------------------------------------------------------------------
    # Apply Filters to Dataset
    # -----------------------------------------------------------------------
    work_df = df_full.copy()
    if f_cat != "All Categories":
        work_df = work_df[work_df["category"] == f_cat]
    work_df = work_df[(work_df["year"] >= f_year_span[0]) & (work_df["year"] <= f_year_span[1])]

    if work_df.empty:
        st.warning("No records match the selected filters. Try widening the year range or category filter.")
        return

    # -----------------------------------------------------------------------
    # Show Me (Automatic Chart Type Inference)
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

    # -----------------------------------------------------------------------
    # Chart Renderers (Following Tableau's Aggregated & Fail-Safe Logic)
    # -----------------------------------------------------------------------
    fig = go.Figure()
    transformed_df = pd.DataFrame()
    invert_y = y_field == "rank"

    # CASE 1: SCATTER PLOT (Two Continuous Measures)
    if resolved_mark == "Scatter":
        scat_df = work_df[[x_field, y_field, "name", "state", "year"]].dropna()
        if len(scat_df) > 500:
            scat_df = scat_df.sample(500, random_state=42)

        corr_stats = calculate_pearson(scat_df, x_field, y_field)
        if corr_stats:
            r, r2 = corr_stats["r"], corr_stats["r2"]
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
                st.metric("Statistical Relationship", strength)

        fig.add_trace(
            go.Scatter(
                x=scat_df[x_field],
                y=scat_df[y_field],
                mode="markers",
                marker=dict(size=8, color=LINE_COLOR, opacity=0.8, line=dict(width=1, color="#0f1013")),
                text=scat_df["name"],
                customdata=np.stack((scat_df["state"], scat_df["year"]), axis=-1),
                hovertemplate="<b>%{text}</b><br>%{xaxis.title.text}: %{x:.2f}<br>%{yaxis.title.text}: %{y:.2f}<br>State: %{customdata[0]}<br>Year: %{customdata[1]}<extra></extra>",
                name="Institutions",
            )
        )

        # OLS Linear Trendline
        if len(scat_df) >= 3:
            xv = scat_df[x_field].values
            yv = scat_df[y_field].values
            slope, intercept = np.polyfit(xv, yv, 1)
            x_line = np.linspace(xv.min(), xv.max(), 50)
            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=slope * x_line + intercept,
                    mode="lines",
                    name="OLS Fit",
                    line=dict(color="#f97316", width=2, dash="dash"),
                )
            )

        title = f"{row_display_name} vs {col_display_name}"
        fig = apply_theme(fig, title=title, xaxis_title=col_display_name, yaxis_title=row_display_name, invert_yaxis=invert_y)
        transformed_df = scat_df

    # CASE 2: BAR CHART (Dimension + Aggregated Measure)
    elif resolved_mark == "Bar":
        group_cols = [x_field]
        if color_field and color_field != x_field and color_field in work_df.columns:
            group_cols.append(color_field)

        agg_df = work_df.groupby(group_cols)[y_field].agg(agg_code).reset_index()
        metric_col_name = f"{agg_code}_{y_field}"
        agg_df.rename(columns={y_field: metric_col_name}, inplace=True)

        if f_sort == "Descending (Highest First)":
            agg_df = agg_df.sort_values(by=metric_col_name, ascending=False)
        elif f_sort == "Ascending (Lowest First)":
            agg_df = agg_df.sort_values(by=metric_col_name, ascending=True)
        elif f_sort == "Alphabetical":
            agg_df = agg_df.sort_values(by=x_field, ascending=True)

        agg_df = agg_df.head(f_top_n)

        # Horizontal layout for readable labels when names or states are long
        use_horizontal = x_field in ["name", "state", "city"]

        if use_horizontal:
            # Sort ascending so highest appears at the top in horizontal bars
            agg_df = agg_df.iloc[::-1]
            fig.add_trace(
                go.Bar(
                    y=agg_df[x_field].astype(str),
                    x=agg_df[metric_col_name],
                    orientation="h",
                    marker_color=LINE_COLOR if not color_field else None,
                    hovertemplate="<b>%{y}</b><br>%{x:.2f}<extra></extra>",
                )
            )
            title = f"{agg_display} of {row_display_name} by {col_display_name} (Top {f_top_n})"
            fig = apply_theme(fig, title=title, xaxis_title=f"{agg_display} of {row_display_name}", yaxis_title=col_display_name)
        else:
            fig.add_trace(
                go.Bar(
                    x=agg_df[x_field].astype(str),
                    y=agg_df[metric_col_name],
                    marker_color=LINE_COLOR if not color_field else None,
                    hovertemplate="<b>%{x}</b><br>%{y:.2f}<extra></extra>",
                )
            )
            title = f"{agg_display} of {row_display_name} by {col_display_name} (Top {f_top_n})"
            fig = apply_theme(fig, title=title, xaxis_title=col_display_name, yaxis_title=f"{agg_display} of {row_display_name}", invert_yaxis=invert_y)

        transformed_df = agg_df

    # CASE 3: LINE CHART (Time-series / Trends)
    elif resolved_mark == "Line":
        if color_field and color_field in work_df.columns:
            top_groups = work_df[color_field].value_counts().head(8).index
            filtered_groups = work_df[work_df[color_field].isin(top_groups)]
            agg_df = filtered_groups.groupby([x_field, color_field])[y_field].agg(agg_code).reset_index()
            metric_col_name = f"{agg_code}_{y_field}"
            agg_df.rename(columns={y_field: metric_col_name}, inplace=True)

            for idx, g in enumerate(top_groups):
                sub = agg_df[agg_df[color_field] == g].sort_values(by=x_field)
                c = QUALITATIVE_PALETTE[idx % len(QUALITATIVE_PALETTE)]
                sym = MARKER_SYMBOLS[idx % len(MARKER_SYMBOLS)]
                fig.add_trace(
                    go.Scatter(
                        x=sub[x_field],
                        y=sub[metric_col_name],
                        mode="lines+markers",
                        name=str(g),
                        line=dict(color=c, width=2.5),
                        marker=dict(symbol=sym, size=7, color=c),
                    )
                )
            title = f"{agg_display} of {row_display_name} over {col_display_name} (by {color_choice})"
            fig = apply_theme(fig, title=title, xaxis_title=col_display_name, yaxis_title=f"{agg_display} of {row_display_name}", invert_yaxis=invert_y, show_legend=True)
            transformed_df = agg_df
        else:
            agg_df = work_df.groupby(x_field)[y_field].agg(agg_code).reset_index()
            metric_col_name = f"{agg_code}_{y_field}"
            agg_df.rename(columns={y_field: metric_col_name}, inplace=True)
            agg_df = agg_df.sort_values(by=x_field)

            fig.add_trace(
                go.Scatter(
                    x=agg_df[x_field],
                    y=agg_df[metric_col_name],
                    mode="lines+markers",
                    line=dict(color=LINE_COLOR, width=3),
                    marker=dict(size=8, color=LINE_COLOR),
                    hovertemplate="<b>%{x}</b>: %{y:.2f}<extra></extra>",
                )
            )
            title = f"{agg_display} of {row_display_name} over {col_display_name}"
            fig = apply_theme(fig, title=title, xaxis_title=col_display_name, yaxis_title=f"{agg_display} of {row_display_name}", invert_yaxis=invert_y)
            transformed_df = agg_df

    # CASE 4: BOX & WHISKER PLOT
    elif resolved_mark == "Box Plot":
        valid_box = work_df[[x_field, y_field]].dropna()
        top_cats = valid_box[x_field].value_counts().head(8).index
        sub_box = valid_box[valid_box[x_field].isin(top_cats)]

        for idx, cat_val in enumerate(top_cats):
            cat_data = sub_box[sub_box[x_field] == cat_val][y_field]
            c = QUALITATIVE_PALETTE[idx % len(QUALITATIVE_PALETTE)]
            fig.add_trace(go.Box(y=cat_data, name=str(cat_val), marker_color=c))

        title = f"{row_display_name} Distribution across {col_display_name}"
        fig = apply_theme(fig, title=title, xaxis_title=col_display_name, yaxis_title=row_display_name, invert_yaxis=invert_y)
        transformed_df = sub_box

    # CASE 5: HISTOGRAM
    elif resolved_mark == "Histogram":
        clean_vals = work_df[y_field].dropna()
        fig.add_trace(go.Histogram(x=clean_vals, marker_color=LINE_COLOR, opacity=0.85))
        title = f"Distribution of {row_display_name}"
        fig = apply_theme(fig, title=title, xaxis_title=row_display_name, yaxis_title="Institution Count")
        transformed_df = pd.DataFrame({y_field: clean_vals})

    # CASE 6: TABLE (Crosstab)
    else:
        st.dataframe(work_df[[x_field, y_field, "name", "state", "year"]].head(100), use_container_width=True)
        return

    # Render Plotly Chart
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # -----------------------------------------------------------------------
    # Spec JSON & CSV Export
    # -----------------------------------------------------------------------
    spec_export = {
        "title": title,
        "engine": "Tableau-Style Workbench",
        "shelves": {
            "columns": x_field,
            "rows": y_field,
            "aggregation": agg_code,
            "mark": resolved_mark,
            "color": color_field,
        },
        "filters": {
            "category": f_cat,
            "year_range": f_year_span,
            "top_n": f_top_n,
            "sort": f_sort,
        },
    }

    with st.expander("📝 View Underlying Query Spec JSON", expanded=False):
        st.json(spec_export)

    with st.expander("📥 Inspect Result Data & Export CSV", expanded=False):
        st.dataframe(transformed_df, use_container_width=True)
        csv_bytes = transformed_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_bytes, file_name="tableau_explore_export.csv", mime="text/csv")
