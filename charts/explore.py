"""
charts/explore.py

Level 2 Explore Tab Module — NIRF Interactive Dashboard.
Allows interactive construction of ChartSpec queries to answer unseen analytical questions:
  - Dynamic chart types (scatter, bar, line, box, histogram, violin, table, auto)
  - Two-variable correlation analysis (Pearson r, R²) with OLS trendline
  - Interactive filter builder (column, operator, value)
  - GroupBy aggregations and safe AST arithmetic column derivations
  - Row limiting (Top N)
  - Quick-load unseen analytical presets
  - Raw JSON spec editor / runner
  - Live query execution through the central executor (core/executor.py)
  - JSON query spec viewer and CSV export
"""

import json
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy import stats

from core.executor import execute_spec


def compute_pearson_r(df: pd.DataFrame, x_col: str, y_col: str) -> Optional[Dict[str, Any]]:
    """Calculates Pearson correlation coefficient and R-squared between two numeric columns."""
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
        r_val, p_val = stats.pearsonr(x_vals.loc[common_idx], y_vals.loc[common_idx])
        return {
            "r": float(r_val),
            "r2": float(r_val ** 2),
            "p_value": float(p_val),
            "n": int(len(common_idx)),
        }
    except Exception:
        return None


UNSEEN_PRESETS = {
    "None (Custom Query)": None,
    "1. Top 10 Engineering institutions by Graduation Outcomes (GO)": {
        "title": "Top 10 Engineering Institutions by GO",
        "takeaway": "Institutions with highest graduation outcomes in engineering",
        "steps": [
            {"op": "filter", "where": [{"field": "category", "op": "eq", "value": "Engineering"}]},
            {"op": "sort", "by": "GO", "order": "desc"},
            {"op": "top", "n": 10},
        ],
        "chart": {"type": "bar", "x": "name", "y": "GO", "orientation": "v"},
    },
    "2. Correlation: TLR vs Perception (PR) in Universities": {
        "title": "Teaching (TLR) vs Perception (PR) in Universities",
        "takeaway": "Evaluating how strongly perception correlates with teaching resources",
        "steps": [
            {"op": "filter", "where": [{"field": "category", "op": "eq", "value": "University"}]},
            {"op": "top", "n": 100},
        ],
        "chart": {"type": "scatter", "x": "TLR", "y": "PR", "trendline": "ols"},
    },
    "3. Average Research (RPC) score by State in 2024": {
        "title": "Average Research Score (RPC) by State (2024)",
        "takeaway": "State-wise breakdown of research productivity",
        "steps": [
            {"op": "filter", "where": [{"field": "year", "op": "eq", "value": 2024}]},
            {"op": "aggregate", "by": ["state"], "metrics": [{"field": "RPC", "fn": "mean", "as": "avg_RPC"}]},
            {"op": "sort", "by": "avg_RPC", "order": "desc"},
            {"op": "top", "n": 12},
        ],
        "chart": {"type": "bar", "x": "state", "y": "avg_RPC", "orientation": "v"},
    },
    "4. Score Distribution across Disciplines (2024)": {
        "title": "Score Spread across Categories (2024)",
        "takeaway": "Cross-discipline comparison of NIRF total scores",
        "steps": [
            {"op": "filter", "where": [{"field": "year", "op": "eq", "value": 2024}]},
            {"op": "top", "n": 200},
        ],
        "chart": {"type": "box", "x": "category", "y": "score"},
    },
}


def render_explore_tab(df_full: pd.DataFrame, df_filtered: pd.DataFrame):
    """Renders the Level 2 Explore query and visualization builder."""
    st.markdown("### Explore: Interactive Query & Chart Builder")
    st.caption("Answer unseen analytical questions live via interactive query parameters or custom JSON specs.")

    # 1. Preset or Pre-loaded Query Bar
    col_preset, col_reset = st.columns([2.5, 1])
    with col_preset:
        preset_choice = st.selectbox(
            "⚡ Quick-load Unseen Analytical Query Preset",
            options=list(UNSEEN_PRESETS.keys()),
            index=0,
            key="exp_preset_choice",
        )

    # Check if a spec was sent via 'Open in Explore' from Trends tab
    loaded_spec = st.session_state.get("explore_spec", None)
    if loaded_spec:
        st.info(f"Loaded query spec from Trends: **{loaded_spec.get('title', 'Trends Spec')}**")
        with col_reset:
            if st.button("Reset Explore View", key="reset_exp_spec"):
                del st.session_state["explore_spec"]
                st.rerun()

    active_preset_spec = UNSEEN_PRESETS[preset_choice] if preset_choice != "None (Custom Query)" else loaded_spec

    all_cols = list(df_filtered.columns)
    num_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df_filtered[c])]
    cat_cols = [c for c in all_cols if not pd.api.types.is_numeric_dtype(df_filtered[c])]

    # Allow toggling between Visual Query Builder and Raw JSON Editor
    mode = st.radio("Configuration Mode", ["Visual Form Builder", "Raw JSON Spec Editor"], horizontal=True, key="exp_mode")

    if mode == "Raw JSON Spec Editor":
        default_json = json.dumps(active_preset_spec or {
            "title": "Custom Query Spec",
            "steps": [
                {"op": "filter", "where": [{"field": "year", "op": "eq", "value": 2024}]},
                {"op": "top", "n": 20}
            ],
            "chart": {"type": "bar", "x": "name", "y": "score"}
        }, indent=2)
        raw_json_str = st.text_area("JSON ChartSpec", value=default_json, height=260, key="exp_raw_json")
        try:
            parsed_spec = json.loads(raw_json_str)
            fig_raw, res_raw, err_raw, _ = execute_spec(df_full, parsed_spec)
            if err_raw:
                st.warning(f"⚠️ {err_raw}")
            else:
                st.plotly_chart(fig_raw, use_container_width=True, config={"displayModeBar": False})
            st.dataframe(res_raw, use_container_width=True)
        except Exception as e:
            st.error(f"JSON Parsing Error: {e}")
        return

    # Visual Form Builder Layout
    col_controls, col_preview = st.columns([1.1, 1.9], gap="large")

    with col_controls:
        st.markdown("##### 1. Chart Configuration")
        chart_types = ["auto", "scatter", "bar", "line", "histogram", "box", "violin", "table"]
        c_type = st.selectbox(
            "Chart Type",
            options=chart_types,
            index=1,
            help="Select visualization family or 'auto' to infer from variable types",
            key="exp_chart_type",
        )

        col_x, col_y = st.columns(2)
        with col_x:
            default_x_idx = all_cols.index("TLR") if "TLR" in all_cols else 0
            exp_x = st.selectbox("X Axis (Primary)", options=all_cols, index=default_x_idx, key="exp_x_col")

        with col_y:
            default_y_idx = num_cols.index("score") if "score" in num_cols else 0
            exp_y = st.selectbox("Y Axis (Metric)", options=num_cols, index=default_y_idx, key="exp_y_col")

        color_options = ["None"] + [c for c in all_cols if c not in [exp_x, exp_y]]
        exp_color = st.selectbox("Color / Grouping (Optional)", options=color_options, index=0, key="exp_color_col")

        trendline_opt = None
        invert_yaxis = False
        orientation_opt = "v"
        if c_type == "bar":
            bar_orient_choice = st.radio("Orientation", ["Vertical", "Horizontal"], index=0, horizontal=True, key="exp_bar_orient")
            orientation_opt = "h" if bar_orient_choice == "Horizontal" else "v"

        if c_type in ["scatter", "auto"]:
            add_trendline = st.checkbox("Fit OLS Linear Trendline", value=True, key="exp_ols_toggle")
            if add_trendline:
                trendline_opt = "ols"

        if "rank" in exp_y.lower() or exp_y == "rank":
            invert_yaxis = st.checkbox("Invert Y-Axis (Rank 1 at Top)", value=True, key="exp_invert_yaxis")

        st.markdown("##### 2. Filters & Transformations")

        # Custom Filter Step
        use_filter = st.checkbox("Add Custom Filter", value=False, key="exp_use_filter")
        filter_step = None
        if use_filter:
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                filter_col = st.selectbox("Column", options=all_cols, key="exp_filter_col")
            with fc2:
                filter_op = st.selectbox("Op", options=["eq", "gte", "lte", "in"], key="exp_filter_op")
            with fc3:
                if filter_op in ["gte", "lte"]:
                    filter_val = st.number_value if filter_col in num_cols else 50.0
                    filter_val = st.number_input("Value", value=50.0, key="exp_filter_num_val")
                else:
                    filter_val = st.text_input("Value", value="Tamil Nadu", key="exp_filter_txt_val")
            filter_step = {"op": "filter", "where": [{"field": filter_col, "op": filter_op, "value": filter_val}]}

        # Aggregation Step
        use_agg = st.checkbox("Aggregate / Group By", value=False, key="exp_use_agg")
        agg_step = None
        if use_agg:
            agg_by = st.multiselect(
                "Group By Column(s)",
                options=cat_cols + ["year"],
                default=[exp_x] if exp_x in cat_cols else [],
                key="exp_agg_by",
            )
            agg_metric = st.selectbox(
                "Metric Column",
                options=num_cols,
                index=num_cols.index(exp_y) if exp_y in num_cols else 0,
                key="exp_agg_metric",
            )
            agg_fn = st.selectbox(
                "Aggregation Function",
                options=["mean", "median", "sum", "count", "max", "min", "std"],
                index=0,
                key="exp_agg_fn",
            )
            if agg_by and agg_metric:
                agg_step = {
                    "op": "aggregate",
                    "by": agg_by,
                    "metrics": [{"field": agg_metric, "fn": agg_fn, "as": f"{agg_fn}_{agg_metric}"}],
                }

        # Safe AST Derive Step
        use_derive = st.checkbox("Derive Column (Safe AST Arithmetic)", value=False, key="exp_use_derive")
        derive_step = None
        if use_derive:
            der_name = st.text_input("New Column Name", value="param_avg", key="exp_der_name")
            der_expr = st.text_input("Arithmetic Expression (e.g. (TLR + RPC) / 2)", value="(TLR + RPC) / 2", key="exp_der_expr")
            if der_name and der_expr:
                derive_step = {"op": "derive", "name": der_name, "expr": der_expr}

        top_limit = st.slider("Row Limit (Top N)", min_value=5, max_value=100, value=25, key="exp_top_n")

    # If preset was selected, use preset spec directly
    if active_preset_spec:
        custom_spec = active_preset_spec
    else:
        pipeline_steps = []
        if filter_step:
            pipeline_steps.append(filter_step)
        if derive_step:
            pipeline_steps.append(derive_step)
        if agg_step:
            pipeline_steps.append(agg_step)
            exp_y_field = f"{agg_step['metrics'][0]['fn']}_{agg_step['metrics'][0]['field']}"
        else:
            exp_y_field = exp_y

        pipeline_steps.append({"op": "top", "n": top_limit})

        resolved_chart_type = c_type
        if c_type == "auto":
            is_x_time = exp_x == "year"
            is_x_num = pd.api.types.is_numeric_dtype(df_filtered[exp_x])
            if is_x_time:
                resolved_chart_type = "line"
            elif is_x_num:
                resolved_chart_type = "scatter"
            else:
                resolved_chart_type = "bar"

        custom_spec = {
            "title": f"Explore: {exp_y_field} vs {exp_x}",
            "takeaway": "Custom query generated via Level 2 Explore interface",
            "steps": pipeline_steps,
            "chart": {
                "type": resolved_chart_type,
                "x": exp_x,
                "y": exp_y_field,
                "color": exp_color if exp_color != "None" else None,
                "trendline": trendline_opt,
                "invert_yaxis": invert_yaxis,
                "orientation": orientation_opt,
            },
        }

    with col_preview:
        # Display Bivariate Pearson Correlation Metric if applicable
        if exp_x in num_cols and exp_y in num_cols and exp_x != exp_y and not active_preset_spec:
            corr_info = compute_pearson_r(df_filtered, exp_x, exp_y)
            if corr_info:
                r_val = corr_info["r"]
                r2_val = corr_info["r2"]
                strength = (
                    "Strong Positive" if r_val >= 0.7
                    else "Moderate Positive" if r_val >= 0.4
                    else "Weak Positive" if r_val > 0.1
                    else "Negligible" if abs(r_val) <= 0.1
                    else "Weak Negative" if r_val >= -0.4
                    else "Moderate Negative" if r_val >= -0.7
                    else "Strong Negative"
                )
                metric_c1, metric_c2, metric_c3 = st.columns(3)
                with metric_c1:
                    st.metric("Pearson Correlation (r)", f"{r_val:+.3f}")
                with metric_c2:
                    st.metric("R² (Variance Explained)", f"{r2_val * 100:.1f}%")
                with metric_c3:
                    st.metric("Relationship", strength)

        # Run through Executor with 5-stage Fallback Ladder
        fig, res_df, err, takeaway = execute_spec(df_full if active_preset_spec else df_filtered, custom_spec)

        if err:
            st.warning(f"⚠️ {err}")
        else:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with st.expander("📝 View / Copy Query Spec JSON", expanded=False):
            st.json(custom_spec)

        with st.expander("📥 Inspect Result Data & Export CSV", expanded=False):
            st.dataframe(res_df, use_container_width=True)
            csv_bytes = res_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv_bytes, file_name="explore_results.csv", mime="text/csv")
