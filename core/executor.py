"""
core/executor.py

The single, unified query and chart execution engine for NIRF Dashboard.
Accepts a JSON spec or ChartSpec, validates with Pydantic and schema whitelists,
executes data transformation steps, auto-downgrades incompatible charts, and renders.

Implements the 5-stage Fallback Ladder:
  1. Validation Error -> Clear message with fuzzy column suggestion ("Did you mean Y?")
  2. Empty Result -> Specific explanation of which filter condition emptied the data
  3. Incompatible Chart -> Auto-downgrade (e.g. numeric+numeric -> scatter, time+num -> line, cat+num -> bar)
  4. Failure -> Transformed tabular view with CSV export
  5. Logging -> Every failure logged to logs/failures.jsonl
"""

import os
import json
import difflib
from datetime import datetime
from typing import Tuple, Optional, Any, Dict, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pydantic import ValidationError

from core.schema import ChartSpec, Step, ChartConfig
from core.safe_expr import evaluate_safe_expr, UnsafeExpressionError
from charts.registry import render_chart_from_spec

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
FAILURES_LOG_PATH = os.path.join(LOGS_DIR, "failures.jsonl")
os.makedirs(LOGS_DIR, exist_ok=True)


def log_failure(spec_data: Any, error_type: str, details: str):
    """Appends failure details to logs/failures.jsonl."""
    record = {
        "timestamp": datetime.now().isoformat(),
        "error_type": error_type,
        "details": details,
        "spec": spec_data if isinstance(spec_data, dict) else str(spec_data),
    }
    try:
        with open(FAILURES_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


def suggest_closest_column(missing_col: str, available_cols: List[str]) -> Optional[str]:
    """Finds closest column name using difflib for helpful error suggestions."""
    matches = difflib.get_close_matches(missing_col, available_cols, n=1, cutoff=0.5)
    return matches[0] if matches else None


def execute_filter_step(df: pd.DataFrame, step) -> Tuple[pd.DataFrame, Optional[str]]:
    """Applies filter conditions sequentially. Reports the exact condition that emptied the data."""
    filtered = df.copy()
    for cond in step.where:
        field = cond.field
        op = cond.op
        val = cond.value

        if field not in filtered.columns:
            suggestion = suggest_closest_column(field, list(filtered.columns))
            hint = f" Did you mean '{suggestion}'?" if suggestion else ""
            return filtered, f"Filter column '{field}' does not exist in dataset.{hint}"

        initial_count = len(filtered)
        if op == "in":
            val_list = val if isinstance(val, (list, tuple, set)) else [val]
            filtered = filtered[filtered[field].isin(val_list)]
        elif op == "eq":
            filtered = filtered[filtered[field] == val]
        elif op == "neq":
            filtered = filtered[filtered[field] != val]
        elif op == "gte":
            filtered = filtered[filtered[field] >= val]
        elif op == "lte":
            filtered = filtered[filtered[field] <= val]
        elif op == "gt":
            filtered = filtered[filtered[field] > val]
        elif op == "lt":
            filtered = filtered[filtered[field] < val]
        elif op == "between":
            if isinstance(val, (list, tuple)) and len(val) == 2:
                filtered = filtered[filtered[field].between(val[0], val[1])]
        elif op == "notnull":
            filtered = filtered[filtered[field].notna()]
        elif op == "contains":
            filtered = filtered[filtered[field].astype(str).str.contains(str(val), case=False, na=False)]

        if len(filtered) == 0 and initial_count > 0:
            return filtered, f"Filter condition '{field} {op} {val}' matched 0 rows (dataset emptied)."

    return filtered, None


def execute_derive_step(df: pd.DataFrame, step) -> Tuple[pd.DataFrame, Optional[str]]:
    """Applies safe arithmetic expression or statistical transform."""
    name = step.name
    res = df.copy()

    if step.expr:
        try:
            series = evaluate_safe_expr(step.expr, res)
            res[name] = series
        except (UnsafeExpressionError, KeyError, Exception) as e:
            return res, f"Failed to compute expression '{step.expr}': {e}"

    elif step.fn:
        fn = step.fn
        target_col = step.expr or (step.by[0] if step.by else None)
        if not target_col or target_col not in res.columns:
            return res, f"Derive function '{fn}' requires valid target column."

        if fn == "zscore":
            mean = res[target_col].mean()
            std = res[target_col].std()
            res[name] = (res[target_col] - mean) / std if std > 0 else 0.0
        elif fn == "pct_rank":
            res[name] = res[target_col].rank(pct=True)
        elif fn == "rank":
            res[name] = res[target_col].rank(ascending=(step.order == "asc"))
        elif fn == "diff":
            if step.by:
                res[name] = res.groupby(step.by)[target_col].diff()
            else:
                res[name] = res[target_col].diff()
        elif fn == "shift":
            res[name] = res[target_col].shift(1)
        elif fn == "std":
            res[name] = res[target_col].std()
        elif fn == "bin" and step.edges:
            res[name] = pd.cut(res[target_col], bins=step.edges, labels=step.labels)

    return res, None


def execute_aggregate_step(df: pd.DataFrame, step) -> Tuple[pd.DataFrame, Optional[str]]:
    """Performs groupby aggregations for specified metrics."""
    by_cols = step.by
    if not by_cols:
        # Whole dataframe aggregation
        agg_dict = {}
        for m in step.metrics:
            agg_dict[m.as_] = pd.Series([df[m.field].agg(m.fn)])
        return pd.DataFrame(agg_dict), None

    missing_by = [c for c in by_cols if c not in df.columns]
    if missing_by:
        return df, f"Aggregation groupby column(s) {missing_by} missing from data."

    grouped = df.groupby(by_cols)
    agg_res = {}

    for m in step.metrics:
        field = m.field
        fn = m.fn
        out_name = m.as_

        if field not in df.columns:
            return df, f"Aggregation metric field '{field}' not found."

        if fn == "iqr":
            agg_res[out_name] = grouped[field].apply(lambda s: s.quantile(0.75) - s.quantile(0.25))
        else:
            agg_res[out_name] = grouped[field].agg(fn)

    res_df = pd.DataFrame(agg_res).reset_index()
    return res_df, None


def auto_downgrade_chart(df: pd.DataFrame, spec: ChartSpec) -> ChartSpec:
    """
    Ladder step 3: Auto-downgrades chart type if current data columns do not fit the requested chart.
    """
    chart = spec.chart
    ctype = chart.type
    x_col = chart.x
    y_col = chart.y

    # Check if requested chart encodings are missing
    missing = chart.validate_encodings()
    if not missing:
        # Check if column types fit
        if x_col in df.columns and y_col in df.columns:
            x_is_num = pd.api.types.is_numeric_dtype(df[x_col])
            y_is_num = pd.api.types.is_numeric_dtype(df[y_col])
            x_is_time = "year" in x_col.lower() or "date" in x_col.lower()

            # Time + Numeric -> Line
            if x_is_time and y_is_num and ctype not in ["line", "bar"]:
                spec.chart.type = "line"
            # Category + Numeric -> Bar
            elif not x_is_num and y_is_num and ctype not in ["bar", "box", "violin"]:
                spec.chart.type = "bar"
            # Numeric + Numeric -> Scatter
            elif x_is_num and y_is_num and not x_is_time and ctype not in ["scatter", "histogram"]:
                spec.chart.type = "scatter"
        return spec

    # Missing encodings -> Pick best available
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

    if categorical_cols and numeric_cols:
        spec.chart.type = "bar"
        spec.chart.x = categorical_cols[0]
        spec.chart.y = numeric_cols[0]
    elif len(numeric_cols) >= 2:
        spec.chart.type = "scatter"
        spec.chart.x = numeric_cols[0]
        spec.chart.y = numeric_cols[1]
    else:
        spec.chart.type = "table"

    return spec


def execute_spec(
    df: pd.DataFrame,
    spec_input: Any,
) -> Tuple[go.Figure, pd.DataFrame, Optional[str], Optional[str]]:
    """
    Executes a ChartSpec against the dataset.
    Returns: (fig, result_df, error_message, takeaway)
    """
    # 1. Validation Stage
    try:
        if isinstance(spec_input, str):
            spec_dict = json.loads(spec_input)
            spec = ChartSpec.model_validate(spec_dict)
        elif isinstance(spec_input, dict):
            spec = ChartSpec.model_validate(spec_input)
        elif isinstance(spec_input, ChartSpec):
            spec = spec_input
        else:
            err = f"Invalid spec input type: {type(spec_input)}"
            log_failure(spec_input, "ValidationError", err)
            fig = go.Figure()
            return fig, df, err, None
    except ValidationError as e:
        err = f"Chart Spec Validation Error: {e}"
        log_failure(spec_input, "ValidationError", err)
        fig = go.Figure()
        return fig, df, err, None
    except Exception as e:
        err = f"Failed to parse spec: {e}"
        log_failure(spec_input, "ParseError", err)
        fig = go.Figure()
        return fig, df, err, None

    # 2. Run Steps sequentially
    curr_df = df.copy()
    for idx, step in enumerate(spec.steps):
        step_op = getattr(step, "op", None)

        if step_op == "filter":
            curr_df, filter_err = execute_filter_step(curr_df, step)
            if filter_err:
                log_failure(spec_input, "EmptyResultError", filter_err)
                fig = go.Figure()
                return fig, curr_df, filter_err, None

        elif step_op == "derive":
            curr_df, derive_err = execute_derive_step(curr_df, step)
            if derive_err:
                log_failure(spec_input, "DeriveError", derive_err)
                fig = go.Figure()
                return fig, curr_df, derive_err, None

        elif step_op == "aggregate":
            curr_df, agg_err = execute_aggregate_step(curr_df, step)
            if agg_err:
                log_failure(spec_input, "AggregateError", agg_err)
                fig = go.Figure()
                return fig, curr_df, agg_err, None

        elif step_op == "sort":
            order = step.order
            by = step.by
            asc = order == "asc" if isinstance(order, str) else [o == "asc" for o in order]
            if isinstance(by, list):
                valid_by = [c for c in by if c in curr_df.columns]
                if valid_by:
                    curr_df = curr_df.sort_values(by=valid_by, ascending=asc)
            elif by in curr_df.columns:
                curr_df = curr_df.sort_values(by=by, ascending=asc)

        elif step_op == "top":
            by = step.by or (curr_df.columns[1] if len(curr_df.columns) > 1 else curr_df.columns[0])
            if by in curr_df.columns:
                curr_df = curr_df.sort_values(by=by, ascending=(step.order == "asc")).head(step.n)
            else:
                curr_df = curr_df.head(step.n)

        elif step_op == "dropna":
            curr_df = curr_df.dropna(subset=step.subset)

        elif step_op == "fillna":
            curr_df = curr_df.fillna(value=step.value)

    # 3. Auto-downgrade check
    spec = auto_downgrade_chart(curr_df, spec)

    # 4. Render Chart
    try:
        fig = render_chart_from_spec(curr_df, spec)
        takeaway = spec.takeaway
        return fig, curr_df, None, takeaway
    except Exception as e:
        err = f"Rendering error: {e}. Falling back to tabular view."
        log_failure(spec_input, "RenderError", str(e))
        # Fallback to table
        spec.chart.type = "table"
        fallback_fig = render_chart_from_spec(curr_df, spec)
        return fallback_fig, curr_df, err, spec.takeaway
