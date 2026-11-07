"""Data-aware visualization selection for the NIRF analytical assistant.

The policy chooses a chart family from analytical intent and result structure.
It is intentionally separate from the Plotly renderer.
"""
from __future__ import annotations
from typing import Any
import pandas as pd

from .chart_specs import normalize_chart_type

INTENT_CHART_POLICY = {
    "ranking": {"default": "dot_plot"},
    "trend_analysis": {"default": "line", "rank_metric": "line"},
    "rank_movement": {"default": "diverging_bar"},
    "geographical_analysis": {"default": "horizontal_bar"},
    "distribution_analysis": {"default": "histogram", "grouped": "box"},
    "statistical_summary": {"default": "none"},
    "correlation_analysis": {"default": "scatter"},
    "multivariate_analysis": {"default": "heatmap", "two_metrics": "scatter_quadrant"},
    "transition_analysis": {"default": "transition_heatmap"},
    "outlier_analysis": {"default": "scatter"},
    "outlier_detection": {"default": "scatter"},
    "comparison": {"default": "horizontal_bar"},
}

CATALOG_CHARTS = {
    "none": "none",
    "rank_trend": "line",
    "line_trend": "line",
    "diverging_rank_movement": "diverging_bar",
    "membership_heatmap": "heatmap",
    "state_count_bar": "horizontal_bar",
    "state_year_heatmap": "heatmap",
    "state_change_bar": "diverging_bar",
    "state_rank_box": "box",
    "state_new_entries_bar": "horizontal_bar",
    "histogram_ecdf": "histogram",
    "rank_band_bar": "horizontal_bar",
    "quartile_bar": "horizontal_bar",
    "category_box": "box",
    "scatter_correlation": "scatter",
    "correlation_bar": "horizontal_bar",
    "correlation_trend": "line",
    "parameter_heatmap": "heatmap",
    "quadrant_scatter": "scatter_quadrant",
    "change_heatmap": "heatmap",
    "parameter_gap_bar": "horizontal_bar",
    "grouped_box": "box",
    "year_box": "box",
    "variability_bar": "horizontal_bar",
    "median_iqr_trend": "line",
    "state_box": "box",
    "transition_heatmap": "transition_heatmap",
    "entry_exit_table": "table",
    "normalized_transition_heatmap": "transition_heatmap",
    "movement_stacked_bar": "stacked_bar",
    "outlier_scatter": "scatter",
    "difference_plot": "difference_plot",
    "change_outlier_plot": "box",
    "metric_rank_scatter": "scatter_rank",
    "volatility_dot_plot": "dot_plot",
}


def profile_result(result: dict[str, Any]) -> dict[str, Any]:
    """Derive measurable chart-selection properties from an analytics result."""
    chart = result.get("chart") or {}
    records = chart.get("data") or result.get("data") or []
    frame = pd.DataFrame(records)
    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    categorical = [c for c in frame.columns if c not in numeric]
    x, y = chart.get("x"), chart.get("y")
    return {
        "n_rows": int(len(frame)),
        "n_groups": int(frame[x].nunique(dropna=True)) if x in frame.columns else 0,
        "has_two_metrics": bool(x in numeric and y in numeric),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "has_year": "year" in frame.columns,
        "has_rank": "rank_num" in frame.columns or "rank" in frame.columns,
        "has_score": "score" in frame.columns,
        "has_missing_values": bool(frame.isna().any().any()) if not frame.empty else False,
    }


def select_visualization(prediction: dict[str, Any],
                         data_profile: dict[str, Any] | None = None,
                         result: dict[str, Any] | None = None) -> dict[str, Any]:
    """Select a supported chart family; a one-row result is an answer card/table."""
    if prediction.get("status") != "success":
        return {"chart": "none", "render": False,
                "reason": "Question intent is not sufficiently confident."}

    profile = data_profile or (profile_result(result) if result else {})
    intent = prediction.get("intent") or (result or {}).get("intent")
    if intent == "outlier_detection":
        intent = "outlier_analysis"
    chart_name = prediction.get("catalog_chart") or (result or {}).get("chart", {}).get("type")
    chart = CATALOG_CHARTS.get(chart_name, None)
    policy = INTENT_CHART_POLICY.get(intent, {})

    if chart is None:
        chart = policy.get("default", "table")

    n_rows = profile.get("n_rows")
    if n_rows is not None and int(n_rows) <= 1:
        chart = "none"
    elif intent == "multivariate_analysis" and profile.get("has_two_metrics") and chart_name == "quadrant_scatter":
        chart = "scatter_quadrant"

    chart = normalize_chart_type(chart, intent=intent,
                                 x=(result or {}).get("chart", {}).get("x"),
                                 y=(result or {}).get("chart", {}).get("y")) if chart != "none" else "none"
    return {
        "chart": chart,
        "render": chart != "none",
        "intent": intent,
        "question_id": prediction.get("question_id"),
        "notes": policy.get("notes", []),
        "reason": "Selected from analytical intent and the returned data profile.",
    }
