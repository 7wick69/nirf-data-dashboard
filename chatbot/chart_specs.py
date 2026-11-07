"""Chart family aliases and presentation defaults for the NIRF chatbot."""
from __future__ import annotations

# Names emitted by analytics_engine.py, question_catalog.py and the learned
# visualization policy are normalized here to builders implemented by the renderer.
CHART_ALIASES = {
    "ranking": "dot_plot",
    "ranking_leaderboard": "dot_plot",
    "rank_trend": "line",
    "rank_line_inverted": "line",
    "line_trend": "line",
    "small_multiples_or_top_n": "line",
    "diverging_horizontal_bar": "diverging_bar",
    "sorted_horizontal_bar": "horizontal_bar",
    "state_count_bar": "horizontal_bar",
    "ordered_bar": "horizontal_bar",
    "quartile_bar": "horizontal_bar",
    "correlation_bar": "horizontal_bar",
    "parameter_gap_bar": "horizontal_bar",
    "variability_bar": "horizontal_bar",
    "state_change_bar": "diverging_bar",
    "state_new_entries_bar": "horizontal_bar",
    "membership_heatmap": "heatmap",
    "state_year_heatmap": "heatmap",
    "parameter_heatmap": "heatmap",
    "change_heatmap": "heatmap",
    "diverging_heatmap": "heatmap",
    "normalized_transition_heatmap": "transition_heatmap",
    "grouped_box_plot": "box",
    "box_plot": "box",
    "category_box": "box",
    "state_rank_box": "box",
    "state_box": "box",
    "year_box": "box",
    "box_plot_with_outlier_table": "box",
    "histogram_ecdf": "histogram",
    "scatter_with_optional_trendline": "scatter",
    "scatter_with_highlighted_outliers": "scatter",
    "quadrant_scatter": "scatter_quadrant",
    "scatter_with_quadrants": "scatter_quadrant",
    "scatter_with_inverted_rank_axis": "scatter_rank",
    "metric_rank_scatter": "scatter_rank",
    "volatility_dot_plot": "dot_plot",
    "entry_exit_table": "table",
    "movement_stacked_bar": "stacked_bar",
    "map": "map",
    "radar": "heatmap",  # Heatmap is more accurately perceived for NIRF parameters.
}

SUPPORTED_CHARTS = {
    "bar", "horizontal_bar", "diverging_bar", "line", "scatter",
    "scatter_quadrant", "scatter_rank", "histogram", "box", "map",
    "heatmap", "transition_heatmap", "stacked_bar", "dot_plot",
    "difference_plot", "sankey", "table",
}

DEFAULT_LIMITS = {
    "ranking": 15,
    "comparison": 15,
    "state": 15,
    "institutions_in_line_chart": 8,
    "heatmap_rows": 30,
}


def normalize_chart_type(chart_type: str | None, intent: str | None = None,
                         x: str | None = None, y: str | None = None) -> str:
    """Normalize legacy/policy chart names into a supported renderer builder."""
    name = str(chart_type or "table").strip().lower()
    if name in CHART_ALIASES:
        name = CHART_ALIASES[name]
    if name == "bar" and y in {"rank_num", "rank"}:
        return "dot_plot"
    if intent in {"outlier_detection", "outlier_analysis"} and x in {"display_name", "name", "institution", "institution_name"}:
        return "dot_plot"
    if name == "bar" and intent in {"trend_analysis", "rank_movement"} and "change" in {x, y}:
        return "diverging_bar"
    if name not in SUPPORTED_CHARTS:
        return "table"
    return name
