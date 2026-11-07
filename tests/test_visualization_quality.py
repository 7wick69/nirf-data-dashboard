import pandas as pd
import pytest

from chatbot.visualization_engine import NIRFVisualizationEngine, VisualizationError
from chatbot.visualization_policy import profile_result, select_visualization


def ranking_result(n=100, include_score=True):
    records = []
    for i in range(1, n + 1):
        row = {
            "display_name": f"Institution {i:03d}",
            "rank_num": i,
            "state": "State A" if i % 2 else "State B",
            "city": "City",
        }
        if include_score:
            row["score"] = 100 - i * 0.5
        records.append(row)
    return {
        "status": "success", "intent": "ranking", "title": "Engineering 2025 rankings",
        "summary": "Synthetic ranking test", "data": records,
        "chart": {"type": "bar", "x": "display_name", "y": "rank_num", "data": records},
        "statistics": {}, "notes": [],
    }


def test_ranking_chart_limits_visible_points_and_preserves_result_data():
    result = ranking_result(100)
    output = NIRFVisualizationEngine().create_figure(result)
    assert output["rendered_type"] == "ranking_leaderboard"
    assert output["visible_record_count"] == 15
    assert len(result["data"]) == 100


def test_rank_only_ranking_uses_compact_dot_plot_without_text_labels():
    result = ranking_result(25, include_score=False)
    output = NIRFVisualizationEngine().create_figure(result)
    assert output["visible_record_count"] == 15
    trace = output["figure"].data[0]
    assert trace.mode == "markers"
    assert trace.text is None


def test_rank_change_uses_diverging_bar():
    records = [
        {"display_name": "A", "start_value": 100, "end_value": 40, "change": 60},
        {"display_name": "B", "start_value": 20, "end_value": 70, "change": -50},
    ]
    result = {"status": "success", "intent": "trend_analysis", "title": "Movement",
              "data": records, "chart": {"type": "bar", "x": "display_name", "y": "change", "data": records},
              "statistics": {}, "notes": []}
    out = NIRFVisualizationEngine().create_figure(result)
    assert out["rendered_type"] == "diverging_bar"
    assert list(out["figure"].data[0].x) == [-50, 60]


def test_policy_returns_supported_chart_for_catalog_alias():
    pred = {"status": "success", "intent": "geographical_analysis", "catalog_chart": "state_count_bar"}
    result = {"status": "success", "intent": "geographical_analysis", "data": [{"state": "Karnataka", "institution_count": 7}],
              "chart": {"type": "map", "x": "state", "y": "institution_count", "data": [{"state": "Karnataka", "institution_count": 7}]}}
    profile = profile_result(result)
    rec = select_visualization(pred, profile)
    assert rec["chart"] in {"horizontal_bar", "none"}


def test_single_record_policy_disables_chart():
    pred = {"status": "success", "intent": "ranking", "catalog_chart": "state_count_bar"}
    rec = select_visualization(pred, {"n_rows": 1})
    assert rec["render"] is False
    assert rec["chart"] == "none"


def test_scatter_correlation_does_not_depend_on_statsmodels():
    records = [{"display_name": f"I{i}", "rpc": i * 2.0, "tlr": i * 3.0} for i in range(1, 8)]
    result = {"status": "success", "intent": "correlation_analysis", "title": "RPC vs TLR",
              "data": records, "chart": {"type": "scatter", "x": "rpc", "y": "tlr", "data": records},
              "statistics": {"pearson_correlation": 1.0, "spearman_correlation": 1.0, "sample_size": 7}, "notes": []}
    out = NIRFVisualizationEngine().create_figure(result)
    assert out["rendered_type"] == "scatter"
    assert len(out["figure"].data) >= 2  # observations + numpy fitted trend


def test_heatmap_uses_available_parameters_only():
    records = [{"display_name": "IIT A", "TLR": 80, "RPC": 70, "GO": 60, "score": 75},
               {"display_name": "IIT B", "TLR": 75, "RPC": 65, "GO": 70, "score": 72}]
    result = {"status": "success", "intent": "multivariate_analysis", "title": "Parameters",
              "data": records, "chart": {"type": "radar", "x": "display_name", "y": None, "data": records},
              "statistics": {}, "notes": []}
    out = NIRFVisualizationEngine().create_figure(result)
    assert out["rendered_type"] == "heatmap"
    assert list(out["figure"].data[0].x) == ["TLR", "RPC", "GO"]


def test_empty_data_is_rejected():
    result = {"status": "success", "intent": "ranking", "data": [], "chart": {"type": "bar", "data": []}}
    with pytest.raises(VisualizationError):
        NIRFVisualizationEngine().create_figure(result)


def test_outlier_records_with_institution_labels_use_metric_dot_plot():
    records = [
        {"display_name": "Institution A", "RPC": 92.0, "rank_num": 4},
        {"display_name": "Institution B", "RPC": 88.0, "rank_num": 8},
        {"display_name": "Institution C", "RPC": 85.0, "rank_num": 12},
    ]
    result = {"status": "success", "intent": "outlier_detection", "title": "RPC outliers",
              "data": records, "chart": {"type": "scatter", "x": "display_name", "y": "RPC", "data": records},
              "statistics": {}, "notes": []}
    out = NIRFVisualizationEngine().create_figure(result)
    assert out["rendered_type"] == "dot_plot"
    assert out["visible_record_count"] == 3


def test_geographical_map_uses_bar_fallback_with_actual_analytics_xy_convention():
    records = [
        {"state": "Karnataka", "institution_count": 9},
        {"state": "Tamil Nadu", "institution_count": 14},
        {"state": "Maharashtra", "institution_count": 11},
    ]
    result = {"status": "success", "intent": "geographical_analysis", "title": "State counts",
              "data": records, "chart": {"type": "map", "x": "state", "y": "institution_count", "data": records},
              "statistics": {}, "notes": []}
    out = NIRFVisualizationEngine().create_figure(result)
    assert out["rendered_type"] == "horizontal_bar_fallback"
    assert list(out["figure"].data[0].x) == [9, 11, 14]
