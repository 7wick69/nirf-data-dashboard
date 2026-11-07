from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "clean" / "data_nirf_clean2.csv"
sys.path.insert(0, str(ROOT))

from chatbot.analytics_engine import NIRFAnalyticsEngine
from chatbot.visualization_engine import NIRFVisualizationEngine
from chatbot.visualization_policy import profile_result, select_visualization
from chatbot.chart_specs import normalize_chart_type


@pytest.fixture(scope="module")
def engine():
    if not DATA.exists():
        pytest.skip("Clean NIRF CSV is not available at data/clean/data_nirf_clean2.csv")
    return NIRFAnalyticsEngine(pd.read_csv(DATA, low_memory=False))


@pytest.fixture(scope="module")
def renderer():
    return NIRFVisualizationEngine()


def run_operation(engine, operation, intent, **fields):
    query = {"intent": intent, "operation": operation, **fields}
    query.setdefault("states", [])
    query.setdefault("institutions", [])
    return engine.execute({"status": "ready", "query": query, "institution_keys": []})


def render(result, renderer, chart):
    prediction = {
        "status": "success",
        "intent": result["intent"],
        "operation": result["operation"],
        "catalog_chart": chart,
        "question_id": result["operation"],
    }
    policy = select_visualization(
        prediction,
        data_profile=profile_result(result),
        result=result,
    )
    assert policy["render"]
    normalized = normalize_chart_type(
        policy["chart"], result["intent"],
        result["chart"].get("x"), result["chart"].get("y"),
    )
    return renderer.create_figure(result, normalized)


def test_bi03_parameter_correlation_ranking(engine, renderer):
    result = run_operation(
        engine, "parameter_correlation_ranking", "correlation_analysis",
        category="Engineering", year=2025,
    )
    assert result["status"] == "success"
    assert result["operation"] == "parameter_correlation_ranking"
    assert len(result["data"]) >= 3
    assert result["data"][0]["absolute_pearson"] >= result["data"][-1]["absolute_pearson"]
    assert all(row["sample_size"] >= 3 for row in result["data"])
    assert render(result, renderer, "correlation_bar")["rendered_type"] == "horizontal_bar"


def test_bi05_correlation_over_time(engine, renderer):
    result = run_operation(
        engine, "correlation_over_time", "correlation_analysis",
        category="Engineering", start_year=2017, end_year=2025,
        metric_x="RPC", metric_y="score",
    )
    assert result["status"] == "success"
    assert result["operation"] == "correlation_over_time"
    assert {row["correlation_type"] for row in result["data"]} == {"Pearson", "Spearman"}
    yearly = result["statistics"]["yearly_results"]
    assert len(yearly) >= 2
    assert all(row["sample_size"] >= 3 for row in yearly)
    rendered = render(result, renderer, "correlation_trend")
    assert rendered["rendered_type"] == "line"
    assert len(rendered["figure"].data) == 2


@pytest.mark.parametrize(
    "question_id,metric_x,metric_y,chart",
    [
        ("MUL02", "RPC", "TLR", "quadrant_scatter"),
        ("MUL03", "TLR", "PR", "quadrant_scatter"),
    ],
)
def test_high_low_metric_quadrant(engine, renderer, question_id, metric_x, metric_y, chart):
    result = run_operation(
        engine, "high_low_metric_quadrant", "multivariate_analysis",
        category="Engineering", year=2025,
        question_id=question_id,
    )
    assert result["status"] == "success"
    assert result["operation"] == "high_low_metric_quadrant"
    assert result["statistics"]["metric_x"] == metric_x
    assert result["statistics"]["metric_y"] == metric_y
    assert len(result["data"]) > 0
    assert len(result["chart"]["data"]) > len(result["data"])
    assert render(result, renderer, chart)["rendered_type"] == "scatter_quadrant"


def test_overall_vs_parameter_change(engine, renderer):
    result = run_operation(
        engine, "overall_vs_parameter_change", "multivariate_analysis",
        category="Engineering", start_year=2020, end_year=2025,
    )
    assert result["status"] == "success"
    assert result["operation"] == "overall_vs_parameter_change"
    assert all(row["score_change"] > 0 for row in result["data"])
    delta_fields = [c for c in result["data"][0] if c.startswith("delta_")]
    assert delta_fields
    assert all(any((row.get(c) is not None and row[c] < 0) for c in delta_fields) for row in result["data"])
    rendered = render(result, renderer, "change_heatmap")
    assert rendered["rendered_type"] == "heatmap"


def test_parameter_gap(engine, renderer):
    result = run_operation(
        engine, "parameter_gap", "multivariate_analysis",
        category="Engineering", year=2025,
    )
    assert result["status"] == "success"
    assert result["operation"] == "parameter_gap"
    assert len(result["data"]) > 0
    gaps = [row["parameter_gap"] for row in result["data"]]
    assert gaps == sorted(gaps, reverse=True)
    assert all(row["parameters_observed"] >= 2 for row in result["data"])
    assert render(result, renderer, "parameter_gap_bar")["rendered_type"] == "horizontal_bar"
