from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_CANDIDATES = [ROOT / "data" / "clean" / "data_nirf_clean2.csv", ROOT / "data" / "data_nirf_clean2.csv"]
DATA = next((candidate for candidate in DATA_CANDIDATES if candidate.exists()), None)
sys.path.insert(0, str(ROOT))

from chatbot.analytics_engine import NIRFAnalyticsEngine
from chatbot.visualization_engine import NIRFVisualizationEngine
from chatbot.visualization_policy import profile_result, select_visualization
from chatbot.chart_specs import normalize_chart_type


@pytest.fixture(scope="module")
def engine():
    if DATA is None:
        pytest.skip("Place data_nirf_clean2.csv in data/clean/ or data/.")
    return NIRFAnalyticsEngine(pd.read_csv(DATA, low_memory=False))


@pytest.fixture(scope="module")
def renderer():
    return NIRFVisualizationEngine()


def execute(engine, operation, intent, **fields):
    query = {"intent": intent, "operation": operation, **fields}
    if "states" not in query:
        query["states"] = []
    if "institutions" not in query:
        query["institutions"] = []
    return engine.execute({"status": "ready", "query": query, "institution_keys": []})


def render_for_operation(result, renderer, chart_name):
    prediction = {
        "status": "success",
        "intent": result["intent"],
        "operation": result["operation"],
        "catalog_chart": chart_name,
        "question_id": result["operation"],
    }
    profile = profile_result(result)
    policy = select_visualization(prediction, data_profile=profile, result=result)
    assert policy["render"] is True
    normalized = normalize_chart_type(
        policy["chart"], result["intent"], result["chart"].get("x"), result["chart"].get("y")
    )
    return renderer.create_figure(result, normalized)


def test_uni02_rank_distribution(engine, renderer):
    result = execute(engine, "rank_distribution", "distribution_analysis",
                     category="Overall", year=2025)
    assert result["status"] == "success"
    assert result["operation"] == "rank_distribution"
    assert result["statistics"]["ranked_institutions"] == sum(r["institution_count"] for r in result["data"])
    assert {"1-10", "11-50", "51-100"}.issubset({r["rank_band"] for r in result["data"]})
    rendered = render_for_operation(result, renderer, "rank_band_bar")
    assert rendered["rendered_type"] == "horizontal_bar"


def test_uni04_score_quartile_share(engine, renderer):
    result = execute(engine, "score_quartile_share", "distribution_analysis",
                     category="Engineering", year=2025, metric="score")
    assert result["status"] == "success"
    assert result["operation"] == "score_quartile_share"
    assert abs(result["statistics"]["percentage_total"] - 100.0) < 0.02
    assert all(0 <= row["percentage"] <= 100 for row in result["data"])
    rendered = render_for_operation(result, renderer, "quartile_bar")
    assert rendered["rendered_type"] == "horizontal_bar"


def test_uni05_score_spread_by_category(engine, renderer):
    result = execute(engine, "score_spread_by_category", "distribution_analysis",
                     year=2025, metric="score")
    assert result["status"] == "success"
    assert result["statistics"]["categories_compared"] >= 5
    assert len({r["category"] for r in result["data"]}) >= 5
    rendered = render_for_operation(result, renderer, "category_box")
    assert rendered["rendered_type"] == "box"


def test_dist01_top10_vs_rest(engine, renderer):
    result = execute(engine, "top10_vs_rest_distribution", "distribution_analysis",
                     category="Engineering", year=2025, metric="score")
    assert result["status"] == "success"
    assert {r["rank_group"] for r in result["data"]} == {"Top 10", "Remaining institutions"}
    assert result["statistics"]["top10_count"] > 0
    assert result["statistics"]["remaining_count"] > 0
    rendered = render_for_operation(result, renderer, "grouped_box")
    assert rendered["rendered_type"] == "box"


def test_dist02_score_dispersion_by_year(engine, renderer):
    result = execute(engine, "score_dispersion_by_year", "distribution_analysis",
                     category="Engineering", start_year=2017, end_year=2025, metric="score")
    assert result["status"] == "success"
    assert result["statistics"]["years_compared"] >= 2
    assert all(row["count"] > 0 for row in result["statistics"]["year_summaries"])
    rendered = render_for_operation(result, renderer, "year_box")
    assert rendered["rendered_type"] == "box"


def test_dist03_parameter_variability(engine, renderer):
    result = execute(engine, "parameter_variability", "statistical_summary",
                     category="Engineering", year=2025)
    assert result["status"] == "success"
    assert result["statistics"]["parameters_compared"] >= 3
    assert result["data"][0]["standard_deviation"] >= result["data"][-1]["standard_deviation"]
    rendered = render_for_operation(result, renderer, "variability_bar")
    assert rendered["rendered_type"] == "horizontal_bar"


def test_dist04_median_iqr_over_time(engine, renderer):
    result = execute(engine, "median_iqr_over_time", "distribution_analysis",
                     category="Engineering", start_year=2017, end_year=2025, metric="score")
    assert result["status"] == "success"
    assert len(result["data"]) >= 2
    assert all(row["q1"] <= row["median"] <= row["q3"] for row in result["data"])
    rendered = render_for_operation(result, renderer, "median_iqr_trend")
    assert rendered["rendered_type"] == "line"
    assert any(getattr(trace, "fill", None) == "tonexty" for trace in rendered["figure"].data)


def test_dist05_score_distribution_by_state(engine, renderer):
    result = execute(engine, "score_distribution_by_state", "distribution_analysis",
                     category="Overall", year=2025, metric="score")
    assert result["status"] == "success"
    assert result["statistics"]["states_compared"] >= 10
    assert len({r["state"] for r in result["data"]}) >= 10
    rendered = render_for_operation(result, renderer, "state_box")
    assert rendered["rendered_type"] == "box"
    assert rendered["visible_record_count"] <= rendered["record_count"]


def test_cross_category_metric_does_not_use_a_single_category_filter(engine):
    result = execute(engine, "score_spread_by_category", "distribution_analysis",
                     category="Engineering", year=2025, metric="score")
    assert result["status"] == "success"
    assert result["statistics"]["categories_compared"] > 1


def test_missing_parameter_values_are_not_imputed_as_zero(engine):
    result = execute(engine, "parameter_variability", "statistical_summary",
                     category="Research", year=2025)
    assert result["status"] == "success"
    metrics = {row["metric"] for row in result["data"]}
    assert metrics.issubset({"TLR", "RPC", "GO", "OI", "PR", "QNR", "QLR", "SFC"})
    assert all(row["count"] >= 2 for row in result["data"])