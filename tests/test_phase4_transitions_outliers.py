from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "clean" / "data_nirf_clean2.csv"
sys.path.insert(0, str(ROOT))

from chatbot.analytics_engine import NIRFAnalyticsEngine, AnalyticsError
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


def execute(engine, operation, intent, **fields):
    query = {"intent": intent, "operation": operation, **fields}
    query.setdefault("states", [])
    query.setdefault("institutions", [])
    return engine.execute({"status": "ready", "query": query, "institution_keys": []})


def render(result, renderer, catalog_chart):
    prediction = {
        "status": "success",
        "intent": "outlier_analysis" if result["intent"] == "outlier_detection" else result["intent"],
        "operation": result.get("operation"),
        "catalog_chart": catalog_chart,
        "question_id": result.get("operation"),
    }
    policy = select_visualization(prediction, profile_result(result), result)
    assert policy["render"] is True
    chart_type = normalize_chart_type(
        policy["chart"], result["intent"], result["chart"].get("x"), result["chart"].get("y")
    )
    return renderer.create_figure(result, chart_type)


def test_trans01_rank_band_transition_count(engine, renderer):
    result = execute(engine, "rank_band_transition_count", "transition_analysis",
                     category="Engineering", start_year=2020, end_year=2025)
    assert result["status"] == "success"
    assert result["operation"] == "rank_band_transition_count"
    assert result["statistics"]["institutions_compared"] > 0
    assert result["statistics"]["moved_51_100_to_top50"] >= 0
    assert {"start_band", "end_band", "institution_count"} <= set(result["chart"]["data"][0])
    assert render(result, renderer, "transition_heatmap")["rendered_type"] == "transition_heatmap"


def test_trans02_top10_entries_exits(engine, renderer):
    result = execute(engine, "top10_entries_exits", "transition_analysis",
                     category="Engineering", start_year=2017, end_year=2025)
    assert result["status"] == "success"
    assert result["operation"] == "top10_entries_exits"
    assert result["statistics"]["year_pairs_compared"] >= 1
    assert result["statistics"]["total_entrants"] >= 0
    assert result["statistics"]["total_exits"] >= 0
    assert "transition" in result["data"][0]
    assert render(result, renderer, "entry_exit_table")["rendered_type"] == "table"


def test_trans03_same_band_percentage(engine, renderer):
    result = execute(engine, "same_band_percentage", "transition_analysis",
                     category="Engineering", start_year=2020, end_year=2025)
    assert result["status"] == "success"
    assert result["operation"] == "same_band_percentage"
    pct = result["statistics"]["same_band_percentage"]
    assert 0 <= pct <= 100
    assert result["data"]
    assert all(0 <= row["transition_pct"] <= 100 for row in result["data"])
    assert render(result, renderer, "normalized_transition_heatmap")["rendered_type"] == "transition_heatmap"


def test_trans04_rank_movement_extremes(engine, renderer):
    result = execute(engine, "rank_movement_extremes", "rank_movement",
                     category="Engineering", start_year=2020, end_year=2025)
    assert result["status"] == "success"
    assert result["operation"] == "rank_movement_extremes"
    assert result["statistics"]["institutions_compared"] > 0
    assert all("change" in row for row in result["data"])
    assert render(result, renderer, "diverging_rank_movement")["rendered_type"] == "diverging_bar"


def test_trans05_movement_counts_by_year(engine, renderer):
    result = execute(engine, "movement_counts_by_year", "transition_analysis",
                     category="Engineering", start_year=2017, end_year=2025)
    assert result["status"] == "success"
    assert result["operation"] == "movement_counts_by_year"
    assert result["statistics"]["year_pairs_compared"] >= 1
    assert {"Moved up", "Moved down", "Retained rank"}.issuperset({r["movement_status"] for r in result["data"]})
    assert render(result, renderer, "movement_stacked_bar")["rendered_type"] == "stacked_bar"


def test_out01_research_score_residual_outliers(engine, renderer):
    result = execute(engine, "research_score_residual_outliers", "outlier_detection",
                     category="Engineering", year=2025)
    assert result["status"] == "success"
    assert result["operation"] == "research_score_residual_outliers"
    assert result["statistics"]["observations"] >= 5
    assert "is_outlier" in result["chart"]["data"][0]
    assert render(result, renderer, "outlier_scatter")["rendered_type"] == "scatter"


def test_out02_metric_difference_outliers(engine, renderer):
    result = execute(engine, "metric_difference_outliers", "outlier_detection",
                     category="Engineering", year=2025, metric_x="RPC", metric_y="TLR")
    assert result["status"] == "success"
    assert result["operation"] == "metric_difference_outliers"
    assert result["statistics"]["sample_size"] >= 5
    assert all(row["change"] == pytest.approx(row["RPC"] - row["TLR"]) for row in result["data"])
    assert render(result, renderer, "difference_plot")["rendered_type"] == "difference_plot"


def test_out03_score_change_outliers(engine, renderer):
    result = execute(engine, "score_change_outliers", "outlier_detection",
                     category="Engineering", start_year=2017, end_year=2025, metric="score")
    assert result["status"] == "success"
    assert result["operation"] == "score_change_outliers"
    assert result["statistics"]["matched_year_pairs"] >= 5
    assert "score_change" in result["chart"]["data"][0]
    assert render(result, renderer, "change_outlier_plot")["rendered_type"] == "box"


def test_out04_metric_rank_outliers_uses_strongest_available_parameter(engine, renderer):
    result = execute(engine, "metric_rank_outliers", "outlier_detection",
                     category="Engineering", year=2025)
    assert result["status"] == "success"
    assert result["operation"] == "metric_rank_outliers"
    assert result["statistics"]["observations"] >= 5
    assert all(row["parameters_observed"] >= 2 for row in result["chart"]["data"])
    assert render(result, renderer, "metric_rank_scatter")["rendered_type"] == "scatter_rank"


def test_out05_rank_volatility_outliers(engine, renderer):
    result = execute(engine, "rank_volatility_outliers", "outlier_detection",
                     category="Engineering", start_year=2017, end_year=2025)
    assert result["status"] == "success"
    assert result["operation"] == "rank_volatility_outliers"
    assert result["statistics"]["institutions_compared"] >= 5
    assert all(row["transitions"] >= 2 for row in result["chart"]["data"])
    assert render(result, renderer, "volatility_dot_plot")["rendered_type"] == "dot_plot"


def test_rank_transition_uses_only_institutions_present_in_both_endpoint_years(engine):
    result = execute(engine, "rank_band_transition_count", "transition_analysis",
                     category="Engineering", start_year=2020, end_year=2025)
    assert sum(row["institution_count"] for row in result["chart"]["data"]) == result["statistics"]["institutions_compared"]


def test_outlier_missing_parameters_are_not_filled_with_zero(engine):
    with pytest.raises(AnalyticsError, match="pairwise-complete"):
        execute(engine, "metric_difference_outliers", "outlier_detection",
                category="Research", year=2025, metric_x="RPC", metric_y="TLR")
