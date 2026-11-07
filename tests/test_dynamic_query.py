from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chatbot.dynamic_query import DynamicQueryEngine, DynamicQueryError, PLAN_SCHEMA
from chatbot.visualization_engine import NIRFVisualizationEngine

DATA = ROOT / "data" / "clean" / "data_nirf_clean2.csv"


@pytest.fixture(scope="module")
def engine():
    frame = pd.read_csv(DATA)
    return DynamicQueryEngine(frame)


def plan(operation, **overrides):
    base = {
        "status": "ready", "clarification": "", "operation": operation,
        "category": "Engineering", "year": 2025,
        "start_year": None, "end_year": None, "use_latest_year": False,
        "states": [], "institutions": [], "rank_min": None, "rank_max": None,
        "metric": "score", "metric_x": None, "metric_y": None,
        "metric_filters": [], "group_by": None, "aggregation": "mean",
        "top_n": 20, "sort_by": "score", "sort_order": "descending",
        "ranked_only": True, "bins": 12,
    }
    base.update(overrides)
    return base


def test_dynamic_ranking_and_render(engine):
    result = engine.execute_plan(plan("ranking", states=["Karnataka"], top_n=10))
    assert result["status"] == "success"
    assert len(result["data"]) <= 10
    assert all(r["state"] == "Karnataka" for r in result["data"])
    fig = NIRFVisualizationEngine().create_figure(result, chart_type="horizontal_bar")["figure"]
    assert len(fig.data) >= 1


def test_dynamic_numeric_filter(engine):
    result = engine.execute_plan(plan(
        "filter", states=["Karnataka"],
        metric_filters=[{"metric": "RPC", "operator": ">=", "value": 40}],
        top_n=50,
    ))
    assert result["status"] == "success"
    assert all(r["RPC"] >= 40 for r in result["data"])
    assert all(r["state"] == "Karnataka" for r in result["data"])


def test_dynamic_state_aggregate(engine):
    result = engine.execute_plan(plan(
        "aggregate", group_by="state", metric="TLR", aggregation="mean", top_n=15,
    ))
    assert result["status"] == "success"
    assert result["chart"]["type"] == "horizontal_bar"
    assert result["statistics"]["group_by"] == "state"
    assert all("metric_value" in r for r in result["data"])
    NIRFVisualizationEngine().create_figure(result, chart_type="horizontal_bar")


def test_dynamic_year_trend(engine):
    result = engine.execute_plan(plan(
        "trend", year=None, start_year=2017, end_year=2025,
        metric="score", aggregation="mean", category="Engineering",
    ))
    years = [r["year"] for r in result["data"]]
    assert years == sorted(years)
    assert len(years) >= 2
    assert result["chart"]["type"] == "line"
    NIRFVisualizationEngine().create_figure(result, chart_type="line")


def test_dynamic_correlation(engine):
    result = engine.execute_plan(plan(
        "correlation", metric=None, metric_x="RPC", metric_y="TLR",
    ))
    assert result["statistics"]["sample_size"] >= 2
    assert result["statistics"]["pearson_correlation"] is not None
    assert result["chart"]["type"] == "scatter"
    NIRFVisualizationEngine().create_figure(result, chart_type="scatter")


def test_dynamic_distribution(engine):
    result = engine.execute_plan(plan("distribution", metric="PR"))
    assert result["statistics"]["count"] > 0
    assert result["chart"]["type"] == "histogram"
    NIRFVisualizationEngine().create_figure(result, chart_type="histogram")


def test_dynamic_state_comparison(engine):
    result = engine.execute_plan(plan(
        "compare", metric="score", group_by="state", states=["Karnataka", "Tamil Nadu"],
    ))
    assert result["chart"]["type"] == "box"
    states = {r["state"] for r in result["data"]}
    assert states == {"Karnataka", "Tamil Nadu"}
    NIRFVisualizationEngine().create_figure(result, chart_type="box")


def test_latest_year_resolution(engine):
    result = engine.execute_plan(plan("ranking", year=None, use_latest_year=True, top_n=5))
    assert result["status"] == "success"
    assert all(r["year"] == engine.years[-1] for r in result["data"])


def test_missing_year_requires_clarification(engine):
    raw = plan("ranking", year=None, use_latest_year=False)
    with pytest.raises(DynamicQueryError, match="specify a year"):
        engine.validate_plan(raw)


def test_rejects_unsafe_group_field(engine):
    raw = plan("aggregate", group_by="__import__", metric="score")
    with pytest.raises(DynamicQueryError, match="Unsupported grouping field"):
        engine.validate_plan(raw)


def test_rejects_unknown_category(engine):
    raw = plan("ranking", category="Unknown category")
    with pytest.raises(DynamicQueryError, match="Unknown NIRF category"):
        engine.validate_plan(raw)


def test_rejects_metric_filter_injection(engine):
    raw = plan("filter", metric_filters=[{"metric": "score", "operator": "; DROP TABLE", "value": 1}])
    with pytest.raises(DynamicQueryError, match="comparison operator"):
        engine.validate_plan(raw)


def test_handles_clarification_and_out_of_scope(engine):
    clarify = engine.execute_plan({"status": "needs_clarification", "clarification": "Which year?"})
    off_topic = engine.execute_plan({"status": "out_of_scope", "clarification": "This is outside NIRF."})
    assert clarify["status"] == "needs_clarification"
    assert off_topic["status"] == "out_of_scope"


def test_plan_question_uses_strict_json_schema(engine, monkeypatch):
    import sys
    import types
    response = SimpleNamespace(message=SimpleNamespace(content='''{
      "status":"ready","clarification":"","operation":"ranking",
      "category":"Engineering","year":2025,"start_year":null,"end_year":null,
      "use_latest_year":false,"states":["Karnataka"],"institutions":[],
      "rank_min":null,"rank_max":null,"metric":"score","metric_x":null,"metric_y":null,
      "metric_filters":[],"group_by":null,"aggregation":"mean","top_n":5,
      "sort_by":"rank_num","sort_order":"ascending","ranked_only":true,"bins":12
    }'''))
    captured = {}

    def fake_chat(**kwargs):
        captured.update(kwargs)
        return response

    fake_ollama = types.ModuleType("ollama")
    fake_ollama.chat = fake_chat
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)
    parsed = engine.plan_question("Show top Engineering institutions in Karnataka for 2025")
    assert parsed["status"] == "ready"
    assert parsed["category"] == "Engineering"
    assert captured["format"] == PLAN_SCHEMA
    assert captured["options"]["temperature"] == 0


def test_dynamic_count_aggregate_does_not_require_metric(engine):
    result = engine.execute_plan(plan(
        "aggregate", metric=None, group_by="state", aggregation="count", top_n=20,
    ))
    assert result["status"] == "success"
    assert all("institution_count" in row for row in result["data"])


def test_dynamic_route_preserves_exact_catalogue_match():
    from chatbot.dynamic_query import should_use_dynamic_query
    known = {
        "status": "success", "similarity": 1.0, "confidence": 0.45,
        "question_id": "GEO01", "operation": "top_n_count_by_state",
    }
    assert should_use_dynamic_query(known) is False


def test_dynamic_route_catches_unseen_query_even_if_classifier_guesses_intent():
    from chatbot.dynamic_query import should_use_dynamic_query
    unseen = {
        "status": "success", "similarity": 0.31, "confidence": 0.67,
        "question_id": "OUT04", "operation": "metric_rank_outliers",
    }
    assert should_use_dynamic_query(unseen) is True


def test_supplemental_rank_examples_keep_top10_query_on_catalogue_path():
    from chatbot.intelligence.learner import QuestionLearner
    from chatbot.dynamic_query import should_use_dynamic_query
    learner = QuestionLearner()
    prediction = learner.predict("Show the top 10 institutions in Engineering in 2025.")
    assert prediction["status"] == "success"
    assert prediction["intent"] == "ranking"
    assert prediction["operation"] == "ranking"
    assert should_use_dynamic_query(prediction) is False


def test_low_match_correlation_question_uses_dynamic_fallback():
    from chatbot.intelligence.learner import QuestionLearner
    from chatbot.dynamic_query import should_use_dynamic_query
    learner = QuestionLearner()
    prediction = learner.predict("Compare average TLR across states for Engineering in 2025")
    assert should_use_dynamic_query(prediction) is True
