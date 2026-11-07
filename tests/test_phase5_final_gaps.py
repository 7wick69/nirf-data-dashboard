from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "clean" / "data_nirf_clean2.csv"
sys.path.insert(0, str(ROOT))

from chatbot.analytics_engine import NIRFAnalyticsEngine
from chatbot.chart_specs import normalize_chart_type
from chatbot.intelligence.learner import QuestionLearner
from chatbot.visualization_engine import NIRFVisualizationEngine
from chatbot.visualization_policy import profile_result, select_visualization


@pytest.fixture(scope="module")
def engine():
    if not DATA.exists():
        pytest.skip("Clean NIRF CSV is not available")
    return NIRFAnalyticsEngine(pd.read_csv(DATA, low_memory=False))


@pytest.fixture(scope="module")
def renderer():
    return NIRFVisualizationEngine()


def execute(engine, operation, intent, **fields):
    query = {"intent": intent, "operation": operation, **fields}
    query.setdefault("states", [])
    query.setdefault("institutions", [])
    return engine.execute({"status": "ready", "query": query, "institution_keys": []})


def policy_chart(result, question_id, catalog_chart):
    prediction = {
        "status": "success",
        "intent": "rank_movement" if question_id == "TS03" else "geographical_analysis",
        "question_id": question_id,
        "operation": result["operation"],
        "catalog_chart": catalog_chart,
    }
    profile = profile_result(result)
    policy = select_visualization(prediction, profile, result)
    return policy, normalize_chart_type(
        policy["chart"], result["intent"], result["chart"].get("x"), result["chart"].get("y")
    )


def test_ts03_rank_decline_returns_only_decliners_and_renders(engine, renderer):
    result = execute(
        engine, "rank_decline", "trend_analysis",
        category="Engineering", start_year=2017, end_year=2025, rank_limit=10,
    )
    assert result["status"] == "success"
    assert result["operation"] == "rank_decline"
    assert result["data"]
    assert all(row["rank_change"] < 0 for row in result["data"])
    assert all(row["decline_positions"] > 0 for row in result["data"])
    assert result["data"] == sorted(result["data"], key=lambda r: r["decline_positions"], reverse=True)
    assert result["statistics"]["declined_institutions"] >= len(result["data"])
    assert result["statistics"]["largest_decline_positions"] == max(
        row["decline_positions"] for row in result["data"]
    )

    policy, chart_type = policy_chart(result, "TS03", "diverging_rank_movement")
    assert policy["render"] is True
    assert chart_type == "diverging_bar"
    output = renderer.create_figure(result, chart_type)
    assert output["rendered_type"] == "diverging_bar"


def test_geo04_two_state_rank_distribution_uses_institution_level_records(engine, renderer):
    result = execute(
        engine, "compare_state_rank_distributions", "geographical_analysis",
        category="Engineering", year=2025, states=["Karnataka", "Tamil Nadu"],
    )
    assert result["status"] == "success"
    assert result["operation"] == "compare_state_rank_distributions"
    assert result["chart"]["x"] == "state"
    assert result["chart"]["y"] == "rank_num"
    assert {row["state"] for row in result["data"]} == {"Karnataka", "Tamil Nadu"}
    assert all(row["rank_num"] is not None for row in result["data"])

    summaries = result["statistics"]["state_summaries"]
    assert len(summaries) == 2
    assert all(row["institution_count"] > 0 for row in summaries)
    assert all(row["q1_rank"] <= row["median_rank"] <= row["q3_rank"] for row in summaries)
    assert result["statistics"]["lower_median_rank_state"] in {"Karnataka", "Tamil Nadu"}

    policy, chart_type = policy_chart(result, "GEO04", "state_rank_box")
    assert policy["render"] is True
    assert chart_type == "box"
    output = renderer.create_figure(result, chart_type)
    assert output["rendered_type"] == "box"
    assert output["figure"].layout.yaxis.autorange == "reversed"


def test_geo04_defaults_to_the_two_official_states_when_filter_is_omitted(engine):
    result = execute(
        engine, "compare_state_rank_distributions", "geographical_analysis",
        category="Engineering", year=2025,
    )
    assert {row["state"] for row in result["data"]} == {"Karnataka", "Tamil Nadu"}


def test_geo04_rejects_one_or_more_than_two_states(engine):
    with pytest.raises(ValueError, match="exactly two states"):
        execute(engine, "compare_state_rank_distributions", "geographical_analysis",
                category="Engineering", year=2025, states=["Karnataka"])
    with pytest.raises(ValueError, match="exactly two states"):
        execute(engine, "compare_state_rank_distributions", "geographical_analysis",
                category="Engineering", year=2025,
                states=["Karnataka", "Tamil Nadu", "Kerala"])


def test_last_two_catalogue_questions_are_learned_as_their_official_ids():
    learner = QuestionLearner()
    ts03 = learner.predict("Which institutions have experienced the largest decline in ranking?")
    geo04 = learner.predict("How do the rankings of institutions in Karnataka compare with those in Tamil Nadu?")
    assert (ts03["status"], ts03["intent"], ts03["question_id"], ts03["operation"]) == (
        "success", "rank_movement", "TS03", "rank_decline"
    )
    assert (geo04["status"], geo04["intent"], geo04["question_id"], geo04["operation"]) == (
        "success", "geographical_analysis", "GEO04", "compare_state_rank_distributions"
    )
