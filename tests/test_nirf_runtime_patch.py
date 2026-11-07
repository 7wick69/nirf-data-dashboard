from chatbot.intelligence import QuestionLearner
from chatbot.visualization_policy import profile_result, select_visualization
from chatbot.chart_specs import normalize_chart_type, SUPPORTED_CHARTS
from chatbot.visualization_engine import NIRFVisualizationEngine


def _result(intent, chart_type, x, y, rows):
    return {
        "status": "success",
        "intent": intent,
        "title": "NIRF test result",
        "summary": "Synthetic result for routing verification.",
        "data": rows,
        "chart": {"type": chart_type, "x": x, "y": y, "data": rows},
        "statistics": {},
        "notes": [],
    }


def test_top_ten_question_is_ranking_and_renders_as_leaderboard():
    learner = QuestionLearner()
    prediction = learner.predict("how the top 10 institutions in Engineering in 2025.")
    assert prediction["status"] == "success"
    assert prediction["intent"] == "ranking"

    rows = [
        {"display_name": f"Institution {i}", "rank_num": i, "score": 90 - i,
         "state": "Tamil Nadu"}
        for i in range(1, 11)
    ]
    result = _result("ranking", "bar", "display_name", "rank_num", rows)
    policy = select_visualization(prediction, profile_result(result), result)
    chart_type = normalize_chart_type(policy["chart"], "ranking", "display_name", "rank_num")
    assert chart_type == "dot_plot"
    rendered = NIRFVisualizationEngine().create_figure(result, chart_type)
    assert rendered["rendered_type"] == "ranking_leaderboard"
    assert rendered["visible_record_count"] == 10


def test_rpc_tlr_question_resolves_classifier_nearest_template_conflict():
    learner = QuestionLearner()
    prediction = learner.predict(
        "How are RPC and TLR related across Engineering institutions in 2025?"
    )
    assert prediction["status"] == "success"
    assert prediction["intent"] == "correlation_analysis"
    assert prediction["intent_source"] == "nearest_template_conflict_resolution"
    assert prediction["slots"]["metrics"] == ["RPC", "TLR"]

    rows = [
        {"display_name": f"Institution {i}", "state": "Karnataka",
         "RPC": 60 + i, "TLR": 50 + 2 * i}
        for i in range(1, 21)
    ]
    result = _result("correlation_analysis", "scatter", "RPC", "TLR", rows)
    policy = select_visualization(prediction, profile_result(result), result)
    chart_type = normalize_chart_type(policy["chart"], "correlation_analysis", "RPC", "TLR")
    assert chart_type == "scatter"
    rendered = NIRFVisualizationEngine().create_figure(result, chart_type)
    assert rendered["rendered_type"] == "scatter"
    assert rendered["visible_record_count"] == 20


def test_state_count_recommendation_is_a_supported_horizontal_bar():
    learner = QuestionLearner()
    prediction = learner.predict(
        "Which states have the most institutions in the Overall top 100 in 2025"
    )
    rows = [
        {"state": "Tamil Nadu", "institution_count": 17},
        {"state": "Maharashtra", "institution_count": 11},
        {"state": "Uttar Pradesh", "institution_count": 9},
    ]
    result = _result("geographical_analysis", "map", "state", "institution_count", rows)
    policy = select_visualization(prediction, profile_result(result), result)
    chart_type = normalize_chart_type(policy["chart"], "geographical_analysis", "state", "institution_count")
    assert chart_type == "horizontal_bar"
    assert chart_type in SUPPORTED_CHARTS
    rendered = NIRFVisualizationEngine().create_figure(result, chart_type)
    assert rendered["rendered_type"] == "horizontal_bar"
    assert rendered["visible_record_count"] == 3


def test_one_ranked_institution_does_not_get_a_scatter_chart():
    learner = QuestionLearner()
    prediction = learner.predict("Who is ranked first in Engineering in 2025?")
    result = _result(
        "ranking", "bar", "display_name", "rank_num",
        [{"display_name": "IIT Madras", "rank_num": 1, "score": 88.72}],
    )
    policy = select_visualization(prediction, profile_result(result), result)
    assert policy["render"] is False
    assert policy["chart"] == "none"
