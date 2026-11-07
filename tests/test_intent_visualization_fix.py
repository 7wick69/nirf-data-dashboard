from chatbot.intelligence import QuestionLearner, select_visualization
from chatbot.visualization_policy import profile_result
from chatbot.chart_specs import normalize_chart_type, SUPPORTED_CHARTS


def test_known_ranking_questions_are_not_rejected():
    learner = QuestionLearner()
    prompts = [
        "Who is ranked first in Engineering in 2025?",
        "Show the top 10 institutions in Engineering in 2025.",
    ]
    for prompt in prompts:
        prediction = learner.predict(prompt)
        assert prediction["status"] == "success", prediction
        assert prediction["intent"] == "ranking", prediction


def test_unseen_ranking_wording_can_use_broad_intent_without_template_match():
    learner = QuestionLearner()
    prediction = learner.predict("Give me the Engineering leaderboard for 2025.")
    assert prediction["status"] == "success"
    assert prediction["intent"] == "ranking"


def test_horizontal_bar_is_a_supported_chart_family():
    chart_type = normalize_chart_type("sorted_horizontal_bar")
    assert chart_type == "horizontal_bar"
    assert chart_type in SUPPORTED_CHARTS


def test_legacy_intelligence_import_uses_canonical_visualization_policy():
    result = {
        "status": "success",
        "intent": "geographical_analysis",
        "chart": {"type": "map", "x": "state", "y": "institution_count",
                  "data": [{"state": "Karnataka", "institution_count": 5},
                           {"state": "Tamil Nadu", "institution_count": 8}]},
    }
    prediction = {"status": "success", "intent": "geographical_analysis",
                  "catalog_chart": "state_count_bar"}
    rec = select_visualization(prediction, profile_result(result), result)
    assert rec["chart"] == "horizontal_bar"
    assert rec["render"] is True
