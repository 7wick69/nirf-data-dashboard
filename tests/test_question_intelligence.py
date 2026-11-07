from chatbot.intelligence import QuestionLearner, select_visualization, validate_result
from chatbot.intelligence.catalog import QUESTION_CATALOG

def test_catalog_has_all_40_questions():
    assert len(QUESTION_CATALOG) == 40
    assert len({q["id"] for q in QUESTION_CATALOG}) == 40

def test_catalog_has_nine_model_intents_for_eight_project_sections():
    intents = {q["intent"] for q in QUESTION_CATALOG}
    assert len(intents) == 9  # rank movement is a separate analytical intent

def test_model_trains_and_predicts():
    model = QuestionLearner(feedback_path="/tmp/nirf_test_feedback.jsonl")
    result = model.predict("Show the yearly average NIRF score trend")
    assert result["status"] == "success"
    assert result["intent"] in {"trend_analysis", "statistical_summary"}

def test_uncertain_question_does_not_force_chart():
    model = QuestionLearner(feedback_path="/tmp/nirf_test_feedback.jsonl")
    result = model.predict("Tell me a random fact about space travel")
    chart = select_visualization(result)
    assert chart["render"] is False

def test_empty_success_result_is_rejected():
    out = validate_result({"status": "success", "data": []})
    assert out["valid"] is False
