import pytest

from chatbot.intelligence.catalog import QUESTION_CATALOG
from chatbot.intelligence.learner import QuestionLearner


@pytest.fixture(scope="module")
def learner():
    # Train once for the module, not once per example.
    return QuestionLearner()


def _cases():
    cases = []
    for item in QUESTION_CATALOG:
        cases.append((item, item["question"], "canonical"))
        for index, example in enumerate(item.get("examples", []), start=1):
            cases.append((item, example, f"paraphrase-{index}"))
    return cases


@pytest.mark.parametrize(
    "item,text,case_type",
    _cases(),
    ids=[f"{item['id']}-{case_type}" for item, _, case_type in _cases()],
)
def test_catalog_questions_and_paraphrases_predict_expected_intent(
    learner, item, text, case_type
):
    prediction = learner.predict(text)

    assert prediction["status"] == "success", (
        f"{item['id']} ({case_type}) was rejected: {prediction}"
    )
    assert prediction["intent"] == item["intent"], (
        f"{item['id']} ({case_type}) expected {item['intent']}, "
        f"got {prediction['intent']} for {text!r}"
    )
    assert prediction.get("question_id") == item["id"], (
        f"{item['id']} ({case_type}) matched the wrong catalogue template: "
        f"{prediction.get('question_id')}"
    )


def test_learner_reports_catalog_coverage_count(learner):
    assert len(learner.rows) >= 160
