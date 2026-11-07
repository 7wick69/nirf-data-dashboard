"""Reviewed feedback storage for incremental model learning."""
from .learner import QuestionLearner

def record_feedback(question: str, corrected_intent: str,
                    question_id: str | None = None,
                    chart: str | None = None,
                    operation: str | None = None,
                    required_slots: list[str] | None = None,
                    learner: QuestionLearner | None = None):
    """Record a human-reviewed correction and retrain the supplied learner."""
    model = learner or QuestionLearner()
    training_info = model.learn(
        question=question,
        corrected_intent=corrected_intent,
        question_id=question_id,
        chart=chart,
        operation=operation,
        required_slots=required_slots,
    )
    return {"recorded": True, **training_info}
