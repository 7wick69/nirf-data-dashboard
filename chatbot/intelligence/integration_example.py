"""
Minimal integration example for the existing Streamlit Ask tab.

Do not replace your existing analytics engine with this module. Use this
learner to classify the question, then pass the validated intent/slots to your
existing query_pipeline and analytics_engine.
"""
from functools import lru_cache
from .learner import QuestionLearner
from .visualization_policy import select_visualization
from .answer_validator import validate_result

@lru_cache(maxsize=1)
def get_question_learner():
    return QuestionLearner()

def understand_question(question: str):
    learner = get_question_learner()
    prediction = learner.predict(question)
    chart = select_visualization(prediction)
    return {"prediction": prediction, "visualization": chart}

def validate_analytics_output(result, prediction, visualization):
    return validate_result(result, prediction, visualization)
