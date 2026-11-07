"""Orchestration layer for the NIRF AI Analyst chatbot.

The learner predicts intent, the query pipeline validates filters, the
analytics engine computes deterministic results, and the canonical
visualization policy selects a chart only after the result exists.
"""
from __future__ import annotations

import json
import os
from typing import Any

import ollama
import pandas as pd

from .query_parser import (
    NIRFQuery,
    check_clarification,
    interpret_question,
    select_visualization as query_visualization,
)
from .query_pipeline import validate_query, df
from .analytics_engine import NIRFAnalyticsEngine
from .visualization_engine import NIRFVisualizationEngine
from .intelligence import QuestionLearner, validate_result
from .visualization_policy import profile_result, select_visualization
from .chart_specs import normalize_chart_type
from .dynamic_query import DynamicQueryEngine, DynamicQueryError, should_use_dynamic_query


METRIC_GLOSSARY = {
    "TLR": "Teaching, Learning & Resources",
    "RPC": "Research and Professional Practice",
    "GO": "Graduation Outcomes",
    "OI": "Outreach and Inclusivity",
    "PR": "Perception",
    "QNR": "Quantitative Research",
    "QLR": "Qualitative Research",
    "SFC": "Student and Faculty Contribution",
}

MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:latest")
analytics = NIRFAnalyticsEngine(df)
visualizer = NIRFVisualizationEngine()
question_learner = QuestionLearner()
dynamic_query_engine = DynamicQueryEngine(df, model=MODEL)
conversation_context = None
DYNAMIC_SIMILARITY_THRESHOLD = 0.55
DYNAMIC_CONFIDENCE_FLOOR = 0.30

ANALYTICS_INTENTS = {
    "rank_movement": "trend_analysis",
    "outlier_analysis": "outlier_detection",
}
SUPPORTED_ANALYTICS_INTENTS = {
    "ranking",
    "comparison",
    "trend_analysis",
    "geographical_analysis",
    "distribution_analysis",
    "correlation_analysis",
    "multivariate_analysis",
    "transition_analysis",
    "outlier_detection",
    "statistical_summary",
}


def _merge_question_analysis(parsed_query, question_analysis: dict[str, Any]):
    """Merge learned intent/slots into the parsed query without stale filters."""
    query = parsed_query.model_dump()
    predicted_intent = question_analysis.get("intent")
    intent = ANALYTICS_INTENTS.get(predicted_intent, predicted_intent)
    if intent not in SUPPORTED_ANALYTICS_INTENTS:
        raise ValueError(f"Unsupported learned intent: {predicted_intent}")

    predicted_slots = question_analysis.get("slots", {}) or {}
    intent_changed = query.get("intent") != intent

    if intent_changed:
        # Clear intent-specific values inherited from the previous turn.
        for field in ("metric", "metric_x", "metric_y", "aggregation", "group_by"):
            if field in query:
                query[field] = [] if field == "group_by" else None
        for field in ("rank_limit", "exact_rank", "start_year", "end_year"):
            if field in query and predicted_slots.get(field) is None:
                query[field] = None
        if "states" in query and not predicted_slots.get("states"):
            query["states"] = []
        # Do not carry a previously selected institution into a new analysis.
        for field in ("institutions", "institution_keys"):
            if field in query:
                query[field] = []

    query["intent"] = intent

    for field in ("year", "start_year", "end_year", "rank_limit", "exact_rank"):
        if predicted_slots.get(field) is not None:
            query[field] = predicted_slots[field]

    if predicted_slots.get("states"):
        query["states"] = predicted_slots["states"]

    metrics = predicted_slots.get("metrics", []) or []
    if len(metrics) >= 2:
        query["metric_x"], query["metric_y"] = metrics[:2]
        query["metric"] = None
    elif metrics and query.get("metric") is None:
        query["metric"] = metrics[0]

    # The question learner supplies the operation label separately from the
    # broad intent. Use the category's available years as defaults for catalogue
    # questions that request a period but do not name endpoint years.
    operation = question_analysis.get("operation")
    period_operations = {
        "mean_score_by_year", "top10_consistency", "state_distribution_by_year",
        "state_representation_change", "new_entries_by_state",
        "score_dispersion_by_year", "median_iqr_over_time",
        "correlation_over_time", "overall_vs_parameter_change",
        "rank_band_transition_count", "top10_entries_exits", "same_band_percentage",
        "rank_movement_extremes", "rank_decline", "movement_counts_by_year",
        "score_change_outliers", "rank_volatility_outliers",
    }
    if operation in period_operations and query.get("category"):
        category_rows = analytics.df[
            analytics.df["category"].astype("string").str.casefold()
            == str(query["category"]).casefold()
        ]
        if "rank_status" in category_rows.columns:
            category_rows = category_rows[
                category_rows["rank_status"].astype("string").str.casefold() == "ranked"
            ]
        available_years = sorted(
            int(y) for y in category_rows["year"].dropna().unique()
        )
        if available_years:
            if query.get("start_year") is None:
                query["start_year"] = available_years[0]
            if query.get("end_year") is None:
                query["end_year"] = available_years[-1]

    # GEO04 is a one-year two-state distribution comparison. If no year is
    # stated, use the latest year with published ranked records in the category.
    if operation == "compare_state_rank_distributions":
        if not query.get("states"):
            query["states"] = ["Karnataka", "Tamil Nadu"]
        if query.get("year") is None and query.get("category"):
            category_rows = analytics.df[
                analytics.df["category"].astype("string").str.casefold()
                == str(query["category"]).casefold()
            ]
            if "rank_status" in category_rows.columns:
                category_rows = category_rows[
                    category_rows["rank_status"].astype("string").str.casefold() == "ranked"
                ]
            available_years = pd.to_numeric(category_rows["year"], errors="coerce").dropna()
            if not available_years.empty:
                query["year"] = int(available_years.max())

    # Catalogue semantics supply deliberate metric-pair defaults where the
    # wording describes a concept rather than spelling out both abbreviations.
    question_id = question_analysis.get("question_id")
    if operation == "correlation_over_time":
        query["metric_x"] = query.get("metric_x") or "RPC"
        query["metric_y"] = query.get("metric_y") or "score"
        query["metric"] = None
    elif operation == "high_low_metric_quadrant":
        if question_id == "MUL03":
            query["metric_x"], query["metric_y"] = "TLR", "PR"
        else:
            query["metric_x"], query["metric_y"] = "RPC", "TLR"
        query["metric"] = None
    elif operation == "metric_difference_outliers":
        # Annexure OUT02 asks for research versus teaching in that order.
        query["metric_x"], query["metric_y"] = "RPC", "TLR"
        query["metric"] = None
    elif operation == "research_score_residual_outliers":
        query["metric"] = "RPC"
    elif operation == "score_change_outliers":
        query["metric"] = query.get("metric") or "score"

    query["visualization"] = query_visualization(intent)
    return check_clarification(NIRFQuery.model_validate(query))


def _supported_chart_type(result: dict[str, Any], visualization_policy: dict[str, Any]):
    """Normalize one policy recommendation through the shared chart registry."""
    if not visualization_policy or not visualization_policy.get("render", False):
        return None

    recommendation = visualization_policy.get("chart")
    if not recommendation or recommendation == "none":
        return None

    chart = result.get("chart") or {}
    normalized = normalize_chart_type(
        recommendation,
        intent=result.get("intent") or visualization_policy.get("intent"),
        x=chart.get("x"),
        y=chart.get("y"),
    )
    return normalized


def build_takeaway(result: dict[str, Any], query: dict[str, Any]):
    intent = result.get("intent")
    records = result.get("data", [])
    category = query.get("category") or "the selected category"
    year = query.get("year")

    if intent == "ranking" and records:
        ranked = sorted(
            records,
            key=lambda record: float(record.get("rank_num") or 999999),
        )
        first = ranked[0]
        name = first.get("display_name") or "The institution"
        rank = first.get("rank_num")
        score = first.get("score")
        rank_label = f"{rank:g}" if isinstance(rank, (int, float)) else rank
        score_text = f", with a score of {score}" if score is not None else ""
        year_text = f" for {year}" if year is not None else ""

        if len(ranked) == 1:
            return f"{name} holds rank #{rank_label} in {category}{year_text}{score_text}."
        return f"{name} leads the {category} rankings{year_text} at rank #{rank_label}{score_text}."

    return None


def _process_dynamic_chat_message(question: str, previous_context=None):
    """Execute an unseen question through a strict JSON plan and deterministic pandas."""
    try:
        result = dynamic_query_engine.answer_question(
            question,
            context=previous_context,
            model=MODEL,
        )
        if result.get("status") != "success":
            status = result.get("status", "needs_clarification")
            return {
                "status": status,
                "answer": result.get("message") or "Please clarify the NIRF filters for this question.",
                "context": previous_context,
                "result": None,
                "figure": None,
                "takeaway": None,
                "chart_type": "none",
                "question_analysis": {
                    "status": "dynamic",
                    "intent": "dynamic_analysis",
                    "question_id": None,
                    "operation": "dynamic_query_plan",
                },
            }

        chart = result.get("chart") or {}
        chart_type = normalize_chart_type(
            chart.get("type"),
            intent=result.get("intent"),
            x=chart.get("x"),
            y=chart.get("y"),
        )
        figure = None
        if chart_type != "none":
            rendered = visualizer.create_figure(result, chart_type=chart_type)
            figure = rendered["figure"]

        answer = generate_explanation(question, result)
        return {
            "status": "success",
            "answer": answer,
            "context": {"dynamic_plan": result.get("dynamic_plan")},
            "result": result,
            "figure": figure,
            "takeaway": None,
            "chart_type": chart_type,
            "visualization_policy": {
                "chart": chart_type,
                "render": figure is not None,
                "reason": "Dynamic chart selected from the validated result schema.",
            },
            "question_analysis": {
                "status": "dynamic",
                "intent": result.get("intent"),
                "question_id": None,
                "operation": result.get("operation"),
                "similarity_route": "unseen_question_fallback",
            },
        }
    except (DynamicQueryError, ValueError) as error:
        return {
            "status": "needs_clarification",
            "answer": str(error),
            "context": previous_context,
            "result": None,
            "figure": None,
            "takeaway": None,
            "chart_type": "none",
            "question_analysis": {"status": "dynamic", "intent": None, "question_id": None},
        }
    except Exception as error:
        return {
            "status": "error",
            "answer": f"Unable to process the dynamic NIRF question: {error}",
            "context": previous_context,
            "result": None,
            "figure": None,
            "takeaway": None,
            "chart_type": "none",
        }


def process_chat_message(question: str, previous_context=None):
    """Process one Streamlit chat turn and return answer, verified data and figure."""
    updated_context = previous_context
    result = None
    figure = None
    takeaway = None
    chart_type = "none"

    try:
        # 1. Learn the analytical intent and extract likely slots.
        question_analysis = question_learner.predict(question)
        if should_use_dynamic_query(
            question_analysis,
            similarity_threshold=DYNAMIC_SIMILARITY_THRESHOLD,
            confidence_floor=DYNAMIC_CONFIDENCE_FLOOR,
        ):
            return _process_dynamic_chat_message(question, previous_context)

        if question_analysis.get("status") != "success":
            return {
                "status": "needs_clarification",
                "answer": question_analysis.get("clarification")
                or "Please enter a question about NIRF analysis.",
                "context": previous_context,
                "result": None,
                "figure": None,
                "takeaway": None,
                "chart_type": "none",
                "question_analysis": question_analysis,
            }

        # 2. Parse the question, then merge the ML intent and slots.
        parsed_query = interpret_question(question, previous_context)
        parsed_query = _merge_question_analysis(parsed_query, question_analysis)
        updated_context = parsed_query.model_dump()

        # 3. Validate filters against the actual dataset.
        validated_query = validate_query(parsed_query, question)
        if validated_query.get("status") != "ready":
            issues = validated_query.get("issues", [])
            if not issues:
                issues = [
                    parsed_query.clarification_question
                    or "Please provide more information."
                ]
            return {
                "status": "needs_clarification",
                "answer": "\n".join(f"- {issue}" for issue in issues),
                "context": updated_context,
                "result": None,
                "figure": None,
                "takeaway": None,
                "chart_type": "none",
                "question_analysis": question_analysis,
            }

        # 4. Pass the matched Annexure A operation into the analytics request.
        # This metadata is attached after Pydantic query validation so it does
        # not need to be part of the user-facing NIRFQuery schema.
        engine_request = dict(validated_query)
        engine_request["query"] = dict(validated_query["query"])
        engine_request["query"]["operation"] = question_analysis.get("operation")
        engine_request["query"]["question_id"] = question_analysis.get("question_id")

        # 5. Execute deterministic Pandas analytics.
        result = analytics.execute(engine_request)
        if result.get("status") != "success":
            return {
                "status": "error",
                "answer": result.get("message", "Analytics could not complete this request."),
                "context": updated_context,
                "result": result,
                "figure": None,
                "takeaway": None,
                "chart_type": "none",
                "question_analysis": question_analysis,
            }

        # 6. Select visualization only after the result and its actual data exist.
        data_profile = profile_result(result)
        visualization_policy = select_visualization(
            question_analysis,
            data_profile=data_profile,
            result=result,
        )

        validation = validate_result(result, question_analysis, visualization_policy)
        if not validation["valid"]:
            return {
                "status": "validation_error",
                "answer": "; ".join(validation["errors"]),
                "context": previous_context,
                "result": None,
                "figure": None,
                "takeaway": None,
                "chart_type": "none",
                "question_analysis": question_analysis,
            }

        takeaway = build_takeaway(result, validated_query["query"])
        chart_type = _supported_chart_type(result, visualization_policy) or "none"
        if chart_type != "none":
            rendered = visualizer.create_figure(result, chart_type=chart_type)
            figure = rendered["figure"]

        # 7. LLM explains only the already-verified Python result.
        answer = generate_explanation(question, result)
        return {
            "status": "success",
            "answer": answer,
            "context": updated_context,
            "result": result,
            "figure": figure,
            "takeaway": takeaway,
            "chart_type": chart_type,
            "visualization_policy": visualization_policy,
            "question_analysis": question_analysis,
        }

    except Exception as error:
        return {
            "status": "error",
            "answer": f"Unable to process the request: {error}",
            "context": updated_context,
            "result": result,
            "figure": figure,
            "takeaway": takeaway,
            "chart_type": chart_type,
        }


def generate_explanation(question: str, result: dict[str, Any]) -> str:
    """Generate a concise explanation using only verified analytical evidence."""
    evidence = {
        "title": result.get("title"),
        "summary": result.get("summary"),
        "statistics": result.get("statistics", {}),
        "sample_records": result.get("data", [])[:8],
        "notes": result.get("notes", []),
        "metric_definitions": METRIC_GLOSSARY,
    }
    system_prompt = """You are the explanation generator for NIRF AI Analyst.
You receive verified analytical results from Python.
STRICT ACCURACY RULES:
1. Use only the supplied analytical results.
2. Never invent numerical values or percentages.
3. Never redefine NIRF parameters.
4. Use the exact metric definitions supplied in metric_definitions.
5. Never describe TLR as Teaching Learning Ratio.
6. Never describe RPC as Research Publication Citations.
7. Never describe PR as Placement.
8. Never describe OI as Outcomes.
9. Do not label a score high, low or moderate without an appropriate comparison.
10. Do not claim causation from correlation.
11. Do not invent institutional performance explanations.
12. Mention limitations only when supplied by Python.
RESPONSE FORMAT:
1. Answer the question directly in 1-3 sentences.
2. Do not generate lengthy headings or sections.
3. Do not repeat the complete institution table.
4. Do not reproduce all numerical statistics.
5. Do not include generic limitations unless relevant.
6. Do not invent performance explanations.
7. The application separately displays the chart, takeaway and data table.
8. Avoid repeating information already shown in the chart and takeaway.
All numerical calculations must come from Python."""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Original question: {question}\n\n"
                    f"Verified analytical results:\n{json.dumps(evidence, default=str)}"
                ),
            },
        ],
        options={"temperature": 0.2},
        stream=False,
    )
    return response.message.content


def display_result(result: dict[str, Any]):
    print("\n" + "=" * 60)
    print("NIRF ANALYTICAL RESULT")
    print("=" * 60)
    print("\nTitle:", result["title"])
    print("\nSummary:", result["summary"])
    print("\nStatistics:")
    for key, value in result.get("statistics", {}).items():
        print(f"  {key}: {value}")
    records = result.get("data", [])
    if records:
        print("\nResult Preview:")
        print(pd.DataFrame(records).head(10).to_string(index=False))
    print("\nVisualization:", result.get("chart", {}).get("type"))
    for note in result.get("notes", []):
        print("-", note)


def process_question(question: str):
    """Legacy console entry point; use the same integrated Streamlit pipeline."""
    global conversation_context
    output = process_chat_message(question, conversation_context)
    conversation_context = output.get("context", conversation_context)
    if output.get("result"):
        display_result(output["result"])
    print("\nNIRF AI Analyst:")
    print(output.get("answer", "No response was generated."))
    if output.get("takeaway"):
        print("\nTakeaway:", output["takeaway"])
    return output


def main():
    global conversation_context
    print("=" * 60)
    print("             NIRF AI ANALYST")
    print("=" * 60)
    print("Model:", MODEL)
    print("Dataset records:", len(df))
    print("Type 'reset' to clear conversation.")
    print("Type 'exit' to quit.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if question.lower() == "reset":
            conversation_context = None
            print("Conversation context cleared.")
            continue
        if question:
            try:
                process_question(question)
            except Exception as error:
                print("\nError processing question:", error)
                print("Please check the query, dataset and local Ollama connection.")


if __name__ == "__main__":
    main()
