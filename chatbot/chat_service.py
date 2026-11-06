
import json
import os
import pandas as pd
import ollama

from .query_parser import interpret_question
from .query_pipeline import validate_query, df
from .analytics_engine import NIRFAnalyticsEngine
from .visualization_engine import NIRFVisualizationEngine


METRIC_GLOSSARY = {
    "TLR": "Teaching, Learning & Resources",
    "RPC": "Research and Professional Practice",
    "GO": "Graduation Outcomes",
    "OI": "Outreach and Inclusivity",
    "PR": "Perception",
    "QNR": "Quantitative Research",
    "QLR": "Qualitative Research",
    "SFC": "Student and Faculty Contribution"
}


# -----------------------------------------
# CONFIGURATION
# -----------------------------------------

MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:latest")

analytics = NIRFAnalyticsEngine(df)
visualizer = NIRFVisualizationEngine()

conversation_context = None


def build_takeaway(result, query):
    intent = result.get("intent")
    records = result.get("data", [])
    category = query.get("category") or "the selected category"
    year = query.get("year")

    if intent == "ranking" and records:
        ranked = sorted(
            records,
            key=lambda record: float(
                record.get("rank_num") or 999999
            )
        )
        first = ranked[0]
        name = first.get("display_name") or "The institution"
        rank = first.get("rank_num")
        score = first.get("score")
        rank_label = f"{rank:g}" if isinstance(rank, (int, float)) else rank
        score_text = f", with a score of {score}" if score is not None else ""

        if len(ranked) == 1:
            year_text = f" in {year}" if year is not None else ""
            return (
                f"{name} holds rank #{rank_label} in "
                f"{category}{year_text}{score_text}."
            )

        year_text = f" for {year}" if year is not None else ""
        return (
            f"{name} leads the {category} rankings{year_text} "
            f"at rank #{rank_label}{score_text}."
        )

    return None


def process_chat_message(question, previous_context=None):
    updated_context = previous_context
    result = None
    figure = None
    takeaway = None

    try:
        # 1. Interpret natural language
        parsed_query = interpret_question(
            question,
            previous_context
        )
        updated_context = parsed_query.model_dump()

        # 2. Validate against NIRF dataset
        validated_query = validate_query(
            parsed_query,
            question
        )
        if validated_query["status"] != "ready":
            issues = validated_query.get("issues", [])
            if not issues:
                issues = [
                    parsed_query.clarification_question
                    or "Please provide more information."
                ]
            return {
                "status": "needs_clarification",
                "answer": "\n".join(
                    f"- {issue}" for issue in issues
                ),
                "context": updated_context,
                "result": None,
                "figure": None,
                "takeaway": None
            }

        # 3. Execute analysis and build its visualization
        result = analytics.execute(validated_query)
        if result.get("status") != "success":
            return {
                "status": "error",
                "answer": result.get(
                    "message",
                    "Analytics could not complete this request."
                ),
                "context": updated_context,
                "result": result,
                "figure": None,
                "takeaway": None
            }

        takeaway = build_takeaway(
            result,
            validated_query["query"]
        )
        figure = visualizer.create_figure(result)["figure"]

        # 4. Explain the verified result
        answer = generate_explanation(question, result)
        return {
            "status": "success",
            "answer": answer,
            "context": updated_context,
            "result": result,
            "figure": figure,
            "takeaway": takeaway
        }
    except Exception as error:
        return {
            "status": "error",
            "answer": f"Unable to process the request: {error}",
            "context": updated_context,
            "result": result,
            "figure": figure,
            "takeaway": takeaway
        }


# -----------------------------------------
# LOCAL LLM RESPONSE GENERATION
# -----------------------------------------

def generate_explanation(question, result):

    # Send only verified analytical results to the LLM
    evidence = {
        "title": result.get("title"),
        "summary": result.get("summary"),
        "statistics": result.get("statistics", {}),
        "sample_records": result.get("data", [])[:8],
        "notes": result.get("notes", []),
        "metric_definitions": METRIC_GLOSSARY
    }

    system_prompt = """You are the explanation generator for NIRF AI Analyst.
You receive verified analytical results from Python.
STRICT ACCURACY RULES:
1. Use only the supplied analytical results.
2. Never invent numerical values or percentages.
3. Never redefine NIRF parameters.
4. Use the exact metric definitions supplied
   in metric_definitions.
5. Never describe TLR as Teaching Learning Ratio.
6. Never describe RPC as Research Publication Citations.
7. Never describe PR as Placement.
8. Never describe OI as Outcomes.
9. Do not label a score high, low or moderate
   without an appropriate comparison.
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
7. The application will separately display the chart,
   takeaway line and underlying data table.
8. Avoid repeating information that the chart
   and takeaway already display.
Explain the results clearly and concisely.
All numerical calculations must come from Python."""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": (
                    f"Original question: {question}\n\n"
                    f"Verified analytical results:\n"
                    f"{json.dumps(evidence, default=str)}"
                )
            }
        ],
        options={"temperature": 0.2},
        stream=False
    )

    return response.message.content


# -----------------------------------------
# DISPLAY ANALYTICAL RESULTS
# -----------------------------------------

def display_result(result):

    print("\n" + "=" * 60)
    print("NIRF ANALYTICAL RESULT")
    print("=" * 60)

    print("\nTitle:", result["title"])
    print("\nSummary:", result["summary"])

    print("\nStatistics:")

    for key, value in result.get(
        "statistics", {}
    ).items():

        print(f"  {key}: {value}")

    records = result.get("data", [])

    if records:

        print("\nResult Preview:")

        frame = pd.DataFrame(records)

        print(
            frame.head(10).to_string(index=False)
        )

    print("\nVisualization:", result["chart"]["type"])

    if result.get("notes"):
        print("\nNotes:")
        for note in result["notes"]:
            print("-", note)


# -----------------------------------------
# MAIN CHAT SERVICE
# -----------------------------------------

def process_question(question):

    global conversation_context

    # STEP 1: NLP interpretation

    parsed_query = interpret_question(
        question,
        conversation_context
    )

    # Preserve context for follow-up questions
    conversation_context = parsed_query.model_dump()

    print("\n[1/4] Natural-language interpretation complete.")

    # STEP 2: Dataset/entity validation

    validated_query = validate_query(
        parsed_query,
        question
    )

    print("[2/4] Dataset validation complete.")

    if validated_query["status"] != "ready":

        print("\nClarification required:")

        if validated_query.get("issues"):

            for issue in validated_query["issues"]:
                print("-", issue)

        elif parsed_query.clarification_question:

            print(parsed_query.clarification_question)

        return

    # STEP 3: Analytical execution

    print("[3/4] Executing analytical query...")

    result = analytics.execute(validated_query)

    if result.get("status") != "success":

        print("Analytics could not complete this request.")
        print(result)
        return

    display_result(result)

    # STEP 4: Natural-language explanation

    print("\n[4/4] Generating explanation...")

    explanation = generate_explanation(
        question,
        result
    )

    print("\nNIRF AI Analyst:")
    print(explanation)


# -----------------------------------------
# INTERACTIVE CHATBOT
# -----------------------------------------

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

        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if question.lower() == "reset":

            conversation_context = None
            print("Conversation context cleared.")
            continue

        if not question:
            continue

        try:

            process_question(question)

        except Exception as error:

            print("\nError processing question:", error)

            print(
                "Please check the query, dataset "
                "and local Ollama connection."
            )


if __name__ == "__main__":
    main()
