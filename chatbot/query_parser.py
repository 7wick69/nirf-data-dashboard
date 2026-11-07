import json
import os
import re

import ollama

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:latest")


# -------------------------------------------------
# 1. Query schema
# -------------------------------------------------

class NIRFQuery(BaseModel):

    model_config = ConfigDict(extra="forbid")

    intent: Literal[
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
        "general_question",
        "clarification"
    ]

    category: str | None = None
    year: int | None = Field(default=None, ge=2017, le=2025)

    start_year: int | None = Field(default=None, ge=2017, le=2025)
    end_year: int | None = Field(default=None, ge=2017, le=2025)

    states: list[str] = Field(default_factory=list)
    institutions: list[str] = Field(default_factory=list)

    metric: str | None = None
    metric_x: str | None = None
    metric_y: str | None = None

    group_by: list[str] = Field(default_factory=list)

    aggregation: str | None = None

    rank_limit: int | None = Field(
        default=None, ge=1, le=1000
    )
    exact_rank: int | None = Field(
        default=None,
        ge=1,
        le=1000
    )

    visualization: Literal[
        "bar", "line", "scatter", "histogram",
        "box", "map", "radar", "table",
        "sankey", "none"
    ] = "none"

    needs_clarification: bool = False
    clarification_question: str | None = None


# -------------------------------------------------
# 2. NIRF dictionaries
# -------------------------------------------------

CATEGORIES = {
    "overall": "Overall",
    "university": "University",
    "universities": "University",
    "engineering": "Engineering",
    "management": "Management",
    "pharmacy": "Pharmacy",
    "college": "College",
    "colleges": "College",
    "architecture": "Architecture",
    "law": "Law",
    "medical": "Medical",
    "medicine": "Medical",
    "dental": "Dental",
    "research": "Research",
    "agriculture": "Agriculture and Allied Sectors",
    "innovation": "Innovation",
    "open university": "Open University",
    "skill university": "Skill University"
}

STATES = {
    "karnataka": "Karnataka",
    "tamil nadu": "Tamil Nadu",
    "kerala": "Kerala",
    "telangana": "Telangana",
    "andhra pradesh": "Andhra Pradesh",
    "maharashtra": "Maharashtra",
    "delhi": "Delhi",
    "uttar pradesh": "Uttar Pradesh",
    "west bengal": "West Bengal",
    "gujarat": "Gujarat",
    "rajasthan": "Rajasthan",
    "punjab": "Punjab",
    "haryana": "Haryana",
    "bihar": "Bihar",
    "odisha": "Odisha",
    "assam": "Assam",
    "goa": "Goa"
}

STATES.update({
    "tamilnadu": "Tamil Nadu",
    "tamil-nadu": "Tamil Nadu",
    "tamil nadu": "Tamil Nadu"
})

METRICS = {
    "TLR": r"\bTLR\b|teaching learning resources",
    "RPC": r"\bRPC\b|research and professional practice",
    "GO": r"\bGO\b|graduation outcomes",
    "OI": r"\bOI\b|outreach and inclusivity",
    "PR": r"\bPR\b|perception",
    "QNR": r"\bQNR\b",
    "QLR": r"\bQLR\b",
    "SFC": r"\bSFC\b"
}


# -------------------------------------------------
# 3. NLP: Extract explicit information
# -------------------------------------------------

def extract_slots(text):

    t = text.lower()

    result = {
        "category": None,
        "year": None,
        "start_year": None,
        "end_year": None,
        "states": [],
        "rank_limit": None,
        "exact_rank": None,
        "metrics": [],
        "aggregation": None
    }

    # Extract years
    years = [
        int(x) for x in re.findall(
            r"\b(?:201[7-9]|202[0-5])\b", t
        )
    ]

    if len(years) >= 2:
        result["start_year"] = years[0]
        result["end_year"] = years[1]

    elif len(years) == 1:
        result["year"] = years[0]

    # Extract category
    for alias in sorted(CATEGORIES, key=len, reverse=True):

        if re.search(
            r"(?<!\w)" + re.escape(alias) + r"(?!\w)", t
        ):
            result["category"] = CATEGORIES[alias]
            break

    # Extract states
    for alias, canonical in STATES.items():

        if re.search(
            r"(?<!\w)" + re.escape(alias) + r"(?!\w)", t
        ):
            result["states"].append(canonical)

    # Extract metrics in the order they appear in the question
    found_metrics = []
    for metric, pattern in METRICS.items():
        for match in re.finditer(
            pattern, t, re.IGNORECASE
        ):
            found_metrics.append(
                (match.start(), metric)
            )

    # Sort by actual position in sentence
    found_metrics.sort(key=lambda x: x[0])

    # Remove duplicate metrics while preserving order
    result["metrics"] = list(dict.fromkeys(
        metric for _, metric in found_metrics
    ))

    # Extract ranking limit
    match = re.search(r"\btop\s+(\d+)\b", t)

    if match:
        result["rank_limit"] = int(match.group(1))

    # Exact rank detection
    ordinal_words = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "sixth": 6,
        "seventh": 7,
        "eighth": 8,
        "ninth": 9,
        "tenth": 10
    }

    # Example: 1st place, 2nd rank, 3rd position
    numeric_match = re.search(
        r"\b(\d+)\s*(?:st|nd|rd|th)\s+"
        r"(?:place|rank|position)\b",
        t
    )

    if numeric_match is None:
        numeric_match = re.search(
            r"\b(?:ranked|ranks?)\s+(?:in\s+)?"
            r"(\d+)\s*(?:st|nd|rd|th)\b",
            t
        )

    if numeric_match:
        result["exact_rank"] = int(numeric_match.group(1))
    else:
        for word, number in ordinal_words.items():
            patterns = [
                rf"\b{word}\s+(?:place|rank|position)\b",
                rf"\b(?:ranked|ranks?)\s+(?:in\s+)?{word}\b"
            ]
            if any(re.search(pattern, t) for pattern in patterns):
                result["exact_rank"] = number
                break

    # Extract aggregation
    if "average" in t or "mean" in t:
        result["aggregation"] = "mean"

    elif "median" in t:
        result["aggregation"] = "median"

    elif "maximum" in t:
        result["aggregation"] = "max"

    elif "minimum" in t:
        result["aggregation"] = "min"

    elif "count" in t or "number of" in t:
        result["aggregation"] = "count"

    return result


# -------------------------------------------------
# 4. Detect analytical intent
# -------------------------------------------------

def detect_intent(text):

    t = text.lower()

    if any(x in t for x in [
        "correlation", "relationship",
        "associated", "versus", " vs "
    ]):
        return "correlation_analysis"

    if any(x in t for x in [
        "distribution", "histogram"
    ]):
        return "distribution_analysis"

    if any(x in t for x in [
        "outlier", "unusually high",
        "unusual performance"
    ]):
        return "outlier_detection"

    if any(x in t for x in [
        "transition", "moved from",
        "entered the top", "exited the top",
        "rank band", "rank-band"
    ]):
        return "transition_analysis"

    if any(x in t for x in [
        "improved", "declined", "improvement",
        "over time", "between", "trend",
        "year-wise", "year wise"
    ]):
        return "trend_analysis"

    if any(x in t for x in [
        "which states", "state-wise",
        "geographical", "geographic",
        "regional distribution"
    ]):
        return "geographical_analysis"

    if any(x in t for x in [
        "compare", "comparison"
    ]):
        return "comparison"

    if any(x in t for x in [
        "parameter", "five parameters",
        "tlr and rpc", "multivariate"
    ]):
        return "multivariate_analysis"

    if (
        re.search(
            r"\b(show|find|list|get|give|identify)\b"
            r".*\b(top|bottom)\b",
            t
        )
        or any(x in t for x in [
            "highest rank",
            "lowest rank",
            "ranking",
            "ranked institutions"
        ])
    ):
        return "ranking"

    return None


# -------------------------------------------------
# 5. Detect follow-up questions
# -------------------------------------------------

def is_follow_up(text, previous_context=None):
    if previous_context is None:
        return False

    t = text.strip().lower()

    # Explicit contextual follow-ups
    contextual_prefixes = (
        "now compare ",
        "now show ",
        "now plot ",
        "now calculate ",
        "what about ",
        "compare them",
        "show them",
        "change ",
        "set the ",
        "set year ",
        "switch to ",
        "make it ",
        "instead ",
        "only ",
        "for ",
        "in ",
        "and "
    )

    if t.startswith(contextual_prefixes):
        return True

    # These are independent, complete requests.
    independent_prefixes = (
        "find ",
        "show ",
        "which ",
        "what ",
        "list ",
        "give ",
        "identify ",
        "compare ",
        "calculate ",
        "plot ",
        "how many ",
        "tell me "
    )

    if t.startswith(independent_prefixes):
        return False

    # Detect parameter-only follow-ups
    slots = extract_slots(text)
    has_explicit_slot = any([
        slots["category"] is not None,
        slots["year"] is not None,
        slots["start_year"] is not None,
        slots["end_year"] is not None,
        slots["states"],
        slots["rank_limit"] is not None,
        slots["exact_rank"] is not None,
        slots["metrics"],
        slots["aggregation"] is not None
    ])

    return has_explicit_slot


# -------------------------------------------------
# 6. Determine visualization
# -------------------------------------------------

def select_visualization(intent):

    mapping = {
        "ranking": "bar",
        "comparison": "bar",
        "trend_analysis": "line",
        "geographical_analysis": "map",
        "distribution_analysis": "histogram",
        "correlation_analysis": "scatter",
        "multivariate_analysis": "radar",
        "transition_analysis": "sankey",
        "outlier_detection": "scatter",
        "statistical_summary": "table",
        "general_question": "none",
        "clarification": "none"
    }

    return mapping.get(intent, "none")


# -------------------------------------------------
# 7. Build clarification questions
# -------------------------------------------------

def check_clarification(query):

    required = []

    if query.intent in [
        "ranking",
        "comparison",
        "distribution_analysis",
        "geographical_analysis"
    ]:

        if not query.category:
            required.append("NIRF category")

        if not query.year:
            required.append("year")

    elif query.intent == "correlation_analysis":

        if not query.category:
            required.append("NIRF category")

        if not query.year:
            required.append("year")

        if not query.metric_x or not query.metric_y:
            required.append("two numerical parameters")

    elif query.intent == "trend_analysis":

        if not query.category:
            required.append("NIRF category")

        if not query.start_year or not query.end_year:
            required.append("start year and end year")

    query.needs_clarification = bool(required)

    if required:

        query.clarification_question = (
            "Could you specify the " +
            " and ".join(required) + "?"
        )

    else:
        query.clarification_question = None

    return query


# -------------------------------------------------
# 8. LLM interpretation
# -------------------------------------------------

SYSTEM_PROMPT = """
You are the query interpretation component of NIRF AI Analyst.

Convert the user's analytical question into the supplied JSON schema.

Do not answer the question itself.

IMPORTANT RULES:

1. Never invent a year.
2. Never invent a category.
3. Never invent a state.
4. Never assume Overall or University when unspecified.
5. Extract only explicitly supported information.
6. Use the correct intent.
7. For correlation, identify metric_x and metric_y.
8. For a top N question, identify rank_limit.
9. If information is missing, leave it null.
10. Never calculate numerical results.
11. Use RPC as the dataset's research parameter column.
12. Preserve previous conversational context when
    the current message is a follow-up.
13. Only update information explicitly changed by
    the current message.
14. If the user asks for an institution at an exact
    rank or ordinal position, use exact_rank.
15. If the user asks for the top N institutions,
    use rank_limit instead.
16. Never confuse exact rank with top N.
"""


def interpret_question(question, previous_context=None):

    follow_up = is_follow_up(
        question, previous_context
    )

    if follow_up and previous_context:
        prompt = f"""Previous analytical context:{json.dumps(previous_context)}
The user's new message is a follow-up.
Preserve relevant previous filters.
Update only the information explicitly changed.
If the analytical intent changes, clear incompatible
analysis-specific parameters.
Current user message:{question}"""
    else:
        prompt = f"""This is a NEW INDEPENDENT analytical request.
Ignore all previous conversation filters and selections.
Do not inherit previous categories, years, states,
institutions or metrics.
Current user message:{question}"""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        format=NIRFQuery.model_json_schema(),
        options={"temperature": 0},
        stream=False
    )

    candidate = NIRFQuery.model_validate_json(
        response.message.content
    )

    # Extract explicit details deterministically
    slots = extract_slots(question)

    detected_intent = detect_intent(question)

    if follow_up and previous_context:
        data = previous_context.copy()
    else:
        data = NIRFQuery(
            intent=candidate.intent
        ).model_dump()

    if (
        follow_up
        and detected_intent
        and detected_intent != previous_context.get("intent")
    ):
        data["metric"] = None
        data["metric_x"] = None
        data["metric_y"] = None
        data["aggregation"] = None
        data["group_by"] = []

    # Preserve previous intent for follow-ups
    if detected_intent:
        data["intent"] = detected_intent
    elif follow_up:
        data["intent"] = previous_context["intent"]
    else:
        data["intent"] = candidate.intent

    # Apply explicitly identified entities
    for field in [
        "category",
        "year",
        "start_year",
        "end_year",
        "rank_limit",
        "exact_rank"
    ]:

        if slots[field] is not None:
            data[field] = slots[field]

        elif not follow_up:
            data[field] = None

    if slots["exact_rank"] is not None:
        data["rank_limit"] = None
    elif slots["rank_limit"] is not None:
        data["exact_rank"] = None

    if slots["states"]:
        data["states"] = slots["states"]

    elif not follow_up:
        data["states"] = []

    if slots["aggregation"]:
        data["aggregation"] = slots["aggregation"]

    # Process two numerical parameters
    if len(slots["metrics"]) >= 2:

        data["metric_x"] = slots["metrics"][0]
        data["metric_y"] = slots["metrics"][1]
        data["metric"] = None

    elif len(slots["metrics"]) == 1:

        data["metric"] = slots["metrics"][0]
        data["metric_x"] = None
        data["metric_y"] = None

    elif not follow_up:

        data["metric"] = None
        data["metric_x"] = None
        data["metric_y"] = None

    # Visualization is controlled by the analytical intent
    data["visualization"] = select_visualization(
        data["intent"]
    )

    data["group_by"] = (
        ["state"]
        if data["intent"] == "geographical_analysis"
        else data.get("group_by", [])
    )

    # Final validation
    query = NIRFQuery.model_validate(data)

    query = check_clarification(query)

    return query


# -------------------------------------------------
# 9. Interactive query engine
# -------------------------------------------------

def main():

    print("=" * 55)
    print("       NIRF NATURAL LANGUAGE QUERY ENGINE V2")
    print("=" * 55)

    print("Model:", MODEL)
    print("Context memory enabled.")
    print("Type 'reset' to clear context.")
    print("Type 'exit' to quit.")

    previous_context = None

    while True:

        question = input("\nUser question: ").strip()

        if question.lower() in ["exit", "quit"]:
            break

        if question.lower() == "reset":

            previous_context = None
            print("Conversation context cleared.")
            continue

        if not question:
            continue

        try:

            query = interpret_question(
                question,
                previous_context
            )

            print("\nGenerated analytical query:\n")

            print(query.model_dump_json(indent=4))

            previous_context = query.model_dump()

            if query.needs_clarification:

                print(
                    "\nClarification needed:",
                    query.clarification_question
                )

        except Exception as error:

            print("\nError:", error)


if __name__ == "__main__":
    main()
