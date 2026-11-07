
import re
import json

from .data_loader import load_dataset
from .nirf_metadata import NIRFMetadata
from .entity_resolver import InstitutionResolver, normalize_name
from .query_parser import interpret_question


# --------------------------------------------------
# 1. Initialize the NIRF data system
# --------------------------------------------------

df = load_dataset()

metadata = NIRFMetadata(df)
resolver = InstitutionResolver(df)

print("NIRF dataset loaded:", len(df), "records")


# --------------------------------------------------
# 2. Canonicalize category and state names
# --------------------------------------------------

def resolve_category(value):

    if not value:
        return None

    for category in metadata.categories:
        if normalize_name(category) == normalize_name(value):
            return category

    return None


def resolve_state(value):

    if not value:
        return None

    for state in metadata.states:
        if normalize_name(state) == normalize_name(value):
            return state

    return None


def normalize_entity_text(value):
    value = re.sub(
        r"(?i)(?<=\w)['’]s\b",
        "",
        str(value)
    )
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def build_institution_aliases():
    aliases = {"iit", "nit", "iiit", "iim", "srm"}

    if hasattr(resolver, "alias_to_keys"):
        aliases.update(
            normalize_entity_text(alias)
            for alias in resolver.alias_to_keys
        )

    abbreviation_prefixes = (
        ("indian institute of technology ", "iit "),
        ("national institute of technology ", "nit "),
        ("international institute of information technology ", "iiit "),
        ("indian institute of information technology ", "iiit "),
        ("indian institute of management ", "iim ")
    )

    for column in [
        "name",
        "display_name",
        "name_in_source"
    ]:
        for value in df[column].dropna().astype(str).unique():
            normalized = normalize_entity_text(value)
            if not normalized:
                continue

            aliases.add(normalized)
            for full_prefix, abbreviation in abbreviation_prefixes:
                if normalized.startswith(full_prefix):
                    aliases.add(
                        abbreviation + normalized[len(full_prefix):]
                    )

            if re.search(r"\bsrm\b", normalized):
                aliases.add("srm")

    return aliases


INSTITUTION_ALIASES = build_institution_aliases()


# --------------------------------------------------
# 3. Find institution names explicitly mentioned
#    in the user's natural-language question
# --------------------------------------------------

def find_institution_mentions(question):

    normalized_question = normalize_entity_text(question)

    aliases = sorted(
        INSTITUTION_ALIASES,
        key=len,
        reverse=True
    )

    mentions = []
    occupied_spans = []

    for alias in aliases:

        pattern = (
            r"(?<!\w)" +
            re.escape(alias) +
            r"(?!\w)"
        )

        for match in re.finditer(pattern, normalized_question):

            span = match.span()

            # Avoid matching a short alias inside
            # an already detected longer institution name.
            overlaps = any(
                span[0] < end and span[1] > start
                for start, end in occupied_spans
            )

            if overlaps:
                continue

            result = resolver.resolve(alias)

            if result["status"] in [
                "matched",
                "ambiguous",
                "needs_clarification"
            ]:
                mentions.append(alias)
                occupied_spans.append(span)

    # Preserve unique mentions in their detected order
    return list(dict.fromkeys(mentions))


# --------------------------------------------------
# 4. Validate and resolve the query
# --------------------------------------------------

def validate_query(query, question=""):

    data = query.model_dump()

    issues = []
    resolved_institutions = []
    institution_keys = []
    resolved_names = []

    # Resolve category
    category = resolve_category(data.get("category"))

    if data.get("category") and not category:
        issues.append(
            f"Unknown category: {data['category']}"
        )

    data["category"] = category

    # Resolve states
    canonical_states = []

    for state in data.get("states", []):

        resolved = resolve_state(state)

        if resolved:
            canonical_states.append(resolved)
        else:
            issues.append(
                f"Unknown state: {state}"
            )

    data["states"] = list(dict.fromkeys(canonical_states))

    # Validate institution names
    institution_names = list(
        data.get("institutions", [])
    )

    institution_names.extend(
        find_institution_mentions(question)
    )

    institution_names = list(
        dict.fromkeys(institution_names)
    )

    location_state = (
        data["states"][0]
        if len(data["states"]) == 1
        else None
    )

    for name in institution_names:

        result = resolver.resolve(
            name,
            state=location_state
        )

        resolved_institutions.append(result)

        if result["status"] == "matched":

            institution_keys.append(
                result["institution"]["institute_key"]
            )
            resolved_names.append(
                result["institution"].get("name")
                or result["institution"].get("display_name")
                or name
            )

        elif result["status"] in [
            "ambiguous",
            "needs_clarification"
        ]:

            issues.append(
                f"Institution '{name}' requires clarification."
            )
            resolved_names.append(name)

        else:

            issues.append(
                f"Institution '{name}' could not be found."
            )
            resolved_names.append(name)

    institution_keys = list(dict.fromkeys(institution_keys))
    data["institutions"] = list(dict.fromkeys(resolved_names))

    # Check year and category availability
    category = data.get("category")

    years_to_validate = []

    if data.get("year") is not None:
        years_to_validate.append(data["year"])

    if data.get("start_year") is not None:
        years_to_validate.append(data["start_year"])

    if data.get("end_year") is not None:
        years_to_validate.append(data["end_year"])

    if category:

        available_years = metadata.years_for_category(category)

        for year in years_to_validate:

            if year not in available_years:
                issues.append(
                    f"{category} has no records for {year}."
                )

    # Check numerical parameters
    metrics = []

    for field in ["metric", "metric_x", "metric_y"]:

        value = data.get(field)

        if value:
            metrics.append(value)

    if category:
        for year in years_to_validate or [None]:

            validation_errors = metadata.validate(
                category=category,
                year=year,
                metrics=metrics
            )

            issues.extend(validation_errors)

    # Check valid year range
    start = data.get("start_year")
    end = data.get("end_year")

    if start is not None and end is not None:

        if start > end:
            issues.append(
                "Start year cannot be greater than end year."
            )

    # Correlation requires two distinct attributes
    if data["intent"] == "correlation_analysis":

        if not data.get("metric_x") or not data.get("metric_y"):
            issues.append(
                "Correlation requires two numerical parameters."
            )

        elif data["metric_x"] == data["metric_y"]:
            issues.append(
                "Choose two different parameters for correlation."
            )

    if (
        data["intent"] == "ranking"
        and re.search(
            r"\b(IIT|NIT|IIIT|IIM|SRM)\b",
            question,
            re.IGNORECASE
        )
        and not institution_keys
    ):
        issues.append(
            "An institution was mentioned but could not be "
            "resolved. Please specify its full name or campus."
        )

    # Update readiness state
    needs_clarification = bool(
        issues or data.get("needs_clarification")
    )

    if needs_clarification:
        status = "needs_clarification"
    else:
        status = "ready"

    return {
        "status": status,
        "query": data,
        "institution_keys": institution_keys,
        "institution_matches": resolved_institutions,
        "issues": list(dict.fromkeys(issues))
    }


# --------------------------------------------------
# 5. Interactive pipeline
# --------------------------------------------------

def main():

    print("\n" + "=" * 55)
    print("       NIRF INTEGRATED QUERY PIPELINE")
    print("=" * 55)

    previous_context = None

    while True:

        question = input("\nYou: ").strip()

        if question.lower() in ["exit", "quit"]:
            break

        if question.lower() == "reset":

            previous_context = None
            print("Context cleared.")
            continue

        if not question:
            continue

        try:

            # Parse natural language
            parsed_query = interpret_question(
                question,
                previous_context
            )

            # Validate against actual dataset
            result = validate_query(
                parsed_query,
                question
            )

            previous_context = parsed_query.model_dump()

            print("\nPIPELINE RESULT:")
            print(
                json.dumps(
                    result,
                    indent=4,
                    ensure_ascii=False
                )
            )

        except Exception as error:

            print("\nPipeline error:", error)


if __name__ == "__main__":
    main()
