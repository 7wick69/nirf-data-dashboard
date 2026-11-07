"""Run the catalogue intent benchmark and produce a machine-readable report.

Run from the project root:
    python tests/benchmark/run_level1_benchmark.py

This stage tests classifier/template retrieval. It deliberately does not call
Ollama or claim that analytics operations are implemented merely because their
intent can be classified.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chatbot.intelligence.catalog import QUESTION_CATALOG
from chatbot.intelligence.learner import QuestionLearner

COVERAGE_PATH = HERE / "operation_coverage.json"
REPORT_DIR = HERE / "reports"


def main() -> int:
    learner = QuestionLearner()
    cases = []
    per_question = {}

    for item in QUESTION_CATALOG:
        texts = [(item["question"], "canonical")]
        texts.extend((example, f"paraphrase_{i}")
                     for i, example in enumerate(item.get("examples", []), start=1))
        question_results = []
        for text, case_type in texts:
            prediction = learner.predict(text)
            passed = (
                prediction.get("status") == "success"
                and prediction.get("intent") == item["intent"]
                and prediction.get("question_id") == item["id"]
            )
            row = {
                "question_id": item["id"],
                "case_type": case_type,
                "text": text,
                "expected_intent": item["intent"],
                "predicted_intent": prediction.get("intent"),
                "predicted_question_id": prediction.get("question_id"),
                "nearest_question_id": prediction.get("nearest_question_id"),
                "status": prediction.get("status"),
                "confidence": prediction.get("confidence"),
                "similarity": prediction.get("similarity"),
                "pass": bool(passed),
            }
            cases.append(row)
            question_results.append(row)
        per_question[item["id"]] = {
            "intent": item["intent"],
            "operation": item["operation"],
            "pass": all(x["pass"] for x in question_results),
            "cases_passed": sum(x["pass"] for x in question_results),
            "cases_total": len(question_results),
        }

    coverage = json.loads(COVERAGE_PATH.read_text(encoding="utf-8"))
    status_counts = Counter(v["status"] for v in coverage["operations"].values())
    question_status_counts = Counter(
        coverage["operations"][item["operation"]]["status"]
        for item in QUESTION_CATALOG
    )
    intent_counts = defaultdict(lambda: {"passed": 0, "total": 0})
    for item, row in zip(
        [x for q in QUESTION_CATALOG for x in [q] * (1 + len(q.get("examples", [])))],
        cases,
    ):
        intent_counts[item["intent"]]["total"] += 1
        intent_counts[item["intent"]]["passed"] += int(row["pass"])

    passed_cases = sum(row["pass"] for row in cases)
    report = {
        "catalogue_questions": len(QUESTION_CATALOG),
        "text_cases_tested": len(cases),
        "text_cases_passed": passed_cases,
        "text_case_accuracy": round(passed_cases / len(cases), 4) if cases else 0,
        "catalogue_questions_all_cases_passed": sum(v["pass"] for v in per_question.values()),
        "catalogue_questions_with_any_failure": [k for k, v in per_question.items() if not v["pass"]],
        "per_intent": dict(intent_counts),
        "operation_coverage_unique_operations": len(coverage["operations"]),
        "operation_coverage_by_unique_operation": dict(status_counts),
        "operation_coverage_by_question": dict(question_status_counts),
        "operation_coverage_note": coverage["scope"],
        "cases": cases,
        "question_results": per_question,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "level1_benchmark_report.json"
    md_path = REPORT_DIR / "level1_benchmark_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# NIRF Level 1 benchmark report",
        "",
        f"- Official catalogue questions: **{len(QUESTION_CATALOG)}**",
        f"- Classifier/template cases: **{len(cases)}**",
        f"- Correct cases: **{passed_cases}/{len(cases)} ({report['text_case_accuracy']:.1%})**",
        f"- Questions with all examples passing: **{report['catalogue_questions_all_cases_passed']}/40**",
        "",
        "## Classifier results by intent",
        "",
        "| Intent | Passed | Total | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for intent, values in sorted(intent_counts.items()):
        pct = values["passed"] / values["total"] if values["total"] else 0
        lines.append(f"| {intent} | {values['passed']} | {values['total']} | {pct:.1%} |")
    lines += [
        "",
        "## Analytics operation coverage audit",
        "",
        "| Status | Unique operations | Questions |",
        "|---|---:|---:|",
    ]
    for status in ("implemented", "partial", "not_implemented"):
        lines.append(f"| {status} | {status_counts.get(status, 0)} | {question_status_counts.get(status, 0)} |")
    lines += [
        "",
        "> Important: classifier accuracy is not end-to-end analytics accuracy. The operation coverage is a static audit against the current analytics-engine dispatch/methods. A question is not considered fully implemented until query parsing, slot validation, analytics, chart rendering and explanation are each verified.",
        "",
        "## Failed classification/template cases",
        "",
    ]
    failures = [row for row in cases if not row["pass"]]
    if failures:
        for row in failures:
            lines.append(
                f"- **{row['question_id']} ({row['case_type']})** expected "
                f"`{row['expected_intent']}`, got `{row['predicted_intent']}` "
                f"(template `{row['predicted_question_id']}`): {row['text']}"
            )
    else:
        lines.append("None.")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Questions: {len(QUESTION_CATALOG)}")
    print(f"Classifier/template cases: {passed_cases}/{len(cases)} passed ({report['text_case_accuracy']:.1%})")
    print(f"Questions with all cases passing: {report['catalogue_questions_all_cases_passed']}/40")
    print("Unique operation coverage:", dict(status_counts))
    print("Question-level operation coverage:", dict(question_status_counts))
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    return 0 if passed_cases == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
