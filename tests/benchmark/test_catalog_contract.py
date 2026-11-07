from collections import Counter
import json
from pathlib import Path

import pytest

from chatbot.intelligence.catalog import QUESTION_CATALOG, INTENT_DESCRIPTIONS


def test_catalog_contains_exactly_40_unique_questions():
    assert len(QUESTION_CATALOG) == 40
    ids = [item["id"] for item in QUESTION_CATALOG]
    assert len(ids) == len(set(ids))


def test_catalog_has_eight_groups_of_five():
    prefixes = [item["id"][:3] for item in QUESTION_CATALOG]
    counts = Counter(prefixes)
    assert counts == {
        "TS0": 5,
        "GEO": 5,
        "UNI": 5,
        "BI0": 5,
        "MUL": 5,
        "DIS": 5,
        "TRA": 5,
        "OUT": 5,
    }


def test_every_question_has_required_catalog_fields():
    required = {
        "id", "intent", "question", "examples",
        "operation", "chart", "required_slots",
    }
    for item in QUESTION_CATALOG:
        assert required.issubset(item), item.get("id")
        assert item["question"].strip()
        assert item["operation"].strip()
        assert isinstance(item["examples"], list)
        assert len(item["examples"]) >= 3
        assert isinstance(item["required_slots"], list)
        assert item["intent"] in INTENT_DESCRIPTIONS


def test_catalog_question_ids_follow_annexure_groups():
    expected = (
        [f"TS{i:02d}" for i in range(1, 6)]
        + [f"GEO{i:02d}" for i in range(1, 6)]
        + [f"UNI{i:02d}" for i in range(1, 6)]
        + [f"BI{i:02d}" for i in range(1, 6)]
        + [f"MUL{i:02d}" for i in range(1, 6)]
        + [f"DIST{i:02d}" for i in range(1, 6)]
        + [f"TRANS{i:02d}" for i in range(1, 6)]
        + [f"OUT{i:02d}" for i in range(1, 6)]
    )
    assert [item["id"] for item in QUESTION_CATALOG] == expected


def test_operation_coverage_accounts_for_every_catalogue_operation():
    coverage_path = Path(__file__).with_name("operation_coverage.json")
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    catalogue_operations = {item["operation"] for item in QUESTION_CATALOG}
    assert catalogue_operations == set(coverage["operations"])
    assert len(catalogue_operations) == 37


def test_operation_coverage_statuses_are_explicit_and_valid():
    coverage_path = Path(__file__).with_name("operation_coverage.json")
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    statuses = {item["status"] for item in coverage["operations"].values()}
    assert statuses <= {"implemented", "partial", "not_implemented"}
    counts = Counter(
        coverage["operations"][item["operation"]]["status"]
        for item in QUESTION_CATALOG
    )
    assert counts.get("implemented", 0) == 40
    assert counts.get("partial", 0) == 0
    assert counts.get("not_implemented", 0) == 0
