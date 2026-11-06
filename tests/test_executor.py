"""
tests/test_executor.py

Tests for Pydantic Schema, Safe Expression Evaluator, and Executor Fallback Ladder.
"""

import json
import pytest
import pandas as pd
from core.data import load_nirf_data
from core.schema import ChartSpec
from core.safe_expr import evaluate_safe_expr, UnsafeExpressionError
from core.executor import execute_spec


def test_safe_expression_evaluator():
    df = pd.DataFrame({
        "TLR": [80.0, 90.0],
        "RPC": [70.0, 85.0],
        "score": [75.0, 88.0],
    })

    # Valid arithmetic
    res = evaluate_safe_expr("(TLR + RPC) / 2", df)
    assert res.iloc[0] == 75.0
    assert res.iloc[1] == 87.5

    # Safe subtraction and scalar multiplication
    res2 = evaluate_safe_expr("score - 50", df)
    assert res2.iloc[0] == 25.0

    # Unsafe function calls must be blocked
    with pytest.raises(UnsafeExpressionError):
        evaluate_safe_expr("__import__('os').system('ls')", df)

    with pytest.raises(UnsafeExpressionError):
        evaluate_safe_expr("eval('2 + 2')", df)


def test_executor_valid_spec():
    df = load_nirf_data()

    spec = {
        "title": "Average TLR by state, 2022",
        "takeaway": "Average TLR across states in 2022",
        "steps": [
            {"op": "filter", "where": [{"field": "year", "op": "eq", "value": 2022}, {"field": "category", "op": "eq", "value": "Overall"}]},
            {"op": "aggregate", "by": ["state"], "metrics": [{"field": "TLR", "fn": "mean", "as": "avg_TLR"}]},
            {"op": "sort", "by": "avg_TLR", "order": "desc"},
            {"op": "top", "n": 5},
        ],
        "chart": {"type": "bar", "x": "state", "y": "avg_TLR"},
    }

    fig, res_df, err, takeaway = execute_spec(df, spec)
    assert err is None
    assert len(res_df) <= 5
    assert "avg_TLR" in res_df.columns
    assert fig is not None


def test_executor_empty_filter_ladder():
    df = load_nirf_data()

    # Filter with non-existent year (e.g. 1900)
    spec = {
        "title": "Empty Year Test",
        "steps": [
            {"op": "filter", "where": [{"field": "year", "op": "eq", "value": 1900}]},
        ],
        "chart": {"type": "bar", "x": "name", "y": "score"},
    }

    fig, res_df, err, takeaway = execute_spec(df, spec)
    assert err is not None
    assert "matched 0 rows" in err


def test_executor_auto_downgrade():
    df = load_nirf_data()

    # Spec requests a line chart with non-time categorical x
    spec = {
        "title": "Auto Downgrade Test",
        "steps": [
            {"op": "filter", "where": [{"field": "year", "op": "eq", "value": 2024}, {"field": "category", "op": "eq", "value": "Overall"}]},
            {"op": "top", "n": 5, "by": "score", "order": "desc"},
        ],
        "chart": {"type": "line", "x": "name", "y": "score"},
    }

    fig, res_df, err, takeaway = execute_spec(df, spec)
    assert err is None
    # Category + numeric auto-downgrades to bar
    assert fig.data[0].type == "bar"
