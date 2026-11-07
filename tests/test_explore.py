"""
tests/test_explore.py

Unit and acceptance tests for Level 2 Explore Module (Tableau Workbench):
  1. Pearson correlation and R2 statistical calculations (pairwise dropna)
  2. Aggregation logic for dimensions & measures (Tableau Shelves)
  3. Unique group guarantees (no overlapping bars)
  4. validate_shelves correctness
  5. Top N pre-aggregation logic before color split
  6. Presets dictionary validation
"""

import pytest
import numpy as np
import pandas as pd
from core.data import load_nirf_data
from charts.explore import (
    calculate_pearson,
    compute_pearson_r,
    validate_shelves,
    PRESETS,
    ALL_DIMENSIONS,
    ALL_MEASURES,
    AGG_FUNCS,
)


def test_compute_pearson_r():
    df = load_nirf_data()
    # Test valid correlation between TLR and score
    res = calculate_pearson(df, "TLR", "score")
    assert res is not None
    assert "r" in res
    assert "r2" in res
    assert -1.0 <= res["r"] <= 1.0
    assert 0.0 <= res["r2"] <= 1.0
    assert res["n"] > 500

    # Non-numeric or identical columns should return None
    assert calculate_pearson(df, "TLR", "TLR") is None
    assert calculate_pearson(df, "name", "score") is None


def test_tableau_shelf_aggregation():
    df = load_nirf_data()
    # Simulate Tableau Columns = state (Dimension), Rows = score (Measure, AVG)
    sub = df[df["year"] == 2024]
    agg = sub.groupby("state")["score"].mean().reset_index()
    assert not agg.empty
    assert "state" in agg.columns
    assert "score" in agg.columns
    assert agg["score"].notna().all()

    # Verify no duplicate states exist in aggregated output (eliminates glitchy overlapping bars)
    assert agg["state"].is_unique


def test_validate_shelves():
    # Scatter: requires two distinct measures
    ok, msg = validate_shelves(x_is_dim=False, x_field="TLR", y_field="RPC", mark="Scatter")
    assert ok is True
    assert msg == ""

    # Scatter with identical measures -> invalid
    ok, msg = validate_shelves(x_is_dim=False, x_field="TLR", y_field="TLR", mark="Scatter")
    assert ok is False
    assert "two different measures" in msg

    # Scatter with dimension on X -> invalid
    ok, msg = validate_shelves(x_is_dim=True, x_field="name", y_field="RPC", mark="Scatter")
    assert ok is False
    assert "continuous measures" in msg

    # Bar with Dimension on X and Measure on Y -> valid
    ok, msg = validate_shelves(x_is_dim=True, x_field="state", y_field="RPC", mark="Bar")
    assert ok is True

    # Bar with Measure on X -> invalid
    ok, msg = validate_shelves(x_is_dim=False, x_field="TLR", y_field="RPC", mark="Bar")
    assert ok is False
    assert "require a Dimension on X" in msg

    # Line with Dimension on X -> valid
    ok, msg = validate_shelves(x_is_dim=True, x_field="year", y_field="score", mark="Line")
    assert ok is True

    # Histogram only requires Y measure -> valid regardless of X
    ok, msg = validate_shelves(x_is_dim=True, x_field="state", y_field="score", mark="Histogram")
    assert ok is True

    # Table is always valid
    ok, msg = validate_shelves(x_is_dim=False, x_field="score", y_field="TLR", mark="Table")
    assert ok is True


def test_top_n_before_color_split():
    df = load_nirf_data()
    eng = df[df["category"] == "Engineering"].copy()

    # Suppose we want Top 5 States by mean score, grouped by category/state
    # Fix 6: compute order on X categories first
    order = eng.groupby("state")["score"].mean().sort_values(ascending=False)
    top_5_states = order.head(5).index.tolist()
    assert len(top_5_states) == 5

    # Filter to those 5 states
    filtered = eng[eng["state"].isin(top_5_states)]
    # Group by [state, category]
    grouped = filtered.groupby(["state", "category"])["score"].mean().reset_index()
    # Check that exactly 5 distinct states exist in the result
    assert grouped["state"].nunique() == 5


def test_presets_keys_and_values():
    for name, p in PRESETS.items():
        assert "tab_col_type" in p
        if p["tab_col_type"] == "Dimension":
            assert p["tab_col_dim"] in ALL_DIMENSIONS
        else:
            assert p["tab_col_meas"] in ALL_MEASURES
        assert p["tab_row_meas"] in ALL_MEASURES
        assert p["tab_mark_type"] in ["Bar", "Line", "Scatter", "Box Plot", "Histogram", "Table"]
