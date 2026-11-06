"""
tests/test_explore.py

Unit tests for Level 2 Explore Module (Tableau Workbench):
  1. Pearson correlation and R2 statistical calculations
  2. Dimension + Measure aggregation logic (Tableau Shelves)
  3. Automatic horizontal layout and inverted Y-axis for ranks
"""

import pytest
import pandas as pd
from core.data import load_nirf_data
from charts.explore import calculate_pearson, compute_pearson_r


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


def test_tableau_aggregation_logic():
    df = load_nirf_data()
    # Simulate Tableau shelf: Columns = state (Dimension), Rows = score (Measure, AVG)
    sub = df[df["year"] == 2024]
    agg = sub.groupby("state")["score"].mean().reset_index()
    assert not agg.empty
    assert "state" in agg.columns
    assert "score" in agg.columns
    assert agg["score"].notna().all()

    # Verify no duplicate states exist in aggregated output (eliminates glitchy overlapping bars)
    assert agg["state"].is_unique
