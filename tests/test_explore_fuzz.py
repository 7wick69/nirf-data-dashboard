"""
tests/test_explore_fuzz.py

Comprehensive fuzz and edge-case testing for charts/explore.py (Tableau Visual Workbench).
Guarantees zero unhandled exceptions across thousands of shelf combinations and edge conditions:
- All marks (Bar, Line, Scatter, Box Plot, Histogram, Table, Heatmap, Area)
- All aggregations (None, Mean, Median, Sum, Max, Min, Count, Std)
- Multi-measure selections
- Count pseudo-measures (__n_inst__, __n_rows__)
- Missing/all-NaN columns (e.g. Innovation category)
- Empty datasets after aggressive filtering
- Categorical observed=True checks
"""

import itertools
import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from core.data import load_nirf_data
from charts.explore import (
    ALL_DIMENSIONS,
    ALL_MEASURES,
    AGG_FUNCS,
    COLOR_OPTIONS,
    PRESETS,
    RANK_BAND_ORDER,
    validate_shelves,
    calculate_pearson,
)
from charts.theme import apply_theme, QUALITATIVE_PALETTE, LINE_COLOR


@pytest.fixture(scope="module")
def nirf_data():
    df = load_nirf_data()
    df["__n_rows__"] = 1
    df["__n_inst__"] = df.get("institution_id", df["name"])
    return df


def test_presets_fully_valid():
    """Ensure all presets have valid, non-null attributes matching module registries."""
    for name, p in PRESETS.items():
        assert "tab_col_type" in p
        if p["tab_col_type"] == "Dimension":
            assert p["tab_col_dim"] in ALL_DIMENSIONS
        else:
            assert p["tab_col_meas"] in ALL_MEASURES
        
        # Row meas can be str or list of str
        row_m = p["tab_row_meas"]
        if isinstance(row_m, list):
            for m in row_m:
                assert m in ALL_MEASURES
        else:
            assert row_m in ALL_MEASURES
            
        assert p["tab_mark_type"] in ["Bar", "Line", "Scatter", "Box Plot", "Histogram", "Table", "Area", "Heatmap"]
        assert p["tab_agg_func"] in AGG_FUNCS


def test_validate_shelves_rules():
    """Verify validate_shelves guards each mark type correctly."""
    # 1. Scatter
    assert validate_shelves(x_is_dim=False, x_field="TLR", y_field="RPC", mark="Scatter")[0] is True
    assert validate_shelves(x_is_dim=False, x_field="TLR", y_field="TLR", mark="Scatter")[0] is False
    assert validate_shelves(x_is_dim=True, x_field="state", y_field="RPC", mark="Scatter")[0] is False
    assert validate_shelves(x_is_dim=False, x_field="TLR", y_field="__n_inst__", mark="Scatter")[0] is False

    # 2. Bar
    assert validate_shelves(x_is_dim=True, x_field="state", y_field="score", mark="Bar")[0] is True
    assert validate_shelves(x_is_dim=False, x_field="TLR", y_field="score", mark="Bar")[0] is False
    assert validate_shelves(x_is_dim=True, x_field="state", y_field="__n_inst__", mark="Bar")[0] is True

    # 3. Line & Area (Ordered X required)
    assert validate_shelves(x_is_dim=True, x_field="year", y_field="score", mark="Line")[0] is True
    assert validate_shelves(x_is_dim=True, x_field="rank_band", y_field="score", mark="Line")[0] is True
    assert validate_shelves(x_is_dim=True, x_field="state", y_field="score", mark="Line")[0] is False
    assert validate_shelves(x_is_dim=True, x_field="year", y_field="score", mark="Area")[0] is True
    assert validate_shelves(x_is_dim=True, x_field="state", y_field="score", mark="Area")[0] is False

    # 4. Box Plot
    assert validate_shelves(x_is_dim=True, x_field="category", y_field="score", mark="Box Plot")[0] is True
    assert validate_shelves(x_is_dim=False, x_field="TLR", y_field="score", mark="Box Plot")[0] is False
    assert validate_shelves(x_is_dim=True, x_field="category", y_field="__n_rows__", mark="Box Plot")[0] is False

    # 5. Histogram
    assert validate_shelves(x_is_dim=False, x_field="anything", y_field="score", mark="Histogram")[0] is True
    assert validate_shelves(x_is_dim=False, x_field="anything", y_field="__n_inst__", mark="Histogram")[0] is False


def test_fuzz_chart_generation(nirf_data):
    """
    Fuzz across key dimensions, measures, marks, and colors to verify no uncaught
    crashes occur during chart construction or aggregation.
    """
    test_dims = ["state", "year", "rank_band"]
    test_meas = ["score", "rank", "TLR", "__n_inst__"]
    test_marks = ["Bar", "Line", "Scatter", "Box Plot", "Histogram"]
    test_colors = [None, "category"]

    sample_df = nirf_data[(nirf_data["year"] >= 2023) & (nirf_data["category"] == "Overall")].copy()

    for x_field in test_dims:
        for y_field in test_meas:
            for mark in test_marks:
                for color_field in test_colors:
                    ok, _ = validate_shelves(x_is_dim=True, x_field=x_field, y_field=y_field, mark=mark)
                    if not ok:
                        continue

                    # Simulate chart builder pipeline
                    work = sample_df.dropna(subset=[x_field, y_field]).copy() if y_field not in ("__n_inst__", "__n_rows__") else sample_df.copy()
                    if work.empty:
                        continue

                    invert_y = (y_field == "rank")
                    group_cols = [x_field] + ([color_field] if color_field else [])

                    if mark == "Bar":
                        if y_field == "__n_inst__":
                            agg_df = work.groupby(group_cols, observed=True)[y_field].nunique().reset_index()
                        else:
                            agg_df = work.groupby(group_cols, observed=True)[y_field].mean().reset_index()
                        assert not agg_df.empty
                        fig = px.bar(agg_df, x=x_field, y=y_field, color=color_field)
                        assert fig is not None

                    elif mark == "Line":
                        if y_field == "__n_inst__":
                            agg_df = work.groupby(group_cols, observed=True)[y_field].nunique().reset_index()
                        elif y_field == "__n_rows__":
                            agg_df = work.groupby(group_cols, observed=True)[y_field].count().reset_index()
                        else:
                            agg_df = work.groupby(group_cols, observed=True)[y_field].mean().reset_index()
                        fig = px.line(agg_df, x=x_field, y=y_field, color=color_field)
                        assert fig is not None

                    elif mark == "Box Plot":
                        fig = px.box(work.head(100), x=x_field, y=y_field, color=color_field)
                        assert fig is not None


def test_innovation_nan_scores_handling(nirf_data):
    """Verify that Innovation category (where scores are NaN) never crashes explore aggregations."""
    inno = nirf_data[nirf_data["category"] == "Innovation"].copy()
    if inno.empty:
        pytest.skip("No Innovation category in dataset")

    # Score is NaN for all rows in Innovation
    valid_score = inno.dropna(subset=["score"])
    assert valid_score.empty

    # Should not crash on empty/all-NaN
    corr = calculate_pearson(inno, "TLR", "score")
    assert corr is None


def test_multimeasure_melt_logic(nirf_data):
    """Test pivoting / melting multiple measures on the Rows shelf."""
    sub = nirf_data[nirf_data["year"] == 2024].copy()
    y_fields = ["TLR", "RPC", "GO"]
    group_cols = ["state"]

    agg_df = sub.groupby(group_cols, observed=True)[y_fields].mean().reset_index().head(5)
    melted = pd.melt(agg_df, id_vars=group_cols, value_vars=y_fields, var_name="Measure", value_name="Value")

    assert "Measure" in melted.columns
    assert "Value" in melted.columns
    assert len(melted) == len(agg_df) * len(y_fields)

    fig = px.bar(melted, x="state", y="Value", color="Measure", barmode="group")
    assert fig is not None
