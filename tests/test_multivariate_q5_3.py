"""
tests/test_multivariate_q5_3.py

Tests for Q5.3: 'Which institutions have high teaching scores but relatively low perception scores?'
Validates:
- Quadrant scatter plot (TLR on X, PR on Y) with percentile crosshairs and shaded target quadrant
- Companion ranked dumbbell chart and diverging gap bar chart
- Percentile-based gap ranking (TLR %ile - PR %ile)
- Edge cases (missing categories, NaN parameters)
"""

import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from core.data import load_nirf_data
from charts.multivariate import (
    q5_3_scatter_tlr_pr,
    q5_3_dumbbell_chart,
    q5_3_diverging_bar,
    shorten_institution_name,
)


@pytest.fixture(scope="module")
def df_nirf():
    return load_nirf_data()


def test_shorten_institution_name():
    assert shorten_institution_name("Indian Institute of Technology Madras") == "IIT Madras"
    assert shorten_institution_name("National Institute of Technology Silchar") == "NIT Silchar"
    assert shorten_institution_name("All India Institute of Medical Sciences Jodhpur") == "AIIMS Jodhpur"
    assert shorten_institution_name("Jawaharlal Nehru University") == "JNU"


def test_q5_3_scatter_percentile_mode(df_nirf):
    fig, quad_df = q5_3_scatter_tlr_pr(
        df_nirf,
        year=2024,
        category="Overall",
        threshold_mode="percentile",
        tlr_percentile=75.0,
        pr_percentile=25.0,
        top_n_labels=10,
        chart_height=680,
    )

    assert isinstance(fig, go.Figure)
    assert not quad_df.empty
    assert "pct_gap" in quad_df.columns
    assert "raw_gap" in quad_df.columns
    assert "TLR_pct" in quad_df.columns
    assert "PR_pct" in quad_df.columns
    assert "quadrant" in quad_df.columns

    # Verify descending sort by pct_gap
    gaps = quad_df["pct_gap"].dropna().tolist()
    assert gaps == sorted(gaps, reverse=True)

    # Verify shaded shape exists on scatter plot
    shapes = fig.layout.shapes
    assert len(shapes) >= 1
    rect_shapes = [s for s in shapes if s.type == "rect"]
    assert len(rect_shapes) >= 1

    # Verify layout height
    assert fig.layout.height == 680


def test_q5_3_scatter_raw_mode(df_nirf):
    fig, quad_df = q5_3_scatter_tlr_pr(
        df_nirf,
        year=2024,
        category="Overall",
        threshold_mode="raw",
        x_threshold=70.0,
        y_threshold=15.0,
    )

    assert isinstance(fig, go.Figure)
    assert not quad_df.empty


def test_q5_3_dumbbell_chart(df_nirf):
    _, quad_df = q5_3_scatter_tlr_pr(
        df_nirf, year=2024, category="Overall"
    )

    fig_db = q5_3_dumbbell_chart(
        quad_df, category="Overall", year=2024, top_n=10, chart_height=680
    )

    assert isinstance(fig_db, go.Figure)
    # Should have connector line trace, PR markers, and TLR markers
    assert len(fig_db.data) >= 3
    trace_names = [t.name for t in fig_db.data if t.name]
    assert "Perception (PR)" in trace_names
    assert "Teaching (TLR)" in trace_names
    assert fig_db.layout.height == 680


def test_q5_3_diverging_bar(df_nirf):
    _, quad_df = q5_3_scatter_tlr_pr(
        df_nirf, year=2024, category="Overall"
    )

    fig_bar = q5_3_diverging_bar(
        quad_df, category="Overall", year=2024, top_n=10, chart_height=680
    )

    assert isinstance(fig_bar, go.Figure)
    assert len(fig_bar.data) == 1
    assert fig_bar.data[0].type == "bar"
    assert fig_bar.data[0].orientation == "h"


def test_q5_3_missing_category_graceful(df_nirf):
    # Innovation category has no TLR/PR published
    fig, quad_df = q5_3_scatter_tlr_pr(
        df_nirf, year=2024, category="Innovation"
    )
    assert isinstance(fig, go.Figure)
    assert quad_df.empty

    # Companion chart should also not crash on empty dataframe
    fig_db = q5_3_dumbbell_chart(quad_df, category="Innovation", year=2024)
    assert isinstance(fig_db, go.Figure)
