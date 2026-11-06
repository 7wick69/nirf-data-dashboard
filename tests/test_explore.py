"""
tests/test_explore.py

Unit tests for Level 2 Explore Module:
  1. Pearson correlation and R2 statistical calculations
  2. Execution of unseen analytical query presets
"""

import pytest
import pandas as pd
from core.data import load_nirf_data
from core.executor import execute_spec
from charts.explore import compute_pearson_r, UNSEEN_PRESETS


def test_compute_pearson_r():
    df = load_nirf_data()
    # Test valid correlation between TLR and score
    res = compute_pearson_r(df, "TLR", "score")
    assert res is not None
    assert "r" in res
    assert "r2" in res
    assert -1.0 <= res["r"] <= 1.0
    assert 0.0 <= res["r2"] <= 1.0
    assert res["n"] > 500

    # Non-numeric or identical columns should return None
    assert compute_pearson_r(df, "TLR", "TLR") is None
    assert compute_pearson_r(df, "name", "score") is None


def test_unseen_presets_execution():
    df = load_nirf_data()
    for name, spec in UNSEEN_PRESETS.items():
        if spec is None:
            continue
        fig, res_df, err, takeaway = execute_spec(df, spec)
        assert err is None, f"Preset '{name}' returned error: {err}"
        assert not res_df.empty, f"Preset '{name}' returned empty dataframe"
        assert fig is not None, f"Preset '{name}' produced None figure"
        assert len(fig.data) > 0, f"Preset '{name}' produced 0 traces"
