"""
tests/test_transitions.py

Unit tests for Section 7: "Ranking Transitions and Institutional Mobility".
Validates both pure data calculation functions (core/transitions.py)
and chart generation functions (charts/transitions.py), including:
  - Configurable rank parsing and band assignment
  - Shared prepare_ranked_frame pipeline helper
  - Hand-checkable 12-institution state re-ranking sanity test
  - Q1–Q5 analytical calculations under both 'Within selected state' and 'National rank'
"""

import os
import sys

# Ensure root directory is on python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from core.data import load_nirf_data
from core.transitions import (
    DEFAULT_RANK_BANDS,
    parse_rank_value,
    assign_custom_rank_band,
    get_category_valid_years,
    get_category_consecutive_pairs,
    prepare_ranked_frame,
    compare_two_years,
    q1_band_transition,
    q2_top10_mobility,
    q3_band_retention,
    q4_biggest_movers,
    q5_annual_mobility_summary,
)


def test_rank_parsing():
    """Validates robust parsing of integer, float, banded range, and string ranks."""
    assert parse_rank_value(1) == 1.0
    assert parse_rank_value(50.0) == 50.0
    assert parse_rank_value("44") == 44.0
    assert parse_rank_value("44A") == 44.0
    
    # Banded ranges with midpoint vs lower bound
    assert parse_rank_value("101-150", band_rank_method="midpoint") == 125.5
    assert parse_rank_value("101-150", band_rank_method="lower") == 101.0
    assert parse_rank_value("151-200", band_rank_method="midpoint") == 175.5
    assert parse_rank_value("151-200", band_rank_method="lower") == 151.0
    
    # Missing, unranked, and withdrawn
    assert np.isnan(parse_rank_value(None))
    assert np.isnan(parse_rank_value(np.nan))
    assert np.isnan(parse_rank_value("Withdrawn"))
    assert np.isnan(parse_rank_value("Not published"))
    assert np.isnan(parse_rank_value("Unranked"))


def test_custom_rank_band_assignment():
    """Validates configurable rank bands."""
    assert assign_custom_rank_band(1) == "Top 10"
    assert assign_custom_rank_band(10) == "Top 10"
    assert assign_custom_rank_band(11) == "11-50"
    assert assign_custom_rank_band(50) == "11-50"
    assert assign_custom_rank_band(51) == "51-100"
    assert assign_custom_rank_band(100) == "51-100"
    assert assign_custom_rank_band(125.5) == "101-150"
    assert assign_custom_rank_band(180) == "151-200"
    assert assign_custom_rank_band(250) == "201+"
    assert assign_custom_rank_band(np.nan) == "Unranked"


def test_category_years_and_pairs():
    """Validates category-aware year validation."""
    df = load_nirf_data()
    
    # Overall exists from 2017 to 2025 (9 years, 8 pairs)
    overall_years = get_category_valid_years(df, "Overall")
    assert overall_years == list(range(2017, 2026))
    overall_pairs = get_category_consecutive_pairs(df, "Overall")
    assert len(overall_pairs) == 8
    assert overall_pairs[0] == (2017, 2018)
    assert overall_pairs[-1] == (2024, 2025)
    
    # Agriculture started in 2023
    agri_years = get_category_valid_years(df, "Agriculture and Allied Sectors")
    assert agri_years == [2023, 2024, 2025]
    agri_pairs = get_category_consecutive_pairs(df, "Agriculture and Allied Sectors")
    assert agri_pairs == [(2023, 2024), (2024, 2025)]


def test_prepare_ranked_frame_basis_modes():
    """Validates that prepare_ranked_frame correctly toggles between State and National basis."""
    df = load_nirf_data()
    
    # Bihar 2024 has 2 institutions in Overall (IIT Patna #73, AIIMS Patna #99)
    bihar_state_basis = prepare_ranked_frame(
        df, "Overall", 2024, state="Bihar", basis="Within selected state"
    )
    assert len(bihar_state_basis) == 2
    # In state basis, IIT Patna is rank 1 (Top 10), AIIMS Patna is rank 2 (Top 10)
    assert sorted(bihar_state_basis["rank"].tolist()) == [1.0, 2.0]
    assert (bihar_state_basis["rank_band"] == "Top 10").all()
    
    # In national basis, they retain national ranks 73 and 99 (band 51-100)
    bihar_nat_basis = prepare_ranked_frame(
        df, "Overall", 2024, state="Bihar", basis="National rank"
    )
    assert len(bihar_nat_basis) == 2
    assert sorted(bihar_nat_basis["rank"].tolist()) == [73.0, 99.0]
    assert (bihar_nat_basis["rank_band"] == "51-100").all()


def test_within_state_re_ranking_sanity():
    """
    Hand-checkable sanity test on a controlled 12-institution dataset.
    
    Scenario:
      State 'TestState' has 12 institutions across 2023 and 2024:
      - 2023:
          Institutes 1..10 have national ranks 10, 20, ..., 100 -> state ranks 1..10 ('Top 10')
          Institutes 11..12 have national ranks 110, 120 -> state ranks 11..12 ('11-50')
      - 2024:
          Institute 11 jumps to national rank 45 -> state rank 5 ('Top 10') [CLIMBER +6]
          Institute 10 drops to national rank 115 -> state rank 11 ('11-50') [FALLER -1]
          Institutes 1..9 maintain Top 10 standing
          Institute 12 stays at state rank 12
          
    Hand-verified assertions:
      1. Origin band '11-50' in 2023 has exactly 2 institutions (Inst 11, Inst 12).
      2. Exactly 1 institution (Inst 11) transitions from 11-50 into Top 10.
      3. Success rate is exactly 50.0% (1 of 2).
      4. Average rank jump for that mover is exactly +6.0 places (11.0 -> 5.0).
      5. In Top 10 mobility: Inst 11 entered, Inst 10 exited, 9 retained.
    """
    rows = []
    for i in range(1, 13):
        # 2023
        rows.append({
            "institution_id": f"INST_{i:02d}",
            "name": f"University {i}",
            "state": "TestState",
            "category": "Overall",
            "year": 2023,
            "rank_num": float(i * 10),
        })
        # 2024
        if i == 11:
            r24 = 45.0  # Climbs between Inst 4 (40) and Inst 5 (50) -> state rank 5
        elif i == 10:
            r24 = 115.0 # Drops past Inst 11's old spot -> state rank 11
        else:
            r24 = float(i * 10)
            
        rows.append({
            "institution_id": f"INST_{i:02d}",
            "name": f"University {i}",
            "state": "TestState",
            "category": "Overall",
            "year": 2024,
            "rank_num": r24,
        })
        
    df_synthetic = pd.DataFrame(rows)
    
    # Run Q1: 11-50 -> Top 10 within TestState
    q1 = q1_band_transition(
        df=df_synthetic,
        category="Overall",
        y1=2023,
        y2=2024,
        from_band="11-50",
        to_band="Top 10",
        state="TestState",
        basis="Within selected state",
    )
    
    assert q1["kpi_total_from"] == 2
    assert q1["kpi_count"] == 1
    assert q1["kpi_rate"] == 50.0
    assert q1["kpi_rate_formatted"] == "50.0% (1 of 2)"
    assert q1["avg_rank_jump"] == 6.0
    assert q1["is_small_sample"] is True  # 2 < 5
    
    mover_row = q1["movers_df"].iloc[0]
    assert mover_row["name"] == "University 11"
    assert mover_row["rank_y1"] == 11.0
    assert mover_row["rank_y2"] == 5.0
    assert mover_row["rank_change"] == 6.0
    
    # Run Q2: State Top 10 mobility
    q2 = q2_top10_mobility(
        df=df_synthetic,
        category="Overall",
        y1=2023,
        y2=2024,
        state="TestState",
        basis="Within selected state",
    )
    assert q2["num_entered"] == 1
    assert q2["num_exited"] == 1
    assert q2["num_retained"] == 9
    assert q2["entered_df"].iloc[0]["name"] == "University 11"
    assert q2["exited_df"].iloc[0]["name"] == "University 10"


def test_compare_two_years():
    """Validates merging logic, delta formula, and status classification."""
    df = load_nirf_data()
    merged = compare_two_years(df, "Overall", 2023, 2024)
    
    assert not merged.empty
    assert "institution_id" in merged.columns
    assert "rank_change" in merged.columns
    assert "mobility_status" in merged.columns
    
    continuing = merged[merged["rank_y1"].notna() & merged["rank_y2"].notna()]
    assert ((continuing["rank_change"] > 0) == (continuing["mobility_status"] == "Moved Up")).all()
    assert ((continuing["rank_change"] < 0) == (continuing["mobility_status"] == "Moved Down")).all()
    assert ((continuing["rank_change"] == 0) == (continuing["mobility_status"] == "Retained")).all()
    
    new_entries = merged[merged["mobility_status"] == "New Entry"]
    assert (new_entries["rank_y1"].isna() & new_entries["rank_y2"].notna()).all()
    
    dropouts = merged[merged["mobility_status"] == "Dropped Out"]
    assert (dropouts["rank_y1"].notna() & dropouts["rank_y2"].isna()).all()


def test_q1_band_transition():
    """Question 1: Validates movers from 51-100 into Top 50 nationally."""
    df = load_nirf_data()
    q1 = q1_band_transition(df, "Overall", 2023, 2024, from_band="51-100", to_band="Top 50")
    
    assert q1["kpi_count"] >= 1
    assert q1["kpi_total_from"] > 0
    assert 0.0 <= q1["kpi_rate"] <= 100.0
    assert q1["avg_rank_jump"] > 0
    
    movers = q1["movers_df"]
    assert (movers["band_y1"] == "51-100").all()
    assert (movers["rank_y2"] <= 50).all()
    assert (movers["rank_change"] > 0).all()


def test_q2_top10_mobility():
    """Question 2: Validates Top 10 entrance and exit analysis."""
    df = load_nirf_data()
    
    # 2017 to 2018 had Anna University enter and IIT Guwahati exit
    q2 = q2_top10_mobility(df, "Overall", 2017, 2018)
    assert q2["num_entered"] == 1
    assert q2["num_exited"] == 1
    assert q2["num_retained"] == 9
    assert "Anna University" in q2["entered_df"]["name"].values
    assert "Indian Institute of Technology Guwahati" in q2["exited_df"]["name"].values


def test_q3_band_retention():
    """Question 3: Validates retention rate calculation across consecutive years."""
    df = load_nirf_data()
    q3 = q3_band_retention(df, "Overall")
    
    assert len(q3) == 8  # 8 pairs between 2017 and 2025
    assert (q3["retention_rate_pct"] > 70.0).all()
    assert (q3["retention_rate_pct"] <= 100.0).all()
    assert ((q3["same_band_count"] + q3["promoted_count"] + q3["demoted_count"]) == q3["continuing_total"]).all()


def test_q4_biggest_movers():
    """Question 4: Validates Top N risers and fallers."""
    df = load_nirf_data()
    q4 = q4_biggest_movers(df, "Overall", 2023, 2024, top_n=5)
    
    risers = q4["risers_df"]
    fallers = q4["fallers_df"]
    
    assert len(risers) == 5
    assert len(fallers) == 5
    assert (risers["rank_change"] > 0).all()
    assert (fallers["rank_change"] < 0).all()
    assert risers["rank_change"].is_monotonic_decreasing
    assert fallers["rank_change"].is_monotonic_increasing


def test_q5_annual_mobility_summary():
    """Question 5: Validates annual mobility breakdown (Up, Down, Same, New, Exited)."""
    df = load_nirf_data()
    q5 = q5_annual_mobility_summary(df, "Overall")
    
    assert len(q5) == 8
    for _, row in q5.iterrows():
        parts_sum = row["Moved Up"] + row["Moved Down"] + row["Retained"] + row["New Entry"] + row["Dropped Out"]
        assert parts_sum == row["Total Tracked"]
        pct_sum = row["Moved Up (%)"] + row["Moved Down (%)"] + row["Retained (%)"] + row["New Entry (%)"] + row["Dropped Out (%)"]
        assert 99.0 <= pct_sum <= 101.0


def test_chart_functions_produce_valid_takeaways():
    """Validates that all plotting functions execute and produce meaningful takeaways when plotly is installed."""
    try:
        import plotly
    except ImportError:
        print("Note: Plotly is not installed in the current environment. Skipping chart visual tests.")
        return

    df = load_nirf_data()
    from charts.transitions import (
        plot_mover_slope_graph,
        plot_band_destination_bar,
        plot_mover_dumbbell,
        plot_band_transition_heatmap,
        plot_band_transition_sankey,
        plot_top10_mobility_slope,
        plot_top10_turnover_bars,
        plot_band_retention_trend,
        plot_band_retention_stacked_bar,
        plot_diverging_movers,
        plot_mobility_composition_bar,
    )
    
    # Q1: Slope Graph, 100% Destination Bar and Dumbbell
    fig_slope, t1_slope = plot_mover_slope_graph(df, "Overall", 2023, 2024)
    assert len(t1_slope) > 10
    assert "institution" in t1_slope.lower()
    assert len(fig_slope.data) >= 3

    _, t1_bar = plot_band_destination_bar(df, "Overall", 2023, 2024)
    assert len(t1_bar) > 10
    assert "reached" in t1_bar.lower() or "institutions" in t1_bar.lower()

    _, t1_db = plot_mover_dumbbell(df, "Overall", 2023, 2024)
    assert len(t1_db) > 10

    _, t1_h = plot_band_transition_heatmap(df, "Overall", 2023, 2024)
    assert len(t1_h) > 10
    
    _, t1_s = plot_band_transition_sankey(df, "Overall", 2023, 2024)
    assert len(t1_s) > 10
    
    # Q2: Slope chart and turnover bars
    _, t2 = plot_top10_mobility_slope(df, "Overall", 2017, 2018)
    assert "Anna University" in t2 or "entered" in t2
    
    _, t2_bars = plot_top10_turnover_bars(df, "Overall")
    assert "turnover" in t2_bars.lower() or "top 10" in t2_bars.lower()

    # Q3: Line trend and stacked bar
    _, t3 = plot_band_retention_trend(df, "Overall")
    assert "retention" in t3.lower() or "stability" in t3.lower()
    
    _, t3_bar = plot_band_retention_stacked_bar(df, "Overall")
    assert "composition" in t3_bar.lower() or "retained" in t3_bar.lower()

    # Q4: Diverging movers
    _, t4 = plot_diverging_movers(df, "Overall", 2023, 2024, top_n=5)
    assert "climber" in t4.lower() or "largest" in t4.lower()
    
    # Q5: Mobility composition bar
    _, t5 = plot_mobility_composition_bar(df, "Overall", mode="count")
    assert "climbed" in t5.lower() or "overview" in t5.lower()

    _, t5_pct = plot_mobility_composition_bar(df, "Overall", mode="percent")
    assert len(t5_pct) > 10


def test_differential_category_and_state_outputs():
    """
    Verification test ensuring that changing Category and State produces
    distinct, correctly filtered results across all Section 7 questions (Q1–Q5).
    """
    df = load_nirf_data()

    # Q1: Overall / All India vs Engineering / Andhra Pradesh
    q1_all = q1_band_transition(df, "Overall", 2023, 2024, state="All states", basis="National rank")
    q1_ap = q1_band_transition(df, "Engineering", 2023, 2024, state="Andhra Pradesh", basis="Within selected state", from_band="Top 10", to_band="Top 10")
    assert q1_all["kpi_total_from"] != q1_ap["kpi_total_from"]
    assert q1_ap["kpi_total_from"] == 4  # Exactly 4 engineering colleges in AP

    # Q2: Top 10 mobility
    q2_all = q2_top10_mobility(df, "Overall", 2023, 2024, state="All states", basis="National rank")
    q2_ap = q2_top10_mobility(df, "Engineering", 2023, 2024, state="Andhra Pradesh", basis="Within selected state")
    assert q2_all["num_retained"] != q2_ap["num_retained"]
    assert q2_ap["num_retained"] == 4

    # Q3: Band retention rate
    q3_all = q3_band_retention(df, "Overall", state="All states", basis="National rank")
    q3_ap = q3_band_retention(df, "Engineering", state="Andhra Pradesh", basis="Within selected state")
    assert q3_all["retention_rate_pct"].mean() != q3_ap["retention_rate_pct"].mean()

    # Q4: Top risers
    q4_all = q4_biggest_movers(df, "Overall", 2023, 2024, state="All states", basis="National rank")
    q4_ap = q4_biggest_movers(df, "Engineering", 2023, 2024, state="Andhra Pradesh", basis="Within selected state")
    assert q4_all["risers_df"].iloc[0]["name"] != q4_ap["risers_df"].iloc[0]["name"]

    # Q5: Annual mobility breakdown
    q5_all = q5_annual_mobility_summary(df, "Overall", state="All states", basis="National rank")
    q5_ap = q5_annual_mobility_summary(df, "Engineering", state="Andhra Pradesh", basis="Within selected state")
    assert q5_all.iloc[-1]["Total Tracked"] != q5_ap.iloc[-1]["Total Tracked"]


if __name__ == "__main__":
    test_rank_parsing()
    test_custom_rank_band_assignment()
    test_category_years_and_pairs()
    test_prepare_ranked_frame_basis_modes()
    test_within_state_re_ranking_sanity()
    test_differential_category_and_state_outputs()
    test_compare_two_years()
    test_q1_band_transition()
    test_q2_top10_mobility()
    test_q3_band_retention()
    test_q4_biggest_movers()
    test_q5_annual_mobility_summary()
    test_chart_functions_produce_valid_takeaways()
    print("ALL 13 TRANSITION UNIT TESTS PASSED SUCCESSFULLY!")
