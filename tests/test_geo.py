"""
tests/test_geo.py
"""

from core.data import load_nirf_data
from charts.geo import (
    plot_top_states_top100,
    plot_geo_distribution_over_time,
    plot_state_representation_improvement,
    plot_state_rank_comparison,
    plot_new_entrant_states,
)


def test_geo_plots():
    df = load_nirf_data()

    # 1. Top States in Top 100
    fig1, takeaway1 = plot_top_states_top100(df, "Overall", 2025)
    assert fig1 is not None
    assert "Tamil Nadu" in takeaway1
    assert "17" in takeaway1

    # 2. Geo Distribution Over Time (Choropleth & Stacked)
    fig2a, takeaway2a = plot_geo_distribution_over_time(df, "Overall", 2025, view_mode="India Choropleth Map")
    assert fig2a is not None
    assert "clustered" in takeaway2a or "Tamil Nadu" in takeaway2a

    fig2b, takeaway2b = plot_geo_distribution_over_time(df, "Overall", 2025, view_mode="State-wise Stacked Bar Chart")
    assert fig2b is not None
    assert "Tamil Nadu" in takeaway2b

    # 3. State Representation Improvement
    fig3, takeaway3 = plot_state_representation_improvement(df, "Overall", 2017, 2025)
    assert fig3 is not None
    assert "Top improvers" in takeaway3

    # 4. Regional Head-to-Head Comparison
    fig4, takeaway4 = plot_state_rank_comparison(df, "Overall", "Karnataka", "Tamil Nadu")
    assert fig4 is not None
    assert "median rank" in takeaway4
    assert fig4.layout.yaxis.autorange == "reversed"

    # 5. Newly Appearing Institutions
    fig5, takeaway5 = plot_new_entrant_states(df, "Overall")
    assert fig5 is not None
    assert "Tamil Nadu" in takeaway5
    assert "Karnataka" in takeaway5

    print("All geospatial chart tests passed on clean NIRF dataset!")


if __name__ == "__main__":
    test_geo_plots()

