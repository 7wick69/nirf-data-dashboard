"""
tests/test_outliers.py

Unit tests for Annexure A Section 8: Outlier Detection and Performance Gaps.
"""

from core.data import load_nirf_data
from charts.outliers import (
    plot_connected_scatter_research_outliers,
    plot_research_vs_teaching_gap,
    plot_score_volatility_trends,
    plot_high_param_low_rank,
    plot_ranking_volatility,
)


def test_outlier_charts():
    df = load_nirf_data()

    # 1. Q1: Connected scatter plot (Research vs Overall trajectory)
    fig1, takeaway1 = plot_connected_scatter_research_outliers(df, category="Overall", top_n=5)
    assert fig1 is not None
    assert len(fig1.data) >= 2
    assert "research" in takeaway1.lower()

    # 2. Q2: Research vs Teaching Gap
    fig2, takeaway2 = plot_research_vs_teaching_gap(df, year=2024, category="Overall")
    assert fig2 is not None
    assert len(fig2.data) >= 2
    assert "teaching" in takeaway2.lower()

    # 3. Q3: Score Volatility Multi-line Chart
    fig3, takeaway3 = plot_score_volatility_trends(df, category="Overall", top_n=5)
    assert fig3 is not None
    assert len(fig3.data) >= 1
    assert "score" in takeaway3.lower()

    # 4. Q4: High Parameter, Low Rank Anomaly Scatter
    fig4, takeaway4 = plot_high_param_low_rank(df, year=2024, category="Overall", param="RPC")
    assert fig4 is not None
    assert len(fig4.data) >= 1
    assert fig4.layout.yaxis.autorange == "reversed"
    assert "anomaly" in takeaway4.lower() or "score" in takeaway4.lower()

    # 5. Q5: Ranking Volatility - Option A: Box Plot
    fig5_box, takeaway5_box = plot_ranking_volatility(df, category="Overall", top_n=6, mode="boxplot")
    assert fig5_box is not None
    assert len(fig5_box.data) >= 1
    assert fig5_box.layout.yaxis.autorange == "reversed"

    # 5. Q5: Ranking Volatility - Option B: Line Chart
    fig5_line, takeaway5_line = plot_ranking_volatility(df, category="Overall", top_n=6, mode="line")
    assert fig5_line is not None
    assert len(fig5_line.data) >= 1
    assert fig5_line.layout.yaxis.autorange == "reversed"

    print("All Outlier Detection & Performance Gaps tests passed successfully!")


if __name__ == "__main__":
    test_outlier_charts()

