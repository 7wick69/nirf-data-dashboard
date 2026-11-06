"""
tests/test_trends.py
"""

from core.data import load_nirf_data
from charts.trends import (
    plot_institution_rank_trend,
    plot_biggest_improvers_decliners,
    plot_average_score_trend,
    plot_consistently_top10,
)


def test_trends_plots():
    df = load_nirf_data()

    # 1. Institution Rank Trend
    fig1, takeaway1 = plot_institution_rank_trend(df, "Indian Institute of Technology Madras", "Overall")
    assert fig1 is not None
    assert "improved from 2 to 1" in takeaway1
    assert fig1.layout.yaxis.autorange == "reversed"

    # 2. Biggest Improvers & Decliners
    fig2, takeaway2 = plot_biggest_improvers_decliners(df, "Overall", 2017, 2025)
    assert fig2 is not None
    assert "Top improver:" in takeaway2

    # 3. Average Score Trend
    fig3, takeaway3 = plot_average_score_trend(df, "Overall")
    assert fig3 is not None
    assert "Average score" in takeaway3

    # 4. Consistently Top 10
    fig4, takeaway4 = plot_consistently_top10(df, "Overall")
    assert fig4 is not None
    assert "institutions maintained Top 10" in takeaway4

    print("All trend chart tests passed on clean uploaded CSV!")


if __name__ == "__main__":
    test_trends_plots()
