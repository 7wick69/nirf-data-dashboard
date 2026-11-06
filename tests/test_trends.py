"""
tests/test_trends.py

Comprehensive tests for Trend visualizations (Annexure Part 1):
  1. Rank trajectory over time (with state filter check)
  2. Greatest improvers
  3. Largest decliners
  4. Average score trend (with benchmark check)
  5. Consistently top 10 institutions (with textbook qualitative palette check)
"""

from core.data import load_nirf_data, get_institutions_list
from charts.trends import (
    plot_institution_rank_trend,
    plot_greatest_improvers,
    plot_largest_decliners,
    plot_average_score_trend,
    plot_consistently_top10,
)
from charts.theme import QUALITATIVE_PALETTE, MARKER_SYMBOLS


def test_trends_plots():
    df = load_nirf_data()

    # 1. Institution Rank Trend
    fig1, takeaway1 = plot_institution_rank_trend(df, "Indian Institute of Technology Madras", "Overall")
    assert fig1 is not None
    assert "improved from 2 to 1" in takeaway1
    assert fig1.layout.yaxis.autorange == "reversed"

    # 1b. Institution Rank Trend with State Filter (Karnataka -> IISc)
    karnataka_insts = get_institutions_list(df, "Overall", state="Karnataka")
    assert "Indian Institute of Science" in karnataka_insts
    assert "Indian Institute of Technology Madras" not in karnataka_insts  # IITM is Tamil Nadu

    fig1_state, takeaway1_state = plot_institution_rank_trend(
        df, "Indian Institute of Science", "Overall", state="Karnataka"
    )
    assert fig1_state is not None
    assert "1 to 2" in takeaway1_state

    # 2. Greatest Improvers
    fig2, takeaway2 = plot_greatest_improvers(df, "Overall", 2017, 2025, top_n=5)
    assert fig2 is not None
    assert "Top improver:" in takeaway2

    # 2b. Greatest Improvers with State Filter (Tamil Nadu)
    fig2_tn, takeaway2_tn = plot_greatest_improvers(df, "Overall", 2017, 2025, top_n=5, state="Tamil Nadu")
    assert fig2_tn is not None
    assert len(fig2_tn.data) > 0

    # 3. Largest Decliners
    fig3, takeaway3 = plot_largest_decliners(df, "Overall", 2017, 2025, top_n=5)
    assert fig3 is not None
    assert "Largest decline:" in takeaway3

    # 4. Average Score Trend (All India vs State Filter)
    fig4, takeaway4 = plot_average_score_trend(df, "Overall")
    assert fig4 is not None
    assert "Average score" in takeaway4

    fig4_state, takeaway4_state = plot_average_score_trend(df, "Overall", state="Tamil Nadu")
    assert fig4_state is not None
    assert "Tamil Nadu" in takeaway4_state
    # Check that state vs national benchmark trace exists
    assert any("All-India" in t.name for t in fig4_state.data)

    # 5. Consistently Top 10 (All India vs State Level Top 10)
    fig5, takeaway5 = plot_consistently_top10(df, "Overall", min_years=5, state="All states")
    assert fig5 is not None
    assert "institutions maintained Top 10" in takeaway5
    assert fig5.layout.yaxis.autorange == "reversed"

    # 5b. State-Level Top 10 (Karnataka should return 10 institutions, Telangana should return 6)
    fig5_kar, takeaway5_kar = plot_consistently_top10(df, "Overall", state="Karnataka")
    assert len(fig5_kar.data) == 10
    assert "Top 10 institutions in Karnataka" in takeaway5_kar

    fig5_tel, takeaway5_tel = plot_consistently_top10(df, "Overall", state="Telangana")
    assert len(fig5_tel.data) == 6
    assert "Top 6 institutions in Telangana" in takeaway5_tel

    # Check textbook qualitative palette & redundant marker symbols
    traces = fig5.data
    assert len(traces) > 0
    for idx, trace in enumerate(traces):
        expected_color = QUALITATIVE_PALETTE[idx % len(QUALITATIVE_PALETTE)]
        expected_symbol = MARKER_SYMBOLS[idx % len(MARKER_SYMBOLS)]
        assert trace.line.color == expected_color
        assert trace.marker.symbol == expected_symbol

    print("All 5 trend chart tests passed with state filtering and textbook visual encoding!")


if __name__ == "__main__":
    test_trends_plots()
