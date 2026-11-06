"""
tests/test_bivariate.py

Unit tests for Annexure A §4 Bivariate Analysis & Correlation visualizations.
"""

from core.data import load_nirf_data
from charts.bivariate import (
    plot_scatter_marginals,
    plot_correlation_vs_weight,
    plot_param_correlation_heatmap,
    plot_tlr_rpc_quadrants,
    plot_tlr_rpc_by_category,
    plot_param_corr_over_years,
    plot_param_year_heatmap,
    plot_explorer,
)


def test_bivariate_charts():
    df = load_nirf_data()

    # 1. Q1: RPC vs Overall with marginals & outliers
    fig1, takeaway1 = plot_scatter_marginals(
        df=df,
        x_param="RPC",
        year=2024,
        category="Overall",
        corr_type="pearson",
        top_n_outliers=4,
    )
    assert fig1 is not None
    assert len(fig1.data) > 0
    assert "correlation" in takeaway1.lower()

    # 2. Q2: PR vs Overall with marginals (Pearson vs Spearman divergence)
    fig2, takeaway2 = plot_scatter_marginals(
        df=df,
        x_param="PR",
        year=2024,
        category="Overall",
        corr_type="spearman",
        top_n_outliers=4,
    )
    assert fig2 is not None
    assert len(fig2.data) > 0

    # 3. Q3: Correlation vs weight + 5x5 heatmap
    fig3_bar, takeaway3_bar = plot_correlation_vs_weight(df=df, category="Overall")
    assert fig3_bar is not None
    assert len(fig3_bar.data) >= 2  # bar + diamond marker

    fig3_heat, _ = plot_param_correlation_heatmap(df=df, category="Overall")
    assert fig3_heat is not None
    assert len(fig3_heat.data) == 1

    # 4. Q4: TLR vs RPC quadrants + discipline lollipop
    fig4_quad, takeaway4_quad = plot_tlr_rpc_quadrants(df=df, year=2024, category="Overall")
    assert fig4_quad is not None
    assert "quadrant" in takeaway4_quad.lower()

    fig4_lolli, takeaway4_lolli = plot_tlr_rpc_by_category(df=df, year=2024)
    assert fig4_lolli is not None
    assert "discipline" in takeaway4_lolli.lower() or "category" in takeaway4_lolli.lower()

    # 5. Q5: Multi-parameter correlation lines with 95% CI + heatmap strip
    fig5_lines, takeaway5_lines = plot_param_corr_over_years(df=df, category="Overall")
    assert fig5_lines is not None
    assert len(fig5_lines.data) >= 5

    fig5_strip, _ = plot_param_year_heatmap(df=df, category="Overall")
    assert fig5_strip is not None

    # 6. Explorer: Select any two variables
    fig_exp, stats_dict, auto_sentence = plot_explorer(
        df=df,
        x_var="RPC",
        y_var="score",
        year=2024,
        category="Overall",
        show_lowess=True,
    )
    assert fig_exp is not None
    assert stats_dict["pearson_r"] is not None
    assert stats_dict["n"] > 0
    assert "relationship" in auto_sentence.lower()

    print("All Bivariate Analysis and Correlation tests passed successfully!")


if __name__ == "__main__":
    test_bivariate_charts()

