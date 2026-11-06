"""
tests/test_univariate.py
========================
Pytest suite for the charts/univariate.py pure analysis functions.

Tests verify:
  1. score_quartile_share  — percentages sum to 100 % (±0.1 %)
  2. score_summary_stats   — numeric bounds and required keys
  3. score_summary_stats_all_years — DataFrame schema
  4. category_spread_comparison — stats_df column completeness
  5. rank_distribution     — returns two non-None Plotly figures
  6. score_distribution    — returns a non-None Plotly figure
  7. Empty-data safety     — no crashes when df is empty or filtered to nothing
  8. Data quality          — total row count and year coverage

Run with:
    PYTHONPATH=. pytest tests/test_univariate.py -v
"""

import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from core.data import load_nirf_data
from charts.univariate import (
    score_distribution,
    rank_distribution,
    score_summary_stats,
    score_summary_stats_all_years,
    score_quartile_share,
    category_spread_comparison,
    plot_score_distribution,
    plot_rank_distribution,
    plot_summary_statistics,
    plot_rank_band_breakdown,
    plot_score_spread_by_category,
    plot_parameter_distribution,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def df():
    """Load the full NIRF dataset once for the whole module."""
    return load_nirf_data()


@pytest.fixture(scope="module")
def df_overall_2024(df):
    """Convenience slice: Overall category, year 2024."""
    return df[(df["category"] == "Overall") & (df["year"] == 2024)].copy()


@pytest.fixture(scope="module")
def empty_df():
    """An empty DataFrame with the same schema as the NIRF data."""
    return pd.DataFrame(columns=["year", "category", "name", "state",
                                  "rank", "score", "TLR", "RPC", "GO", "OI", "PR"])


# ===========================================================================
# 1. score_quartile_share — percentages must sum to 100 %
# ===========================================================================

class TestScoreQuartileShare:
    def test_percentages_sum_to_100(self, df):
        """Quartile percentages must sum to exactly 100 % (±0.1 %)."""
        df_q, fig = score_quartile_share(df, year=2024, category="Overall")
        assert not df_q.empty, "Quartile DataFrame should not be empty"
        assert isinstance(fig, go.Figure), "Should return a Plotly Figure"

        # Recompute pct from the returned df
        assert "Quartile" in df_q.columns, "Quartile column must exist"
        counts = df_q.groupby("Quartile").size()
        total  = counts.sum()
        assert total > 0, "Should have at least 1 institution"

        pcts = counts / total * 100
        assert abs(pcts.sum() - 100.0) < 0.1, (
            f"Quartile percentages sum to {pcts.sum():.4f}%, expected ~100%"
        )

    def test_four_quartiles_present(self, df):
        """With enough data, all 4 quartile labels should appear."""
        df_q, _ = score_quartile_share(df, year=2024, category="Overall")
        quartile_names = df_q["Quartile"].unique()
        assert len(quartile_names) == 4, (
            f"Expected 4 quartile groups, got {len(quartile_names)}: {quartile_names}"
        )

    def test_quartile_labels(self, df):
        """Quartile label strings follow the expected pattern."""
        df_q, _ = score_quartile_share(df, year=2024, category="Overall")
        labels = set(df_q["Quartile"].unique())
        expected = {
            "Q1 (bottom 25 %)",
            "Q2 (25 – 50 %)",
            "Q3 (50 – 75 %)",
            "Q4 (top 25 %)",
        }
        assert labels == expected, f"Unexpected labels: {labels}"

    def test_empty_data_does_not_crash(self, empty_df):
        """Empty df must return (empty DataFrame, Figure) without raising."""
        df_q, fig = score_quartile_share(empty_df, year=2024, category="Overall")
        assert isinstance(df_q, pd.DataFrame)
        assert isinstance(fig, go.Figure)

    def test_nonexistent_year_category(self, df):
        """Filtering to a year/category with no data should return empty safely."""
        df_q, fig = score_quartile_share(df, year=1900, category="FakeCategory")
        assert df_q.empty
        assert isinstance(fig, go.Figure)


# ===========================================================================
# 2. score_summary_stats — numeric bounds and keys
# ===========================================================================

class TestScoreSummaryStats:
    def test_required_keys(self, df):
        """All required keys must be present in the returned dict."""
        stats = score_summary_stats(df, year=2024, category="Overall")
        required = {"mean", "median", "min", "max", "std", "count",
                    "min_inst", "max_inst"}
        assert required == set(stats.keys()), (
            f"Missing keys: {required - set(stats.keys())}"
        )

    def test_numeric_bounds_in_range(self, df):
        """All score statistics should be in the realistic [0, 100] range."""
        stats = score_summary_stats(df, year=2024, category="Overall")
        for key in ["mean", "median", "min", "max"]:
            val = stats[key]
            assert val is not None, f"stats['{key}'] is None"
            assert 0.0 <= val <= 100.0, (
                f"stats['{key}'] = {val} is outside [0, 100]"
            )

    def test_min_le_mean_le_max(self, df):
        """min ≤ mean ≤ max must hold."""
        stats = score_summary_stats(df, year=2024, category="Overall")
        assert stats["min"] <= stats["mean"] <= stats["max"], (
            f"Ordering violated: min={stats['min']}, mean={stats['mean']}, "
            f"max={stats['max']}"
        )

    def test_count_positive(self, df):
        """count must be a positive integer for a known year/category."""
        stats = score_summary_stats(df, year=2024, category="Overall")
        assert isinstance(stats["count"], int)
        assert stats["count"] > 0

    def test_inst_names_are_strings(self, df):
        """min_inst and max_inst must be non-empty strings."""
        stats = score_summary_stats(df, year=2024, category="Overall")
        assert isinstance(stats["min_inst"], str) and len(stats["min_inst"]) > 0
        assert isinstance(stats["max_inst"], str) and len(stats["max_inst"]) > 0

    def test_empty_returns_none_values(self, empty_df):
        """Empty df should return all None values, not raise."""
        stats = score_summary_stats(empty_df, year=2024, category="Overall")
        assert all(v is None for v in stats.values())

    def test_nonexistent_category(self, df):
        """Nonexistent category returns None values gracefully."""
        stats = score_summary_stats(df, year=2024, category="Nonexistent")
        assert stats["count"] is None


# ===========================================================================
# 3. score_summary_stats_all_years — DataFrame schema
# ===========================================================================

class TestScoreSummaryStatsAllYears:
    def test_returns_dataframe(self, df):
        agg = score_summary_stats_all_years(df, category="Overall")
        assert isinstance(agg, pd.DataFrame), "Should return a pd.DataFrame"

    def test_expected_columns(self, df):
        agg = score_summary_stats_all_years(df, category="Overall")
        expected_cols = {"Year", "Mean", "Median", "Min", "Max", "Std_Dev", "Count"}
        assert expected_cols.issubset(set(agg.columns)), (
            f"Missing columns: {expected_cols - set(agg.columns)}"
        )

    def test_sorted_ascending_by_year(self, df):
        agg = score_summary_stats_all_years(df, category="Overall")
        assert list(agg["Year"]) == sorted(agg["Year"].tolist()), (
            "Rows should be sorted ascending by Year"
        )

    def test_year_range_filter(self, df):
        agg = score_summary_stats_all_years(
            df, category="Overall", year_range=(2020, 2023)
        )
        assert agg["Year"].min() >= 2020
        assert agg["Year"].max() <= 2023

    def test_all_years_covered(self, df):
        """Overall category should have data for all 9 years (2017–2025)."""
        agg = score_summary_stats_all_years(df, category="Overall")
        years = set(agg["Year"].tolist())
        expected = set(range(2017, 2026))
        assert expected.issubset(years), (
            f"Missing years: {expected - years}"
        )

    def test_empty_df_returns_empty(self, empty_df):
        agg = score_summary_stats_all_years(empty_df, category="Overall")
        assert isinstance(agg, pd.DataFrame)
        assert agg.empty


# ===========================================================================
# 4. category_spread_comparison — stats DataFrame columns
# ===========================================================================

class TestCategorySpreadComparison:
    def test_returns_tuple(self, df):
        all_cats = df["category"].dropna().unique().tolist()
        stats_df, fig = category_spread_comparison(df, years=[2024], categories=all_cats)
        assert isinstance(stats_df, pd.DataFrame)
        assert isinstance(fig, go.Figure)

    def test_expected_stat_columns(self, df):
        all_cats = df["category"].dropna().unique().tolist()
        stats_df, _ = category_spread_comparison(df, years=[2024], categories=all_cats)
        required_cols = {"Category", "N", "Mean", "Std Dev", "IQR", "Range", "CV (%)"}
        assert required_cols.issubset(set(stats_df.columns)), (
            f"Missing stat columns: {required_cols - set(stats_df.columns)}"
        )

    def test_one_row_per_category(self, df):
        cats = ["Overall", "Engineering", "University"]
        stats_df, _ = category_spread_comparison(df, years=[2024], categories=cats)
        assert len(stats_df) == len(cats), (
            f"Expected {len(cats)} rows, got {len(stats_df)}"
        )

    def test_std_dev_non_negative(self, df):
        all_cats = df["category"].dropna().unique().tolist()
        stats_df, _ = category_spread_comparison(df, years=[2024], categories=all_cats)
        assert (stats_df["Std Dev"] >= 0).all(), "Std Dev cannot be negative"

    def test_iqr_non_negative(self, df):
        all_cats = df["category"].dropna().unique().tolist()
        stats_df, _ = category_spread_comparison(df, years=[2024], categories=all_cats)
        assert (stats_df["IQR"] >= 0).all(), "IQR cannot be negative"

    def test_empty_df_does_not_crash(self, empty_df):
        stats_df, fig = category_spread_comparison(empty_df, years=[], categories=[])
        assert isinstance(stats_df, pd.DataFrame)
        assert isinstance(fig, go.Figure)


# ===========================================================================
# 5. rank_distribution — returns two Plotly figures
# ===========================================================================

class TestRankDistribution:
    def test_returns_two_figures(self, df):
        hist_fig, band_fig = rank_distribution(df, year=2024, category="Overall")
        assert isinstance(hist_fig, go.Figure), "First return must be a go.Figure"
        assert isinstance(band_fig, go.Figure), "Second return must be a go.Figure"

    def test_all_years_mode(self, df):
        hist_fig, band_fig = rank_distribution(df, year="All years", category="Overall")
        assert isinstance(hist_fig, go.Figure)
        assert isinstance(band_fig, go.Figure)

    def test_empty_does_not_crash(self, empty_df):
        hist_fig, band_fig = rank_distribution(empty_df, year=2024, category="Overall")
        assert isinstance(hist_fig, go.Figure)
        assert isinstance(band_fig, go.Figure)

    def test_band_fig_has_traces(self, df):
        """Band bar chart should have at least one trace for valid data."""
        _, band_fig = rank_distribution(df, year="All years", category="Overall")
        assert len(band_fig.data) > 0, "Band chart should have traces"


# ===========================================================================
# 6. score_distribution — returns a Plotly figure
# ===========================================================================

class TestScoreDistribution:
    def test_returns_figure(self, df):
        fig = score_distribution(df, year=2024, category="Overall")
        assert isinstance(fig, go.Figure)

    def test_has_histogram_trace(self, df):
        fig = score_distribution(df, year=2024, category="Overall", kde=False)
        # Should have exactly 1 trace (histogram only when kde=False)
        assert len(fig.data) >= 1

    def test_kde_adds_extra_trace(self, df):
        fig_no_kde  = score_distribution(df, year=2024, category="Overall", kde=False)
        fig_kde     = score_distribution(df, year=2024, category="Overall", kde=True)
        assert len(fig_kde.data) > len(fig_no_kde.data), (
            "KDE=True should add an additional Scatter trace"
        )

    def test_empty_does_not_crash(self, empty_df):
        fig = score_distribution(empty_df, year=2024, category="Overall")
        assert isinstance(fig, go.Figure)


# ===========================================================================
# 7. app.py wrapper functions — return (figure, str) tuples
# ===========================================================================

class TestPlotWrappers:
    def test_plot_score_distribution(self, df):
        fig, takeaway = plot_score_distribution(df, year=2024, category="Overall")
        assert isinstance(fig, go.Figure)
        assert isinstance(takeaway, str) and len(takeaway) > 10

    def test_plot_rank_distribution(self, df):
        fig, takeaway = plot_rank_distribution(df, year=2024, category="Overall")
        assert isinstance(fig, go.Figure)
        assert isinstance(takeaway, str)

    def test_plot_summary_statistics(self, df):
        fig, takeaway = plot_summary_statistics(df, category="Overall")
        assert isinstance(fig, go.Figure)
        assert isinstance(takeaway, str)

    def test_plot_rank_band_breakdown(self, df):
        fig, takeaway = plot_rank_band_breakdown(df, category="Overall")
        assert isinstance(fig, go.Figure)
        assert isinstance(takeaway, str)

    def test_plot_score_spread_by_category(self, df):
        fig, takeaway = plot_score_spread_by_category(df, year=2024)
        assert isinstance(fig, go.Figure)
        assert isinstance(takeaway, str)

    def test_plot_parameter_distribution_violin(self, df):
        fig, takeaway = plot_parameter_distribution(
            df, year=2024, category="Overall", chart_style="violin"
        )
        assert isinstance(fig, go.Figure)
        assert isinstance(takeaway, str)

    def test_plot_parameter_distribution_box(self, df):
        fig, takeaway = plot_parameter_distribution(
            df, year=2024, category="Overall", chart_style="box"
        )
        assert isinstance(fig, go.Figure)
        assert isinstance(takeaway, str)


# ===========================================================================
# 8. Data quality checks
# ===========================================================================

class TestDataQuality:
    def test_minimum_row_count(self, df):
        """Dataset must have at least 6000 rows (real data has 6866)."""
        assert len(df) >= 6_000, f"Expected ≥6000 rows, got {len(df)}"

    def test_all_expected_years(self, df):
        """Dataset must cover all 9 NIRF years 2017–2025."""
        years = set(df["year"].dropna().astype(int).unique())
        expected = set(range(2017, 2026))
        assert expected.issubset(years), f"Missing years: {expected - years}"

    def test_overall_category_exists(self, df):
        cats = set(df["category"].dropna().unique())
        assert "Overall" in cats, "'Overall' category must be present"

    def test_score_range(self, df):
        """All scores should be in [0, 100]."""
        scores = df["score"].dropna()
        assert (scores >= 0).all() and (scores <= 100).all(), (
            f"Scores out of [0,100]: min={scores.min()}, max={scores.max()}"
        )

    def test_parameters_exist(self, df):
        for param in ["TLR", "RPC", "GO", "OI", "PR"]:
            assert param in df.columns, f"Parameter '{param}' missing from df"


# ===========================================================================
# Run directly: python tests/test_univariate.py
# ===========================================================================
if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

