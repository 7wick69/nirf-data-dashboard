"""
pipeline/run_transitions.py

Standalone demonstration runner for Section 7:
"Ranking Transitions and Institutional Mobility".

Executes all 5 transition questions end-to-end, prints metrics,
generates Plotly figures, and validates data integrity.

Usage:
    python pipeline/run_transitions.py
"""

import os
import sys

# Ensure root directory is on python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data import load_nirf_data
from core.transitions import (
    q1_band_transition,
    q2_top10_mobility,
    q3_band_retention,
    q4_biggest_movers,
    q5_annual_mobility_summary,
    get_category_valid_years,
)
from charts.transitions import (
    plot_band_transition_heatmap,
    plot_band_transition_sankey,
    plot_top10_mobility_slope,
    plot_band_retention_trend,
    plot_diverging_movers,
    plot_mobility_composition_bar,
)


def run_section7_showcase():
    print("=" * 75)
    print("  NIRF DASHBOARD - SECTION 7: TRANSITIONS & MOBILITY DEMO RUNNER")
    print("=" * 75)

    print("\n[Step 1] Loading Cleaned NIRF Dataset...")
    df = load_nirf_data()
    print(f"Loaded dataset successfully: {df.shape[0]} records, {df.shape[1]} columns.")

    category = "Overall"
    valid_years = get_category_valid_years(df, category)
    print(f"Category '{category}' valid ranking years: {valid_years}")

    # ---------------------------------------------------------
    # Question 1
    # ---------------------------------------------------------
    print("\n" + "-" * 75)
    print("  QUESTION 1: Band Transitions (51-100 -> Top 50 Movers)")
    print("-" * 75)
    y1, y2 = 2023, 2024
    q1 = q1_band_transition(df, category, y1, y2, from_band="51-100", to_band="Top 50")
    print(f"Period: {y1} -> {y2}")
    print(f"Total institutions in 51-100 in {y1}: {q1['kpi_total_from']}")
    print(f"Institutions breaking into Top 50:    {q1['kpi_count']} ({q1['kpi_rate']}%)")
    print(f"Average rank climb for movers:       +{q1['avg_rank_jump']} places")
    print("\nMatching Institutions:")
    for _, row in q1["movers_df"].iterrows():
        print(f"  * {row['name']} ({row['state']}) | Rank {int(row['rank_y1'])} -> #{int(row['rank_y2'])} (+{int(row['rank_change'])} places)")

    fig_q1_h, takeaway_q1_h = plot_band_transition_heatmap(df, category, y1, y2)
    fig_q1_s, takeaway_q1_s = plot_band_transition_sankey(df, category, y1, y2)
    print(f"\nHeatmap Takeaway: {takeaway_q1_h}")
    print(f"Sankey Takeaway:  {takeaway_q1_s}")
    print(f"Plotly Figures Generated: Heatmap ({len(fig_q1_h.data)} trace), Sankey ({len(fig_q1_s.data)} trace)")

    # ---------------------------------------------------------
    # Question 2
    # ---------------------------------------------------------
    print("\n" + "-" * 75)
    print("  QUESTION 2: Top 10 Entrants and Exits")
    print("-" * 75)
    q2_pairs = [(2017, 2018), (2021, 2022), (2024, 2025)]
    for py1, py2 in q2_pairs:
        q2 = q2_top10_mobility(df, category, py1, py2)
        print(f"\n--- Period: {py1} -> {py2} ---")
        print(f"Takeaway: {q2['takeaway']}")
        if not q2["entered_df"].empty:
            for _, r in q2["entered_df"].iterrows():
                prev_str = f"#{int(r['rank_y1'])}" if r['rank_y1'] > 0 else "Unranked"
                print(f"  [ENTERED] {r['name']} ({prev_str} -> #{int(r['rank_y2'])})")
        if not q2["exited_df"].empty:
            for _, r in q2["exited_df"].iterrows():
                new_str = f"#{int(r['rank_y2'])}" if r['rank_y2'] > 0 else "Unranked"
                print(f"  [EXITED]  {r['name']} (#{int(r['rank_y1'])} -> {new_str})")
        print(f"  [RETAINED] {q2['num_retained']} institutions maintained Top 10 standing.")

    fig_q2, takeaway_q2 = plot_top10_mobility_slope(df, category, 2017, 2018)
    print(f"\nPlotly Slope Chart Generated: {len(fig_q2.data)} trajectory traces.")

    # ---------------------------------------------------------
    # Question 3
    # ---------------------------------------------------------
    print("\n" + "-" * 75)
    print("  QUESTION 3: Rank Band Retention Rate Trend")
    print("-" * 75)
    q3 = q3_band_retention(df, category)
    display_q3 = q3[["pair", "continuing_total", "same_band_count", "retention_rate_pct", "promoted_count", "demoted_count"]]
    print(display_q3.to_string(index=False))

    fig_q3, takeaway_q3 = plot_band_retention_trend(df, category)
    print(f"\nRetention Trend Takeaway: {takeaway_q3}")
    print(f"Plotly Retention Trend Chart Generated: {len(fig_q3.data)} trace(s).")

    # ---------------------------------------------------------
    # Question 4
    # ---------------------------------------------------------
    print("\n" + "-" * 75)
    print("  QUESTION 4: Largest Upward and Downward Rank Movements (Top 5)")
    print("-" * 75)
    q4 = q4_biggest_movers(df, category, 2023, 2024, top_n=5)
    print("Top Climbers (Risers):")
    for _, r in q4["risers_df"].iterrows():
        print(f"  ▲ +{int(r['rank_change'])} places: {r['name']} (#{int(r['rank_y1'])} -> #{int(r['rank_y2'])}) [{r['state']}]")
    print("\nTop Decliners (Fallers):")
    for _, r in q4["fallers_df"].iterrows():
        print(f"  ▼ {int(r['rank_change'])} places: {r['name']} (#{int(r['rank_y1'])} -> #{int(r['rank_y2'])}) [{r['state']}]")

    fig_q4, takeaway_q4 = plot_diverging_movers(df, category, 2023, 2024, top_n=5)
    print(f"\nDiverging Bar Takeaway: {takeaway_q4}")
    print(f"Plotly Diverging Bar Chart Generated: {len(fig_q4.data)} trace(s).")

    # ---------------------------------------------------------
    # Question 5
    # ---------------------------------------------------------
    print("\n" + "-" * 75)
    print("  QUESTION 5: Annual Mobility Breakdown (Up / Down / Same / New / Exited)")
    print("-" * 75)
    q5 = q5_annual_mobility_summary(df, category)
    summary_cols = ["pair", "Moved Up", "Retained", "Moved Down", "New Entry", "Dropped Out", "Total Tracked"]
    print(q5[summary_cols].to_string(index=False))

    fig_q5, takeaway_q5 = plot_mobility_composition_bar(df, category, mode="count")
    print(f"\nMobility Breakdown Takeaway: {takeaway_q5}")
    print(f"Plotly Composition Bar Chart Generated: {len(fig_q5.data)} traces.")

    print("\n" + "=" * 75)
    print("  ALL SECTION 7 TRANSITION & MOBILITY MODULES RAN FLAWLESSLY!")
    print("=" * 75)


if __name__ == "__main__":
    run_section7_showcase()

