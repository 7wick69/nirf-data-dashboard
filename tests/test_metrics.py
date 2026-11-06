"""
tests/test_metrics.py
"""

from core.data import load_nirf_data


def test_compute_derived_metrics():
    df = load_nirf_data()

    # Check that required derived columns exist
    expected_cols = [
        "rank_band",
        "rank_change",
        "score_change_yoy",
        "param_gap",
        "best_param",
        "worst_param",
        "first_year",
        "last_year",
        "years_in_top10",
        "volatility",
        "score_zscore",
        "score_pct_rank",
        "TLR_zscore",
        "RPC_zscore",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing derived column: {col}"

    # Check IIT Madras Overall
    iit_m = df[(df["name"] == "Indian Institute of Technology Madras") & (df["category"] == "Overall")]
    assert len(iit_m) == 9  # 2017 to 2025
    assert iit_m["years_in_top10"].iloc[0] == 9
    print("Metrics test passed on clean uploaded CSV!")


if __name__ == "__main__":
    test_compute_derived_metrics()
