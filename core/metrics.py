"""
core/metrics.py

Computes derived metrics for NIRF ranking data:
  - YoY rank_change (positive means improved rank, e.g. 8 -> 2 is +6)
  - YoY score_change_yoy
  - rank_band ('1-10', '11-50', '51-100', '101+')
  - years_in_top10, first_year, last_year per institution
  - param_gap (max - min across the 5 parameters)
  - best_param and worst_param
  - volatility (std of rank across years per institution and category)
  - per-year zscore and pct_rank for score and each parameter (TLR, RPC, GO, OI, PR)
"""

import numpy as np
import pandas as pd

PARAMETERS = ["TLR", "RPC", "GO", "OI", "PR"]


def assign_rank_band(rank: float) -> str:
    """Categorizes numeric rank into standard NIRF tiers."""
    if pd.isna(rank):
        return "Unranked"
    r = float(rank)
    if r <= 10:
        return "1-10"
    elif r <= 50:
        return "11-50"
    elif r <= 100:
        return "51-100"
    else:
        return "101+"


def compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes nirf_long DataFrame and returns an enriched copy with all derived metrics.
    Preserves all original columns and adds derived columns.
    """
    df = df.copy()

    # Ensure correct data types
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    for col in PARAMETERS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort deterministically
    df = df.sort_values(by=["category", "institution_id", "year"]).reset_index(drop=True)

    # 1. Rank band
    df["rank_band"] = df["rank"].apply(assign_rank_band)

    # 2. YoY Changes (rank_change and score_change_yoy)
    # Group by category + institution_id
    grouped_inst = df.groupby(["category", "institution_id"])

    # rank_change: prev_rank - curr_rank (positive = improved rank, e.g. 8 to 2 is +6)
    df["prev_rank"] = grouped_inst["rank"].shift(1)
    df["rank_change"] = df["prev_rank"] - df["rank"]

    # score_change_yoy: curr_score - prev_score (positive = improved score)
    df["prev_score"] = grouped_inst["score"].shift(1)
    df["score_change_yoy"] = df["score"] - df["prev_score"]

    # 3. Parameter gap, best_param, worst_param
    available_params = [p for p in PARAMETERS if p in df.columns]
    if available_params:
        param_vals = df[available_params]
        df["param_gap"] = param_vals.max(axis=1) - param_vals.min(axis=1)
        
        all_na_mask = param_vals.isna().all(axis=1)
        best_param = pd.Series(index=df.index, dtype="object")
        worst_param = pd.Series(index=df.index, dtype="object")
        valid_indices = df.index[~all_na_mask]
        if len(valid_indices) > 0:
            best_param.loc[valid_indices] = param_vals.loc[valid_indices].idxmax(axis=1)
            worst_param.loc[valid_indices] = param_vals.loc[valid_indices].idxmin(axis=1)
        df["best_param"] = best_param
        df["worst_param"] = worst_param
    else:
        df["param_gap"] = np.nan
        df["best_param"] = None
        df["worst_param"] = None

    # 4. Institution-level metrics across years per category
    # first_year, last_year
    first_last = grouped_inst["year"].agg(first_year="min", last_year="max").reset_index()
    df = df.merge(first_last, on=["category", "institution_id"], how="left")

    # years_in_top10
    top10_counts = (
        df[df["rank"] <= 10]
        .groupby(["category", "institution_id"])["year"]
        .nunique()
        .reset_index()
        .rename(columns={"year": "years_in_top10"})
    )
    df = df.merge(top10_counts, on=["category", "institution_id"], how="left")
    df["years_in_top10"] = df["years_in_top10"].fillna(0).astype(int)

    # volatility: std of rank across years
    volatility = (
        grouped_inst["rank"]
        .std()
        .reset_index()
        .rename(columns={"rank": "volatility"})
    )
    df = df.merge(volatility, on=["category", "institution_id"], how="left")
    df["volatility"] = df["volatility"].fillna(0.0).round(2)

    # 5. Per-year zscore and pct_rank for score and parameters
    metrics_to_scale = ["score"] + available_params
    for col in metrics_to_scale:
        # Group by category and year
        grp = df.groupby(["category", "year"])[col]
        
        # zscore: (x - mean) / std
        means = grp.transform("mean")
        stds = grp.transform("std")
        df[f"{col}_zscore"] = np.where(stds > 0, ((df[col] - means) / stds).round(3), 0.0)

        # percentile rank: 0.0 to 1.0
        df[f"{col}_pct_rank"] = grp.transform(lambda s: s.rank(pct=True)).round(3)

    return df

