"""
core/data.py

Centralized data loading and caching for NIRF rankings.
Loads the user-provided cleaned dataset directly without modifying or dropping data.
"""

import os
from typing import Optional, List, Tuple
import pandas as pd
from core.metrics import compute_derived_metrics

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "clean")
CLEAN_CSV_PATH = os.path.join(DATA_DIR, "data_nirf_clean2.csv")
NIRF_PARQUET_DEFAULT = os.path.join(DATA_DIR, "nirf_long.parquet")
INSTITUTIONS_PARQUET_DEFAULT = os.path.join(DATA_DIR, "institutions.parquet")


def load_nirf_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Loads NIRF dataset directly from the clean file without extra cleaning steps.
    Attaches computed derived metrics.
    """
    if filepath:
        path = filepath
    elif os.path.exists(CLEAN_CSV_PATH):
        path = CLEAN_CSV_PATH
    elif os.path.exists(NIRF_PARQUET_DEFAULT):
        path = NIRF_PARQUET_DEFAULT
    else:
        # Fallback to dummy data if neither exists
        from pipeline.make_dummy_data import generate_dummy_data
        df_raw, _ = generate_dummy_data()
        return compute_derived_metrics(df_raw)

    if path.endswith(".parquet"):
        df_raw = pd.read_parquet(path)
    else:
        df_raw = pd.read_csv(path)

    # Standardize column references if using original clean CSV columns
    if "rank" not in df_raw.columns and "rank_num" in df_raw.columns:
        df_raw["rank"] = df_raw["rank_num"]
    if "institution_id" not in df_raw.columns and "institute_key" in df_raw.columns:
        df_raw["institution_id"] = df_raw["institute_key"]
    if "display_name" in df_raw.columns:
        df_raw["name"] = df_raw["display_name"].fillna(df_raw["name"])

    # Compute derived metrics directly on the dataset
    enriched_df = compute_derived_metrics(df_raw)
    return enriched_df


def load_institutions(filepath: Optional[str] = None) -> pd.DataFrame:
    """Loads institution canonical registry directly from the clean dataset."""
    path = filepath or INSTITUTIONS_PARQUET_DEFAULT
    if os.path.exists(path):
        return pd.read_parquet(path)

    # Build dynamically from the clean dataset
    df = load_nirf_data()
    inst_df = (
        df[["institution_id", "name", "state", "city"]]
        .drop_duplicates(subset=["institution_id"])
        .rename(columns={"name": "canonical_name"})
    )
    return inst_df


def get_categories(df: pd.DataFrame) -> List[str]:
    """Returns sorted unique category names, with 'Overall' placed first if present."""
    cats = sorted(df["category"].dropna().unique().tolist())
    if "Overall" in cats:
        cats.remove("Overall")
        cats.insert(0, "Overall")
    return cats


def get_states(df: pd.DataFrame) -> List[str]:
    """Returns sorted unique state names."""
    return sorted(df["state"].dropna().unique().tolist())


def get_year_range(df: pd.DataFrame) -> Tuple[int, int]:
    """Returns (min_year, max_year)."""
    return int(df["year"].min()), int(df["year"].max())


def get_institutions_list(df: pd.DataFrame, category: Optional[str] = None) -> List[str]:
    """Returns sorted list of institution names, optionally filtered by category."""
    sub = df if category is None else df[df["category"] == category]
    return sorted(sub["name"].dropna().unique().tolist())
