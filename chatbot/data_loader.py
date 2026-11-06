"""Load and validate the cleaned NIRF dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.data import load_nirf_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "clean"
    / "data_nirf_clean2.csv"
)


def load_dataset(filepath: str | None = None) -> pd.DataFrame:
    """Load the dashboard's cleaned, preprocessed NIRF dataset."""
    return load_nirf_data(filepath or str(DATA_PATH))


def data_quality_summary(df: pd.DataFrame) -> dict:
    """Return a lightweight report for startup diagnostics."""
    duplicate_key_cols = ["year", "category", "institute_key"]
    duplicate_key_count = int(df.duplicated(duplicate_key_cols).sum())
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "years": sorted(int(x) for x in df["year"].dropna().unique()),
        "categories": sorted(str(x) for x in df["category"].dropna().unique()),
        "unique_institutions": int(df["institute_key"].nunique(dropna=True)),
        "unique_states": int(df["state"].nunique(dropna=True)),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_year_category_institution_keys": duplicate_key_count,
        "missing_institute_keys": int(df["institute_key"].isna().sum()),
        "missing_display_names": int(df["display_name"].isna().sum()),
        "rank_status_counts": {
            str(k): int(v)
            for k, v in df["rank_status"].value_counts(dropna=False).items()
        },
    }


if __name__ == "__main__":
    data = load_dataset()
    import json
    print(json.dumps(data_quality_summary(data), indent=2, ensure_ascii=False))
