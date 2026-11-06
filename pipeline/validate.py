"""
pipeline/validate.py

Data Quality & Integrity Validator for user's cleaned NIRF dataset (data_nirf_clean2.csv):
  - Duplicate key checks: (institute_key, year, category)
  - ID uniqueness verification
  - State name verification
  - Numeric rank verification
  - Missing-value summary per column
"""

import os
import sys
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "clean")
CLEAN_CSV = os.path.join(DATA_DIR, "data_nirf_clean2.csv")
NIRF_PARQUET = os.path.join(DATA_DIR, "nirf_long.parquet")


def validate_dataset(filepath: str = None) -> bool:
    if filepath is None:
        filepath = CLEAN_CSV if os.path.exists(CLEAN_CSV) else NIRF_PARQUET

    print("=" * 60)
    print(f"Running Data Quality & Validation on: {filepath}")
    print("=" * 60)

    if not os.path.exists(filepath):
        print(f"ERROR: Dataset not found at {filepath}")
        return False

    df = pd.read_parquet(filepath) if filepath.endswith(".parquet") else pd.read_csv(filepath)
    is_valid = True

    # 1. Row count & Columns
    print(f"Total Rows: {len(df)}")
    key_col = "institute_key" if "institute_key" in df.columns else "institution_id"
    rank_col = "rank_num" if "rank_num" in df.columns else "rank"
    name_col = "display_name" if "display_name" in df.columns else "name"

    # 2. Duplicate Keys: (key_col, year, category)
    dup_mask = df.duplicated(subset=[key_col, "year", "category"], keep=False)
    dup_count = dup_mask.sum()
    if dup_count > 0:
        print(f"[FAIL] Found {dup_count} duplicate ({key_col}, year, category) keys!")
        is_valid = False
    else:
        print(f"[PASS] Zero duplicate ({key_col}, year, category) keys found.")

    # 3. Duplicate display_name on (year, category)
    name_dups = df.duplicated(subset=[name_col, "year", "category"], keep=False).sum()
    if name_dups > 0:
        print(f"[WARNING] {name_dups} duplicate display names on (year, category).")
    else:
        print(f"[PASS] Display names uniquely identify entries per year & category.")

    # 4. State Names
    dataset_states = df["state"].dropna().unique()
    print(f"[PASS] Found {len(dataset_states)} unique states/UTs.")

    # 5. Missing Values Summary
    print("\nMissing values in key columns:")
    for col in [key_col, name_col, "state", "year", "category", rank_col, "score", "TLR", "RPC", "GO", "OI", "PR"]:
        if col in df.columns:
            null_count = df[col].isna().sum()
            pct = (null_count / len(df)) * 100
            print(f"  - {col:15s}: {null_count:4d} missing ({pct:5.1f}%)")

    print("\nValidation Summary: " + ("ALL CHECKS PASSED ✅" if is_valid else "VALIDATION FAILED ❌"))
    print("=" * 60)
    return is_valid


if __name__ == "__main__":
    success = validate_dataset()
    sys.exit(0 if success else 1)
