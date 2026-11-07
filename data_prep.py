"""
data_prep.py
============
Standalone NIRF data loading and cleaning pipeline.
Called by univariate_app.py; can also be used by any module that needs
a tidy, canonical DataFrame.

Cleaning steps (every action is logged in quality_report["actions"]):
  1. Load from CSV / Parquet
  2. Rename raw columns → canonical output names
  3. Coerce Year → int  (handles "NIRF 2017", "2017", 2017, etc.)
  4. Strip whitespace from all string columns
  5. Standardize State names  (Orissa→Odisha, etc.)
  6. Parse Rank_Original → Rank_Numeric using the **midpoint strategy**
     - Band "101-150" → 125.0, Is_Band_Rank = True
     - Plain "42"    → 42.0,  Is_Band_Rank = False
  7. Coerce numeric columns (Score, TLR, RPC, GO, OI, PR) with error→NaN
  8. Remove exact duplicate rows (key: Year + Category + Institute_ID)
  9. Report missing values per key column
 10. Save cleaned_nirf.csv to project root

Output DataFrame columns:
  Year (int), Category (str), Institute_ID (str), Institute_Name (str),
  City (str), State (str), Rank_Original (str), Rank_Numeric (float),
  Is_Band_Rank (bool), Score (float), TLR (float), RPC (float),
  GO (float), OI (float), PR (float)

Rank-band midpoint assumption:
  Using the midpoint avoids systematic bias toward either the lower or
  upper bound when computing descriptive statistics, ensuring that
  "101-150" institutions are neither over- nor under-estimated.
  The original string is preserved in Rank_Original for full transparency.
"""

import os
import re
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CSV = os.path.join(_ROOT, "data", "clean", "data_nirf_clean2.csv")
_CLEANED_OUT = os.path.join(_ROOT, "cleaned_nirf.csv")

# ---------------------------------------------------------------------------
# State name normalization map
# Source: inconsistencies observed in raw NIRF portal exports
# ---------------------------------------------------------------------------
_STATE_MAP: Dict[str, str] = {
    # Odisha variants
    "Orissa":                           "Odisha",
    "Orisa":                            "Odisha",
    # Tamil Nadu variants
    "Tamilnadu":                        "Tamil Nadu",
    "Tamil Nadu":                       "Tamil Nadu",
    # Delhi variants
    "NCT Of Delhi":                     "Delhi",
    "NCT of Delhi":                     "Delhi",
    "New Delhi":                        "Delhi",
    # Uttarakhand variants
    "Uttaranchal":                      "Uttarakhand",
    "Uttrakhand":                       "Uttarakhand",
    # Puducherry variants
    "Pondicherry":                      "Puducherry",
    "Pondichery":                       "Puducherry",
    # Island/UTs
    "Andaman And Nicobar":              "Andaman and Nicobar Islands",
    "Andaman & Nicobar":                "Andaman and Nicobar Islands",
    "Andaman And Nicobar Islands":      "Andaman and Nicobar Islands",
    # J&K
    "Jammu And Kashmir":                "Jammu and Kashmir",
    "Jammu & Kashmir":                  "Jammu and Kashmir",
    # Other UTs
    "Dadra And Nagar Haveli":           "Dadra and Nagar Haveli",
    "Daman And Diu":                    "Daman and Diu",
    # Chhattisgarh
    "Chattisgarh":                      "Chhattisgarh",
    "Chattisghar":                      "Chhattisgarh",
    # Telangana (sometimes marked as Andhra Pradesh in older data)
    "Telangana":                        "Telangana",
}

# ---------------------------------------------------------------------------
# Column mapping: raw CSV column name → canonical output column name
# Handles both data_nirf_clean2.csv column names and common alternatives
# ---------------------------------------------------------------------------
_COL_MAP = {
    "year":           "Year",
    "category":       "Category",
    "institute_key":  "Institute_ID",
    "name":           "Institute_Name",
    # 'display_name' used as fallback if 'name' absent
    "display_name":   "_display_name_fallback",
    "city":           "City",
    "state":          "State",
    "rank_raw":       "Rank_Original",
    # rank_num → intermediate; becomes Rank_Numeric after band parsing
    "rank_num":       "_rank_num_raw",
    "score":          "Score",
    "TLR":            "TLR",
    "RPC":            "RPC",
    "GO":             "GO",
    "OI":             "OI",
    "PR":             "PR",
}

# Final output column order
_OUT_COLS = [
    "Year", "Category", "Institute_ID", "Institute_Name", "City", "State",
    "Rank_Original", "Rank_Numeric", "Is_Band_Rank",
    "Score", "TLR", "RPC", "GO", "OI", "PR",
]

_NUMERIC_COLS = ["Score", "TLR", "RPC", "GO", "OI", "PR"]


# ---------------------------------------------------------------------------
# Helper: parse rank string → (Rank_Numeric, Is_Band_Rank)
# ---------------------------------------------------------------------------
def _parse_rank_band(raw) -> Tuple[float, bool]:
    """
    Convert a raw rank string into a numeric value and a band flag.

    Strategy: MIDPOINT of band range.
    Rationale: avoids systematic over/under-estimation when computing
    statistics on mixed exact-rank + band-rank datasets.

    Examples
    --------
    "101-150"  → (125.0, True)
    "151-200"  → (175.0, True)
    "201-300"  → (250.0, True)
    "301+"     → (301.0, True)   ← lower bound only; true upper unknown
    "42"       → (42.0,  False)
    None/NaN   → (nan,   False)
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return np.nan, False
    s = str(raw).strip()
    if s == "" or s.lower() in ("nan", "none", "na", "-"):
        return np.nan, False

    # Pattern: "lo-hi" or "lo–hi" (en-dash)
    m = re.match(r'^(\d+)\s*[-\u2013]\s*(\d+)$', s)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return (lo + hi) / 2.0, True

    # Pattern: "lo+" (open-ended upper band)
    m2 = re.match(r'^(\d+)\s*\+$', s)
    if m2:
        return float(m2.group(1)), True

    # Plain integer / float
    try:
        return float(s), False
    except (ValueError, TypeError):
        return np.nan, False


def _standardize_year(val) -> int:
    """
    Convert year values to plain int.
    Handles: 2017, "2017", "NIRF 2017", "NIRF-2017", "2017-18", etc.
    Returns np.nan if conversion fails.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    s = str(val).strip()
    # Extract first 4-digit number
    m = re.search(r'(\d{4})', s)
    if m:
        return int(m.group(1))
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return np.nan


def _standardize_state(state) -> str:
    """Apply state normalization map; return unchanged if not in map."""
    if state is None or (isinstance(state, float) and np.isnan(state)):
        return state
    s = str(state).strip()
    if s in _STATE_MAP:
        return _STATE_MAP[s]
    # Try title-cased lookup
    tc = s.title()
    if tc in _STATE_MAP:
        return _STATE_MAP[tc]
    return s


# ---------------------------------------------------------------------------
# Main cleaning function
# ---------------------------------------------------------------------------
def load_and_clean(
    filepath=None,
    save_cleaned: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Load NIRF raw data and apply the full cleaning pipeline.

    Parameters
    ----------
    filepath : str or file-like, optional
        Path to CSV / Parquet. Defaults to data/clean/data_nirf_clean2.csv.
        Also accepts a file-like object from st.file_uploader.
    save_cleaned : bool
        If True, writes cleaned_nirf.csv to the project root directory.

    Returns
    -------
    df_clean : pd.DataFrame
        Tidy DataFrame with canonical column names.
    quality_report : dict
        Keys: total_rows, duplicates_removed, years, categories,
              actions (list[str]), missing_after (dict col→int).
    """
    actions: list = []

    # ── Step 1: Load ─────────────────────────────────────────────────────────
    path = filepath or _DEFAULT_CSV
    if hasattr(path, "read"):
        # File-like object (e.g. Streamlit uploader)
        fname = getattr(path, "name", "uploaded_file")
        if str(fname).endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, low_memory=False)
        actions.append(f"Loaded uploaded file: {fname}")
    elif isinstance(path, str):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Data file not found: {path}\n"
                f"Expected columns include: year, category, institute_key, name, "
                f"city, state, rank_num, rank_raw, score, TLR, RPC, GO, OI, PR"
            )
        if path.endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, low_memory=False)
        actions.append(f"Loaded {os.path.basename(path)} ({len(df)} rows)")
    else:
        raise TypeError(f"filepath must be a string path or file-like object, got {type(path)}")

    original_rows = len(df)
    actions.append(f"Raw row count: {original_rows}")

    # ── Step 2: Rename columns ────────────────────────────────────────────────
    rename = {}
    for src, dst in _COL_MAP.items():
        # Only rename if source exists AND destination not already set by an earlier mapping
        if src in df.columns and dst not in df.columns:
            rename[src] = dst

    # 'name' takes priority over 'display_name' for Institute_Name
    if "name" in df.columns and "display_name" in df.columns:
        # Will rename 'name' → Institute_Name; skip display_name
        rename.pop("display_name", None)
        rename.pop("_display_name_fallback", None)
    elif "display_name" in df.columns and "name" not in df.columns:
        # Use display_name as the name source
        rename["display_name"] = "Institute_Name"
        rename.pop("_display_name_fallback", None)

    df = df.rename(columns=rename)
    actions.append(f"Renamed {len(rename)} columns: {sorted(rename.keys())}")

    # Ensure required columns exist with sensible fallbacks
    if "Institute_Name" not in df.columns:
        for alt in ["canonical_name", "institution_name", "inst_name"]:
            if alt in df.columns:
                df["Institute_Name"] = df[alt]
                actions.append(f"Used '{alt}' as Institute_Name")
                break
        else:
            df["Institute_Name"] = "Unknown"
            actions.append("WARNING: No institution name column found — set to 'Unknown'")

    if "Institute_ID" not in df.columns:
        for alt in ["institution_id", "inst_id", "inst_key"]:
            if alt in df.columns:
                df["Institute_ID"] = df[alt]
                actions.append(f"Used '{alt}' as Institute_ID")
                break
        else:
            df["Institute_ID"] = ["INST{:04d}".format(i) for i in range(len(df))]
            actions.append("WARNING: No Institute_ID column found — generated synthetic IDs")

    # ── Step 3: Standardize Year ──────────────────────────────────────────────
    if "Year" in df.columns:
        df["Year"] = df["Year"].apply(_standardize_year)
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        # Cast to nullable Int64 to preserve NaN, then we'll convert to int after drop
        df["Year"] = df["Year"].astype("Int64")
        actions.append("Year standardized → int (handles 'NIRF 2017', etc.)")
    else:
        actions.append("WARNING: 'Year' column not found after renaming")

    # ── Step 4: Strip whitespace from string columns ──────────────────────────
    str_cols = ["Category", "Institute_Name", "Institute_ID", "City", "State"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})
    actions.append("Stripped whitespace from string columns")

    # ── Step 5: Standardize State names ──────────────────────────────────────
    if "State" in df.columns:
        before_states = df["State"].unique().tolist()
        df["State"] = df["State"].apply(_standardize_state)
        after_states = df["State"].unique().tolist()
        changed = len(before_states) - len(after_states)
        actions.append(
            f"State names standardized: {len(after_states)} unique states "
            f"({max(0, changed)} variant(s) collapsed)"
        )

    # ── Step 6: Parse ranks → Rank_Numeric + Is_Band_Rank ────────────────────
    # Build Rank_Original from rank_raw if available
    if "Rank_Original" not in df.columns:
        if "_rank_num_raw" in df.columns:
            df["Rank_Original"] = df["_rank_num_raw"].astype(str)
        else:
            df["Rank_Original"] = ""

    # Parse each Rank_Original string
    parsed_ranks = df["Rank_Original"].apply(_parse_rank_band)
    df["Rank_Numeric"] = parsed_ranks.apply(lambda x: x[0])
    df["Is_Band_Rank"] = parsed_ranks.apply(lambda x: x[1])

    # For rows where parsing failed but _rank_num_raw is a valid number, use it
    if "_rank_num_raw" in df.columns:
        fallback_mask = df["Rank_Numeric"].isna() & df["_rank_num_raw"].notna()
        df.loc[fallback_mask, "Rank_Numeric"] = pd.to_numeric(
            df.loc[fallback_mask, "_rank_num_raw"], errors="coerce"
        )
        df.drop(columns=["_rank_num_raw"], inplace=True)

    band_count = int(df["Is_Band_Rank"].sum())
    actions.append(
        f"Rank parsing complete: {band_count} band-rank rows (midpoint strategy), "
        f"{len(df) - band_count} exact numeric ranks"
    )

    # ── Step 7: Coerce numeric columns ───────────────────────────────────────
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Rank_Numeric"] = pd.to_numeric(df["Rank_Numeric"], errors="coerce")
    actions.append(f"Numeric columns coerced to float: {_NUMERIC_COLS}")

    # ── Step 8: Remove duplicates ─────────────────────────────────────────────
    dedup_keys = [k for k in ["Year", "Category", "Institute_ID"] if k in df.columns]
    before_dedup = len(df)
    if dedup_keys:
        df = df.drop_duplicates(subset=dedup_keys, keep="first")
    duplicates_removed = before_dedup - len(df)
    if duplicates_removed > 0:
        actions.append(
            f"Removed {duplicates_removed} duplicate rows "
            f"(dedup key: {dedup_keys}, kept first occurrence)"
        )
    else:
        actions.append(f"No duplicate rows found (checked key: {dedup_keys})")

    # ── Step 9: Select and order output columns ───────────────────────────────
    out_cols = [c for c in _OUT_COLS if c in df.columns]
    df = df[out_cols].reset_index(drop=True)

    # ── Step 10: Missing-value report ─────────────────────────────────────────
    missing_after = {col: int(df[col].isna().sum()) for col in df.columns}
    missing_nonzero = {k: v for k, v in missing_after.items() if v > 0}
    if missing_nonzero:
        actions.append(f"Remaining missing values: {missing_nonzero}")
    else:
        actions.append("No missing values in key output columns after cleaning ✓")

    # ── Step 11: Save cleaned CSV ─────────────────────────────────────────────
    if save_cleaned:
        try:
            df.to_csv(_CLEANED_OUT, index=False)
            actions.append(f"Saved cleaned_nirf.csv → {_CLEANED_OUT}")
        except Exception as e:
            actions.append(f"WARNING: Could not save cleaned_nirf.csv: {e}")

    # ── Build quality report ─────────────────────────────────────────────────
    years = (
        sorted(df["Year"].dropna().astype(int).unique().tolist())
        if "Year" in df.columns else []
    )
    cats = (
        sorted(df["Category"].dropna().unique().tolist())
        if "Category" in df.columns else []
    )

    quality_report = {
        "total_rows":         len(df),
        "duplicates_removed": duplicates_removed,
        "years":              years,
        "categories":         cats,
        "actions":            actions,
        "missing_after":      missing_after,
    }

    return df, quality_report


# ---------------------------------------------------------------------------
# Streamlit-cached wrapper (import streamlit lazily so non-Streamlit code works)
# ---------------------------------------------------------------------------
def get_cached_data(filepath=None):
    """
    Streamlit-cached version of load_and_clean().
    Returns (df, quality_report).
    Usage: df, report = get_cached_data()
    """
    try:
        import streamlit as st

        @st.cache_data(show_spinner="Loading and cleaning NIRF data…")
        def _cached(fp):
            return load_and_clean(fp, save_cleaned=True)

        return _cached(filepath)
    except ImportError:
        # Non-Streamlit context (e.g. pytest)
        return load_and_clean(filepath, save_cleaned=False)


# ---------------------------------------------------------------------------
# CLI entrypoint: python data_prep.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    fp = sys.argv[1] if len(sys.argv) > 1 else None
    df, report = load_and_clean(fp, save_cleaned=True)
    print(f"\n{'='*55}")
    print(f"  NIRF Data Prep — Quality Report")
    print(f"{'='*55}")
    print(f"  Rows:      {report['total_rows']}")
    print(f"  Dupes rm:  {report['duplicates_removed']}")
    print(f"  Years:     {report['years']}")
    print(f"  Categories ({len(report['categories'])}): {report['categories']}")
    print(f"\n  Actions:")
    for a in report["actions"]:
        print(f"    • {a}")
    print(f"\n  Missing values (post-clean):")
    for col, cnt in report["missing_after"].items():
        if cnt > 0:
            print(f"    {col}: {cnt}")
    if not any(v > 0 for v in report["missing_after"].values()):
        print("    None ✓")
    print(f"{'='*55}\n")
    print(df.head(3).to_string())

