"""
core/transitions.py

Pure data-processing and transition calculation engine for Section 7:
"Ranking Transitions and Institutional Mobility" (NIRF 2017-2025).

This module contains zero UI and zero plotting code. Every function is pure,
deterministic, and thoroughly testable with pytest.

Key Capabilities:
  - Configurable rank band segmentation (Top 10, 11-50, 51-100, 101-150, 151-200, 201+, Unranked)
  - Configurable rank parsing for banded/range values (midpoint vs lower bound)
  - Dual rank basis: 'Within selected state' (re-ranked 1..N inside state) vs 'National rank'
  - Shared pipeline helper `prepare_ranked_frame` ensuring identical state ranking logic across Q1-Q5
  - Same-category institution matching between any two years (Y1, Y2)
  - Categorization of mobility: Moved Up, Moved Down, Retained, New Entry, Dropped Out
  - Question 1: From-Band to To-Band transition matrix, counts, and mover details
  - Question 2: Top 10 entrance, exit, and continuous persistence analysis
  - Question 3: Rank band retention rate calculation across consecutive year pairs
  - Question 4: Diverging movers extraction (Top N risers and fallers)
  - Question 5: Annual mobility composition (Up, Down, Same, New Entry, Dropped Out)
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd


# Default standard rank bands as defined by NIRF reporting brackets
DEFAULT_RANK_BANDS: List[Tuple[str, float, float]] = [
    ("Top 10", 1.0, 10.0),
    ("11-50", 11.0, 50.0),
    ("51-100", 51.0, 100.0),
    ("101-150", 101.0, 150.0),
    ("151-200", 151.0, 200.0),
    ("201+", 201.0, 9999.0),
]


def parse_rank_value(val: Any, band_rank_method: str = "midpoint") -> float:
    """
    Parses a rank value into a numeric float.
    
    Handles:
      - Integers / floats directly (e.g., 42 -> 42.0)
      - Strings with trailing letters / suffixes (e.g., '44A' -> 44.0)
      - Strings specifying ranges/bands (e.g., '101-150'):
          * If band_rank_method == 'midpoint': (101 + 150) / 2 = 125.5
          * If band_rank_method == 'lower': 101.0
      - Missing, withdrawn, or unranked strings -> np.nan
    """
    if val is None or pd.isna(val):
        return np.nan
    
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val) if not np.isnan(val) else np.nan
    
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "withdrawn", "not published", "unranked", "-"):
        return np.nan
    
    # Check for range format: e.g. "101-150" or "101 - 150"
    if "-" in s:
        parts = s.split("-")
        try:
            low = float(parts[0].strip())
            high = float(parts[1].strip())
            if band_rank_method == "lower":
                return low
            return (low + high) / 2.0
        except (ValueError, IndexError):
            pass
    
    # Strip any non-digit suffix (e.g. '44A' -> 44)
    cleaned = "".join(ch for ch in s if ch.isdigit() or ch == ".")
    try:
        return float(cleaned) if cleaned else np.nan
    except ValueError:
        return np.nan


def assign_custom_rank_band(
    rank: float,
    bands: Optional[List[Tuple[str, float, float]]] = None,
    unranked_label: str = "Unranked",
) -> str:
    """
    Assigns a numeric rank to a configured rank band.
    
    Args:
        rank: Numeric rank (or NaN)
        bands: List of tuples (band_name, lower_bound, upper_bound)
        unranked_label: Label to return if rank is NaN or does not fit any band
    """
    if pd.isna(rank):
        return unranked_label
    
    active_bands = bands or DEFAULT_RANK_BANDS
    r = float(rank)
    for name, low, high in active_bands:
        if low <= r <= high:
            return name
    
    # If higher than highest band
    max_high = max(b[2] for b in active_bands)
    if r > max_high:
        return f"{int(max_high)}+"
    
    return unranked_label


def get_category_valid_years(df: pd.DataFrame, category: str) -> List[int]:
    """
    Returns the sorted list of valid ranking years present for a specific category.
    Prevents comparing years where a newly added category did not exist.
    """
    sub = df[df["category"] == category]
    years = sorted(sub["year"].dropna().unique().astype(int).tolist())
    return years


def get_category_consecutive_pairs(df: pd.DataFrame, category: str) -> List[Tuple[int, int]]:
    """
    Returns valid consecutive year pairs for the given category,
    e.g. [(2017, 2018), (2018, 2019), ...].
    """
    years = get_category_valid_years(df, category)
    pairs = []
    for i in range(len(years) - 1):
        if years[i + 1] == years[i] + 1:
            pairs.append((years[i], years[i + 1]))
    return pairs


def prepare_ranked_frame(
    df: pd.DataFrame,
    category: str,
    year: int,
    state: Optional[str] = None,
    basis: str = "Within selected state",
    bands: Optional[List[Tuple[str, float, float]]] = None,
    band_rank_method: str = "midpoint",
) -> pd.DataFrame:
    """
    Unified, shared data-preparation helper for Q1–Q5 mobility analysis.
    
    Guarantees consistent ranking and band assignment across all questions:
      1. Filters strictly to the requested category and year.
      2. Standardizes identifiers (`institution_id`) and names.
      3. Parses numeric ranks (handling bands like '101-150' via midpoint or lower bound).
      4. State filtering & Re-ranking:
         - If `state` is active (not None and != 'All states'):
             * Filters to the state BEFORE assigning bands or re-ranking.
             * Computes within-state rank: 1 = best national rank in that state,
               resolving ties with method='min' (standard sporting rank).
             * If `basis == 'Within selected state'`:
                 Primary rank and bands are derived from the within-state rank.
             * If `basis == 'National rank'`:
                 Primary rank and bands remain the official national ranks.
         - If no state is selected ('All states' or None):
             * Primary rank and bands are national (both basis options yield national ranks).
    
    Returns DataFrame containing:
      ['institution_id', 'name', 'city', 'state', 'rank', 'rank_national',
       'rank_state', 'score', 'rank_band', 'rank_band_national']
    """
    sub = df[(df["category"] == category) & (df["year"] == year)].copy()
    
    is_state_filtered = bool(state and state != "All states")
    
    # 1. State filtering occurs BEFORE band assignment or re-ranking
    if is_state_filtered:
        sub = sub[sub["state"] == state].copy()
        
    if sub.empty:
        return pd.DataFrame(columns=[
            "institution_id", "name", "city", "state", "rank", "rank_national",
            "rank_state", "score", "rank_band", "rank_band_national"
        ])
    
    # Standardize identifier
    if "institution_id" not in sub.columns:
        if "institute_key" in sub.columns:
            sub["institution_id"] = sub["institute_key"]
        else:
            sub["institution_id"] = sub["name"]
            
    if "name" not in sub.columns:
        sub["name"] = sub.get("display_name", sub["institution_id"])
    else:
        if "display_name" in sub.columns:
            sub["name"] = sub["display_name"].fillna(sub["name"])

    if "city" not in sub.columns:
        sub["city"] = "Unknown"
    if "score" not in sub.columns:
        sub["score"] = np.nan
            
    # 3. Standardize national rank
    if "rank" in sub.columns:
        rank_series = sub["rank"]
    elif "rank_num" in sub.columns:
        rank_series = sub["rank_num"]
    elif "rank_raw" in sub.columns:
        rank_series = sub["rank_raw"]
    else:
        rank_series = pd.Series(np.nan, index=sub.index)
        
    sub["rank_national"] = rank_series.apply(lambda v: parse_rank_value(v, band_rank_method=band_rank_method))
    
    # Deduplicate in case of duplicate submissions in same year-category
    sub = sub.sort_values(by=["institution_id", "rank_national"]).drop_duplicates(
        subset=["institution_id"], keep="first"
    ).copy()
    
    # 4. Compute within-state re-ranking (method='min', ties get same rank, preserves NaN)
    if is_state_filtered:
        sub["rank_state"] = sub["rank_national"].rank(method="min", ascending=True, na_option="keep")
    else:
        sub["rank_state"] = sub["rank_national"]
        
    # 5. Assign active working rank based on user-selected basis
    if is_state_filtered and basis == "Within selected state":
        sub["rank"] = sub["rank_state"]
    else:
        sub["rank"] = sub["rank_national"]
        
    # 6. Assign rank bands
    sub["rank_band"] = sub["rank"].apply(lambda r: assign_custom_rank_band(r, bands=bands))
    sub["rank_band_national"] = sub["rank_national"].apply(lambda r: assign_custom_rank_band(r, bands=bands))
    
    score_col = sub["score"] if "score" in sub.columns else pd.Series(np.nan, index=sub.index)
    
    return sub[[
        "institution_id", "name", "city", "state", "rank", "rank_national",
        "rank_state", "score", "rank_band", "rank_band_national"
    ]]


def standardize_institutions_subset(
    df: pd.DataFrame,
    category: str,
    year: int,
    state: Optional[str] = None,
    band_rank_method: str = "midpoint",
    bands: Optional[List[Tuple[str, float, float]]] = None,
) -> pd.DataFrame:
    """
    Backward-compatible wrapper around prepare_ranked_frame using national rank.
    """
    return prepare_ranked_frame(
        df=df,
        category=category,
        year=year,
        state=state,
        basis="National rank",
        bands=bands,
        band_rank_method=band_rank_method,
    )


def compare_two_years(
    df: pd.DataFrame,
    category: str,
    y1: int,
    y2: int,
    state: Optional[str] = None,
    basis: str = "Within selected state",
    bands: Optional[List[Tuple[str, float, float]]] = None,
    band_rank_method: str = "midpoint",
) -> pd.DataFrame:
    """
    Compares institution rankings between Year 1 and Year 2 within the SAME category.
    
    Handles:
      - Continuous institutions (present in both years)
      - Newly entered institutions (present in Y2 only, labeled 'New Entry')
      - Exited institutions (present in Y1 only, labeled 'Dropped Out')
      - Rank change formula: Rank_Y1 - Rank_Y2 (positive = improved, negative = declined)
      - State-aware rank basis: 'Within selected state' vs 'National rank'
      - Classification: 'Moved Up', 'Moved Down', 'Retained', 'New Entry', 'Dropped Out'
    """
    df_y1 = prepare_ranked_frame(
        df, category, y1, state=state, basis=basis, bands=bands, band_rank_method=band_rank_method
    )
    df_y2 = prepare_ranked_frame(
        df, category, y2, state=state, basis=basis, bands=bands, band_rank_method=band_rank_method
    )
    
    merged = pd.merge(
        df_y1,
        df_y2,
        on="institution_id",
        how="outer",
        suffixes=("_y1", "_y2")
    )
    
    # Reconcile metadata
    merged["name"] = merged["name_y2"].fillna(merged["name_y1"])
    merged["city"] = merged["city_y2"].fillna(merged["city_y1"])
    merged["state"] = merged["state_y2"].fillna(merged["state_y1"])
    
    # Fill unranked band labels for newly entered / exited
    merged["band_y1"] = merged["rank_band_y1"].fillna("Unranked")
    merged["band_y2"] = merged["rank_band_y2"].fillna("Unranked")
    
    # Calculate rank changes (positive = improved, e.g. 76 -> 43 is +33)
    merged["rank_change"] = merged["rank_y1"] - merged["rank_y2"]
    merged["rank_change_national"] = merged["rank_national_y1"] - merged["rank_national_y2"]
    
    # Classify mobility status
    def classify_mobility(row):
        has_y1 = pd.notna(row["rank_y1"])
        has_y2 = pd.notna(row["rank_y2"])
        
        if not has_y1 and has_y2:
            return "New Entry"
        if has_y1 and not has_y2:
            return "Dropped Out"
        
        diff = row["rank_change"]
        if diff > 0:
            return "Moved Up"
        elif diff < 0:
            return "Moved Down"
        else:
            return "Retained"

    merged["mobility_status"] = merged.apply(classify_mobility, axis=1)
    
    # Classify band transition status
    def classify_band_status(row):
        b1, b2 = row["band_y1"], row["band_y2"]
        if b1 == "Unranked" and b2 != "Unranked":
            return "New Entry"
        if b1 != "Unranked" and b2 == "Unranked":
            return "Dropped Out"
        if b1 == b2:
            return "Retained in Band"
        
        # Determine if band jumped up or down using rank comparison
        if pd.notna(row["rank_change"]):
            return "Promoted Band" if row["rank_change"] > 0 else "Demoted Band"
        return "Changed Band"

    merged["band_status"] = merged.apply(classify_band_status, axis=1)
    
    return merged


# ---------------------------------------------------------
# Question 1: Band Transition Analysis
# ---------------------------------------------------------
def q1_band_transition(
    df: pd.DataFrame,
    category: str,
    y1: int,
    y2: int,
    from_band: str = "51-100",
    to_band: str = "Top 50",
    state: Optional[str] = None,
    basis: str = "Within selected state",
    bands: Optional[List[Tuple[str, float, float]]] = None,
    band_rank_method: str = "midpoint",
) -> Dict[str, Any]:
    """
    Answers Question 1:
    How many institutions moved from the 51-100 band into the top 50 (or any selected
    From-Band -> To-Band pair) between two selected years?
    
    Supports:
      - 'Within selected state' basis: re-ranked inside the state before band creation
      - 'National rank' basis: national bands filtered to the state
      - Configurable bands and midpoint vs lower-bound parsing
      - Small sample size alert (< 5 in origin band)
    
    Returns:
      - 'movers_df': Institutions matching the transition criterion
      - 'transition_matrix': Crosstab of From-Band (Y1) vs To-Band (Y2)
      - 'kpi_count': Total count of matching movers
      - 'kpi_total_from': Total institutions in from_band in Y1
      - 'kpi_rate': Percentage of from_band that achieved the transition
      - 'kpi_rate_formatted': String like '50.0% (1 of 2)'
      - 'avg_rank_jump': Average positive rank improvement
      - 'is_small_sample': Boolean indicating if origin band has < 5 institutions
      - 'basis_label': Human readable label of active rank basis
      - 'subtitle': Standardized subtitle for charts and cards
    """
    is_state_filtered = bool(state and state != "All states")
    effective_basis = basis if is_state_filtered else "National rank"
    
    merged = compare_two_years(
        df, category, y1, y2, state=state, basis=effective_basis, bands=bands, band_rank_method=band_rank_method
    )
    active_bands = bands or DEFAULT_RANK_BANDS
    band_names = [b[0] for b in active_bands] + ["Unranked"]
    
    # Filter matching movers
    # Support 'Top 50' composite filter (rank <= 50) as well as exact band names
    if to_band == "Top 50":
        matching_mask = (merged["band_y1"] == from_band) & (merged["rank_y2"] <= 50)
    elif to_band == "Top 10":
        matching_mask = (merged["band_y1"] == from_band) & (merged["rank_y2"] <= 10)
    else:
        matching_mask = (merged["band_y1"] == from_band) & (merged["band_y2"] == to_band)
        
    movers = merged[matching_mask].copy()
    movers = movers.sort_values(by="rank_change", ascending=False)
    
    # Compute Transition Matrix Crosstab
    # Explicitly categorize using band_names to preserve full matrix structure even if some bands are empty
    y1_series = pd.Categorical(merged["band_y1"], categories=band_names, ordered=True)
    y2_series = pd.Categorical(merged["band_y2"], categories=band_names, ordered=True)
    matrix = pd.crosstab(y1_series, y2_series, dropna=False)
    
    # KPI metrics
    from_total = int((merged["band_y1"] == from_band).sum())
    mover_count = int(len(movers))
    mover_rate = (mover_count / from_total * 100.0) if from_total > 0 else 0.0
    avg_jump = float(movers["rank_change"].mean()) if mover_count > 0 else 0.0

    # Destination breakdown of origin cohort (for 100% destination stacked bar)
    origin_cohort = merged[merged["band_y1"] == from_band].copy()
    target_mask = origin_cohort["institution_id"].isin(movers["institution_id"])
    stayed_mask = (origin_cohort["band_y2"] == from_band) & (~target_mask)
    dropped_mask = (origin_cohort["mobility_status"] == "Dropped Out") | (origin_cohort["band_y2"] == "Unranked")
    other_mask = (~target_mask) & (~stayed_mask) & (~dropped_mask)

    dest_counts = {
        "Moved to Target": int(target_mask.sum()),
        "Stayed in Origin Band": int(stayed_mask.sum()),
        "Moved to Other Bands": int(other_mask.sum()),
        "Dropped Out": int(dropped_mask.sum()),
        "Total Origin": int(len(origin_cohort)),
    }
    
    is_small_sample = from_total > 0 and from_total < 5
    basis_label = "Rank within state" if (is_state_filtered and effective_basis == "Within selected state") else "National rank"
    state_display = state if is_state_filtered else "All India"
    subtitle = f"{category} | {state_display} | {basis_label} | {y1} → {y2}"
    
    return {
        "movers_df": movers[[
            "name", "state", "city", "rank_y1", "rank_y2", "rank_national_y1",
            "rank_national_y2", "rank_change", "band_y1", "band_y2"
        ]],
        "transition_matrix": matrix,
        "dest_counts": dest_counts,
        "kpi_count": mover_count,
        "kpi_total_from": from_total,
        "kpi_rate": round(mover_rate, 1),
        "kpi_rate_formatted": f"{mover_rate:.1f}% ({mover_count} of {from_total})" if from_total > 0 else "0.0% (0 of 0)",
        "avg_rank_jump": round(avg_jump, 1) if mover_count > 0 else None,
        "is_small_sample": is_small_sample,
        "basis_label": basis_label,
        "subtitle": subtitle,
    }


# ---------------------------------------------------------
# Question 2: Top 10 Mobility Analysis
# ---------------------------------------------------------
def q2_top10_mobility(
    df: pd.DataFrame,
    category: str,
    y1: int,
    y2: int,
    state: Optional[str] = None,
    basis: str = "Within selected state",
    band_rank_method: str = "midpoint",
) -> Dict[str, Any]:
    """
    Answers Question 2:
    Which institutions entered or exited the top 10 between consecutive (or selected) years?
    
    Dual behavior based on Rank Basis:
      - 'Within selected state': Analyzes mobility inside the state's own Top 10.
      - 'National rank': Analyzes the National Top 10, highlighting state presence.
    """
    is_state_active = bool(state and state != "All states")
    effective_basis = basis if is_state_active else "National rank"
    
    if is_state_active and effective_basis == "Within selected state":
        # State-level Top 10 mobility
        merged = compare_two_years(
            df, category, y1, y2, state=state, basis="Within selected state", band_rank_method=band_rank_method
        )
        in_top10_y1 = merged["rank_y1"] <= 10.0
        in_top10_y2 = merged["rank_y2"] <= 10.0
        
        entered = merged[~in_top10_y1 & in_top10_y2].copy().sort_values(by="rank_y2")
        exited = merged[in_top10_y1 & ~in_top10_y2].copy().sort_values(by="rank_y1")
        retained = merged[in_top10_y1 & in_top10_y2].copy().sort_values(by="rank_y2")
        all_top10 = merged[in_top10_y1 | in_top10_y2].copy().sort_values(by="rank_y2")
        
        all_top10["is_selected_state"] = True
        entered["is_selected_state"] = True
        exited["is_selected_state"] = True
        retained["is_selected_state"] = True
        
        num_entered = len(entered)
        num_exited = len(exited)
        num_retained = len(retained)
        
        takeaway = (
            f"Within {state} ({category}, {y1}→{y2}, State Rank Basis): {num_entered} institution(s) "
            f"entered the State Top 10, {num_exited} exited, and {num_retained} retained their rank."
        )
        
        cols = ["name", "state", "city", "rank_y1", "rank_y2", "rank_change", "mobility_status", "is_selected_state"]
        return {
            "entered_df": entered[cols] if not entered.empty else pd.DataFrame(columns=cols),
            "exited_df": exited[cols] if not exited.empty else pd.DataFrame(columns=cols),
            "retained_df": retained[cols] if not retained.empty else pd.DataFrame(columns=cols),
            "all_top10_df": all_top10[cols] if not all_top10.empty else pd.DataFrame(columns=cols),
            "takeaway": takeaway,
            "num_entered": num_entered,
            "num_exited": num_exited,
            "num_retained": num_retained,
            "is_state_active": True,
            "basis": "Within selected state",
            "state_name": state,
            "state_top10_count": len(all_top10),
            "state_best_name": all_top10.iloc[0]["name"] if not all_top10.empty else None,
            "state_best_rank_y2": 1.0 if not all_top10.empty else None,
        }

    # National Rank Basis (Default or 'All states')
    merged_all = compare_two_years(
        df, category, y1, y2, state=None, basis="National rank", band_rank_method=band_rank_method
    )
    
    in_top10_y1 = merged_all["rank_y1"] <= 10.0
    in_top10_y2 = merged_all["rank_y2"] <= 10.0
    
    entered_all = merged_all[~in_top10_y1 & in_top10_y2].copy().sort_values(by="rank_y2")
    exited_all = merged_all[in_top10_y1 & ~in_top10_y2].copy().sort_values(by="rank_y1")
    retained_all = merged_all[in_top10_y1 & in_top10_y2].copy().sort_values(by="rank_y2")
    all_top10_cohort = merged_all[in_top10_y1 | in_top10_y2].copy().sort_values(by="rank_y2")
    
    if is_state_active:
        all_top10_cohort["is_selected_state"] = all_top10_cohort["state"] == state
        entered_all["is_selected_state"] = entered_all["state"] == state
        exited_all["is_selected_state"] = exited_all["state"] == state
        retained_all["is_selected_state"] = retained_all["state"] == state
    else:
        all_top10_cohort["is_selected_state"] = False
        entered_all["is_selected_state"] = False
        exited_all["is_selected_state"] = False
        retained_all["is_selected_state"] = False

    cols = ["name", "state", "city", "rank_y1", "rank_y2", "rank_change", "mobility_status", "is_selected_state"]
    
    # State-specific standing info
    state_institutions = merged_all[merged_all["state"] == state].copy() if is_state_active else pd.DataFrame()
    state_top10 = all_top10_cohort[all_top10_cohort["state"] == state] if is_state_active else pd.DataFrame()

    state_best_name = None
    state_best_rank_y1 = None
    state_best_rank_y2 = None
    if not state_institutions.empty:
        if state_institutions["rank_y2"].notna().any():
            best_row_y2 = state_institutions.loc[state_institutions["rank_y2"].idxmin()]
            state_best_name = best_row_y2["name"]
            state_best_rank_y2 = best_row_y2["rank_y2"]
        if state_institutions["rank_y1"].notna().any():
            state_best_rank_y1 = state_institutions["rank_y1"].min()

    num_entered = len(entered_all)
    num_exited = len(exited_all)
    num_retained = len(retained_all)
    state_top10_count = len(state_top10)
    
    if is_state_active:
        if state_top10_count > 0:
            state_names = ", ".join(state_top10["name"].tolist())
            takeaway = (
                f"In '{category}' ({y1}→{y2}), {state_top10_count} institution(s) from {state} placed in the National Top 10: "
                f"{state_names}. Nationally, {num_entered} institution(s) entered the Top 10 and {num_retained} retained their position."
            )
        else:
            contender_str = f" (Highest-ranking contender from {state}: {state_best_name} at #{int(state_best_rank_y2)})" if state_best_rank_y2 else ""
            takeaway = (
                f"In '{category}' ({y1}→{y2}), 0 institutions from {state} placed in the National Top 10{contender_str}. "
                f"Nationally, {num_entered} entered and {num_retained} retained their Top 10 standing."
            )
    else:
        if num_entered == 0 and num_exited == 0:
            takeaway = (
                f"The National Top 10 remained completely intact between {y1} and {y2} in '{category}', "
                f"with all {num_retained} institutions retaining their top-tier standing."
            )
        else:
            entered_names = ", ".join(entered_all["name"].tolist()[:2])
            exited_names = ", ".join(exited_all["name"].tolist()[:2])
            takeaway = (
                f"Between {y1} and {y2} in '{category}', {num_entered} institution(s) entered the National Top 10 "
                f"({entered_names}) while {num_exited} dropped out ({exited_names}). "
                f"{num_retained} institution(s) maintained their Top 10 position."
            )
        
    return {
        "entered_df": entered_all[cols],
        "exited_df": exited_all[cols],
        "retained_df": retained_all[cols],
        "all_top10_df": all_top10_cohort[cols],
        "takeaway": takeaway,
        "num_entered": num_entered,
        "num_exited": num_exited,
        "num_retained": num_retained,
        "is_state_active": is_state_active,
        "basis": "National rank",
        "state_name": state,
        "state_top10_count": state_top10_count,
        "state_best_name": state_best_name,
        "state_best_rank_y1": state_best_rank_y1,
        "state_best_rank_y2": state_best_rank_y2,
    }


# ---------------------------------------------------------
# Question 3: Rank Band Retention Analysis
# ---------------------------------------------------------
def q3_band_retention(
    df: pd.DataFrame,
    category: str,
    state: Optional[str] = None,
    basis: str = "Within selected state",
    bands: Optional[List[Tuple[str, float, float]]] = None,
    band_rank_method: str = "midpoint",
) -> pd.DataFrame:
    """
    Answers Question 3:
    What percentage of institutions remained in the same rank band from one year to the next?
    
    Applies the specified rank basis (state-level re-ranking vs national rank).
    """
    pairs = get_category_consecutive_pairs(df, category)
    rows = []
    
    for y1, y2 in pairs:
        merged = compare_two_years(
            df, category, y1, y2, state=state, basis=basis, bands=bands, band_rank_method=band_rank_method
        )
        
        continuing = merged[pd.notna(merged["rank_y1"]) & pd.notna(merged["rank_y2"])].copy()
        total_continuing = len(continuing)
        
        if total_continuing == 0:
            continue
            
        same_band = (continuing["band_y1"] == continuing["band_y2"]).sum()
        promoted = (continuing["band_status"] == "Promoted Band").sum()
        demoted = (continuing["band_status"] == "Demoted Band").sum()
        rate = (same_band / total_continuing) * 100.0
        
        active_bands = bands or DEFAULT_RANK_BANDS
        band_rates: Dict[str, float] = {}
        for b_name, _, _ in active_bands:
            in_b = continuing[continuing["band_y1"] == b_name]
            b_total = len(in_b)
            b_retained = (in_b["band_y2"] == b_name).sum()
            band_rates[f"{b_name}_retention_pct"] = round((b_retained / b_total * 100), 1) if b_total > 0 else np.nan
        
        row_dict = {
            "pair": f"{y1}→{y2}",
            "y1": y1,
            "y2": y2,
            "continuing_total": total_continuing,
            "same_band_count": same_band,
            "retention_rate_pct": round(rate, 1),
            "promoted_count": promoted,
            "demoted_count": demoted,
        }
        row_dict.update(band_rates)
        rows.append(row_dict)
        
    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Question 4: Largest Upward & Downward Movers
# ---------------------------------------------------------
def q4_biggest_movers(
    df: pd.DataFrame,
    category: str,
    y1: int,
    y2: int,
    top_n: int = 10,
    state: Optional[str] = None,
    basis: str = "Within selected state",
    band_rank_method: str = "midpoint",
) -> Dict[str, Any]:
    """
    Answers Question 4:
    Which institutions had the largest upward and downward rank movements?
    """
    merged = compare_two_years(
        df, category, y1, y2, state=state, basis=basis, band_rank_method=band_rank_method
    )
    
    continuing = merged[pd.notna(merged["rank_y1"]) & pd.notna(merged["rank_y2"])].copy()
    
    risers = (
        continuing[continuing["rank_change"] > 0]
        .sort_values(by="rank_change", ascending=False)
        .head(top_n)
    )
    
    fallers = (
        continuing[continuing["rank_change"] < 0]
        .sort_values(by="rank_change", ascending=True)
        .head(top_n)
    )
    
    cols = ["name", "state", "city", "rank_y1", "rank_y2", "rank_change"]
    
    combined = pd.concat([
        risers.assign(mover_type="Riser"),
        fallers.assign(mover_type="Faller")
    ], ignore_index=True)
    
    if not risers.empty and not fallers.empty:
        top_riser = risers.iloc[0]
        top_faller = fallers.iloc[0]
        takeaway = (
            f"Largest climber: {top_riser['name']} ({top_riser['state']}), climbing +{int(top_riser['rank_change'])} places "
            f"from #{int(top_riser['rank_y1'])} to #{int(top_riser['rank_y2'])}. "
            f"Largest drop: {top_faller['name']} ({top_faller['state']}), falling {int(top_faller['rank_change'])} places "
            f"from #{int(top_faller['rank_y1'])} to #{int(top_faller['rank_y2'])}."
        )
    elif not risers.empty:
        top_riser = risers.iloc[0]
        takeaway = f"Largest climber: {top_riser['name']} (+{int(top_riser['rank_change'])} places). No institutions declined."
    elif not fallers.empty:
        top_faller = fallers.iloc[0]
        takeaway = f"Largest drop: {top_faller['name']} ({int(top_faller['rank_change'])} places). No institutions improved."
    else:
        takeaway = f"All institutions held their exact ranks between {y1} and {y2} in '{category}'."
        
    return {
        "risers_df": risers[cols] if not risers.empty else pd.DataFrame(columns=cols),
        "fallers_df": fallers[cols] if not fallers.empty else pd.DataFrame(columns=cols),
        "combined_df": combined,
        "takeaway": takeaway,
    }


# ---------------------------------------------------------
# Question 5: Annual Mobility Composition Breakdown
# ---------------------------------------------------------
def q5_annual_mobility_summary(
    df: pd.DataFrame,
    category: str,
    state: Optional[str] = None,
    basis: str = "Within selected state",
    bands: Optional[List[Tuple[str, float, float]]] = None,
    band_rank_method: str = "midpoint",
) -> pd.DataFrame:
    """
    Answers Question 5:
    How many institutions moved up, moved down, or retained their positions each year?
    """
    pairs = get_category_consecutive_pairs(df, category)
    records = []
    
    for y1, y2 in pairs:
        merged = compare_two_years(
            df, category, y1, y2, state=state, basis=basis, bands=bands, band_rank_method=band_rank_method
        )
        
        up = (merged["mobility_status"] == "Moved Up").sum()
        down = (merged["mobility_status"] == "Moved Down").sum()
        retained = (merged["mobility_status"] == "Retained").sum()
        new_entry = (merged["mobility_status"] == "New Entry").sum()
        dropped = (merged["mobility_status"] == "Dropped Out").sum()
        total = len(merged)
        
        if total == 0:
            continue
            
        records.append({
            "pair": f"{y1}→{y2}",
            "y1": y1,
            "y2": y2,
            "Moved Up": up,
            "Moved Down": down,
            "Retained": retained,
            "New Entry": new_entry,
            "Dropped Out": dropped,
            "Total Tracked": total,
            "Moved Up (%)": round(up / total * 100, 1),
            "Moved Down (%)": round(down / total * 100, 1),
            "Retained (%)": round(retained / total * 100, 1),
            "New Entry (%)": round(new_entry / total * 100, 1),
            "Dropped Out (%)": round(dropped / total * 100, 1),
        })
        
    return pd.DataFrame(records)
