"""Dataset-driven category, year, state and metric metadata."""
from __future__ import annotations

import pandas as pd

METRIC_COLUMNS = {
    "rank": "rank_num",
    "score": "score",
    "TLR": "TLR",
    "RPC": "RPC",
    "GO": "GO",
    "OI": "OI",
    "PR": "PR",
    "QNR": "QNR",
    "QLR": "QLR",
    "SFC": "SFC",
}


class NIRFMetadata:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.categories = sorted(df["category"].dropna().astype(str).unique().tolist())
        self.years = sorted(int(y) for y in df["year"].dropna().unique())
        self.states = sorted(df["state"].dropna().astype(str).unique().tolist())

    def years_for_category(self, category: str) -> list[int]:
        subset = self.df[self.df["category"].str.casefold() == category.casefold()]
        return sorted(int(y) for y in subset["year"].dropna().unique())

    def categories_for_year(self, year: int) -> list[str]:
        subset = self.df[self.df["year"] == year]
        return sorted(subset["category"].dropna().astype(str).unique().tolist())

    def metrics_for(self, category: str | None = None,
                    year: int | None = None) -> list[str]:
        subset = self.df
        if category:
            subset = subset[subset["category"].str.casefold() == category.casefold()]
        if year is not None:
            subset = subset[subset["year"] == year]

        available = []
        for metric, col in METRIC_COLUMNS.items():
            if col in subset.columns and subset[col].notna().any():
                available.append(metric)
        return available

    def validate(self, category: str | None = None,
                 year: int | None = None,
                 metrics: list[str] | None = None) -> list[str]:
        """Return validation issues; an empty list means the supplied fields are valid."""
        issues: list[str] = []
        subset = self.df

        if category:
            canonical = next(
                (c for c in self.categories if c.casefold() == category.casefold()),
                None,
            )
            if canonical is None:
                issues.append(
                    f"Unknown category '{category}'. Available categories: "
                    + ", ".join(self.categories)
                )
                return issues
            subset = subset[subset["category"] == canonical]

        if year is not None:
            if year not in self.years:
                issues.append(f"Year {year} is not present in the dataset.")
                return issues
            subset = subset[subset["year"] == year]
            if category and subset.empty:
                issues.append(f"Category '{category}' has no records in {year}.")
                return issues

        for metric in metrics or []:
            metric_key = metric.upper() if metric.lower() != "rank" and metric.lower() != "score" else metric.lower()
            if metric_key not in METRIC_COLUMNS:
                issues.append(f"Unknown metric '{metric}'.")
                continue
            column = METRIC_COLUMNS[metric_key]
            if subset.empty or subset[column].notna().sum() == 0:
                scope = f" for {category}" if category else ""
                scope += f" in {year}" if year is not None else ""
                issues.append(f"Metric '{metric_key}' is unavailable{scope}.")

        return issues

    def summary(self) -> dict:
        category_years = {
            category: self.years_for_category(category)
            for category in self.categories
        }
        category_metrics = {
            category: self.metrics_for(category=category)
            for category in self.categories
        }
        return {
            "years": self.years,
            "categories": self.categories,
            "states": self.states,
            "category_years": category_years,
            "category_metrics": category_metrics,
        }
