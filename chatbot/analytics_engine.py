"""Dataset-backed analytics engine for the NIRF conversational assistant.

The engine performs deterministic calculations with pandas. It does not call an LLM.
Expected input is the validated query dict produced by query_pipeline.validate_query().
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
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

DISPLAY_COLUMNS = [
    "year", "category", "institute_key", "display_name", "state", "city",
    "rank_num", "rank_raw", "rank_band", "score", "TLR", "RPC", "GO",
    "OI", "PR", "QNR", "QLR", "SFC",
]


class AnalyticsError(ValueError):
    """Raised when a query cannot be executed against the dataset."""


def _plain(value: Any) -> Any:
    """Recursively convert numpy/pandas values into JSON-safe Python values."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    # pd.isna may return an array for non-scalar objects; only evaluate it
    # as a boolean when the result is itself scalar.
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    return value


def _records(frame: pd.DataFrame, columns: list[str] | None = None) -> list[dict]:
    if columns is not None:
        use = [c for c in columns if c in frame.columns]
        frame = frame[use]
    return [
        {str(k): _plain(v) for k, v in row.items()}
        for row in frame.to_dict(orient="records")
    ]


@dataclass
class AnalyticsResult:
    status: str
    intent: str
    title: str
    summary: str
    data: list[dict]
    chart: dict
    statistics: dict
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


class NIRFAnalyticsEngine:
    """Run ranking, trend, geographic and statistical queries on cleaned NIRF data."""

    def __init__(self, df: pd.DataFrame):
        required = {
            "year", "category", "institute_key", "display_name", "state",
            "rank_num", "rank_status", "score",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"NIRF dataset missing columns: {sorted(missing)}")

        self.df = df.copy()
        self.df["year"] = pd.to_numeric(self.df["year"], errors="coerce")
        self.df["rank_num"] = pd.to_numeric(self.df["rank_num"], errors="coerce")
        for metric, col in METRIC_COLUMNS.items():
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

    @staticmethod
    def _query_dict(query: Any) -> dict:
        if hasattr(query, "model_dump"):
            return query.model_dump()
        if isinstance(query, dict):
            return query
        raise TypeError("query must be a dictionary or a Pydantic model")

    def _base_filter(
        self,
        query: dict,
        institution_keys: list[str] | None = None,
        year: int | None = None,
        require_ranked: bool = False,
    ) -> pd.DataFrame:
        frame = self.df

        category = query.get("category")
        if category:
            frame = frame[frame["category"].astype("string").str.casefold() == str(category).casefold()]

        selected_year = year if year is not None else query.get("year")
        if selected_year is not None:
            frame = frame[frame["year"] == int(selected_year)]

        states = query.get("states") or []
        if states:
            state_norm = {str(s).strip().casefold() for s in states}
            frame = frame[frame["state"].astype("string").str.strip().str.casefold().isin(state_norm)]

        keys = institution_keys if institution_keys is not None else query.get("institution_keys")
        if keys:
            frame = frame[frame["institute_key"].astype(str).isin({str(k) for k in keys})]

        institutions = query.get("institutions") or []
        if institutions and not keys:
            names = {str(n).strip().casefold() for n in institutions}
            frame = frame[frame["display_name"].astype("string").str.strip().str.casefold().isin(names)]

        if require_ranked:
            frame = frame[frame["rank_num"].notna()]
            if "rank_status" in frame.columns:
                frame = frame[frame["rank_status"].astype("string").str.casefold() == "ranked"]

        return frame.copy()

    @staticmethod
    def _metric_column(metric: str | None) -> str:
        if not metric:
            metric = "score"
        key = metric if metric in ("TLR", "RPC", "GO", "OI", "PR", "QNR", "QLR", "SFC") else metric.lower()
        if key not in METRIC_COLUMNS:
            raise AnalyticsError(f"Unsupported metric: {metric}")
        return METRIC_COLUMNS[key]

    @staticmethod
    def _need(query: dict, fields: list[str]) -> None:
        missing = [field for field in fields if query.get(field) is None]
        if missing:
            raise AnalyticsError("Query needs: " + ", ".join(missing))

    def _result(self, intent: str, title: str, summary: str,
                frame: pd.DataFrame, chart_type: str,
                x: str | None = None, y: str | None = None,
                statistics: dict | None = None, notes: list[str] | None = None,
                data_columns: list[str] | None = None) -> dict:
        chart = {
            "type": chart_type,
            "x": x,
            "y": y,
            "data": _records(frame, data_columns),
        }
        output = AnalyticsResult(
            status="success",
            intent=intent,
            title=title,
            summary=summary,
            data=_records(frame, data_columns),
            chart=chart,
            statistics={k: _plain(v) for k, v in (statistics or {}).items()},
            notes=notes or [],
        )
        return output.to_dict()

    def ranking(self, query: dict, institution_keys: list[str] | None = None) -> dict:
        self._need(query, ["category", "year"])
        frame = self._base_filter(query, institution_keys, require_ranked=True)
        if frame.empty:
            raise AnalyticsError("No ranked institutions match the selected filters.")
        frame = frame.sort_values(["rank_num", "display_name"], ascending=[True, True])
        exact_rank = query.get("exact_rank")
        limit = query.get("rank_limit")
        # Exact rank takes priority
        if exact_rank is not None:
            frame = frame[
                frame["rank_num"] == exact_rank
            ]
        elif limit is not None:
            frame = frame.head(int(limit))
        if frame.empty:
            raise AnalyticsError(
                f"No institution found at the requested rank "
                f"in {query['category']} for {query['year']}."
            )
        metric = query.get("metric") or "rank"
        title = f"{query['category']} rankings – {query['year']}"
        summary = f"Found {len(frame)} ranked institution record(s) for {query['category']} in {query['year']}."
        cols = [c for c in DISPLAY_COLUMNS if c in frame.columns]
        return self._result("ranking", title, summary, frame, "bar", "display_name", "rank_num",
                             {"records": len(frame), "best_rank": frame["rank_num"].min()},
                             ["Lower numerical rank indicates a better position."], cols)

    def geographical(self, query: dict) -> dict:
        self._need(query, ["category", "year"])
        frame = self._base_filter(query, require_ranked=True)
        rank_limit = query.get("rank_limit")
        if rank_limit is not None:
            frame = frame[frame["rank_num"] <= int(rank_limit)]
        grouped = (frame.groupby("state", dropna=False)
                   .agg(institution_count=("institute_key", "nunique"))
                   .reset_index()
                   .sort_values("institution_count", ascending=False))
        if grouped.empty:
            raise AnalyticsError("No ranked records match the geographical filters.")
        threshold_text = f" in the top {rank_limit}" if rank_limit else ""
        summary = f"Counted distinct institutions by state for {query['category']} in {query['year']}{threshold_text}."
        return self._result("geographical_analysis", f"State-wise institutions – {query['category']} {query['year']}",
                            summary, grouped, "map", "state", "institution_count",
                            {"states_represented": len(grouped),
                             "total_distinct_institutions": int(frame["institute_key"].nunique())},
                            ["Counts use distinct canonical institute_key values.",
                             "Only records with valid published ranks are included."],
                            ["state", "institution_count"])

    def comparison(self, query: dict, institution_keys: list[str] | None = None) -> dict:
        self._need(query, ["category", "year"])
        frame = self._base_filter(query, institution_keys, require_ranked=True)
        if frame.empty:
            raise AnalyticsError("No ranked records match the comparison filters.")

        states = query.get("states") or []
        if len(states) >= 2:
            rank_limit = query.get("rank_limit")
            if rank_limit is not None:
                frame = frame[frame["rank_num"] <= int(rank_limit)]
                grouped = (frame.groupby("state")["institute_key"].nunique()
                           .rename("institution_count").reset_index())
                y = "institution_count"
                metric_description = f"distinct institutions in top {rank_limit}"
            else:
                metric = query.get("metric") or "score"
                col = self._metric_column(metric)
                frame = frame[frame[col].notna()]
                grouped = (frame.groupby("state")[col]
                           .agg(["mean", "median", "count"]).reset_index())
                grouped = grouped.rename(columns={"mean": "mean_value", "median": "median_value", "count": "institution_count"})
                y = "mean_value"
                metric_description = f"mean {metric} score"
            grouped = grouped.sort_values(y, ascending=False)
            return self._result("comparison", f"State comparison – {query['category']} {query['year']}",
                                f"Compared {', '.join(states)} using {metric_description}.",
                                grouped, "bar", "state", y,
                                {"states_compared": len(grouped)},
                                ["The comparison uses only records matching the selected category and year."],
                                list(grouped.columns))

        metric = query.get("metric") or "score"
        col = self._metric_column(metric)
        frame = frame[frame[col].notna()]
        cols = [c for c in DISPLAY_COLUMNS if c in frame.columns]
        frame = frame.sort_values(col, ascending=(metric.lower() == "rank"))
        return self._result("comparison", f"Institution comparison – {query['category']} {query['year']}",
                            f"Returned {len(frame)} institution record(s) for comparison using {metric}.",
                            frame, "bar", "display_name", col,
                            {"records": len(frame)}, [], cols)

    def trend(self, query: dict, institution_keys: list[str] | None = None) -> dict:
        self._need(query, ["category", "start_year", "end_year"])
        start, end = int(query["start_year"]), int(query["end_year"])
        if start > end:
            raise AnalyticsError("start_year must be less than or equal to end_year.")
        frame = self._base_filter(query, institution_keys, require_ranked=True)
        frame = frame[frame["year"].between(start, end)]
        metric = query.get("metric") or "rank"
        col = self._metric_column(metric)
        frame = frame[frame[col].notna()]
        if frame.empty:
            raise AnalyticsError("No records found in the selected trend period.")

        if institution_keys or query.get("institutions"):
            frame = frame.sort_values("year")
            cols = [c for c in ["year", "display_name", "state", "rank_num", "score", col] if c in frame.columns]
            return self._result("trend_analysis", f"{metric} trend – {query['category']} ({start}–{end})",
                                f"Returned {len(frame)} year-wise observation(s) for the selected institution(s).",
                                frame, "line", "year", col, {"observations": len(frame)}, [], cols)

        start_rows = frame[frame["year"] == start][["institute_key", "display_name", col]].rename(columns={col: "start_value"})
        end_rows = frame[frame["year"] == end][["institute_key", "display_name", col]].rename(columns={col: "end_value"})
        joined = start_rows.merge(end_rows, on="institute_key", suffixes=("_start", "_end"))
        if joined.empty:
            raise AnalyticsError("No institutions have comparable records in both selected years.")
        joined["display_name"] = joined.get("display_name_end", joined.get("display_name_start"))
        if metric.lower() == "rank":
            joined["change"] = joined["start_value"] - joined["end_value"]
            joined = joined.sort_values("change", ascending=False)
            improved = int((joined["change"] > 0).sum())
            declined = int((joined["change"] < 0).sum())
            summary = f"Compared {len(joined)} institutions present in both years: {improved} improved, {declined} declined, and {(joined['change'] == 0).sum()} retained the same rank."
            notes = ["Positive rank change means improvement because a lower rank number is better.",
                     "Only institutions with valid ranked records in both endpoint years are compared."]
        else:
            joined["change"] = joined["end_value"] - joined["start_value"]
            joined = joined.sort_values("change", ascending=False)
            summary = f"Compared {len(joined)} institutions present in both years by {metric} change."
            notes = ["Positive score change means improvement.", "Only institutions with valid metric values in both endpoint years are compared."]
        result_cols = ["institute_key", "display_name", "start_value", "end_value", "change"]
        return self._result("trend_analysis", f"{metric} change – {query['category']} ({start}–{end})",
                            summary, joined, "bar", "display_name", "change",
                            {"institutions_compared": len(joined),
                             "improved": int((joined["change"] > 0).sum()),
                             "declined": int((joined["change"] < 0).sum()),
                             "unchanged": int((joined["change"] == 0).sum())}, notes, result_cols)

    def correlation(self, query: dict, institution_keys: list[str] | None = None) -> dict:
        self._need(query, ["category", "year", "metric_x", "metric_y"])
        x_col = self._metric_column(query["metric_x"])
        y_col = self._metric_column(query["metric_y"])
        if x_col == y_col:
            raise AnalyticsError("Choose two different metrics for correlation.")
        frame = self._base_filter(query, institution_keys)
        frame = frame.dropna(subset=[x_col, y_col])
        if len(frame) < 2:
            raise AnalyticsError("At least two complete institution records are needed for correlation.")
        pearson = frame[x_col].corr(frame[y_col], method="pearson")
        spearman = frame[x_col].corr(frame[y_col], method="spearman")
        corr_data = frame[["display_name", "state", x_col, y_col]].copy()
        return self._result("correlation_analysis", f"{query['metric_x']} vs {query['metric_y']} – {query['category']} {query['year']}",
                            f"Calculated correlation using {len(frame)} complete institution records.",
                            corr_data, "scatter", x_col, y_col,
                            {"pearson_correlation": pearson,
                             "spearman_correlation": spearman,
                             "sample_size": len(frame)},
                            ["Correlation is calculated only on pairwise-complete observations.",
                             "Correlation does not imply causation."],
                            ["display_name", "state", x_col, y_col])

    def distribution(self, query: dict) -> dict:
        self._need(query, ["category", "year"])
        metric = query.get("metric") or "score"
        col = self._metric_column(metric)
        frame = self._base_filter(query)
        values = frame[col].dropna()
        if values.empty:
            raise AnalyticsError(f"No valid {metric} values exist for the selected filters.")
        stats = values.describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
        hist_counts, bin_edges = np.histogram(values.to_numpy(), bins="auto")
        hist = pd.DataFrame({
            "bin_start": bin_edges[:-1],
            "bin_end": bin_edges[1:],
            "count": hist_counts,
        })
        return self._result("distribution_analysis", f"Distribution of {metric} – {query['category']} {query['year']}",
                            f"Analyzed {len(values)} valid {metric} observations.", hist,
                            "histogram", "bin_start", "count",
                            {"count": len(values), "mean": stats.get("mean"),
                             "median": stats.get("50%"), "std": stats.get("std"),
                             "min": stats.get("min"), "max": stats.get("max"),
                             "q1": stats.get("25%"), "q3": stats.get("75%"),
                             "iqr": stats.get("75%", 0) - stats.get("25%", 0)},
                            ["Missing metric values are excluded; no imputation is performed."],
                            ["bin_start", "bin_end", "count"])

    def statistical_summary(self, query: dict) -> dict:
        self._need(query, ["category", "year"])
        metric = query.get("metric") or "score"
        col = self._metric_column(metric)
        frame = self._base_filter(query)
        values = frame[col].dropna()
        if values.empty:
            raise AnalyticsError(f"No valid {metric} values exist for the selected filters.")
        stats = {
            "count": int(values.count()),
            "mean": values.mean(),
            "median": values.median(),
            "minimum": values.min(),
            "maximum": values.max(),
            "standard_deviation": values.std(),
            "q1": values.quantile(0.25),
            "q3": values.quantile(0.75),
            "iqr": values.quantile(0.75) - values.quantile(0.25),
        }
        table = pd.DataFrame([{"metric": metric, **stats}])
        return self._result("statistical_summary", f"Statistical summary – {metric} ({query['category']} {query['year']})",
                            f"Calculated descriptive statistics from {len(values)} valid observations.",
                            table, "table", "metric", "mean", stats,
                            ["Missing metric values are excluded."], list(table.columns))

    def multivariate(self, query: dict, institution_keys: list[str] | None = None) -> dict:
        self._need(query, ["category", "year"])
        frame = self._base_filter(query, institution_keys, require_ranked=True)
        if frame.empty:
            raise AnalyticsError("No ranked records match the selected filters.")
        available = [m for m in ["TLR", "RPC", "GO", "OI", "PR"] if m in frame.columns and frame[m].notna().any()]
        if not available:
            available = [m for m in ["OI", "PR", "QNR", "QLR", "SFC"] if m in frame.columns and frame[m].notna().any()]
        if len(available) < 2:
            raise AnalyticsError("Not enough parameter columns are available for multivariate analysis.")
        limit = query.get("rank_limit") or 5
        if not institution_keys and not query.get("institutions"):
            frame = frame.sort_values("rank_num").head(int(limit))
        selected = [c for c in ["display_name", "state", "rank_num", "score"] + available if c in frame.columns]
        return self._result("multivariate_analysis", f"Parameter comparison – {query['category']} {query['year']}",
                            f"Compared {len(frame)} institutions across {', '.join(available)}.",
                            frame[selected], "radar", "display_name", None,
                            {"institutions": len(frame), "parameters": available},
                            ["Only parameter columns available for this category/year are included."], selected)

    def transitions(self, query: dict) -> dict:
        self._need(query, ["category", "start_year", "end_year"])
        start, end = int(query["start_year"]), int(query["end_year"])
        base = self._base_filter(query, require_ranked=True)
        base = base[base["year"].isin([start, end])]
        start_df = base[base["year"] == start][["institute_key", "display_name", "rank_num", "rank_band"]].rename(
            columns={"rank_num": "start_rank", "rank_band": "start_band"})
        end_df = base[base["year"] == end][["institute_key", "display_name", "rank_num", "rank_band"]].rename(
            columns={"rank_num": "end_rank", "rank_band": "end_band"})
        merged = start_df.merge(end_df, on="institute_key", suffixes=("_start", "_end"))
        if merged.empty:
            raise AnalyticsError("No institutions have ranked records in both selected years.")
        merged["display_name"] = merged.get("display_name_end", merged.get("display_name_start"))
        merged["movement"] = merged["start_rank"] - merged["end_rank"]
        merged["movement_status"] = np.select(
            [merged["movement"] > 0, merged["movement"] < 0],
            ["Moved up", "Moved down"], default="Retained rank"
        )
        summary = merged["movement_status"].value_counts().to_dict()
        return self._result("transition_analysis", f"Ranking transitions – {query['category']} ({start}–{end})",
                            f"Compared {len(merged)} institutions with records in both years.",
                            merged.sort_values("movement", ascending=False), "sankey",
                            "start_band", "end_band", summary,
                            ["Only institutions with valid ranked records in both endpoint years are included."],
                            ["institute_key", "display_name", "start_rank", "end_rank", "start_band", "end_band", "movement", "movement_status"])

    def outliers(self, query: dict) -> dict:
        self._need(query, ["category", "year"])
        metric = query.get("metric") or "score"
        col = self._metric_column(metric)
        frame = self._base_filter(query)
        valid = frame.dropna(subset=[col]).copy()
        if len(valid) < 4:
            raise AnalyticsError("At least four valid observations are needed for IQR outlier detection.")
        q1, q3 = valid[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out = valid[(valid[col] < lower) | (valid[col] > upper)].copy()
        out = out.sort_values(col, ascending=False)
        cols = [c for c in DISPLAY_COLUMNS if c in out.columns]
        return self._result("outlier_detection", f"{metric} outliers – {query['category']} {query['year']}",
                            f"Detected {len(out)} IQR outlier(s) among {len(valid)} valid observations.",
                            out, "scatter", "display_name", col,
                            {"q1": q1, "q3": q3, "iqr": iqr,
                             "lower_fence": lower, "upper_fence": upper,
                             "outlier_count": len(out), "observations": len(valid)},
                            ["Outliers are identified using the 1.5 × IQR rule; an outlier is not automatically an error."], cols)


    # ------------------------------------------------------------------
    # Annexure A: dedicated time-series and geographical operations
    # ------------------------------------------------------------------

    def _period_frame(self, query: dict, require_ranked: bool = True) -> tuple[pd.DataFrame, int, int, list[int]]:
        self._need(query, ["category", "start_year", "end_year"])
        start, end = int(query["start_year"]), int(query["end_year"])
        if start > end:
            raise AnalyticsError("start_year must be less than or equal to end_year.")
        frame = self._base_filter(query, require_ranked=require_ranked)
        frame = frame[frame["year"].between(start, end)].copy()
        years = sorted(int(y) for y in frame["year"].dropna().unique())
        if not years:
            raise AnalyticsError("No records are available in the selected category and year range.")
        return frame, start, end, years

    def mean_score_by_year(self, query: dict) -> dict:
        """Annual mean score and sample count for the selected category."""
        frame, start, end, years = self._period_frame(query, require_ranked=True)
        metric = query.get("metric") or "score"
        col = self._metric_column(metric)
        valid = frame.dropna(subset=[col]).copy()
        if valid.empty:
            raise AnalyticsError(f"No valid {metric} values exist in the selected period.")
        yearly = (valid.groupby("year", as_index=False)
                  .agg(mean_score=(col, "mean"), median_score=(col, "median"),
                       institution_count=("institute_key", "nunique"))
                  .sort_values("year"))
        # Retain actual year observations only; do not interpolate missing years.
        yearly["year"] = yearly["year"].astype(int)
        result = self._result(
            "trend_analysis",
            f"Average {metric} score over time – {query['category']} ({start}–{end})",
            f"Calculated annual mean {metric} values from {int(yearly['institution_count'].sum())} year-institution observations across {len(yearly)} published years.",
            yearly, "line", "year", "mean_score",
            {"years_observed": len(yearly),
             "first_year_mean": yearly.iloc[0]["mean_score"],
             "last_year_mean": yearly.iloc[-1]["mean_score"],
             "change_in_mean": yearly.iloc[-1]["mean_score"] - yearly.iloc[0]["mean_score"],
             "total_year_institution_observations": int(yearly["institution_count"].sum())},
            ["Annual means use valid published ranked records only.",
             "Missing metric values are excluded without imputation.",
             "No smoothing or interpolation is applied."],
            ["year", "mean_score", "median_score", "institution_count"],
        )
        result["operation"] = "mean_score_by_year"
        return result

    def top10_consistency(self, query: dict) -> dict:
        """Find institutions repeatedly present in the top 10 across the period.

        Consistency is defined transparently as top-10 membership in at least
        60% of the category's observed ranked years, with at least two years.
        The year-by-institution binary matrix powers the membership heatmap.
        """
        frame, start, end, years = self._period_frame(query, require_ranked=True)
        if len(years) < 2:
            raise AnalyticsError("Top-10 consistency requires at least two published ranking years.")
        ranked = frame[["year", "institute_key", "display_name", "state", "rank_num"]].dropna(subset=["rank_num"]).copy()
        ranked["is_top10"] = (ranked["rank_num"] <= 10).astype(int)
        all_keys = ranked[["institute_key", "display_name", "state"]].drop_duplicates("institute_key")
        membership = ranked.pivot_table(index="institute_key", columns="year", values="is_top10", aggfunc="max", fill_value=0)
        membership = membership.reindex(columns=years, fill_value=0).fillna(0).astype(int)
        membership = all_keys.set_index("institute_key").join(membership, how="left").fillna(0)
        membership["top10_year_count"] = membership[years].sum(axis=1).astype(int)
        membership["years_observed"] = len(years)
        membership["consistency_pct"] = (membership["top10_year_count"] / len(years) * 100).round(2)
        threshold = float(query.get("consistency_threshold", 0.60))
        if not 0 < threshold <= 1:
            raise AnalyticsError("consistency_threshold must be greater than 0 and at most 1.")
        minimum_years = max(2, int(np.ceil(threshold * len(years))))
        selected = membership[membership["top10_year_count"] >= minimum_years].copy()
        if selected.empty:
            # Return recurrent top-10 members, while being explicit that none
            # met the configured consistency threshold.
            selected = membership[membership["top10_year_count"] >= 2].copy()
        selected = selected.sort_values(["top10_year_count", "display_name"], ascending=[False, True]).reset_index()
        year_cols = [str(y) for y in years]
        selected = selected.rename(columns={y: str(y) for y in years})
        for y in year_cols:
            selected[y] = pd.to_numeric(selected[y], errors="coerce").fillna(0).astype(int)
        notes = [
            f"Consistent membership threshold: at least {minimum_years} of {len(years)} observed ranking years ({threshold:.0%}).",
            "A value of 1 means the institution ranked in the top 10 that year; 0 means it did not.",
            "Tied ranks are retained because membership is determined using rank_num <= 10.",
        ]
        result = self._result(
            "trend_analysis", f"Institutions consistently in the top 10 – {query['category']} ({start}–{end})",
            f"Found {len(selected)} institutions with recurrent top-10 membership across {len(years)} observed ranking years.",
            selected, "membership_heatmap", "year", "top10_year_count",
            {"institutions_returned": len(selected), "years_observed": len(years),
             "minimum_years_for_consistency": minimum_years,
             "consistency_threshold": threshold,
             "institutions_ever_top10": int((membership["top10_year_count"] > 0).sum())},
            notes, ["institute_key", "display_name", "state", *year_cols,
                    "top10_year_count", "years_observed", "consistency_pct"],
        )
        result["operation"] = "top10_consistency"
        return result

    def state_distribution_by_year(self, query: dict) -> dict:
        """State-wise count of distinct institutions in a rank band, by year."""
        frame, start, end, years = self._period_frame(query, require_ranked=True)
        rank_limit = int(query.get("rank_limit") or 100)
        if rank_limit < 1:
            raise AnalyticsError("rank_limit must be a positive integer.")
        frame = frame[frame["rank_num"] <= rank_limit]
        if frame.empty:
            raise AnalyticsError("No institutions meet the requested rank limit in this period.")
        grouped = (frame.groupby(["state", "year"], dropna=False)["institute_key"]
                   .nunique().rename("institution_count").reset_index())
        pivot = grouped.pivot(index="state", columns="year", values="institution_count").fillna(0)
        pivot = pivot.reindex(columns=years, fill_value=0)
        pivot["period_total"] = pivot.sum(axis=1)
        # Preserve every state in the returned data. The renderer may limit
        # visible heatmap rows for readability, but the table keeps all rows.
        pivot = pivot.sort_values("period_total", ascending=False).reset_index()
        pivot = pivot.rename(columns={y: str(y) for y in years})
        totals_by_year = grouped.groupby("year")["institution_count"].sum().to_dict()
        result = self._result(
            "geographical_analysis", f"State distribution of top {rank_limit} institutions – {query['category']} ({start}–{end})",
            f"Counted distinct institutions by state and year across {len(years)} observed years.",
            pivot, "heatmap", "year", "institution_count",
            {"rank_limit": rank_limit, "states_shown": len(pivot),
             "years_observed": len(years),
             "institutions_by_year": {str(int(k)): int(v) for k, v in totals_by_year.items()}},
            ["Counts use canonical institute_key values.",
             "Only institutions with published ranks at or above the selected rank threshold are included.",
             "Ties at the rank cutoff are retained, so a year's count can exceed the nominal top-N limit.",
             "Zero means no qualifying institution was recorded for that state and year; it is not an imputed score.",
             "All state rows remain in the returned data; the chart renderer may display a readable subset."],
            ["state", *[str(y) for y in years], "period_total"],
        )
        result["operation"] = "state_distribution_by_year"
        return result

    def state_representation_change(self, query: dict) -> dict:
        """Compare each state's share of top-N institutions at the endpoints."""
        frame, start, end, years = self._period_frame(query, require_ranked=True)
        rank_limit = int(query.get("rank_limit") or 100)
        top = frame[frame["rank_num"] <= rank_limit].copy()
        if top.empty:
            raise AnalyticsError("No institutions meet the requested rank limit in this period.")
        start_rows = top[top["year"] == start]
        end_rows = top[top["year"] == end]
        if start_rows.empty or end_rows.empty:
            raise AnalyticsError("Both endpoint years need published ranked records for this comparison.")
        start_counts = start_rows.groupby("state")["institute_key"].nunique().rename("start_count")
        end_counts = end_rows.groupby("state")["institute_key"].nunique().rename("end_count")
        comparison = pd.concat([start_counts, end_counts], axis=1).fillna(0).reset_index()
        comparison["start_count"] = comparison["start_count"].astype(int)
        comparison["end_count"] = comparison["end_count"].astype(int)
        start_total = int(start_rows["institute_key"].nunique())
        end_total = int(end_rows["institute_key"].nunique())
        comparison["start_share_pct"] = comparison["start_count"] / start_total * 100
        comparison["end_share_pct"] = comparison["end_count"] / end_total * 100
        comparison["count_change"] = comparison["end_count"] - comparison["start_count"]
        comparison["share_change_pp"] = comparison["end_share_pct"] - comparison["start_share_pct"]
        comparison = comparison.sort_values("share_change_pp", ascending=False).reset_index(drop=True)
        comparison = comparison.round({"start_share_pct": 2, "end_share_pct": 2, "share_change_pp": 2})
        improved = int((comparison["share_change_pp"] > 0).sum())
        declined = int((comparison["share_change_pp"] < 0).sum())
        result = self._result(
            "geographical_analysis", f"State representation change in top {rank_limit} – {query['category']} ({start}–{end})",
            f"Compared state representation in the top {rank_limit} between {start} and {end}; {improved} states gained share and {declined} lost share.",
            comparison, "state_change_bar", "state", "share_change_pp",
            {"start_institutions": start_total, "end_institutions": end_total,
             "states_gained_share": improved, "states_lost_share": declined,
             "largest_gain_state": comparison.iloc[0]["state"],
             "largest_gain_pp": comparison.iloc[0]["share_change_pp"],
             "largest_decline_state": comparison.iloc[-1]["state"],
             "largest_decline_pp": comparison.iloc[-1]["share_change_pp"]},
            ["Share is calculated as a state's distinct top-N institutions divided by the total distinct top-N institutions in that endpoint year.",
             "share_change_pp is the end-year minus start-year change in percentage points."],
            ["state", "start_count", "end_count", "count_change", "start_share_pct", "end_share_pct", "share_change_pp"],
        )
        result["operation"] = "state_representation_change"
        return result

    def new_entries_by_state(self, query: dict) -> dict:
        """Count institutions newly present in the end-year ranking by state."""
        frame, start, end, years = self._period_frame(query, require_ranked=True)
        start_rows = frame[frame["year"] == start].copy()
        end_rows = frame[frame["year"] == end].copy()
        if start_rows.empty or end_rows.empty:
            raise AnalyticsError("Both endpoint years need published ranked records to identify new entries.")
        start_keys = set(start_rows["institute_key"].astype(str))
        entrants = end_rows[~end_rows["institute_key"].astype(str).isin(start_keys)].copy()
        if entrants.empty:
            raise AnalyticsError(f"No newly appearing institutions were found between {start} and {end}.")
        entrants = entrants.sort_values(["rank_num", "display_name"], ascending=[True, True])
        grouped = (entrants.groupby("state", dropna=False)
                   .agg(new_institution_count=("institute_key", "nunique"))
                   .reset_index()
                   .sort_values("new_institution_count", ascending=False))
        result = self._result(
            "geographical_analysis", f"Newly appearing ranked institutions by state – {query['category']} ({start}–{end})",
            f"Found {entrants['institute_key'].nunique()} institutions ranked in {end} that were not ranked in {start}, across {len(grouped)} states.",
            grouped, "state_new_entries_bar", "state", "new_institution_count",
            {"new_institutions": int(entrants["institute_key"].nunique()),
             "states_with_new_entries": len(grouped), "start_year": start, "end_year": end},
            ["New entry means a canonical institute_key has a valid ranked record in the end year but no valid ranked record in the start year.",
             "Institution identity is matched using institute_key; entrants are assigned to their end-year state."],
            ["state", "new_institution_count"],
        )
        result["details"] = _records(entrants, [c for c in DISPLAY_COLUMNS if c in entrants.columns])
        result["operation"] = "new_entries_by_state"
        return result

    # ------------------------------------------------------------------
    # Annexure A: dedicated univariate and distribution operations
    # ------------------------------------------------------------------

    @staticmethod
    def _group_stats(frame: pd.DataFrame, group_col: str, value_col: str) -> list[dict]:
        """Return JSON-safe descriptive summaries for each observed group."""
        summaries = []
        for group_value, values in frame.groupby(group_col, dropna=False)[value_col]:
            clean = pd.to_numeric(values, errors="coerce").dropna()
            if clean.empty:
                continue
            summaries.append({
                group_col: str(group_value),
                "count": int(clean.count()),
                "mean": float(clean.mean()),
                "median": float(clean.median()),
                "standard_deviation": float(clean.std()) if clean.count() > 1 else None,
                "q1": float(clean.quantile(0.25)),
                "q3": float(clean.quantile(0.75)),
                "iqr": float(clean.quantile(0.75) - clean.quantile(0.25)),
                "minimum": float(clean.min()),
                "maximum": float(clean.max()),
            })
        return summaries

    def rank_distribution(self, query: dict) -> dict:
        """Count distinct ranked institutions in the published rank bands."""
        self._need(query, ["category", "year"])
        frame = self._base_filter(query, require_ranked=True)
        if frame.empty:
            raise AnalyticsError("No ranked institutions are available for the selected category and year.")
        if "rank_band" not in frame.columns:
            frame["rank_band"] = pd.cut(
                frame["rank_num"],
                bins=[0, 10, 50, 100, 150, 200, np.inf],
                labels=["1-10", "11-50", "51-100", "101-150", "151-200", "201+"],
                right=True,
            )
        else:
            missing_band = frame["rank_band"].isna() & frame["rank_num"].notna()
            if missing_band.any():
                fallback = pd.cut(
                    frame.loc[missing_band, "rank_num"],
                    bins=[0, 10, 50, 100, 150, 200, np.inf],
                    labels=["1-10", "11-50", "51-100", "101-150", "151-200", "201+"],
                    right=True,
                )
                frame.loc[missing_band, "rank_band"] = fallback.astype(object)
        grouped = (frame.dropna(subset=["rank_band"])
                   .groupby("rank_band", observed=True)["institute_key"]
                   .nunique().rename("institution_count").reset_index())
        if grouped.empty:
            raise AnalyticsError("No valid rank bands are available for the selected records.")
        order = {name: i for i, name in enumerate(["1-10", "11-50", "51-100", "101-150", "151-200", "201+"])}
        grouped["_order"] = grouped["rank_band"].astype(str).map(order).fillna(999)
        grouped = grouped.sort_values("_order").drop(columns="_order").reset_index(drop=True)
        total = int(frame["institute_key"].nunique())
        grouped["percentage"] = (grouped["institution_count"] / total * 100).round(2)
        result = self._result(
            "distribution_analysis",
            f"Rank distribution – {query['category']} {query['year']}",
            f"Grouped {total} distinct ranked institutions into {len(grouped)} published rank bands.",
            grouped, "rank_band_bar", "rank_band", "institution_count",
            {"ranked_institutions": total, "rank_bands": len(grouped)},
            ["Counts use distinct institute_key values.",
             "Rank bands follow the published rank_band field; fallback bands are used only where that field is missing."],
            ["rank_band", "institution_count", "percentage"],
        )
        result["operation"] = "rank_distribution"
        return result

    def score_quartile_share(self, query: dict) -> dict:
        """Calculate institution shares in data-derived score quartiles."""
        self._need(query, ["category", "year"])
        metric = query.get("metric") or "score"
        col = self._metric_column(metric)
        frame = self._base_filter(query, require_ranked=True)
        valid = frame.dropna(subset=[col]).copy()
        if valid.empty:
            raise AnalyticsError(f"No ranked institutions have a published {metric} value for this selection.")
        values = pd.to_numeric(valid[col], errors="coerce")
        try:
            codes, edges = pd.qcut(values, q=4, labels=False, retbins=True, duplicates="drop")
        except ValueError as exc:
            raise AnalyticsError(f"Could not form score quartiles from the available values: {exc}") from exc
        codes = pd.Series(codes, index=valid.index)
        n_bins = max(1, len(edges) - 1)
        labels = [f"Q{i}" for i in range(1, n_bins + 1)]
        valid["quartile"] = codes.map(lambda v: labels[int(v)] if pd.notna(v) else None)
        valid = valid.dropna(subset=["quartile"])
        grouped = (valid.groupby("quartile", sort=True)
                   .agg(institution_count=("institute_key", "nunique"),
                        lower_bound=(col, "min"), upper_bound=(col, "max"))
                   .reset_index())
        # One row per canonical institution is expected per category/year; use
        # the distinct-institution denominator to make the percentage explicit.
        total = int(valid["institute_key"].nunique())
        grouped["percentage"] = (grouped["institution_count"] / total * 100).round(2)
        result = self._result(
            "distribution_analysis",
            f"{metric} quartile share – {query['category']} {query['year']}",
            f"Distributed {total} ranked institutions with valid {metric} values across {len(grouped)} score quartile(s).",
            grouped, "quartile_bar", "quartile", "percentage",
            {"metric": metric, "institutions_with_metric": total,
             "quartiles_formed": len(grouped), "percentage_total": float(grouped["percentage"].sum())},
            ["Q1 contains the lowest observed metric values and Q4 the highest when four intervals are formed.",
             "Quartile boundaries are calculated from the selected records using empirical quantiles.",
             "Missing metric values are excluded; no score is imputed.",
             "Tied values can cause fewer than four distinct quantile intervals."],
            ["quartile", "institution_count", "percentage", "lower_bound", "upper_bound"],
        )
        result["operation"] = "score_quartile_share"
        return result

    def score_spread_by_category(self, query: dict) -> dict:
        """Compare a metric distribution across all selected NIRF categories."""
        self._need(query, ["year"])
        metric = query.get("metric") or "score"
        col = self._metric_column(metric)
        # This operation is explicitly cross-category, so a single category
        # inferred by the generic parser must not silently narrow the analysis.
        base_query = dict(query)
        base_query.pop("category", None)
        frame = self._base_filter(base_query, require_ranked=True)
        selected_categories = query.get("categories") or []
        if selected_categories:
            wanted = {str(c).strip().casefold() for c in selected_categories}
            frame = frame[frame["category"].astype(str).str.strip().str.casefold().isin(wanted)]
        valid = frame.dropna(subset=[col, "category"]).copy()
        if valid.empty:
            raise AnalyticsError(f"No ranked records with {metric} scores are available across categories in {query['year']}.")
        valid["metric_value"] = pd.to_numeric(valid[col], errors="coerce")
        summaries = self._group_stats(valid, "category", "metric_value")
        count_by_category = valid.groupby("category")["institute_key"].nunique()
        excluded_categories = sorted(
            set(frame["category"].dropna().astype(str))
            - set(valid["category"].dropna().astype(str))
        )
        summary_frame = pd.DataFrame(summaries)
        if summary_frame.empty:
            raise AnalyticsError("No category has enough valid observations for score-spread comparison.")
        result = self._result(
            "distribution_analysis",
            f"{metric} score spread across NIRF categories – {query['year']}",
            f"Compared {metric} distributions across {len(summary_frame)} categories using ranked records with valid metric values.",
            valid, "category_box", "category", "metric_value",
            {"categories_compared": len(summary_frame),
             "institution_count_by_category": {str(k): int(v) for k, v in count_by_category.items()},
             "category_summaries": summaries},
            ["Each box represents the observed institution-level distribution within a category.",
             "Only ranked records with a non-missing metric are included.",
             "Category-specific parameter coverage may differ; missing values are not treated as zero.",
             *([f"Categories excluded because no valid {metric} scores were published: {', '.join(excluded_categories)}."] if excluded_categories else [])],
            ["category", "metric_value", "institute_key", "display_name", "state"],
        )
        result["operation"] = "score_spread_by_category"
        return result

    def top10_vs_rest_distribution(self, query: dict) -> dict:
        """Compare metric distributions for top-10 and remaining ranked institutions."""
        self._need(query, ["category", "year"])
        metric = query.get("metric") or "score"
        col = self._metric_column(metric)
        frame = self._base_filter(query, require_ranked=True)
        valid = frame.dropna(subset=[col]).copy()
        if valid.empty:
            raise AnalyticsError(f"No ranked institutions have a valid {metric} value.")
        valid["rank_group"] = np.where(valid["rank_num"] <= 10, "Top 10", "Remaining institutions")
        valid["metric_value"] = pd.to_numeric(valid[col], errors="coerce")
        summaries = self._group_stats(valid, "rank_group", "metric_value")
        if len(summaries) < 2:
            raise AnalyticsError("Both top-10 and remaining-institution groups are required for this comparison.")
        result = self._result(
            "distribution_analysis",
            f"Top 10 vs remaining institutions – {query['category']} {query['year']}",
            f"Compared {metric} distributions for {int((valid['rank_group'] == 'Top 10').sum())} top-10 records and {int((valid['rank_group'] == 'Remaining institutions').sum())} remaining records.",
            valid, "grouped_box", "rank_group", "metric_value",
            {"group_summaries": summaries,
             "top10_count": summaries[0]["count"] if summaries[0]["rank_group"] == "Top 10" else next((r["count"] for r in summaries if r["rank_group"] == "Top 10"), 0),
             "remaining_count": next((r["count"] for r in summaries if r["rank_group"] == "Remaining institutions"), 0)},
            ["Top-10 membership is based on numeric rank_num <= 10; tied institutions at rank 10 are retained.",
             "Only ranked institutions with a valid selected metric are included."],
            ["rank_group", "metric_value", "rank_num", "display_name", "state", "institute_key"],
        )
        result["operation"] = "top10_vs_rest_distribution"
        return result

    def score_dispersion_by_year(self, query: dict) -> dict:
        """Compare annual institution-level score dispersion across a period."""
        frame, start, end, years = self._period_frame(query, require_ranked=True)
        metric = query.get("metric") or "score"
        col = self._metric_column(metric)
        valid = frame.dropna(subset=[col]).copy()
        if valid.empty:
            raise AnalyticsError(f"No valid {metric} values exist in the selected period.")
        valid["year"] = valid["year"].astype(int)
        summaries = self._group_stats(valid, "year", col)
        if len(summaries) < 2:
            raise AnalyticsError("At least two years with valid metric observations are needed to compare dispersion.")
        result = self._result(
            "distribution_analysis",
            f"{metric} score dispersion by year – {query['category']} ({start}–{end})",
            f"Compared annual {metric} distributions across {len(summaries)} years.",
            valid, "year_box", "year", col,
            {"year_summaries": summaries,
             "years_compared": len(summaries),
             "most_dispersed_year": max(summaries, key=lambda item: item["standard_deviation"] or -1)["year"]},
            ["Dispersion is compared using the standard deviation and interquartile range for each year.",
             "Only ranked records with valid metric values are included; missing years are not interpolated."],
            ["year", col, "display_name", "state", "institute_key"],
        )
        result["operation"] = "score_dispersion_by_year"
        return result

    def parameter_variability(self, query: dict) -> dict:
        """Compare standard deviations across parameters actually published."""
        self._need(query, ["category", "year"])
        frame = self._base_filter(query, require_ranked=True)
        if frame.empty:
            raise AnalyticsError("No ranked records are available for the selected category and year.")
        metric_names = ["TLR", "RPC", "GO", "OI", "PR", "QNR", "QLR", "SFC"]
        summaries = []
        unavailable = []
        for metric in metric_names:
            col = self._metric_column(metric)
            if col not in frame.columns:
                unavailable.append(metric)
                continue
            values = pd.to_numeric(frame[col], errors="coerce").dropna()
            if len(values) < 2:
                unavailable.append(metric)
                continue
            mean = float(values.mean())
            std = float(values.std(ddof=1))
            summaries.append({
                "metric": metric,
                "count": int(values.count()),
                "mean": mean,
                "median": float(values.median()),
                "standard_deviation": std,
                "variance": float(values.var(ddof=1)),
                "coefficient_of_variation": (std / abs(mean) * 100) if mean != 0 else None,
            })
        if not summaries:
            raise AnalyticsError("Fewer than two valid observations exist for every available NIRF parameter.")
        summary_frame = pd.DataFrame(summaries).sort_values("standard_deviation", ascending=False)
        highest = summary_frame.iloc[0]
        result = self._result(
            "statistical_summary",
            f"NIRF parameter variability – {query['category']} {query['year']}",
            f"Compared variability across {len(summary_frame)} parameters with at least two published observations.",
            summary_frame, "variability_bar", "metric", "standard_deviation",
            {"most_variable_parameter": str(highest["metric"]),
             "highest_standard_deviation": float(highest["standard_deviation"]),
             "parameters_compared": len(summary_frame),
             "parameter_summaries": summaries},
            ["Variability is measured using sample standard deviation (ddof=1) on the parameter's observed 0–100 scores.",
             "Only non-missing parameter observations are used; a missing parameter is not treated as zero.",
             *([f"Parameters omitted for insufficient or unavailable data: {', '.join(unavailable)}."] if unavailable else [])],
            ["metric", "count", "mean", "median", "standard_deviation", "variance", "coefficient_of_variation"],
        )
        result["operation"] = "parameter_variability"
        return result

    def median_iqr_over_time(self, query: dict) -> dict:
        """Calculate yearly median, Q1 and Q3 without smoothing or interpolation."""
        frame, start, end, years = self._period_frame(query, require_ranked=True)
        metric = query.get("metric") or "score"
        col = self._metric_column(metric)
        valid = frame.dropna(subset=[col]).copy()
        if valid.empty:
            raise AnalyticsError(f"No valid {metric} observations exist in the selected period.")
        grouped = (valid.groupby("year")[col]
                   .agg(count="count", median="median", q1=lambda s: s.quantile(0.25),
                        q3=lambda s: s.quantile(0.75), minimum="min", maximum="max")
                   .reset_index().sort_values("year"))
        grouped["year"] = grouped["year"].astype(int)
        grouped["iqr"] = grouped["q3"] - grouped["q1"]
        summaries = [{
            "year": int(row.year), "count": int(row.count),
            "median": float(row.median), "q1": float(row.q1), "q3": float(row.q3),
            "iqr": float(row.iqr), "minimum": float(row.minimum), "maximum": float(row.maximum),
        } for row in grouped.itertuples(index=False)]
        result = self._result(
            "distribution_analysis",
            f"Median and IQR of {metric} over time – {query['category']} ({start}–{end})",
            f"Calculated annual median and interquartile range for {metric} across {len(grouped)} observed years.",
            grouped, "median_iqr_trend", "year", "median",
            {"years_observed": len(grouped), "year_summaries": summaries,
             "first_year_median": summaries[0]["median"],
             "last_year_median": summaries[-1]["median"]},
            ["The shaded interval represents Q1 to Q3 for each year.",
             "Only published ranked records with valid metric values are included.",
             "No smoothing, interpolation or missing-value imputation is applied."],
            ["year", "count", "median", "q1", "q3", "iqr", "minimum", "maximum"],
        )
        result["operation"] = "median_iqr_over_time"
        return result

    def score_distribution_by_state(self, query: dict) -> dict:
        """Return institution-level metric observations grouped by state."""
        self._need(query, ["category", "year"])
        metric = query.get("metric") or "score"
        col = self._metric_column(metric)
        frame = self._base_filter(query, require_ranked=True)
        valid = frame.dropna(subset=[col, "state"]).copy()
        if valid.empty:
            raise AnalyticsError(f"No ranked institutions with valid {metric} values and state labels are available.")
        valid["metric_value"] = pd.to_numeric(valid[col], errors="coerce")
        state_counts = valid.groupby("state")["institute_key"].nunique()
        summaries = self._group_stats(valid, "state", "metric_value")
        summary_frame = pd.DataFrame(summaries)
        if summary_frame.empty:
            raise AnalyticsError("No state has valid observations for score distribution.")
        summary_frame = summary_frame.sort_values("median", ascending=False)
        result = self._result(
            "distribution_analysis",
            f"{metric} score distribution by state – {query['category']} {query['year']}",
            f"Compared {metric} distributions for {int(valid['institute_key'].nunique())} ranked institutions across {valid['state'].nunique()} states.",
            valid, "state_box", "state", "metric_value",
            {"states_compared": int(valid["state"].nunique()),
             "institution_count_by_state": {str(k): int(v) for k, v in state_counts.items()},
             "state_summaries": summaries},
            ["Each observation is an institution-level metric score grouped by its published state.",
             "Missing scores and missing state labels are excluded; no values are imputed.",
             "The chart may show a subset of states for readability; all observations remain in the result data."],
            ["state", "metric_value", "institute_key", "display_name", "rank_num", "score"],
        )
        result["operation"] = "score_distribution_by_state"
        return result


    # ------------------------------------------------------------------
    # Annexure A: correlation and multivariate operations
    # ------------------------------------------------------------------

    def parameter_correlation_ranking(self, query: dict) -> dict:
        """Rank available NIRF parameters by association with the overall score.

        Pearson and Spearman coefficients are computed pairwise for each
        parameter. Parameters with insufficient complete observations or no
        variation are excluded rather than assigned a zero coefficient.
        """
        self._need(query, ["category", "year"])
        frame = self._base_filter(query, require_ranked=True)
        score_col = self._metric_column("score")
        if frame.empty:
            raise AnalyticsError("No ranked records are available for this category and year.")

        parameter_names = ["TLR", "RPC", "GO", "OI", "PR", "QNR", "QLR", "SFC"]
        rows: list[dict[str, Any]] = []
        unavailable: list[str] = []
        for metric in parameter_names:
            if metric not in frame.columns:
                unavailable.append(metric)
                continue
            pair = frame[[metric, score_col]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pair) < 3 or pair[metric].nunique() < 2 or pair[score_col].nunique() < 2:
                unavailable.append(metric)
                continue
            pearson = pair[metric].corr(pair[score_col], method="pearson")
            spearman = pair[metric].corr(pair[score_col], method="spearman")
            if pd.isna(pearson) or pd.isna(spearman):
                unavailable.append(metric)
                continue
            rows.append({
                "metric": metric,
                "pearson_correlation": float(pearson),
                "spearman_correlation": float(spearman),
                "absolute_pearson": abs(float(pearson)),
                "sample_size": int(len(pair)),
            })
        if not rows:
            raise AnalyticsError("No NIRF parameter has enough complete, varying observations for correlation ranking.")

        summary = pd.DataFrame(rows).sort_values(
            ["absolute_pearson", "metric"], ascending=[False, True]
        ).reset_index(drop=True)
        summary.insert(0, "correlation_rank", np.arange(1, len(summary) + 1))
        strongest = summary.iloc[0]
        result = self._result(
            "correlation_analysis",
            f"NIRF parameter correlations with overall score – {query['category']} {query['year']}",
            f"Ranked {len(summary)} available parameters by absolute Pearson correlation with the overall score.",
            summary, "correlation_bar", "metric", "pearson_correlation",
            {"strongest_parameter": str(strongest["metric"]),
             "strongest_pearson_correlation": float(strongest["pearson_correlation"]),
             "strongest_spearman_correlation": float(strongest["spearman_correlation"]),
             "parameters_compared": len(summary),
             "parameter_results": rows},
            ["Parameters are ordered by the absolute value of Pearson correlation; the displayed coefficient retains its sign.",
             "Pearson and Spearman use pairwise-complete institution records for each parameter.",
             "Correlation does not imply causation.",
             *([f"Parameters omitted for insufficient, unavailable or constant data: {', '.join(unavailable)}."] if unavailable else [])],
            ["correlation_rank", "metric", "pearson_correlation", "spearman_correlation", "absolute_pearson", "sample_size"],
        )
        result["operation"] = "parameter_correlation_ranking"
        return result

    def correlation_over_time(self, query: dict) -> dict:
        """Calculate yearly research-to-overall-score correlations.

        Defaults to RPC versus overall score as specified by BI05. Each year is
        evaluated independently; years with fewer than three complete pairs or
        constant values are omitted and explained in the result notes.
        """
        frame, start, end, observed_years = self._period_frame(query, require_ranked=True)
        metric_x = query.get("metric_x") or "RPC"
        metric_y = query.get("metric_y") or "score"
        x_col = self._metric_column(metric_x)
        y_col = self._metric_column(metric_y)
        if x_col == y_col:
            raise AnalyticsError("Choose two different metrics for correlation over time.")

        yearly_rows: list[dict[str, Any]] = []
        omitted_years: list[int] = []
        for year in observed_years:
            pair = frame.loc[frame["year"] == year, [x_col, y_col]].apply(
                pd.to_numeric, errors="coerce"
            ).dropna()
            if len(pair) < 3 or pair[x_col].nunique() < 2 or pair[y_col].nunique() < 2:
                omitted_years.append(year)
                continue
            yearly_rows.append({
                "year": int(year),
                "pearson_correlation": float(pair[x_col].corr(pair[y_col], method="pearson")),
                "spearman_correlation": float(pair[x_col].corr(pair[y_col], method="spearman")),
                "sample_size": int(len(pair)),
            })
        if not yearly_rows:
            raise AnalyticsError(
                f"No year in {start}–{end} has at least three complete, varying {metric_x}/{metric_y} observations."
            )

        wide = pd.DataFrame(yearly_rows).sort_values("year")
        long = wide.melt(
            id_vars=["year", "sample_size"],
            value_vars=["pearson_correlation", "spearman_correlation"],
            var_name="correlation_type", value_name="correlation",
        )
        long["correlation_type"] = long["correlation_type"].map({
            "pearson_correlation": "Pearson",
            "spearman_correlation": "Spearman",
        })
        first, last = wide.iloc[0], wide.iloc[-1]
        notes = [
            "Each year is calculated independently using pairwise-complete observations.",
            "Years with fewer than three complete pairs or a constant metric are omitted.",
            "No smoothing, interpolation or missing-year imputation is applied.",
            "Correlation does not imply causation.",
        ]
        if omitted_years:
            notes.append(f"Years omitted for insufficient or non-varying data: {', '.join(map(str, omitted_years))}.")
        result = self._result(
            "correlation_analysis",
            f"{metric_x} vs {metric_y} correlation over time – {query['category']} ({start}–{end})",
            f"Calculated yearly Pearson and Spearman correlations for {len(wide)} observed years.",
            long, "correlation_trend", "year", "correlation",
            {"metric_x": metric_x, "metric_y": metric_y,
             "years_calculated": len(wide),
             "first_year_pearson": float(first["pearson_correlation"]),
             "last_year_pearson": float(last["pearson_correlation"]),
             "pearson_change": float(last["pearson_correlation"] - first["pearson_correlation"]),
             "yearly_results": yearly_rows},
            notes,
            ["year", "correlation_type", "correlation", "sample_size"],
        )
        result["operation"] = "correlation_over_time"
        return result

    def high_low_metric_quadrant(self, query: dict) -> dict:
        """Identify institutions in a high-x/low-y metric quadrant.

        High/low is defined relative to the category/year medians. The chart
        retains the full pairwise-complete population for context, while the
        result table contains only institutions in the requested quadrant.
        """
        self._need(query, ["category", "year"])
        question_id = query.get("question_id")
        metric_x = query.get("metric_x")
        metric_y = query.get("metric_y")
        if not metric_x or not metric_y:
            if question_id == "MUL03":
                metric_x, metric_y = "TLR", "PR"
            else:
                metric_x, metric_y = "RPC", "TLR"
        x_col = self._metric_column(metric_x)
        y_col = self._metric_column(metric_y)
        if x_col == y_col:
            raise AnalyticsError("The two quadrant parameters must be different.")

        frame = self._base_filter(query, require_ranked=True)
        if frame.empty:
            raise AnalyticsError("No ranked institutions match the selected category and year.")
        available = frame[["institute_key", "display_name", "state", "rank_num", "score", x_col, y_col]].copy()
        available[x_col] = pd.to_numeric(available[x_col], errors="coerce")
        available[y_col] = pd.to_numeric(available[y_col], errors="coerce")
        available = available.dropna(subset=[x_col, y_col])
        if len(available) < 4:
            raise AnalyticsError("At least four complete institutions are required for a meaningful median quadrant analysis.")
        x_median = float(available[x_col].median())
        y_median = float(available[y_col].median())
        selected = available[(available[x_col] >= x_median) & (available[y_col] < y_median)].copy()
        if selected.empty:
            raise AnalyticsError("No institutions fall in the high-x/low-y quadrant for these metrics.")
        focus_label = f"High {metric_x} / Low {metric_y}"
        available["quadrant_class"] = np.where(
            (available[x_col] >= x_median) & (available[y_col] < y_median),
            "Target quadrant", "Other quadrants",
        )
        selected["quadrant_gap"] = (selected[x_col] - x_median) + (y_median - selected[y_col])
        selected = selected.sort_values(["quadrant_gap", "rank_num"], ascending=[False, True])

        result = self._result(
            "multivariate_analysis",
            f"Institutions with high {metric_x} and low {metric_y} – {query['category']} {query['year']}",
            f"Found {len(selected)} institutions in the high-{metric_x}/low-{metric_y} quadrant using category/year medians as thresholds.",
            selected, "quadrant_scatter", x_col, y_col,
            {"metric_x": metric_x, "metric_y": metric_y,
             "x_median_threshold": x_median, "y_median_threshold": y_median,
             "qualifying_institutions": len(selected),
             "complete_observations": len(available),
             "focus_quadrant": focus_label},
            [f"High {metric_x} means at or above the category/year median ({x_median:.2f}); low {metric_y} means below its median ({y_median:.2f}).",
             "The scatter chart shows all complete institutions with the qualifying quadrant highlighted; the result table contains qualifying institutions only.",
             "Only pairwise-complete published parameter values are included; missing values are not imputed."],
            ["institute_key", "display_name", "state", "rank_num", "score", x_col, y_col, "quadrant_gap"],
        )
        # Keep the full population in the chart while exposing only the selected
        # high/low group in result.data for the underlying answer table.
        result["chart"]["data"] = _records(available, [
            "institute_key", "display_name", "state", "rank_num", "score",
            x_col, y_col, "quadrant_class",
        ])
        result["operation"] = "high_low_metric_quadrant"
        return result

    def overall_vs_parameter_change(self, query: dict) -> dict:
        """Find score-improving institutions with one or more falling parameters."""
        frame, start, end, observed_years = self._period_frame(query, require_ranked=True)
        start_rows = frame.loc[frame["year"] == start].copy()
        end_rows = frame.loc[frame["year"] == end].copy()
        if start_rows.empty or end_rows.empty:
            raise AnalyticsError("Both endpoint years must contain published ranked records.")

        metrics = [m for m in ["TLR", "RPC", "GO", "OI", "PR"] if m in frame.columns]
        if not metrics:
            raise AnalyticsError("No NIRF parameter columns are available for the selected period.")
        base_cols = ["institute_key", "display_name", "state", "score"] + metrics
        start_data = start_rows[[c for c in base_cols if c in start_rows.columns]].rename(
            columns={c: f"{c}_start" for c in base_cols if c != "institute_key"}
        )
        end_data = end_rows[[c for c in base_cols if c in end_rows.columns]].rename(
            columns={c: f"{c}_end" for c in base_cols if c != "institute_key"}
        )
        matched = start_data.merge(end_data, on="institute_key", how="inner")
        if matched.empty:
            raise AnalyticsError("No institutions have ranked records in both endpoint years.")
        matched["display_name"] = matched.get("display_name_end", matched.get("display_name_start"))
        matched["state"] = matched.get("state_end", matched.get("state_start"))
        matched["score_change"] = pd.to_numeric(matched["score_end"], errors="coerce") - pd.to_numeric(matched["score_start"], errors="coerce")
        delta_cols: list[str] = []
        for metric in metrics:
            start_col, end_col = f"{metric}_start", f"{metric}_end"
            if start_col not in matched.columns or end_col not in matched.columns:
                continue
            delta_col = f"delta_{metric}"
            matched[delta_col] = pd.to_numeric(matched[end_col], errors="coerce") - pd.to_numeric(matched[start_col], errors="coerce")
            delta_cols.append(delta_col)
        valid_score = matched.dropna(subset=["score_change"]).copy()
        has_decline = valid_score[delta_cols].lt(0).any(axis=1)
        qualifying = valid_score[(valid_score["score_change"] > 0) & has_decline].copy()
        if qualifying.empty:
            raise AnalyticsError("No matched institutions increased overall score while at least one comparable parameter declined.")
        qualifying["declining_parameters"] = qualifying[delta_cols].apply(
            lambda row: ", ".join(col.replace("delta_", "") for col, value in row.items() if pd.notna(value) and value < 0), axis=1
        )
        qualifying = qualifying.sort_values("score_change", ascending=False)
        declines_by_metric = {
            col.replace("delta_", ""): int((qualifying[col] < 0).sum()) for col in delta_cols
        }
        result_cols = ["institute_key", "display_name", "state", "score_start", "score_end", "score_change", *delta_cols, "declining_parameters"]
        result = self._result(
            "multivariate_analysis",
            f"Overall score improvement despite parameter declines – {query['category']} ({start}–{end})",
            f"Found {len(qualifying)} matched institutions whose overall score increased while at least one comparable NIRF parameter declined.",
            qualifying, "change_heatmap", "display_name", "score_change",
            {"matched_institutions": len(matched),
             "matched_with_valid_score_change": len(valid_score),
             "overall_score_increased_with_parameter_decline": len(qualifying),
             "parameter_decline_counts": declines_by_metric,
             "mean_overall_score_change": float(qualifying["score_change"].mean())},
            ["Only institutions with ranked records in both endpoint years are compared.",
             "Overall score change is end-year minus start-year score; positive values indicate an increase.",
             "A parameter decline is counted only when that parameter has valid values at both endpoints.",
             "The heatmap shows paired parameter score changes; negative cells indicate declines, positive cells indicate increases.",
             "Missing parameter values are excluded, not imputed as zero."],
            result_cols,
        )
        result["chart"]["data"] = _records(qualifying, result_cols)
        result["operation"] = "overall_vs_parameter_change"
        return result

    def parameter_gap(self, query: dict) -> dict:
        """Rank institutions by the gap between strongest and weakest parameters."""
        self._need(query, ["category", "year"])
        frame = self._base_filter(query, require_ranked=True)
        if frame.empty:
            raise AnalyticsError("No ranked institutions match the selected category and year.")
        metrics = [m for m in ["TLR", "RPC", "GO", "OI", "PR", "QNR", "QLR", "SFC"] if m in frame.columns]
        if len(metrics) < 2:
            raise AnalyticsError("At least two NIRF parameter columns are required for a parameter-gap analysis.")
        values = frame[metrics].apply(pd.to_numeric, errors="coerce")
        has_two = values.notna().sum(axis=1) >= 2
        valid = frame.loc[has_two, ["institute_key", "display_name", "state", "rank_num", "score"]].copy()
        values = values.loc[has_two]
        if valid.empty:
            raise AnalyticsError("No institutions have at least two observed NIRF parameter scores.")
        valid["strongest_parameter"] = values.idxmax(axis=1, skipna=True)
        valid["weakest_parameter"] = values.idxmin(axis=1, skipna=True)
        valid["strongest_score"] = values.max(axis=1, skipna=True)
        valid["weakest_score"] = values.min(axis=1, skipna=True)
        valid["parameter_gap"] = valid["strongest_score"] - valid["weakest_score"]
        valid["parameters_observed"] = values.notna().sum(axis=1).astype(int)
        valid = valid.sort_values(["parameter_gap", "rank_num"], ascending=[False, True])
        limit = int(query.get("result_limit") or 20)
        if limit < 1:
            raise AnalyticsError("result_limit must be a positive integer.")
        selected = valid.head(limit).copy()
        largest = selected.iloc[0]
        result = self._result(
            "multivariate_analysis",
            f"Largest strongest-to-weakest NIRF parameter gaps – {query['category']} {query['year']}",
            f"Ranked institutions by the difference between their strongest and weakest available NIRF parameter scores; showing {len(selected)} of {len(valid)} eligible institutions.",
            selected, "parameter_gap_bar", "display_name", "parameter_gap",
            {"eligible_institutions": len(valid), "institutions_returned": len(selected),
             "largest_gap_institution": str(largest["display_name"]),
             "largest_gap": float(largest["parameter_gap"]),
             "parameters_compared": metrics},
            ["Parameter gap is maximum minus minimum among observed NIRF parameter scores for each institution.",
             "Only institutions with at least two valid parameter scores are included.",
             "Parameter availability differs by category/year; missing metrics are not imputed as zero."],
            ["institute_key", "display_name", "state", "rank_num", "score", "strongest_parameter", "weakest_parameter", "strongest_score", "weakest_score", "parameter_gap", "parameters_observed"],
        )
        result["operation"] = "parameter_gap"
        return result

    # ------------------------------------------------------------------
    # Annexure A: ranking transitions and outlier/performance-gap operations
    # ------------------------------------------------------------------

    @staticmethod
    def _canonical_rank_band(rank: float) -> str:
        """Apply consistent rank bands using the numeric rank (ties retained)."""
        if rank <= 10:
            return "1-10"
        if rank <= 50:
            return "11-50"
        if rank <= 100:
            return "51-100"
        if rank <= 150:
            return "101-150"
        if rank <= 200:
            return "151-200"
        return "201+"

    def _endpoint_rank_pairs(self, query: dict) -> tuple[pd.DataFrame, int, int]:
        self._need(query, ["category", "start_year", "end_year"])
        start, end = int(query["start_year"]), int(query["end_year"])
        if start >= end:
            raise AnalyticsError("Choose two different years with start_year earlier than end_year.")
        frame = self._base_filter(query, require_ranked=True)
        frame = frame[frame["year"].isin([start, end])].copy()
        required = {"institute_key", "display_name", "rank_num", "year"}
        if not required.issubset(frame.columns):
            raise AnalyticsError("The dataset is missing fields required for rank transitions.")
        start_rows = frame[frame["year"] == start][
            ["institute_key", "display_name", "state", "rank_num"]
        ].drop_duplicates("institute_key").rename(columns={
            "display_name": "start_name", "state": "start_state", "rank_num": "start_rank"
        })
        end_rows = frame[frame["year"] == end][
            ["institute_key", "display_name", "state", "rank_num"]
        ].drop_duplicates("institute_key").rename(columns={
            "display_name": "end_name", "state": "end_state", "rank_num": "end_rank"
        })
        merged = start_rows.merge(end_rows, on="institute_key", how="inner", validate="one_to_one")
        if merged.empty:
            raise AnalyticsError("No institutions have valid ranked records in both selected years.")
        merged["display_name"] = merged["end_name"].fillna(merged["start_name"])
        merged["state"] = merged["end_state"].fillna(merged["start_state"])
        merged["start_band"] = merged["start_rank"].map(self._canonical_rank_band)
        merged["end_band"] = merged["end_rank"].map(self._canonical_rank_band)
        merged["rank_change"] = merged["start_rank"] - merged["end_rank"]
        merged["movement_status"] = np.select(
            [merged["rank_change"] > 0, merged["rank_change"] < 0],
            ["Moved up", "Moved down"], default="Retained rank"
        )
        return merged, start, end

    @staticmethod
    def _transition_flow_table(merged: pd.DataFrame, normalized: bool = False) -> pd.DataFrame:
        flows = (merged.groupby(["start_band", "end_band"], dropna=False)
                 .agg(institution_count=("institute_key", "nunique"))
                 .reset_index())
        if normalized:
            row_totals = flows.groupby("start_band")["institution_count"].transform("sum")
            flows["transition_pct"] = np.where(
                row_totals > 0, flows["institution_count"] / row_totals * 100, np.nan
            )
        return flows

    def rank_band_transition_count(self, query: dict) -> dict:
        """Count institutions moving from rank 51–100 into the top 50."""
        merged, start, end = self._endpoint_rank_pairs(query)
        target = merged[(merged["start_rank"].between(51, 100, inclusive="both")) &
                        (merged["end_rank"] <= 50)].copy()
        flows = self._transition_flow_table(merged)
        if target.empty:
            summary = (f"No institutions moved from ranks 51–100 in {start} into the top 50 in {end}. "
                       f"Compared {len(merged)} institutions with valid ranked records in both years.")
            display = merged.sort_values("rank_change", ascending=False).head(20)
        else:
            summary = (f"{len(target)} institution(s) moved from ranks 51–100 in {start} into the top 50 in {end}. "
                       f"Compared {len(merged)} institutions with valid ranked records in both years.")
            display = target.sort_values("rank_change", ascending=False)
        result = self._result(
            "transition_analysis", f"Rank 51–100 to top 50 transitions – {query['category']} ({start}–{end})",
            summary, display, "transition_heatmap", "start_band", "end_band",
            {"institutions_compared": int(len(merged)),
             "moved_51_100_to_top50": int(len(target)),
             "movement_share_pct": float(len(target) / len(merged) * 100),
             "start_year": start, "end_year": end},
            ["Only institutions with valid ranked records in both endpoint years are included.",
             "Rank bands are derived from numeric rank values; tied ranks are retained.",
             "The transition heatmap shows all matched rank-band flows; the result table lists the requested 51–100 to top-50 movers."],
            ["institute_key", "display_name", "state", "start_rank", "end_rank", "start_band", "end_band", "rank_change"],
        )
        result["chart"]["data"] = _records(flows)
        result["details"] = _records(target, ["institute_key", "display_name", "state", "start_rank", "end_rank", "rank_change"])
        result["operation"] = "rank_band_transition_count"
        return result

    def top10_entries_exits(self, query: dict) -> dict:
        """List top-10 membership entries and exits between adjacent years."""
        frame, start, end, observed_years = self._period_frame(query, require_ranked=True)
        years = sorted(y for y in observed_years if start <= y <= end)
        if len(years) < 2:
            raise AnalyticsError("Top-10 entries/exits require at least two observed ranking years.")
        rows: list[dict] = []
        summary_rows: list[dict] = []
        for from_year, to_year in zip(years[:-1], years[1:]):
            if to_year != from_year + 1:
                continue  # Do not treat a multi-year publication gap as a consecutive transition.
            previous = frame[frame["year"] == from_year].copy()
            current = frame[frame["year"] == to_year].copy()
            prev_top = previous[previous["rank_num"] <= 10].drop_duplicates("institute_key")
            curr_top = current[current["rank_num"] <= 10].drop_duplicates("institute_key")
            prev_keys, curr_keys = set(prev_top["institute_key"]), set(curr_top["institute_key"])
            entrant_keys, exit_keys = curr_keys - prev_keys, prev_keys - curr_keys
            for key in entrant_keys:
                r = curr_top[curr_top["institute_key"] == key].iloc[0]
                old = previous[previous["institute_key"] == key]
                rows.append({"year_from": from_year, "year_to": to_year,
                             "transition": "Entered top 10", "institute_key": key,
                             "display_name": r["display_name"], "state": r.get("state"),
                             "start_rank": None, "end_rank": float(r["rank_num"]),
                             "previous_year_rank_status": "No ranked top-10 membership" if old.empty or old["rank_num"].min() > 10 else "Previously outside top 10"})
            for key in exit_keys:
                r = prev_top[prev_top["institute_key"] == key].iloc[0]
                new = current[current["institute_key"] == key]
                next_rank = float(new["rank_num"].iloc[0]) if not new.empty else None
                rows.append({"year_from": from_year, "year_to": to_year,
                             "transition": "Exited top 10", "institute_key": key,
                             "display_name": r["display_name"], "state": r.get("state"),
                             "start_rank": float(r["rank_num"]), "end_rank": next_rank,
                             "next_year_status": "Ranked outside top 10" if next_rank is not None else "No valid ranked record"})
            summary_rows.append({"year_from": from_year, "year_to": to_year,
                                 "entrants": int(len(entrant_keys)), "exits": int(len(exit_keys))})
        if not summary_rows:
            raise AnalyticsError("No adjacent consecutive-year pairs are available in the selected period.")
        events = pd.DataFrame(rows, columns=["year_from", "year_to", "transition", "institute_key",
                                              "display_name", "state", "start_rank", "end_rank",
                                              "previous_year_rank_status", "next_year_status"])
        if events.empty:
            events = pd.DataFrame([{"year_from": start, "year_to": end,
                                    "transition": "No membership changes", "display_name": "No entrants or exits detected"}])
        summary_frame = pd.DataFrame(summary_rows)
        result = self._result(
            "transition_analysis", f"Top-10 entries and exits – {query['category']} ({start}–{end})",
            f"Across {len(summary_rows)} consecutive-year transition(s), found {sum(x['entrants'] for x in summary_rows)} top-10 entr(y/ies) and {sum(x['exits'] for x in summary_rows)} exit(s).",
            events, "table", "transition", "display_name",
            {"year_pairs_compared": len(summary_rows),
             "total_entrants": int(sum(x["entrants"] for x in summary_rows)),
             "total_exits": int(sum(x["exits"] for x in summary_rows)),
             "yearly_counts": summary_rows},
            ["Top-10 membership uses rank_num <= 10, retaining ties.",
             "Only adjacent calendar years are compared; gaps are not bridged.",
             "An institution with no valid ranked record in the next year is labelled as having no valid ranked record, not assumed to have declined."],
            list(events.columns),
        )
        result["operation"] = "top10_entries_exits"
        return result

    def same_band_percentage(self, query: dict) -> dict:
        """Calculate row-normalized rank-band retention between two years."""
        merged, start, end = self._endpoint_rank_pairs(query)
        flows = self._transition_flow_table(merged, normalized=True)
        same_count = int((merged["start_band"] == merged["end_band"]).sum())
        denominator = int(len(merged))
        pct = float(same_count / denominator * 100) if denominator else 0.0
        result = self._result(
            "transition_analysis", f"Rank-band retention – {query['category']} ({start}–{end})",
            f"{same_count} of {denominator} institutions ({pct:.2f}%) remained in the same rank band between {start} and {end}.",
            flows, "transition_heatmap", "start_band", "end_band",
            {"institutions_compared": denominator, "same_band_count": same_count,
             "same_band_percentage": pct,
             "changed_band_count": denominator - same_count,
             "start_year": start, "end_year": end},
            ["Percentages are normalized within each start-year band (each row sums to 100%).",
             "The overall same-band percentage uses institutions with valid ranked records in both endpoint years.",
             "Rank bands are derived from numeric rank values; ties are retained."],
            list(flows.columns),
        )
        result["operation"] = "same_band_percentage"
        return result

    def rank_movement_extremes(self, query: dict) -> dict:
        """Return the strongest upward and downward endpoint rank movers."""
        merged, start, end = self._endpoint_rank_pairs(query)
        limit = max(1, min(int(query.get("rank_limit") or 10), 25))
        upward = merged[merged["rank_change"] > 0].nlargest(limit, "rank_change").copy()
        downward = merged[merged["rank_change"] < 0].nsmallest(limit, "rank_change").copy()
        selected = pd.concat([downward, upward], ignore_index=True).drop_duplicates("institute_key")
        if selected.empty:
            selected = merged.head(min(limit, len(merged))).copy()
        selected["change"] = selected["rank_change"]
        result = self._result(
            "rank_movement", f"Largest upward and downward rank movements – {query['category']} ({start}–{end})",
            f"Compared {len(merged)} institutions: {int((merged['rank_change'] > 0).sum())} moved up, {int((merged['rank_change'] < 0).sum())} moved down, and {int((merged['rank_change'] == 0).sum())} retained the same numerical rank.",
            selected.sort_values("rank_change"), "diverging_bar", "change", "display_name",
            {"institutions_compared": int(len(merged)),
             "improved": int((merged["rank_change"] > 0).sum()),
             "declined": int((merged["rank_change"] < 0).sum()),
             "unchanged": int((merged["rank_change"] == 0).sum()),
             "largest_improvement": float(merged["rank_change"].max()),
             "largest_decline": float(merged["rank_change"].min()),
             "top_n_each_direction": limit},
            ["Positive rank change means improvement because a lower numerical rank is better.",
             "Only institutions with valid ranked records in both endpoint years are compared.",
             f"The chart/table displays up to {limit} institutions in each direction; statistics use the full matched cohort."],
            ["institute_key", "display_name", "state", "start_rank", "end_rank", "rank_change", "change"],
        )
        result["operation"] = "rank_movement_extremes"
        return result

    def movement_counts_by_year(self, query: dict) -> dict:
        """Count up/down/unchanged rank movements for each consecutive year pair."""
        frame, start, end, observed_years = self._period_frame(query, require_ranked=True)
        years = sorted(y for y in observed_years if start <= y <= end)
        yearly_events = []
        yearly_summaries = []
        for from_year, to_year in zip(years[:-1], years[1:]):
            if to_year != from_year + 1:
                continue
            old = frame[frame["year"] == from_year][["institute_key", "rank_num"]].drop_duplicates("institute_key")
            new = frame[frame["year"] == to_year][["institute_key", "rank_num"]].drop_duplicates("institute_key")
            pairs = old.merge(new, on="institute_key", how="inner", suffixes=("_start", "_end"))
            if pairs.empty:
                continue
            change = pairs["rank_num_start"] - pairs["rank_num_end"]
            status = np.select([change > 0, change < 0], ["Moved up", "Moved down"], default="Retained rank")
            counts = pd.Series(status).value_counts().to_dict()
            yearly_summaries.append({"year_from": from_year, "year_to": to_year,
                                     "compared_institutions": int(len(pairs)),
                                     "moved_up": int(counts.get("Moved up", 0)),
                                     "moved_down": int(counts.get("Moved down", 0)),
                                     "unchanged": int(counts.get("Retained rank", 0))})
            for label, count in counts.items():
                yearly_events.append({"year": to_year, "year_from": from_year,
                                       "movement_status": label, "institution_count": int(count)})
        if not yearly_events:
            raise AnalyticsError("No consecutive-year institution pairs are available for movement counts.")
        data = pd.DataFrame(yearly_events).sort_values(["year", "movement_status"])
        result = self._result(
            "transition_analysis", f"Yearly rank movement counts – {query['category']} ({start}–{end})",
            f"Counted upward, downward and unchanged rank movements across {len(yearly_summaries)} consecutive-year transition(s).",
            data, "stacked_bar", "year", "institution_count",
            {"year_pairs_compared": len(yearly_summaries), "yearly_counts": yearly_summaries,
             "total_moved_up": int(sum(x["moved_up"] for x in yearly_summaries)),
             "total_moved_down": int(sum(x["moved_down"] for x in yearly_summaries)),
             "total_unchanged": int(sum(x["unchanged"] for x in yearly_summaries))},
            ["Only institutions with valid ranked records in both adjacent calendar years are counted.",
             "No movement is inferred across missing publication years."],
            ["year", "year_from", "movement_status", "institution_count"],
        )
        result["operation"] = "movement_counts_by_year"
        return result

    @staticmethod
    def _upper_iqr_outliers(values: pd.Series) -> tuple[pd.Series, float, float, float, float]:
        clean = pd.to_numeric(values, errors="coerce")
        q1, q3 = clean.quantile([0.25, 0.75])
        iqr = float(q3 - q1)
        upper = float(q3 + 1.5 * iqr)
        lower = float(q1 - 1.5 * iqr)
        mask = clean > upper
        return mask.fillna(False), float(q1), float(q3), lower, upper

    def research_score_residual_outliers(self, query: dict) -> dict:
        """Find unusually strong RPC values relative to the score–RPC relationship."""
        self._need(query, ["category", "year"])
        frame = self._base_filter(query, require_ranked=True)
        required = ["score", "RPC"]
        if not set(required).issubset(frame.columns):
            raise AnalyticsError("Overall score and RPC columns are required for this analysis.")
        valid = frame.dropna(subset=required).copy()
        if len(valid) < 5 or valid["score"].nunique() < 2:
            raise AnalyticsError("At least five complete institutions with varying overall scores are required for residual analysis.")
        slope, intercept = np.polyfit(valid["score"].to_numpy(float), valid["RPC"].to_numpy(float), 1)
        valid["expected_RPC"] = intercept + slope * valid["score"]
        valid["research_residual"] = valid["RPC"] - valid["expected_RPC"]
        mask, q1, q3, lower, upper = self._upper_iqr_outliers(valid["research_residual"])
        valid["is_outlier"] = mask.astype(bool)
        candidates = valid[valid["is_outlier"]].sort_values("research_residual", ascending=False)
        display = candidates if not candidates.empty else valid.nlargest(min(10, len(valid)), "research_residual")
        is_fallback = candidates.empty
        result = self._result(
            "outlier_detection", f"Research score residual outliers – {query['category']} {query['year']}",
            (f"Detected {len(candidates)} institution(s) with RPC scores unusually above the level expected from overall score "
             f"using an upper 1.5×IQR residual fence among {len(valid)} complete institutions." if not is_fallback else
             f"No institution crossed the upper 1.5×IQR residual fence among {len(valid)} complete institutions; showing the ten highest positive research residual candidates."),
            display, "scatter", "score", "RPC",
            {"observations": int(len(valid)), "outlier_count": int(len(candidates)),
             "residual_q1": q1, "residual_q3": q3, "residual_iqr": q3 - q1,
             "lower_fence": lower, "upper_fence": upper,
             "regression_slope_rpc_on_score": float(slope),
             "regression_intercept": float(intercept), "fallback_candidates_shown": bool(is_fallback)},
            ["Expected RPC is estimated by ordinary least squares using overall score as predictor.",
             "Outliers are unusually positive RPC residuals beyond Q3 + 1.5×IQR; this is an association-based diagnostic, not a causal claim.",
             "Missing score/RPC values are excluded; no imputation is performed."],
            [c for c in ["institute_key", "display_name", "state", "rank_num", "score", "RPC", "expected_RPC", "research_residual", "is_outlier"] if c in display.columns],
        )
        result["chart"]["data"] = _records(valid, [c for c in ["display_name", "state", "rank_num", "score", "RPC", "research_residual", "is_outlier"] if c in valid.columns])
        result["operation"] = "research_score_residual_outliers"
        return result

    def metric_difference_outliers(self, query: dict) -> dict:
        """Find institutions with an unusually large positive x-minus-y metric gap."""
        self._need(query, ["category", "year"])
        metric_x = query.get("metric_x") or "RPC"
        metric_y = query.get("metric_y") or "TLR"
        x_col, y_col = self._metric_column(metric_x), self._metric_column(metric_y)
        if x_col == y_col:
            raise AnalyticsError("Research and teaching must be two different metrics for this comparison.")
        frame = self._base_filter(query, require_ranked=True)
        valid = frame.dropna(subset=[x_col, y_col]).copy()
        if len(valid) < 5:
            raise AnalyticsError("At least five pairwise-complete institutions are required for metric-gap outlier analysis.")
        valid["metric_x"] = valid[x_col]
        valid["metric_y"] = valid[y_col]
        valid["change"] = valid[x_col] - valid[y_col]
        mask, q1, q3, lower, upper = self._upper_iqr_outliers(valid["change"])
        valid["is_outlier"] = mask.astype(bool)
        outliers = valid[valid["is_outlier"]].sort_values("change", ascending=False)
        display = outliers if not outliers.empty else valid.nlargest(min(10, len(valid)), "change")
        no_outlier = outliers.empty
        result = self._result(
            "outlier_detection", f"{metric_x} substantially above {metric_y} – {query['category']} {query['year']}",
            (f"Found {len(outliers)} institution(s) with unusually large positive {metric_x}−{metric_y} gaps, using the upper 1.5×IQR fence." if not no_outlier else
             f"No institution crossed the upper 1.5×IQR fence for {metric_x}−{metric_y}; showing the ten largest positive gaps as candidates."),
            display, "difference_plot", "change", "display_name",
            {"metric_x": metric_x, "metric_y": metric_y, "sample_size": int(len(valid)),
             "outlier_count": int(len(outliers)), "q1": q1, "q3": q3,
             "iqr": q3 - q1, "lower_fence": lower, "upper_fence": upper,
             "fallback_candidates_shown": bool(no_outlier)},
            [f"Gap is calculated as {metric_x} minus {metric_y}; a positive value means {metric_x} is higher.",
             "Outliers are selected using the upper 1.5×IQR fence across pairwise-complete ranked institutions.",
             "Missing parameter values are excluded and never replaced with zero."],
            [c for c in ["institute_key", "display_name", "state", "rank_num", metric_x, metric_y, "change", "is_outlier"] if c in display.columns],
        )
        result["operation"] = "metric_difference_outliers"
        return result

    def score_change_outliers(self, query: dict) -> dict:
        """Detect unusual score jumps/drops between adjacent published years."""
        frame, start, end, observed_years = self._period_frame(query, require_ranked=True)
        metric = query.get("metric") or "score"
        col = self._metric_column(metric)
        if col not in frame.columns:
            raise AnalyticsError(f"Metric {metric} is unavailable in the selected period.")
        years = sorted(y for y in observed_years if start <= y <= end)
        all_changes = []
        for from_year, to_year in zip(years[:-1], years[1:]):
            if to_year != from_year + 1:
                continue
            old = frame[frame["year"] == from_year][["institute_key", "display_name", "state", col]].rename(columns={col: "score_start"})
            new = frame[frame["year"] == to_year][["institute_key", col]].rename(columns={col: "score_end"})
            paired = old.merge(new, on="institute_key", how="inner", validate="one_to_one").dropna(subset=["score_start", "score_end"])
            for row in paired.to_dict("records"):
                row.update({"year_from": from_year, "year": to_year,
                            "score_change": float(row["score_end"] - row["score_start"])})
                all_changes.append(row)
        changes = pd.DataFrame(all_changes)
        if len(changes) < 5:
            raise AnalyticsError("At least five matched institution-year changes are required for score-change outlier detection.")
        mask, q1, q3, lower, upper = self._upper_iqr_outliers(changes["score_change"])
        lower_mask = changes["score_change"] < lower
        changes["is_outlier"] = (mask | lower_mask).astype(bool)
        outliers = changes[changes["is_outlier"]].copy()
        display = outliers.sort_values("score_change", key=lambda s: s.abs(), ascending=False)
        no_outlier = display.empty
        if no_outlier:
            display = changes.reindex(changes["score_change"].abs().sort_values(ascending=False).index).head(10).copy()
        result = self._result(
            "outlier_detection", f"Year-to-year {metric} change outliers – {query['category']} ({start}–{end})",
            (f"Detected {len(outliers)} unusually large positive or negative {metric} changes among {len(changes)} matched institution-year pairs." if not no_outlier else
             f"No matched score change crossed the 1.5×IQR fences among {len(changes)} pairs; showing the ten largest absolute changes as candidates."),
            display, "box", "year", "score_change",
            {"metric": metric, "matched_year_pairs": int(len(changes)),
             "outlier_count": int(len(outliers)), "q1": q1, "q3": q3,
             "iqr": q3 - q1, "lower_fence": lower, "upper_fence": upper,
             "fallback_candidates_shown": bool(no_outlier)},
            ["Only institutions with valid ranked records and non-missing metric values in both adjacent calendar years are compared.",
             "Both unusually positive and negative changes are detected using the 1.5×IQR rule.",
             "Year gaps are not bridged; missing values are not imputed."],
            [c for c in ["institute_key", "display_name", "state", "year_from", "year", "score_start", "score_end", "score_change", "is_outlier"] if c in display.columns],
        )
        result["chart"]["data"] = _records(changes)
        result["operation"] = "score_change_outliers"
        return result

    def metric_rank_outliers(self, query: dict) -> dict:
        """Find strong parameter performance paired with worse-than-expected rank."""
        self._need(query, ["category", "year"])
        frame = self._base_filter(query, require_ranked=True)
        requested_metric = query.get("metric")
        if requested_metric:
            metric = requested_metric
            col = self._metric_column(metric)
            if col not in frame.columns:
                raise AnalyticsError(f"Metric {metric} is unavailable for this category/year.")
            valid = frame.dropna(subset=[col, "rank_num"]).copy()
            valid["parameter_score"] = valid[col]
            valid["strongest_parameter"] = metric
            valid["parameters_observed"] = 1
        else:
            metrics = [m for m in ["TLR", "RPC", "GO", "OI", "PR", "QNR", "QLR", "SFC"]
                       if m in frame.columns and frame[m].notna().any()]
            if not metrics:
                raise AnalyticsError("No NIRF parameter columns are available for this category/year.")
            valid = frame.dropna(subset=["rank_num"]).copy()
            valid["parameter_score"] = valid[metrics].max(axis=1, skipna=True)
            valid["strongest_parameter"] = valid[metrics].idxmax(axis=1, skipna=True)
            valid["parameters_observed"] = valid[metrics].notna().sum(axis=1)
            valid = valid[valid["parameters_observed"] >= 2].copy()
            metric = "strongest available parameter"
        valid = valid.dropna(subset=["parameter_score", "rank_num"]).copy()
        if len(valid) < 5 or valid["parameter_score"].nunique() < 2:
            raise AnalyticsError("At least five institutions with varying parameter scores are required for metric-rank outlier analysis.")
        slope, intercept = np.polyfit(valid["parameter_score"].to_numpy(float), valid["rank_num"].to_numpy(float), 1)
        valid["expected_rank"] = intercept + slope * valid["parameter_score"]
        valid["rank_residual"] = valid["rank_num"] - valid["expected_rank"]
        mask, q1, q3, lower, upper = self._upper_iqr_outliers(valid["rank_residual"])
        valid["is_outlier"] = mask.astype(bool)
        outliers = valid[valid["is_outlier"]].sort_values("rank_residual", ascending=False)
        display = outliers if not outliers.empty else valid.nlargest(min(10, len(valid)), "rank_residual")
        no_outlier = outliers.empty
        result = self._result(
            "outlier_detection", f"Strong parameter score but relatively poor rank – {query['category']} {query['year']}",
            (f"Detected {len(outliers)} institution(s) with worse-than-expected rank residuals despite strong {metric} performance." if not no_outlier else
             f"No institution crossed the upper 1.5×IQR rank-residual fence; showing the ten largest positive residual candidates."),
            display, "scatter_rank", "parameter_score", "rank_num",
            {"metric_used": metric, "observations": int(len(valid)), "outlier_count": int(len(outliers)),
             "residual_q1": q1, "residual_q3": q3, "residual_iqr": q3 - q1,
             "lower_fence": lower, "upper_fence": upper,
             "regression_slope_rank_on_parameter": float(slope),
             "regression_intercept": float(intercept),
             "fallback_candidates_shown": bool(no_outlier)},
            ["Poor rank is represented by a positive residual: actual rank number is worse than predicted from parameter score.",
             "When no individual metric is specified, the strongest available parameter score is used per institution, requiring at least two observed parameters.",
             "Outliers use the upper 1.5×IQR residual fence; missing values are not imputed."],
            [c for c in ["institute_key", "display_name", "state", "rank_num", "score", "strongest_parameter", "parameter_score", "parameters_observed", "expected_rank", "rank_residual", "is_outlier"] if c in display.columns],
        )
        result["chart"]["data"] = _records(valid, [c for c in ["display_name", "state", "rank_num", "score", "strongest_parameter", "parameter_score", "parameters_observed", "rank_residual", "is_outlier"] if c in valid.columns])
        result["operation"] = "metric_rank_outliers"
        return result

    def rank_volatility_outliers(self, query: dict) -> dict:
        """Identify high mean absolute annual rank changes relative to category peers."""
        frame, start, end, years = self._period_frame(query, require_ranked=True)
        years = sorted(y for y in years if start <= y <= end)
        records = []
        for from_year, to_year in zip(years[:-1], years[1:]):
            if to_year != from_year + 1:
                continue
            old = frame[frame["year"] == from_year][["institute_key", "display_name", "state", "rank_num"]].rename(columns={"rank_num": "rank_start"})
            new = frame[frame["year"] == to_year][["institute_key", "rank_num"]].rename(columns={"rank_num": "rank_end"})
            paired = old.merge(new, on="institute_key", how="inner", validate="one_to_one")
            if paired.empty:
                continue
            paired["rank_change"] = paired["rank_end"] - paired["rank_start"]
            paired["absolute_rank_change"] = paired["rank_change"].abs()
            paired["year_from"], paired["year_to"] = from_year, to_year
            records.extend(paired.to_dict("records"))
        changes = pd.DataFrame(records)
        if changes.empty:
            raise AnalyticsError("No consecutive-year matched rank records are available for volatility analysis.")
        summary = (changes.groupby("institute_key", as_index=False)
                   .agg(display_name=("display_name", "last"), state=("state", "last"),
                        transitions=("absolute_rank_change", "count"),
                        mean_absolute_rank_change=("absolute_rank_change", "mean"),
                        rank_change_std=("rank_change", "std"),
                        largest_absolute_rank_change=("absolute_rank_change", "max")))
        summary = summary[summary["transitions"] >= 2].copy()
        if len(summary) < 5:
            raise AnalyticsError("At least five institutions with two or more consecutive-year transitions are required for peer volatility analysis.")
        mask, q1, q3, lower, upper = self._upper_iqr_outliers(summary["mean_absolute_rank_change"])
        summary["is_outlier"] = mask.astype(bool)
        outliers = summary[summary["is_outlier"]].sort_values("mean_absolute_rank_change", ascending=False)
        display = outliers if not outliers.empty else summary.nlargest(min(10, len(summary)), "mean_absolute_rank_change")
        no_outlier = outliers.empty
        result = self._result(
            "outlier_detection", f"Rank volatility outliers – {query['category']} ({start}–{end})",
            (f"Detected {len(outliers)} institution(s) with unusually high average annual absolute rank movement among {len(summary)} comparable institutions." if not no_outlier else
             f"No institution crossed the upper 1.5×IQR volatility fence among {len(summary)} comparable institutions; showing the ten highest-volatility candidates."),
            display, "dot_plot", "mean_absolute_rank_change", "display_name",
            {"institutions_compared": int(len(summary)), "outlier_count": int(len(outliers)),
             "q1": q1, "q3": q3, "iqr": q3 - q1, "upper_fence": upper,
             "lower_fence": lower, "consecutive_year_transitions": int(changes[["year_from", "year_to"]].drop_duplicates().shape[0]),
             "fallback_candidates_shown": bool(no_outlier)},
            ["Volatility is measured as the mean absolute rank change across adjacent calendar years.",
             "Peer comparison uses institutions with at least two valid consecutive-year transitions in the same category and period.",
             "Missing/non-ranked years are not bridged; only published valid ranks are compared.",
             "Unusually high volatility is defined by the upper 1.5×IQR fence."],
            list(display.columns),
        )
        result["chart"]["data"] = _records(summary)
        result["operation"] = "rank_volatility_outliers"
        return result


    # ------------------------------------------------------------------
    # Annexure A: final TS03 and GEO04 operations
    # ------------------------------------------------------------------

    def rank_decline(self, query: dict) -> dict:
        """Return only institutions whose numerical rank worsened.

        rank_change is start_rank - end_rank: positive is improvement,
        negative is decline. The output is restricted to decliners and sorted
        with the largest numerical deterioration first.
        """
        merged, start, end = self._endpoint_rank_pairs(query)
        declined = merged.loc[merged["rank_change"] < 0].copy()
        if declined.empty:
            raise AnalyticsError(
                f"No institutions declined in {query['category']} between {start} and {end}."
            )

        declined["change"] = declined["rank_change"]
        declined["decline_positions"] = declined["end_rank"] - declined["start_rank"]
        limit = max(1, min(int(query.get("rank_limit") or 10), 25))
        display = declined.sort_values(
            ["decline_positions", "display_name"], ascending=[False, True]
        ).head(limit).copy()

        result = self._result(
            "rank_movement",
            f"Largest ranking declines – {query['category']} ({start}–{end})",
            (f"{len(declined)} of {len(merged)} matched institutions declined in rank. "
             f"The largest decline was {int(declined['decline_positions'].max())} positions."),
            display,
            "diverging_bar",
            "change",
            "display_name",
            {
                "institutions_compared": int(len(merged)),
                "declined_institutions": int(len(declined)),
                "improved_institutions": int((merged["rank_change"] > 0).sum()),
                "unchanged_institutions": int((merged["rank_change"] == 0).sum()),
                "largest_decline_positions": int(declined["decline_positions"].max()),
                "mean_decline_positions": float(declined["decline_positions"].mean()),
                "top_n_displayed": int(len(display)),
                "start_year": start,
                "end_year": end,
            },
            [
                "Only institutions with valid ranked records in both endpoint years are compared.",
                "Decliners are defined as institutions whose numerical rank number increased.",
                "The signed change is start_rank minus end_rank; negative values indicate a decline.",
                f"The result displays the {len(display)} largest declines; cohort statistics use all {len(declined)} decliners.",
            ],
            ["institute_key", "display_name", "state", "start_rank", "end_rank",
             "rank_change", "change", "decline_positions"],
        )
        result["operation"] = "rank_decline"
        return result

    def compare_state_rank_distributions(self, query: dict) -> dict:
        """Compare institution-level rank distributions for two selected states.

        When no state filter is supplied, Annexure GEO04 defaults to Karnataka
        and Tamil Nadu. The result keeps institution-level ranks for a true
        two-group box plot and also returns state-level descriptive statistics.
        """
        self._need(query, ["category", "year"])
        selected_states = query.get("states") or ["Karnataka", "Tamil Nadu"]
        if len(selected_states) != 2:
            raise AnalyticsError(
                "State rank-distribution comparison requires exactly two states."
            )

        frame = self._base_filter(query, require_ranked=True)
        frame = frame[frame["year"] == int(query["year"])].copy()
        if frame.empty:
            raise AnalyticsError("No ranked institutions match the selected category and year.")

        # Resolve user spelling/case to the state labels actually present in the data.
        available = {
            str(value).strip().casefold(): str(value).strip()
            for value in frame["state"].dropna().unique()
        }
        resolved = []
        for state in selected_states:
            key = str(state).strip().casefold()
            if key not in available:
                raise AnalyticsError(
                    f"No ranked institutions for state '{state}' in {query['category']} {query['year']}."
                )
            resolved.append(available[key])
        if resolved[0].casefold() == resolved[1].casefold():
            raise AnalyticsError("Choose two different states for the rank comparison.")

        frame = frame[frame["state"].isin(resolved)].copy()
        frame = frame.dropna(subset=["rank_num"])
        # Protect against duplicate rows while preserving one canonical institution.
        frame = (frame.sort_values(["rank_num", "display_name"])
                      .drop_duplicates("institute_key", keep="first"))
        if frame["state"].nunique() != 2:
            raise AnalyticsError("Both selected states need at least one ranked institution.")

        summaries = []
        for state in resolved:
            values = pd.to_numeric(
                frame.loc[frame["state"] == state, "rank_num"], errors="coerce"
            ).dropna()
            if values.empty:
                raise AnalyticsError(f"No valid rank values are available for {state}.")
            q1, median, q3 = values.quantile([0.25, 0.50, 0.75]).tolist()
            summaries.append({
                "state": state,
                "institution_count": int(values.size),
                "mean_rank": float(values.mean()),
                "median_rank": float(median),
                "q1_rank": float(q1),
                "q3_rank": float(q3),
                "iqr_rank": float(q3 - q1),
                "best_rank": float(values.min()),
                "worst_rank": float(values.max()),
            })

        stats_frame = pd.DataFrame(summaries)
        better_state = str(stats_frame.sort_values("median_rank").iloc[0]["state"])
        display = frame.sort_values(["state", "rank_num", "display_name"]).copy()
        result = self._result(
            "comparison",
            f"Institution rank distributions – {resolved[0]} vs {resolved[1]} ({query['category']} {query['year']})",
            (f"Compared {len(display)} ranked institutions across {resolved[0]} "
             f"({summaries[0]['institution_count']}) and {resolved[1]} "
             f"({summaries[1]['institution_count']}). {better_state} has the lower median numerical rank."),
            display,
            "box",
            "state",
            "rank_num",
            {
                "states_compared": 2,
                "institutions_compared": int(len(display)),
                "state_summaries": summaries,
                "lower_median_rank_state": better_state,
                "median_rank_difference": float(abs(summaries[0]["median_rank"] - summaries[1]["median_rank"])),
            },
            [
                "The box plot uses institution-level published numerical ranks, not state averages.",
                "Lower numerical rank is better.",
                "Only valid ranked records for the selected category and year are included.",
                "Institutions are deduplicated using institute_key; missing ranks are excluded.",
            ],
            [c for c in ["institute_key", "display_name", "state", "rank_num", "rank_band", "score"] if c in display.columns],
        )
        result["operation"] = "compare_state_rank_distributions"
        return result

    def execute(self, validated_query: Any, institution_keys: list[str] | None = None) -> dict:
        """Execute a validated query or full pipeline result.

        Accepts either a query dict/Pydantic model, or the dictionary returned
        by query_pipeline.validate_query().
        """
        if isinstance(validated_query, dict) and "query" in validated_query:
            pipeline_status = validated_query.get("status")
            if pipeline_status != "ready":
                return {
                    "status": "needs_clarification",
                    "message": "The query has not passed validation.",
                    "issues": validated_query.get("issues", []),
                }
            query = validated_query["query"]
            if institution_keys is None:
                institution_keys = validated_query.get("institution_keys") or []
        else:
            query = self._query_dict(validated_query)

        operation = query.get("operation") or (validated_query.get("operation") if isinstance(validated_query, dict) else None)
        operation_dispatch = {
            "mean_score_by_year": self.mean_score_by_year,
            "top10_consistency": self.top10_consistency,
            "state_distribution_by_year": self.state_distribution_by_year,
            "state_representation_change": self.state_representation_change,
            "new_entries_by_state": self.new_entries_by_state,
            "rank_distribution": self.rank_distribution,
            "score_quartile_share": self.score_quartile_share,
            "score_spread_by_category": self.score_spread_by_category,
            "top10_vs_rest_distribution": self.top10_vs_rest_distribution,
            "score_dispersion_by_year": self.score_dispersion_by_year,
            "parameter_variability": self.parameter_variability,
            "median_iqr_over_time": self.median_iqr_over_time,
            "score_distribution_by_state": self.score_distribution_by_state,
            "parameter_correlation_ranking": self.parameter_correlation_ranking,
            "correlation_over_time": self.correlation_over_time,
            "high_low_metric_quadrant": self.high_low_metric_quadrant,
            "overall_vs_parameter_change": self.overall_vs_parameter_change,
            "parameter_gap": self.parameter_gap,
            "rank_band_transition_count": self.rank_band_transition_count,
            "top10_entries_exits": self.top10_entries_exits,
            "same_band_percentage": self.same_band_percentage,
            "rank_movement_extremes": self.rank_movement_extremes,
            "rank_decline": self.rank_decline,
            "compare_state_rank_distributions": self.compare_state_rank_distributions,
            "movement_counts_by_year": self.movement_counts_by_year,
            "research_score_residual_outliers": self.research_score_residual_outliers,
            "metric_difference_outliers": self.metric_difference_outliers,
            "score_change_outliers": self.score_change_outliers,
            "metric_rank_outliers": self.metric_rank_outliers,
            "rank_volatility_outliers": self.rank_volatility_outliers,
        }
        if operation in operation_dispatch:
            return operation_dispatch[operation](query)

        intent = query.get("intent")
        dispatch = {
            "ranking": self.ranking,
            "comparison": self.comparison,
            "trend_analysis": self.trend,
            "geographical_analysis": self.geographical,
            "correlation_analysis": self.correlation,
            "distribution_analysis": self.distribution,
            "statistical_summary": self.statistical_summary,
            "multivariate_analysis": self.multivariate,
            "transition_analysis": self.transitions,
            "outlier_detection": self.outliers,
        }
        if intent not in dispatch:
            raise AnalyticsError(f"Unsupported analytical intent: {intent}")
        method = dispatch[intent]
        if intent in {"ranking", "comparison", "trend_analysis", "correlation_analysis", "multivariate_analysis"}:
            return method(query, institution_keys)
        return method(query)
