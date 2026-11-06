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
    """Convert numpy/pandas scalars and missing values to JSON-safe values."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
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
