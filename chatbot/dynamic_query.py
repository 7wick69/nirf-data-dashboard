"""Safe, schema-constrained dynamic NIRF query planning and execution.

The local LLM only converts a natural-language question into a bounded JSON plan.
All filtering, aggregation, statistics and chart data are computed by pandas here.
No Python/SQL/eval expressions supplied by the model are executed.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


METRICS = ("score", "rank", "TLR", "RPC", "GO", "OI", "PR", "QNR", "QLR", "SFC")
METRIC_COLUMNS = {
    "score": "score", "rank": "rank_num", "TLR": "TLR", "RPC": "RPC",
    "GO": "GO", "OI": "OI", "PR": "PR", "QNR": "QNR", "QLR": "QLR", "SFC": "SFC",
}
GROUP_FIELDS = ("year", "category", "state", "city", "rank_band")
OPERATIONS = ("ranking", "filter", "aggregate", "trend", "correlation", "distribution", "compare")
AGGREGATIONS = ("mean", "median", "sum", "count", "min", "max", "std")
FILTER_OPERATORS = (">", ">=", "<", "<=", "==", "!=")


PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["ready", "needs_clarification", "out_of_scope"]},
        "clarification": {"type": "string"},
        "operation": {"type": "string", "enum": list(OPERATIONS)},
        "category": {"type": ["string", "null"]},
        "year": {"type": ["integer", "null"]},
        "start_year": {"type": ["integer", "null"]},
        "end_year": {"type": ["integer", "null"]},
        "use_latest_year": {"type": "boolean"},
        "states": {"type": "array", "items": {"type": "string"}},
        "institutions": {"type": "array", "items": {"type": "string"}},
        "rank_min": {"type": ["number", "null"]},
        "rank_max": {"type": ["number", "null"]},
        "metric": {"type": ["string", "null"], "enum": [*METRICS, None]},
        "metric_x": {"type": ["string", "null"], "enum": [*METRICS, None]},
        "metric_y": {"type": ["string", "null"], "enum": [*METRICS, None]},
        "metric_filters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": list(METRICS)},
                    "operator": {"type": "string", "enum": list(FILTER_OPERATORS)},
                    "value": {"type": "number"},
                },
                "required": ["metric", "operator", "value"],
                "additionalProperties": False,
            },
        },
        "group_by": {"type": ["string", "null"], "enum": [*GROUP_FIELDS, None]},
        "aggregation": {"type": "string", "enum": list(AGGREGATIONS)},
        "top_n": {"type": "integer", "minimum": 1, "maximum": 100},
        "sort_by": {"type": "string", "enum": ["rank_num", "score", "TLR", "RPC", "GO", "OI", "PR", "QNR", "QLR", "SFC", "year"]},
        "sort_order": {"type": "string", "enum": ["ascending", "descending"]},
        "ranked_only": {"type": "boolean"},
        "bins": {"type": "integer", "minimum": 5, "maximum": 40},
    },
    "required": [
        "status", "clarification", "operation", "category", "year", "start_year", "end_year",
        "use_latest_year", "states", "institutions", "rank_min", "rank_max", "metric",
        "metric_x", "metric_y", "metric_filters", "group_by", "aggregation", "top_n",
        "sort_by", "sort_order", "ranked_only", "bins",
    ],
    "additionalProperties": False,
}


class DynamicQueryError(ValueError):
    """Raised when a generated dynamic query plan is unsafe or invalid."""


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_value(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _records(frame: pd.DataFrame, limit: int = 500) -> list[dict[str, Any]]:
    view = frame.head(limit)
    return [{str(k): _json_value(v) for k, v in row.items()}
            for row in view.to_dict(orient="records")]


def aggregation_requires_metric(plan: dict[str, Any]) -> bool:
    return str(plan.get("aggregation", "mean")).lower() != "count"


def _apply_operator(series: pd.Series, operator: str, value: float) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if operator == ">": return numeric > value
    if operator == ">=": return numeric >= value
    if operator == "<": return numeric < value
    if operator == "<=": return numeric <= value
    if operator == "==": return numeric == value
    if operator == "!=": return numeric != value
    raise DynamicQueryError(f"Unsupported filter operator: {operator}")


def should_use_dynamic_query(question_analysis: dict[str, Any],
                             similarity_threshold: float = 0.55,
                             confidence_floor: float = 0.30) -> bool:
    """Return True when a question is not a reliable Annexure A template match."""
    if question_analysis.get("status") != "success":
        return True
    similarity = float(question_analysis.get("similarity") or 0.0)
    confidence = float(question_analysis.get("confidence") or 0.0)
    return (
        similarity < similarity_threshold
        or confidence < confidence_floor
        or not question_analysis.get("operation")
        or not question_analysis.get("question_id")
    )


class DynamicQueryEngine:
    """Converts unseen NIRF questions to a safe plan and executes with pandas."""

    def __init__(self, dataframe: pd.DataFrame, model: str = "qwen2.5:latest"):
        required = {"year", "category", "institute_key", "display_name", "state", "rank_num", "rank_status", "score"}
        missing = required.difference(dataframe.columns)
        if missing:
            raise ValueError(f"NIRF dataframe is missing required columns: {sorted(missing)}")
        self.df = dataframe.copy()
        self.model = model
        self.df["year"] = pd.to_numeric(self.df["year"], errors="coerce")
        self.df["rank_num"] = pd.to_numeric(self.df["rank_num"], errors="coerce")
        for metric in METRICS:
            col = METRIC_COLUMNS[metric]
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        self.categories = sorted(self.df["category"].dropna().astype(str).unique().tolist())
        self.states = sorted(self.df["state"].dropna().astype(str).unique().tolist())
        self.years = sorted(int(y) for y in self.df["year"].dropna().unique())

    def plan_question(self, question: str, context: dict[str, Any] | None = None,
                      model: str | None = None) -> dict[str, Any]:
        """Ask Ollama for schema-constrained JSON; no analytical computation occurs here."""
        if not str(question or "").strip():
            raise DynamicQueryError("Please enter a NIRF question.")
        try:
            import ollama
        except ImportError as exc:
            raise DynamicQueryError("The local Ollama Python package is not installed.") from exc

        system = f"""You are a query planner for a NIRF analytics application.
Convert the user's question into one JSON query plan matching the supplied schema.
You do NOT calculate, estimate, write code, SQL, formulas or expressions.
Allowed categories: {self.categories}
Allowed states: {self.states}
Allowed years: {self.years}
Allowed numeric metrics: {list(METRICS)}
Allowed group_by fields: {list(GROUP_FIELDS)}
Rules:
- Use status=out_of_scope for questions unrelated to NIRF institutional data.
- Use status=needs_clarification when a required category/year/metric or a comparison target is genuinely ambiguous.
- For a question saying latest/current, set use_latest_year=true and leave year null.
- Do not invent category, state, institution, year or metric names.
- For a single-year request, set year. For a period, set start_year and end_year.
- ranking means order institutions by NIRF rank. filter means return institutions matching conditions.
- aggregate means summarize one metric by group_by. trend means aggregate a metric by year.
- correlation requires metric_x and metric_y. distribution requires metric. compare compares distributions across group_by.
- Use metric_filters for explicit numeric conditions such as RPC > 60 or score >= 80.
- Put state names in states and institution names in institutions.
- Set ranked_only=true for rank-based selections; generally true for category rankings.
- top_n defaults to 20 unless the user specifies a number. Never exceed 100.
- sort_by must be one of the allowed fields. sort_order=descending for highest score/metric, ascending for best rank.
- aggregation must be one of mean, median, sum, count, min, max, std.
- Do not add fields outside the schema. Return only the JSON object."""
        response = ollama.chat(
            model=model or self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps({"question": question, "conversation_context": context or {}}, ensure_ascii=False)},
            ],
            format=PLAN_SCHEMA,
            options={"temperature": 0},
            stream=False,
        )
        content = response.message.content
        try:
            plan = json.loads(content)
        except (TypeError, json.JSONDecodeError) as exc:
            raise DynamicQueryError("The local model did not return a valid structured query plan.") from exc
        return self.validate_plan(plan)

    def validate_plan(self, raw_plan: dict[str, Any]) -> dict[str, Any]:
        """Validate every field against explicit allow-lists and dataset values."""
        if not isinstance(raw_plan, dict):
            raise DynamicQueryError("The generated query plan must be a JSON object.")
        plan = dict(raw_plan)
        status = plan.get("status", "ready")
        if status in {"needs_clarification", "out_of_scope"}:
            return {
                "status": status,
                "clarification": str(plan.get("clarification") or "Please clarify the NIRF category, year or metric."),
            }
        if status != "ready":
            raise DynamicQueryError(f"Unsupported query-plan status: {status}")

        operation = plan.get("operation")
        if operation not in OPERATIONS:
            raise DynamicQueryError(f"Unsupported dynamic operation: {operation}")
        category = plan.get("category")
        if category:
            match = next((c for c in self.categories if c.casefold() == str(category).casefold()), None)
            if not match:
                raise DynamicQueryError(f"Unknown NIRF category '{category}'. Available categories include: {', '.join(self.categories[:12])}.")
            plan["category"] = match

        for field in ("year", "start_year", "end_year"):
            value = plan.get(field)
            if value is not None:
                if isinstance(value, bool) or int(value) != value:
                    raise DynamicQueryError(f"{field} must be an integer year.")
                value = int(value)
                if value not in self.years:
                    raise DynamicQueryError(f"Year {value} is not present in the NIRF dataset.")
                plan[field] = value

        if plan.get("year") is not None and (plan.get("start_year") is not None or plan.get("end_year") is not None):
            raise DynamicQueryError("Specify either one year or a year range, not both.")
        if plan.get("year") is not None and plan.get("use_latest_year"):
            raise DynamicQueryError("Use either a specific year or latest-year selection, not both.")
        if operation == "trend":
            if plan.get("start_year") is None or plan.get("end_year") is None:
                raise DynamicQueryError("A trend query needs both start_year and end_year.")
        elif operation in {"ranking", "filter", "aggregate", "correlation", "distribution", "compare"}:
            if plan.get("year") is None and not plan.get("use_latest_year"):
                raise DynamicQueryError("Please specify a year or ask for the latest published year.")
        if plan.get("start_year") is not None and plan.get("end_year") is not None and plan["start_year"] > plan["end_year"]:
            raise DynamicQueryError("start_year must be less than or equal to end_year.")

        states = plan.get("states") or []
        normalized_states = []
        for state in states:
            match = next((s for s in self.states if s.casefold() == str(state).casefold()), None)
            if not match:
                raise DynamicQueryError(f"Unknown state '{state}'. Check the state name and try again.")
            normalized_states.append(match)
        plan["states"] = list(dict.fromkeys(normalized_states))

        institutions = plan.get("institutions") or []
        if not isinstance(institutions, list) or any(not isinstance(x, str) or not x.strip() for x in institutions):
            raise DynamicQueryError("Institution filters must be a list of non-empty names.")
        plan["institutions"] = [x.strip() for x in institutions]

        for field in ("metric", "metric_x", "metric_y"):
            metric = plan.get(field)
            if metric is not None and metric not in METRICS:
                raise DynamicQueryError(f"Unsupported metric '{metric}'. Allowed metrics: {', '.join(METRICS)}.")
        if operation in {"trend", "distribution", "compare"} and not plan.get("metric"):
            raise DynamicQueryError(f"The {operation} operation requires a metric.")
        if operation == "aggregate" and aggregation_requires_metric(plan) and not plan.get("metric"):
            raise DynamicQueryError("An aggregate other than count requires a metric.")
        if operation == "correlation":
            if not plan.get("metric_x") or not plan.get("metric_y"):
                raise DynamicQueryError("A correlation query needs two metrics, such as RPC and TLR.")
            if plan["metric_x"] == plan["metric_y"]:
                raise DynamicQueryError("Choose two different metrics for correlation.")
        if operation in {"aggregate", "compare"} and not plan.get("group_by"):
            raise DynamicQueryError(f"The {operation} operation needs a grouping field such as state, category or year.")
        if plan.get("group_by") is not None and plan["group_by"] not in GROUP_FIELDS:
            raise DynamicQueryError(f"Unsupported grouping field: {plan['group_by']}")
        if plan.get("group_by") is not None and plan["group_by"] not in self.df.columns:
            raise DynamicQueryError(f"Grouping field '{plan['group_by']}' is not available in this dataset.")

        aggregation = plan.get("aggregation", "mean")
        if aggregation not in AGGREGATIONS:
            raise DynamicQueryError(f"Unsupported aggregation: {aggregation}")
        plan["aggregation"] = aggregation
        top_n = int(plan.get("top_n", 20))
        if not 1 <= top_n <= 100:
            raise DynamicQueryError("top_n must be between 1 and 100.")
        plan["top_n"] = top_n
        if plan.get("sort_by") not in {"rank_num", "score", "TLR", "RPC", "GO", "OI", "PR", "QNR", "QLR", "SFC", "year"}:
            plan["sort_by"] = "rank_num" if operation == "ranking" else (plan.get("metric") or "score")
        if plan.get("sort_order") not in {"ascending", "descending"}:
            plan["sort_order"] = "ascending" if operation == "ranking" else "descending"
        plan["ranked_only"] = bool(plan.get("ranked_only", True))
        plan["use_latest_year"] = bool(plan.get("use_latest_year", False))
        plan["bins"] = min(40, max(5, int(plan.get("bins", 12))))

        filters = plan.get("metric_filters") or []
        if not isinstance(filters, list):
            raise DynamicQueryError("metric_filters must be a list.")
        for condition in filters:
            if not isinstance(condition, dict) or condition.get("metric") not in METRICS:
                raise DynamicQueryError("Each metric filter must use an allowed NIRF metric.")
            if condition.get("operator") not in FILTER_OPERATORS:
                raise DynamicQueryError("Each metric filter must use a supported comparison operator.")
            if not isinstance(condition.get("value"), (int, float)) or isinstance(condition.get("value"), bool):
                raise DynamicQueryError("Metric filter values must be numeric.")
        plan["metric_filters"] = filters

        for field in ("rank_min", "rank_max"):
            value = plan.get(field)
            if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value)):
                raise DynamicQueryError(f"{field} must be a finite number.")
        if plan.get("rank_min") is not None and plan.get("rank_max") is not None and float(plan["rank_min"]) > float(plan["rank_max"]):
            raise DynamicQueryError("rank_min cannot be greater than rank_max.")
        if plan.get("rank_min") is not None or plan.get("rank_max") is not None:
            plan["ranked_only"] = True

        return plan

    def _filtered_frame(self, plan: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
        frame = self.df.copy()
        if plan.get("category"):
            frame = frame[frame["category"].astype(str).str.casefold() == plan["category"].casefold()]

        if plan.get("year") is not None:
            frame = frame[frame["year"] == plan["year"]]
        elif plan.get("start_year") is not None and plan.get("end_year") is not None:
            frame = frame[frame["year"].between(plan["start_year"], plan["end_year"])]
        elif plan.get("use_latest_year"):
            latest = frame["year"].dropna().max()
            if pd.isna(latest):
                raise DynamicQueryError("No published years are available for the selected category.")
            plan["year"] = int(latest)
            frame = frame[frame["year"] == int(latest)]

        if plan.get("states"):
            wanted = {s.casefold() for s in plan["states"]}
            frame = frame[frame["state"].astype(str).str.casefold().isin(wanted)]
        if plan.get("institutions"):
            names = [n.casefold() for n in plan["institutions"]]
            normalized = frame["display_name"].astype(str).str.casefold()
            mask = pd.Series(False, index=frame.index)
            for name in names:
                mask |= normalized.eq(name) | normalized.str.contains(name, regex=False)
            frame = frame[mask]
        if plan.get("ranked_only") or plan.get("operation") == "ranking":
            frame = frame[frame["rank_num"].notna()]
            if "rank_status" in frame.columns:
                frame = frame[frame["rank_status"].astype(str).str.casefold() == "ranked"]
        if plan.get("rank_min") is not None:
            frame = frame[frame["rank_num"] >= float(plan["rank_min"])]
        if plan.get("rank_max") is not None:
            frame = frame[frame["rank_num"] <= float(plan["rank_max"])]
        for condition in plan.get("metric_filters", []):
            col = METRIC_COLUMNS[condition["metric"]]
            if col not in frame.columns:
                raise DynamicQueryError(f"Metric {condition['metric']} is not present in this dataset version.")
            keep = _apply_operator(frame[col], condition["operator"], float(condition["value"]))
            frame = frame[keep.fillna(False)]

        # A single canonical institution/category/year row is expected. Keep the
        # first ranked row if duplicate source rows exist, never sum duplicates.
        frame = frame.sort_values(["rank_num", "display_name"], na_position="last")
        frame = frame.drop_duplicates(subset=["year", "category", "institute_key"], keep="first")
        if frame.empty:
            raise DynamicQueryError("No NIRF records match these filters. Check the category, year, states and metric thresholds.")
        return frame.copy(), {"rows_after_filters": int(len(frame)), "institutions": int(frame["institute_key"].nunique())}

    @staticmethod
    def _summary_stats(values: pd.Series) -> dict[str, Any]:
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if numeric.empty:
            return {"count": 0}
        return {
            "count": int(numeric.size), "mean": float(numeric.mean()),
            "median": float(numeric.median()), "std": float(numeric.std(ddof=1)) if numeric.size > 1 else None,
            "min": float(numeric.min()), "q1": float(numeric.quantile(.25)),
            "q3": float(numeric.quantile(.75)), "max": float(numeric.max()),
        }

    def execute_plan(self, raw_plan: dict[str, Any]) -> dict[str, Any]:
        plan = self.validate_plan(raw_plan)
        if plan["status"] != "ready":
            return {"status": plan["status"], "message": plan.get("clarification"), "data": [], "chart": {"type": "none", "data": []}}
        operation = plan["operation"]
        frame, filter_stats = self._filtered_frame(plan)
        metric = plan.get("metric") or "score"
        metric_col = METRIC_COLUMNS[metric]

        if operation in {"ranking", "filter"}:
            if operation == "ranking":
                frame = frame.sort_values("rank_num", ascending=True)
                if plan.get("sort_by") not in {"rank_num", None}:
                    col = METRIC_COLUMNS.get(plan["sort_by"], plan["sort_by"])
                    frame = frame.sort_values(col, ascending=plan["sort_order"] == "ascending", na_position="last")
            else:
                col = METRIC_COLUMNS.get(plan["sort_by"], plan["sort_by"])
                if col in frame.columns:
                    frame = frame.sort_values(col, ascending=plan["sort_order"] == "ascending", na_position="last")
            frame = frame.head(plan["top_n"])
            if operation == "filter" and plan.get("metric_filters"):
                candidate_metric = plan["metric_filters"][0]["metric"]
                candidate_col = METRIC_COLUMNS[candidate_metric]
                value_col = candidate_col if frame[candidate_col].notna().any() else "score"
            else:
                value_col = "score" if frame["score"].notna().any() else "rank_num"
            data = _records(frame, 500)
            return self._result(
                operation, f"Dynamic {operation} results", f"Returned {len(frame)} institution(s) after applying the requested filters.",
                data, "horizontal_bar", "display_name", value_col,
                {**filter_stats, "returned_records": len(frame), "top_n": plan["top_n"]},
                ["This result was generated from a validated dynamic query plan.", "Calculations and filtering were performed by pandas."],
            )

        if operation == "correlation":
            x_col, y_col = METRIC_COLUMNS[plan["metric_x"]], METRIC_COLUMNS[plan["metric_y"]]
            valid = frame.dropna(subset=[x_col, y_col]).copy()
            if len(valid) < 2:
                raise DynamicQueryError("At least two complete observations are required for correlation.")
            pearson = valid[x_col].corr(valid[y_col], method="pearson")
            spearman = valid[x_col].corr(valid[y_col], method="spearman")
            cols = [c for c in ["display_name", "state", "rank_num", x_col, y_col] if c in valid.columns]
            return self._result(
                operation, f"{plan['metric_x']} vs {plan['metric_y']}",
                f"Calculated the association between {plan['metric_x']} and {plan['metric_y']} using {len(valid)} complete observations.",
                _records(valid[cols], 10000), "scatter", x_col, y_col,
                {"pearson_correlation": _json_value(pearson), "spearman_correlation": _json_value(spearman),
                 "sample_size": int(len(valid)), "x_metric": plan["metric_x"], "y_metric": plan["metric_y"]},
                ["Correlation describes association, not causation.", "Rows missing either metric are excluded; no metric values are imputed."],
            )

        if operation == "distribution":
            valid = frame.dropna(subset=[metric_col]).copy()
            if valid.empty:
                raise DynamicQueryError(f"No published {metric} values are available after filtering.")
            stats = self._summary_stats(valid[metric_col])
            counts, edges = np.histogram(
                pd.to_numeric(valid[metric_col], errors="coerce").dropna().to_numpy(),
                bins=plan["bins"],
            )
            bins_frame = pd.DataFrame({
                "bin_start": edges[:-1], "bin_end": edges[1:], "count": counts.astype(int),
            })
            return self._result(
                operation, f"Distribution of {metric}",
                f"Summarized {metric} for {len(valid)} institutions with published values.",
                _records(bins_frame, 100), "histogram", metric_col, None,
                {**filter_stats, **stats, "metric": metric, "bins": plan["bins"]},
                ["Histogram bins are calculated from all complete observations.",
                 "Missing metric values are excluded, not replaced by zero."],
            )

        if operation in {"aggregate", "trend", "compare"}:
            group = "year" if operation == "trend" else plan.get("group_by")
            if operation == "trend" and not group:
                group = "year"
            if not group:
                raise DynamicQueryError(f"{operation} needs a grouping field.")
            if plan["aggregation"] == "count":
                valid = frame.dropna(subset=["institute_key"]).copy()
                grouped = valid.groupby(group, dropna=False)["institute_key"].nunique().rename("institution_count").reset_index()
                value_col = "institution_count"
            else:
                valid = frame.dropna(subset=[metric_col]).copy()
                if valid.empty:
                    raise DynamicQueryError(f"No published {metric} values are available for this aggregation.")
                grouped = valid.groupby(group, dropna=False)[metric_col].agg(plan["aggregation"]).rename("metric_value").reset_index()
                value_col = "metric_value"
            grouped = grouped.sort_values(group)
            if operation in {"aggregate", "compare"}:
                grouped = grouped.sort_values(value_col, ascending=plan["sort_order"] == "ascending")
            grouped = grouped.head(plan["top_n"] if operation != "trend" else 100)
            if operation == "trend" and group == "year":
                grouped = grouped.sort_values("year")
                chart_type = "line"
            elif operation == "compare":
                chart_type = "box"
                # Return raw grouped observations for the box plot; summary table is preserved separately.
                raw_columns = list(dict.fromkeys([group, metric_col, "institute_key", "display_name", "state"]))
                raw = valid[raw_columns].copy()
                result_data = _records(raw, 500)
            else:
                chart_type = "horizontal_bar"
            if operation == "compare":
                data = result_data
                x_value, y_value = group, metric_col
            else:
                data = _records(grouped, 500)
                x_value, y_value = group, value_col
            return self._result(
                operation, f"{plan['aggregation'].title()} {metric} by {group}",
                f"Calculated {plan['aggregation']} {metric} for {grouped[group].nunique()} group(s).",
                data, chart_type, x_value, y_value,
                {**filter_stats, "metric": metric, "aggregation": plan["aggregation"],
                 "group_by": group, "group_summaries": _records(grouped, 100)},
                ["Only valid published metric values are included. Missing values are not imputed."],
            )

        raise DynamicQueryError(f"The operation '{operation}' is not implemented.")

    @staticmethod
    def _result(operation: str, title: str, summary: str, data: list[dict[str, Any]],
                chart_type: str, x: str | None, y: str | None,
                statistics: dict[str, Any], notes: list[str]) -> dict[str, Any]:
        return {
            "status": "success", "intent": "dynamic_analysis", "operation": f"dynamic_{operation}",
            "title": title, "summary": summary, "data": data,
            "chart": {"type": chart_type, "x": x, "y": y, "data": data},
            "statistics": _json_value(statistics), "notes": notes,
            "dynamic": True,
        }

    def answer_question(self, question: str, context: dict[str, Any] | None = None,
                        model: str | None = None) -> dict[str, Any]:
        plan = self.plan_question(question, context=context, model=model)
        if plan["status"] != "ready":
            return {"status": plan["status"], "message": plan.get("clarification"), "plan": plan}
        result = self.execute_plan(plan)
        result["dynamic_plan"] = plan
        return result
