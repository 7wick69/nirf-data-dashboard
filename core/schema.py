"""
core/schema.py

Pydantic v2 specification for NIRF Chart Query Specs.
Every chart across Level 1, Level 2 (Explore), and Level 3 (Ask) is represented
by ONE JSON spec: title + ordered data steps + one chart.
"""

from typing import Any, List, Literal, Optional, Union, Annotated, Dict
from pydantic import BaseModel, Field, model_validator, ConfigDict


# ---------------------------------------------------------
# Step 1: Filter Step
# ---------------------------------------------------------
FilterOp = Literal["in", "eq", "neq", "between", "gte", "lte", "gt", "lt", "notnull", "contains"]


class FilterCondition(BaseModel):
    field: str
    op: FilterOp
    value: Any = None


class FilterStep(BaseModel):
    op: Literal["filter"] = "filter"
    where: List[FilterCondition] = Field(default_factory=list)


# ---------------------------------------------------------
# Step 2: Derive Step
# ---------------------------------------------------------
DeriveFn = Literal["diff", "zscore", "pct_rank", "rank", "shift", "std", "bin"]


class DeriveStep(BaseModel):
    op: Literal["derive"] = "derive"
    name: str
    expr: Optional[str] = None
    fn: Optional[DeriveFn] = None
    by: Optional[List[str]] = None
    order: Optional[Literal["asc", "desc"]] = "asc"
    edges: Optional[List[float]] = None
    labels: Optional[List[str]] = None

    @model_validator(mode="after")
    def check_expr_or_fn(self):
        if not self.expr and not self.fn:
            raise ValueError("DeriveStep requires either 'expr' (safe arithmetic) or 'fn' (transform function)")
        return self


# ---------------------------------------------------------
# Step 3: Aggregate Step
# ---------------------------------------------------------
AggFn = Literal["mean", "median", "min", "max", "count", "nunique", "std", "iqr", "sum"]


class MetricAgg(BaseModel):
    field: str
    fn: AggFn
    as_: str = Field(alias="as")

    model_config = ConfigDict(populate_by_name=True)


class AggregateStep(BaseModel):
    op: Literal["aggregate"] = "aggregate"
    by: List[str] = Field(default_factory=list)
    metrics: List[MetricAgg] = Field(default_factory=list)


# ---------------------------------------------------------
# Step 4: Pivot Step
# ---------------------------------------------------------
class PivotStep(BaseModel):
    op: Literal["pivot"] = "pivot"
    index: Union[str, List[str]]
    columns: str
    values: str


# ---------------------------------------------------------
# Step 5: Sort, Top, Dropna, Fillna Steps
# ---------------------------------------------------------
class SortStep(BaseModel):
    op: Literal["sort"] = "sort"
    by: Union[str, List[str]]
    order: Union[Literal["asc", "desc"], List[Literal["asc", "desc"]]] = "asc"


class TopStep(BaseModel):
    op: Literal["top"] = "top"
    n: int = 10
    by: Optional[str] = None
    order: Literal["asc", "desc"] = "desc"


class DropnaStep(BaseModel):
    op: Literal["dropna"] = "dropna"
    subset: Optional[List[str]] = None


class FillnaStep(BaseModel):
    op: Literal["fillna"] = "fillna"
    value: Any
    subset: Optional[List[str]] = None


# ---------------------------------------------------------
# Discriminated Union for Ordered Pipeline Steps
# ---------------------------------------------------------
Step = Annotated[
    Union[
        FilterStep,
        DeriveStep,
        AggregateStep,
        PivotStep,
        SortStep,
        TopStep,
        DropnaStep,
        FillnaStep,
    ],
    Field(discriminator="op"),
]


# ---------------------------------------------------------
# Chart Configurations & Required Encodings
# ---------------------------------------------------------
ChartType = Literal[
    "line",
    "bar",
    "area",
    "scatter",
    "histogram",
    "box",
    "violin",
    "heatmap",
    "choropleth",
    "sankey",
    "radar",
    "parallel",
    "dumbbell",
    "bump",
    "table",
]

REQUIRED_ENCODINGS: Dict[ChartType, List[str]] = {
    "line": ["x", "y"],
    "bar": ["x", "y"],
    "area": ["x", "y"],
    "scatter": ["x", "y"],
    "histogram": ["x"],
    "box": ["y"],
    "violin": ["y"],
    "heatmap": ["x", "y"],
    "choropleth": ["locations", "color"],
    "sankey": ["source", "target", "value"],
    "radar": ["fields"],
    "parallel": ["fields"],
    "dumbbell": ["label", "start", "end"],
    "bump": ["x", "y", "color"],
    "table": [],
}


class ChartConfig(BaseModel):
    type: ChartType
    x: Optional[str] = None
    y: Optional[str] = None
    color: Optional[str] = None
    size: Optional[str] = None
    facet: Optional[str] = None
    trendline: Optional[Literal["ols", "lowess"]] = None
    locations: Optional[str] = None  # For choropleth
    source: Optional[str] = None     # For sankey
    target: Optional[str] = None     # For sankey
    value: Optional[str] = None      # For sankey / heatmap
    label: Optional[str] = None      # For dumbbell
    start: Optional[str] = None      # For dumbbell
    end: Optional[str] = None        # For dumbbell
    fields: Optional[List[str]] = None  # For radar, parallel, table
    orientation: Optional[Literal["v", "h"]] = "v"
    barmode: Optional[Literal["group", "stack", "relative"]] = None
    invert_yaxis: Optional[bool] = False
    show_legend: Optional[bool] = None

    def validate_encodings(self) -> List[str]:
        """Returns list of missing required encoding attributes for this chart type."""
        required = REQUIRED_ENCODINGS.get(self.type, [])
        missing = []
        for req in required:
            # Check attribute
            val = getattr(self, req, None)
            if val is None:
                # Handle special aliases, e.g. locations can be mapped to x or state
                if req == "locations" and (self.x is not None):
                    continue
                if req == "label" and (self.y is not None):
                    continue
                missing.append(req)
        return missing


# ---------------------------------------------------------
# Master Chart Specification Schema
# ---------------------------------------------------------
class ChartSpec(BaseModel):
    title: str
    takeaway: Optional[str] = None
    steps: List[Step] = Field(default_factory=list)
    chart: ChartConfig

