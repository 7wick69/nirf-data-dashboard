"""Pre-render checks for NIRF Plotly charts."""
from __future__ import annotations
from typing import Any
import pandas as pd
from .chart_specs import SUPPORTED_CHARTS, normalize_chart_type


def validate_chart_input(result: dict[str, Any], chart_type: str | None = None) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if result.get("status") != "success":
        errors.append("Only successful analytical results can be visualized.")
    chart = result.get("chart") or {}
    records = chart.get("data") or result.get("data") or []
    frame = pd.DataFrame(records)
    if frame.empty:
        errors.append("The analytical result contains no chart records.")
    x, y = chart.get("x"), chart.get("y")
    normalized = normalize_chart_type(chart_type or chart.get("type"), result.get("intent"), x, y)
    if normalized not in SUPPORTED_CHARTS:
        errors.append(f"Unsupported chart type: {normalized}")
    if normalized in {"line", "scatter", "scatter_quadrant", "scatter_rank"}:
        if x not in frame.columns or y not in frame.columns:
            errors.append("The selected chart requires valid x and y columns.")
        else:
            if not pd.api.types.is_numeric_dtype(pd.to_numeric(frame[x], errors="coerce")):
                warnings.append(f"X field '{x}' is not numeric; verify it is appropriate for this chart.")
            if not pd.api.types.is_numeric_dtype(pd.to_numeric(frame[y], errors="coerce")):
                warnings.append(f"Y field '{y}' is not numeric; verify it is appropriate for this chart.")
    if normalized == "box" and y not in frame.columns:
        errors.append("Box plot metric column is missing.")
    if len(frame) > 1000:
        warnings.append("Large result set: use aggregation or sampling for interactive rendering.")
    if normalized == "line" and "display_name" in frame and frame["display_name"].nunique() > 8:
        warnings.append("Many institution series may overlap; show a selected subset or small multiples.")
    if normalized == "heatmap" and len(frame) > 30:
        warnings.append("Heatmap has many rows; render a selected subset and retain full data in the table.")
    return {"valid": not errors, "errors": errors, "warnings": warnings,
            "normalized_chart_type": normalized, "row_count": len(frame)}
