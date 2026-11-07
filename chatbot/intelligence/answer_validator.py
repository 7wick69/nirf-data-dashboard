"""Lightweight integrity checks for analytical result objects."""
from __future__ import annotations
import math
from typing import Any

def _is_empty_data(data: Any) -> bool:
    if data is None:
        return True
    if hasattr(data, "empty"):
        return bool(data.empty)
    if isinstance(data, (list, tuple, dict, str)):
        return len(data) == 0
    return False

def validate_result(result: dict[str, Any] | None,
                    prediction: dict[str, Any] | None = None,
                    chart_spec: dict[str, Any] | None = None) -> dict[str, Any]:
    errors, warnings = [], []
    if not isinstance(result, dict):
        return {"valid": False, "errors": ["Result must be a dictionary."], "warnings": []}

    status = result.get("status", "success")
    if status == "success" and _is_empty_data(result.get("data")):
        errors.append("Result status is success but the returned data is empty.")
    if status in {"needs_clarification", "unsupported", "no_data", "validation_error"}:
        warnings.append(f"Result status is {status}; do not present it as a successful analysis.")

    if prediction and prediction.get("status") != "success":
        warnings.append("Question intent is uncertain; do not generate a definitive analytical explanation.")

    if chart_spec and chart_spec.get("render") and _is_empty_data(result.get("data")):
        errors.append("A chart was requested but no data is available.")

    # Check numeric values where the result exposes a statistics dictionary.
    stats = result.get("statistics")
    if isinstance(stats, dict):
        for key, value in stats.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(float(value)):
                    errors.append(f"Statistic '{key}' is not finite.")

    return {"valid": not errors, "errors": errors, "warnings": warnings}
