"""
pipeline/generate_preset_questions.py

Generates the preset ChartSpec JSON files in questions/ for Trends (Section 1).
"""

import os
import json

QUESTIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "questions")
os.makedirs(QUESTIONS_DIR, exist_ok=True)

PRESETS = [
    # Section 1: Time-series / Trends
    {
        "filename": "sec1_rank_trajectory.json",
        "spec": {
            "title": "Overall NIRF Rank Trajectory (2017–2025)",
            "takeaway": "Tracks annual rank trajectory with inverted y-axis",
            "steps": [
                {"op": "filter", "where": [{"field": "category", "op": "eq", "value": "Overall"}]},
                {"op": "sort", "by": "year", "order": "asc"},
            ],
            "chart": {"type": "line", "x": "year", "y": "rank", "invert_yaxis": True},
        },
    },
    {
        "filename": "sec1_greatest_improvers.json",
        "spec": {
            "title": "Greatest Rank Improvers (2017 to 2025)",
            "takeaway": "Institutions showing the highest net rank climbed",
            "steps": [
                {"op": "filter", "where": [{"field": "category", "op": "eq", "value": "Overall"}]},
                {"op": "sort", "by": "rank_change", "order": "desc"},
                {"op": "top", "n": 5},
            ],
            "chart": {"type": "bar", "x": "rank_change", "y": "name", "orientation": "h"},
        },
    },
    {
        "filename": "sec1_largest_decliners.json",
        "spec": {
            "title": "Largest Rank Decliners (2017 to 2025)",
            "takeaway": "Institutions showing steepest net rank drop",
            "steps": [
                {"op": "filter", "where": [{"field": "category", "op": "eq", "value": "Overall"}]},
                {"op": "sort", "by": "rank_change", "order": "asc"},
                {"op": "top", "n": 5},
            ],
            "chart": {"type": "bar", "x": "rank_change", "y": "name", "orientation": "h"},
        },
    },
    {
        "filename": "sec1_avg_score_trend.json",
        "spec": {
            "title": "Average Score Trend Over Time (2017–2025)",
            "takeaway": "Mean and median score progression across ranking years",
            "steps": [
                {"op": "filter", "where": [{"field": "category", "op": "eq", "value": "Overall"}]},
                {"op": "aggregate", "by": ["year"], "metrics": [{"field": "score", "fn": "mean", "as": "avg_score"}]},
                {"op": "sort", "by": "year", "order": "asc"},
            ],
            "chart": {"type": "line", "x": "year", "y": "avg_score"},
        },
    },
    {
        "filename": "sec1_consistently_top10.json",
        "spec": {
            "title": "Consistently Top-Ranked Institutions Across Multiple Years",
            "takeaway": "Multi-institution bump trajectory tracking top tier contenders",
            "steps": [
                {"op": "filter", "where": [{"field": "category", "op": "eq", "value": "Overall"}, {"field": "rank", "op": "lte", "value": 10}]},
                {"op": "sort", "by": "year", "order": "asc"},
            ],
            "chart": {"type": "line", "x": "year", "y": "rank", "color": "name", "invert_yaxis": True},
        },
    },
]


def generate():
    for item in PRESETS:
        path = os.path.join(QUESTIONS_DIR, item["filename"])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(item["spec"], f, indent=2)
    print(f"Generated {len(PRESETS)} Trends preset question specs in {QUESTIONS_DIR}")


if __name__ == "__main__":
    generate()

