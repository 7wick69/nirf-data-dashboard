"""
charts/geo.py

Geospatial and Regional Analysis visualizations for NIRF ranking data:
  1. Top States in Top 100 (Choropleth Map + Horizontal Bar Chart)
  2. Geographical Distribution Over Time (Choropleth Map + Stacked Bar Chart)
  3. State-wise Representation Improvement (Diverging Delta Map + Delta Bar/Slope Chart)
  4. Regional Head-to-Head Comparison: Karnataka vs Tamil Nadu (Focus Map + Boxplot/Violin Plot)
  5. Newly Appearing Institutions by State (Incubator Density Map + New Entrants Bar Chart)
"""

import os
import json
from typing import Tuple, Optional, Union, List
import pandas as pd
import plotly.graph_objects as go
from charts.theme import (
    apply_theme,
    DARK_BG,
    PANEL_BG,
    GRID_COLOR,
    LINE_COLOR,
    ACCENT_GREEN,
    ACCENT_RED,
    TEXT_COLOR,
    MUTED_TEXT,
    PALETTE,
)

# Path to India States GeoJSON files
ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
MAP_STATES_GEOJSON_PATH = os.path.join(ASSETS_DIR, "india_map_states.geojson")
DEFAULT_GEOJSON_PATH = os.path.join(ASSETS_DIR, "india_states.geojson")

# State Name Standardization mappings across different GeoJSON schemas
STATE_GEOJSON_MAPPINGS = {
    "state_name": {
        "Andaman and Nicobar Islands": "A & N Islands",
        "Jammu and Kashmir": "Jammu & Kashmir",
    },
    "NAME_1": {
        "Odisha": "Orissa",
        "Uttarakhand": "Uttaranchal",
        "Andaman and Nicobar Islands": "Andaman and Nicobar",
    },
    "st_nm": {},
}


def _load_india_geojson() -> Optional[dict]:
    """Loads India States GeoJSON from assets directory, preferring india_map_states.geojson."""
    for path in [MAP_STATES_GEOJSON_PATH, DEFAULT_GEOJSON_PATH]:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    d = json.load(f)

                # Normalize any GeometryCollection (e.g. A & N Islands) to MultiPolygon
                for feat in d.get("features", []):
                    geom = feat.get("geometry", {})
                    if geom.get("type") == "GeometryCollection":
                        poly_coords = []
                        for g in geom.get("geometries", []):
                            if g.get("type") == "Polygon":
                                poly_coords.append(g.get("coordinates"))
                            elif g.get("type") == "MultiPolygon":
                                poly_coords.extend(g.get("coordinates"))
                        feat["geometry"] = {
                            "type": "MultiPolygon",
                            "coordinates": poly_coords,
                        }

                # If district-level features are present, aggregate by state MultiPolygon
                if d.get("features") and any("district" in f.get("properties", {}) for f in d["features"]):
                    state_features = {}
                    for feat in d["features"]:
                        st = (
                            feat.get("properties", {}).get("state_name")
                            or feat.get("properties", {}).get("st_nm")
                        )
                        if not st:
                            continue
                        geom = feat.get("geometry")
                        if not geom:
                            continue
                        if st not in state_features:
                            state_features[st] = []
                        gtype = geom.get("type")
                        coords = geom.get("coordinates", [])
                        if gtype == "Polygon":
                            state_features[st].append(coords)
                        elif gtype == "MultiPolygon":
                            state_features[st].extend(coords)
                    new_feats = [
                        {
                            "type": "Feature",
                            "properties": {"state_name": st, "st_nm": st},
                            "geometry": {"type": "MultiPolygon", "coordinates": polys},
                        }
                        for st, polys in sorted(state_features.items())
                    ]
                    return {"type": "FeatureCollection", "features": new_feats}

                return d
            except Exception:
                continue
    return None


def _get_geo_key_and_mappings(geojson: dict) -> Tuple[str, str, dict]:
    """Detects property key and mapping dictionaries for GeoJSON features."""
    first_props = geojson["features"][0].get("properties", {}) if geojson.get("features") else {}
    if "state_name" in first_props:
        feature_id_key = "properties.state_name"
        state_prop_key = "state_name"
    elif "st_nm" in first_props:
        feature_id_key = "properties.st_nm"
        state_prop_key = "st_nm"
    elif "NAME_1" in first_props:
        feature_id_key = "properties.NAME_1"
        state_prop_key = "NAME_1"
    else:
        feature_id_key = "properties.state_name"
        state_prop_key = "state_name"

    mapping = STATE_GEOJSON_MAPPINGS.get(state_prop_key, {})
    reverse_mapping = {v: k for k, v in mapping.items()}
    return feature_id_key, state_prop_key, reverse_mapping


# ---------------------------------------------------------
# 1. Top States in Top 100: Map & Bar Chart
# ---------------------------------------------------------
def plot_top_states_map(
    df: pd.DataFrame,
    category: str = "Overall",
    year: Optional[int] = 2025,
    color_palette: str = "Viridis",
) -> go.Figure:
    """1A. Choropleth Map: States with highest number of institutions in the Top 100."""
    sub = df[(df["category"] == category) & (df["rank"] <= 100)].copy()
    if year is not None:
        sub = sub[sub["year"] == year]

    geojson = _load_india_geojson()
    if geojson is None:
        fig = go.Figure()
        return apply_theme(fig, title="Map data unavailable")

    feature_id_key, state_prop_key, reverse_mapping = _get_geo_key_and_mappings(geojson)
    counts_dict = sub["state"].value_counts().to_dict()

    geo_state_names = []
    for feat in geojson.get("features", []):
        st_name = feat.get("properties", {}).get(state_prop_key)
        if st_name and st_name not in geo_state_names:
            geo_state_names.append(st_name)

    records = []
    for geo_st in geo_state_names:
        csv_st = reverse_mapping.get(geo_st, geo_st)
        cnt = int(counts_dict.get(csv_st, 0))
        records.append({
            "geo_state": geo_st,
            "display_state": csv_st,
            "count": cnt,
        })
    counts = pd.DataFrame(records)

    fig = go.Figure(
        go.Choropleth(
            geojson=geojson,
            featureidkey=feature_id_key,
            locations=counts["geo_state"],
            z=counts["count"],
            colorscale=color_palette,
            zmin=0,
            marker_line_color="#475569",
            marker_line_width=0.8,
            colorbar=dict(
                title=dict(text="Top 100 Count", font=dict(color=TEXT_COLOR, size=11)),
                tickfont=dict(color=MUTED_TEXT, size=10),
                bgcolor="rgba(23, 25, 30, 0.75)",
                bordercolor=GRID_COLOR,
                borderwidth=1,
            ),
            customdata=counts[["display_state"]],
            hovertemplate="<b>%{customdata[0]}</b><br>Top 100 Institutions: %{z}<extra></extra>",
        )
    )
    fig = apply_theme(
        fig,
        title=f"Top 100 Institutional Density: {category} ({year or 'All Years'})",
    )
    fig.update_geos(
        visible=False,
        bgcolor=DARK_BG,
        projection_type="mercator",
        lataxis_range=[6.0, 38.5],
        lonaxis_range=[66.5, 100.5],
        showland=False,
        showocean=False,
        showlakes=False,
        showrivers=False,
        showcountries=False,
        showcoastlines=False,
        framewidth=0,
    )
    fig.update_layout(height=680, margin=dict(l=0, r=10, t=45, b=0))
    return fig


def plot_top_states_top100(
    df: pd.DataFrame,
    category: str = "Overall",
    year: Optional[int] = 2025,
    top_n: int = 15,
) -> Tuple[go.Figure, str]:
    """1B. Horizontal Bar Chart: States with highest number of institutions in the Top 100."""
    sub = df[(df["category"] == category) & (df["rank"] <= 100)].copy()

    if year is not None:
        sub = sub[sub["year"] == year]

    if sub.empty:
        fig = go.Figure()
        fig = apply_theme(fig, title=f"No Top 100 records for {category} ({year or 'All Years'})")
        return fig, f"No Top 100 institutions found for category '{category}' in {year or 'the selected period'}."

    total_count = len(sub)
    state_counts = sub["state"].value_counts().reset_index()
    state_counts.columns = ["state", "count"]
    state_counts["percentage"] = (state_counts["count"] / total_count) * 100.0

    top_states = state_counts.head(top_n).iloc[::-1]

    top_state = state_counts.iloc[0]["state"]
    top_cnt = int(state_counts.iloc[0]["count"])
    top_pct = state_counts.iloc[0]["percentage"]
    runner = state_counts.iloc[1]["state"] if len(state_counts) > 1 else None
    runner_cnt = int(state_counts.iloc[1]["count"]) if len(state_counts) > 1 else 0

    year_str = str(year) if year is not None else "All Years in Range"
    top_share_str = f"{top_pct:.1f}% of Top 100" if year is not None else f"{top_pct:.1f}% of all Top 100 entries"
    if runner:
        takeaway = (
            f"In {year_str} ({category}), {top_state} leads with {top_cnt} institutions "
            f"({top_share_str}), followed by {runner} with {runner_cnt}."
        )
    else:
        takeaway = (
            f"In {year_str} ({category}), {top_state} leads with {top_cnt} institutions "
            f"({top_share_str})."
        )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=top_states["count"],
            y=top_states["state"],
            orientation="h",
            text=[f"{c} ({p:.1f}%)" for c, p in zip(top_states["count"], top_states["percentage"])],
            textposition="auto",
            marker=dict(
                color=PALETTE[0],
                line=dict(color=GRID_COLOR, width=1),
            ),
            customdata=top_states["percentage"],
            hovertemplate=(
                "<b>%{y}</b><br>"
                + "Count in Top 100: %{x}<br>"
                + "Share of Top 100: %{customdata:.1f}%<extra></extra>"
            ),
        )
    )

    fig = apply_theme(
        fig,
        title=f"Leading States in Top 100: {category} ({year or 'All Years'})",
        xaxis_title="Number of Institutions in Top 100",
        yaxis_title="State",
    )
    fig.update_layout(height=650)
    return fig, takeaway


# ---------------------------------------------------------
# 2. Geographical Distribution Over Time: Map & Stacked Chart
# ---------------------------------------------------------
def plot_geo_distribution_over_time(
    df: pd.DataFrame,
    category: str = "Overall",
    selected_year: int = 2025,
    view_mode: str = "India Choropleth Map",
    top_n_states: int = 8,
    color_palette: str = "Viridis",
) -> Tuple[go.Figure, str]:
    """2. Geographical distribution of top-ranked institutions changing from 2017 to 2025."""
    sub_all = df[(df["category"] == category) & (df["rank"] <= 100)].copy()

    if sub_all.empty:
        fig = go.Figure()
        fig = apply_theme(fig, title=f"No data for {category}")
        return fig, f"No Top 100 data available for category '{category}'."

    geojson = _load_india_geojson()

    if view_mode == "India Choropleth Map" and geojson is not None:
        sub_yr = sub_all[sub_all["year"] == selected_year]
        if sub_yr.empty:
            sub_yr = sub_all[sub_all["year"] == sub_all["year"].max()]
            selected_year = int(sub_yr["year"].iloc[0])

        counts_dict = sub_yr["state"].value_counts().to_dict()
        feature_id_key, state_prop_key, reverse_mapping = _get_geo_key_and_mappings(geojson)

        geo_state_names = []
        for feat in geojson.get("features", []):
            st_name = feat.get("properties", {}).get(state_prop_key)
            if st_name and st_name not in geo_state_names:
                geo_state_names.append(st_name)

        records = []
        for geo_st in geo_state_names:
            csv_st = reverse_mapping.get(geo_st, geo_st)
            cnt = int(counts_dict.get(csv_st, 0))
            records.append({
                "geo_state": geo_st,
                "display_state": csv_st,
                "count": cnt,
            })
        counts = pd.DataFrame(records)

        ranked_states = counts.sort_values(by="count", ascending=False)
        top_row = ranked_states.iloc[0]
        runner_row = ranked_states.iloc[1] if len(ranked_states) > 1 else None
        takeaway = (
            f"In {selected_year} ({category}), top institutions are clustered in {top_row['display_state']} ({top_row['count']})"
            + (f" and {runner_row['display_state']} ({runner_row['count']})" if runner_row is not None else "")
            + ", leading the concentration of Top 100 institutions."
        )

        fig = go.Figure(
            go.Choropleth(
                geojson=geojson,
                featureidkey=feature_id_key,
                locations=counts["geo_state"],
                z=counts["count"],
                colorscale=color_palette,
                zmin=0,
                marker_line_color="#475569",
                marker_line_width=0.8,
                colorbar=dict(
                    title=dict(text="Top 100 Count", font=dict(color=TEXT_COLOR, size=11)),
                    tickfont=dict(color=MUTED_TEXT, size=10),
                    bgcolor="rgba(23, 25, 30, 0.75)",
                    bordercolor=GRID_COLOR,
                    borderwidth=1,
                ),
                customdata=counts[["display_state"]],
                hovertemplate="<b>%{customdata[0]}</b><br>Top 100 Institutions: %{z}<extra></extra>",
            )
        )
        fig = apply_theme(
            fig,
            title=f"Geographical Distribution Map: {category} ({selected_year})",
        )
        fig.update_geos(
        visible=False,
        bgcolor=DARK_BG,
        projection_type="mercator",
        lataxis_range=[6.0, 38.5],
        lonaxis_range=[66.5, 100.5],
        showland=False,
        showocean=False,
        showlakes=False,
        showrivers=False,
        showcountries=False,
        showcoastlines=False,
        framewidth=0,
    )
        fig.update_layout(height=680, margin=dict(l=0, r=10, t=45, b=0))
        return fig, takeaway

    else:
        # State-wise Stacked Bar Chart across years (2017-2025)
        piv = sub_all.pivot_table(index="year", columns="state", values="rank", aggfunc="count", fill_value=0)
        total_by_state = piv.sum(axis=0).sort_values(ascending=False)
        top_states = total_by_state.head(top_n_states).index.tolist()

        fig = go.Figure()
        for idx, st in enumerate(top_states):
            color = PALETTE[idx % len(PALETTE)]
            fig.add_trace(
                go.Bar(
                    x=piv.index,
                    y=piv[st],
                    name=st,
                    marker=dict(color=color),
                    hovertemplate=f"<b>{st}</b><br>Year: %{{x}}<br>Count: %{{y}}<extra></extra>",
                )
            )

        other_states = [s for s in piv.columns if s not in top_states]
        if other_states:
            other_series = piv[other_states].sum(axis=1)
            fig.add_trace(
                go.Bar(
                    x=piv.index,
                    y=other_series,
                    name="Other States",
                    marker=dict(color="#475569"),
                    hovertemplate="<b>Other States</b><br>Year: %{x}<br>Count: %{y}<extra></extra>",
                )
            )

        fig = apply_theme(
            fig,
            title=f"Evolution of Top 100 Distribution Over Time: {category} (2017–2025)",
            xaxis_title="Year",
            yaxis_title="Number of Top 100 Institutions",
            show_legend=True,
        )
        fig.update_layout(barmode="stack", height=650)
        fig.update_xaxes(type="category")
        takeaway = (
            f"Between 2017 and 2025 ({category}), {top_states[0]} and {top_states[1] if len(top_states)>1 else ''} "
            f"consistently anchored the highest share of Top 100 institutions."
        )
        return fig, takeaway


# ---------------------------------------------------------
# 3. State Representation Improvement: Diverging Map & Delta Chart
# ---------------------------------------------------------
def plot_state_representation_map(
    df: pd.DataFrame,
    category: str = "Overall",
    start_year: int = 2017,
    end_year: int = 2025,
) -> go.Figure:
    """3A. Diverging Choropleth Map: Green for positive gain, Red for decline."""
    sub = df[(df["category"] == category) & (df["rank"] <= 100)].copy()
    geojson = _load_india_geojson()
    if geojson is None:
        fig = go.Figure()
        return apply_theme(fig, title="Map data unavailable")

    feature_id_key, state_prop_key, reverse_mapping = _get_geo_key_and_mappings(geojson)
    c_start = sub[sub["year"] == start_year]["state"].value_counts().to_dict()
    c_end = sub[sub["year"] == end_year]["state"].value_counts().to_dict()

    geo_state_names = []
    for feat in geojson.get("features", []):
        st_name = feat.get("properties", {}).get(state_prop_key)
        if st_name and st_name not in geo_state_names:
            geo_state_names.append(st_name)

    records = []
    for geo_st in geo_state_names:
        csv_st = reverse_mapping.get(geo_st, geo_st)
        v1 = int(c_start.get(csv_st, 0))
        v2 = int(c_end.get(csv_st, 0))
        records.append({
            "geo_state": geo_st,
            "display_state": csv_st,
            "delta": v2 - v1,
            "start": v1,
            "end": v2,
        })
    counts = pd.DataFrame(records)
    max_d = max(abs(counts["delta"].min()), abs(counts["delta"].max())) or 1

    # Diverging color scale: Red for negative, neutral slate for 0, Emerald green for positive
    diverging_scale = [
        [0.0, "#ef4444"],
        [0.45, "#7f1d1d"],
        [0.5, "#252830"],
        [0.55, "#065f46"],
        [1.0, "#22c55e"],
    ]

    fig = go.Figure(
        go.Choropleth(
            geojson=geojson,
            featureidkey=feature_id_key,
            locations=counts["geo_state"],
            z=counts["delta"],
            colorscale=diverging_scale,
            zmin=-max_d,
            zmax=max_d,
            marker_line_color="#475569",
            marker_line_width=0.8,
            colorbar=dict(
                title=dict(text="Net Delta", font=dict(color=TEXT_COLOR, size=11)),
                tickfont=dict(color=MUTED_TEXT, size=10),
                bgcolor="rgba(23, 25, 30, 0.75)",
                bordercolor=GRID_COLOR,
                borderwidth=1,
            ),
            customdata=counts[["display_state", "start", "end"]],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                + f"Net Representation Shift: %{{z:+d}}<br>"
                + f"{start_year} Count: %{{customdata[1]}}<br>"
                + f"{end_year} Count: %{{customdata[2]}}<extra></extra>"
            ),
        )
    )
    fig = apply_theme(
        fig,
        title=f"Representation Shift Map: {category} ({start_year} vs {end_year})",
    )
    fig.update_geos(
        visible=False,
        bgcolor=DARK_BG,
        projection_type="mercator",
        lataxis_range=[6.0, 38.5],
        lonaxis_range=[66.5, 100.5],
        showland=False,
        showocean=False,
        showlakes=False,
        showrivers=False,
        showcountries=False,
        showcoastlines=False,
        framewidth=0,
    )
    fig.update_layout(height=680, margin=dict(l=0, r=10, t=45, b=0))
    return fig


def plot_state_representation_improvement(
    df: pd.DataFrame,
    category: str = "Overall",
    start_year: int = 2017,
    end_year: int = 2025,
    chart_type: str = "Bar (Delta)",
) -> Tuple[go.Figure, str]:
    """3B. Delta Bar Chart or Slope Chart: Representation improvement over time."""
    sub = df[(df["category"] == category) & (df["rank"] <= 100)].copy()

    c_start = sub[sub["year"] == start_year]["state"].value_counts()
    c_end = sub[sub["year"] == end_year]["state"].value_counts()

    all_states = sorted(set(c_start.index) | set(c_end.index))
    if not all_states:
        fig = go.Figure()
        fig = apply_theme(fig, title=f"No Top 100 data for {category} between {start_year} and {end_year}")
        return fig, "No representation records found for the selected years."

    records = []
    for s in all_states:
        v_start = int(c_start.get(s, 0))
        v_end = int(c_end.get(s, 0))
        records.append({"state": s, "start": v_start, "end": v_end, "delta": v_end - v_start})

    res_df = pd.DataFrame(records).sort_values(by="delta", ascending=True)

    improvers = res_df[res_df["delta"] > 0].sort_values(by="delta", ascending=False)
    decliners = res_df[res_df["delta"] < 0].sort_values(by="delta", ascending=True)

    top_imp_text = (
        f"{improvers.iloc[0]['state']} (+{improvers.iloc[0]['delta']})" if not improvers.empty else "None"
    )
    second_imp_text = (
        f", {improvers.iloc[1]['state']} (+{improvers.iloc[1]['delta']})" if len(improvers) > 1 else ""
    )
    top_dec_text = (
        f"{decliners.iloc[0]['state']} ({decliners.iloc[0]['delta']})" if not decliners.empty else "None"
    )

    if start_year == end_year:
        takeaway = f"Same comparison year selected ({start_year}): No net representation shift for {category}."
    else:
        takeaway = (
            f"From {start_year} to {end_year} ({category}): Top improvers were {top_imp_text}{second_imp_text}; "
            f"largest decline occurred in {top_dec_text}."
        )

    fig = go.Figure()

    if chart_type == "Slope Chart":
        slope_df = res_df.sort_values(by="delta", ascending=False)
        for _, row in slope_df.iterrows():
            st = row["state"]
            d = row["delta"]
            color = ACCENT_GREEN if d > 0 else (ACCENT_RED if d < 0 else MUTED_TEXT)
            # Abbreviate long state names for left label to avoid clipping
            abbrev = st if len(st) <= 12 else st[:11] + "."
            left_label  = f"{abbrev} {row['start']}"
            right_label = f"{row['end']} ({d:+d})"
            fig.add_trace(
                go.Scatter(
                    x=[str(start_year), str(end_year)],
                    y=[row["start"], row["end"]],
                    mode="lines+markers+text",
                    name=st,
                    line=dict(color=color, width=2 if d != 0 else 1),
                    marker=dict(size=7, color=color),
                    text=[left_label, right_label],
                    textposition=["middle left", "middle right"],
                    textfont=dict(
                        color="#e2e8f0",   # bright white-ish so visible on dark bg
                        size=11,
                    ),
                    hovertemplate=f"<b>{st}</b><br>{start_year}: {row['start']}<br>{end_year}: {row['end']}<br>Change: {d:+d}<extra></extra>",
                    showlegend=False,
                )
            )
        fig = apply_theme(
            fig,
            title=f"Representation Trajectory: {category} ({start_year} vs {end_year})",
            xaxis_title="Year",
            yaxis_title="Number of Institutions in Top 100",
        )
        # Full-width layout: moderate margins so labels on both sides are visible.
        # height scales with number of states for readability.
        n_states = len(slope_df)
        chart_height = max(520, min(900, n_states * 28 + 120))
        fig.update_layout(
            height=chart_height,
            margin=dict(l=160, r=140, t=55, b=50),
            autosize=True,
        )
        fig.update_xaxes(
            tickfont=dict(size=13, color="#e2e8f0"),
            title_font=dict(size=13),
        )
        fig.update_yaxes(
            tickfont=dict(size=12, color="#e2e8f0"),
            title_font=dict(size=13),
            zeroline=True,
            zerolinecolor="#475569",
            zerolinewidth=1,
        )
    else:
        bar_colors = [
            ACCENT_GREEN if d > 0 else (ACCENT_RED if d < 0 else MUTED_TEXT) for d in res_df["delta"]
        ]
        fig.add_trace(
            go.Bar(
                x=res_df["delta"],
                y=res_df["state"],
                orientation="h",
                marker=dict(color=bar_colors),
                text=[f"+{d}" if d > 0 else str(d) for d in res_df["delta"]],
                textposition="auto",
                customdata=list(zip(res_df["start"], res_df["end"])),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    + f"Change: %{{x:+d}}<br>"
                    + f"{start_year} Count: %{{customdata[0]}}<br>"
                    + f"{end_year} Count: %{{customdata[1]}}<extra></extra>"
                ),
            )
        )
        fig = apply_theme(
            fig,
            title=f"Net Representation Delta: {category} ({start_year} to {end_year})",
            xaxis_title=f"Net Institution Change ({end_year} - {start_year})",
            yaxis_title="State",
        )
        fig.update_layout(height=650)

    return fig, takeaway


# ---------------------------------------------------------
# 4. Regional Comparison (Karnataka vs Tamil Nadu): Focus Map & Boxplot
# ---------------------------------------------------------
def plot_state_rank_comparison_map(
    df: pd.DataFrame,
    category: str = "Overall",
    state1: str = "Karnataka",
    state2: str = "Tamil Nadu",
    year_range: Optional[Tuple[int, int]] = (2017, 2025),
) -> go.Figure:
    """4A. Choropleth Focus Map highlighting the two compared states with median rankings."""
    sub = df[(df["category"] == category) & (df["state"].isin([state1, state2]))].copy()
    if year_range:
        sub = sub[(sub["year"] >= year_range[0]) & (sub["year"] <= year_range[1])]

    geojson = _load_india_geojson()
    if geojson is None:
        fig = go.Figure()
        return apply_theme(fig, title="Map data unavailable")

    feature_id_key, state_prop_key, reverse_mapping = _get_geo_key_and_mappings(geojson)
    meds = sub.groupby("state")["rank"].median().to_dict()
    counts_map = sub.groupby("state")["rank"].count().to_dict()

    geo_state_names = []
    for feat in geojson.get("features", []):
        st_name = feat.get("properties", {}).get(state_prop_key)
        if st_name and st_name not in geo_state_names:
            geo_state_names.append(st_name)

    records = []
    for geo_st in geo_state_names:
        csv_st = reverse_mapping.get(geo_st, geo_st)
        if csv_st == state1:
            records.append({
                "geo_state": geo_st,
                "display_state": csv_st,
                "val": 2,
                "desc": f"Median Rank: #{meds.get(state1, 0):.1f}<br>Ranked Institutions: {counts_map.get(state1, 0)}",
            })
        elif csv_st == state2:
            records.append({
                "geo_state": geo_st,
                "display_state": csv_st,
                "val": 1,
                "desc": f"Median Rank: #{meds.get(state2, 0):.1f}<br>Ranked Institutions: {counts_map.get(state2, 0)}",
            })
        else:
            records.append({
                "geo_state": geo_st,
                "display_state": csv_st,
                "val": 0,
                "desc": "<i>Not in comparison</i>",
            })
    counts = pd.DataFrame(records)

    focus_colorscale = [
        [0.0, "#16181e"],
        [0.49, "#16181e"],
        [0.5, "#22d3ee"],
        [0.75, "#22d3ee"],
        [0.76, "#3b82f6"],
        [1.0, "#3b82f6"],
    ]

    fig = go.Figure(
        go.Choropleth(
            geojson=geojson,
            featureidkey=feature_id_key,
            locations=counts["geo_state"],
            z=counts["val"],
            colorscale=focus_colorscale,
            showscale=False,
            marker_line_color="#475569",
            marker_line_width=0.8,
            customdata=counts[["display_state", "desc"]],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                + "%{customdata[1]}<extra></extra>"
            ),
        )
    )
    fig = apply_theme(
        fig,
        title=f"Geographic Focus: {state1} & {state2}",
    )
    fig.update_geos(
        visible=False,
        bgcolor=DARK_BG,
        projection_type="mercator",
        lataxis_range=[6.0, 38.5],
        lonaxis_range=[66.5, 100.5],
        showland=False,
        showocean=False,
        showlakes=False,
        showrivers=False,
        showcountries=False,
        showcoastlines=False,
        framewidth=0,
    )
    fig.update_layout(height=680, margin=dict(l=0, r=10, t=45, b=0))
    return fig


def plot_state_rank_comparison(
    df: pd.DataFrame,
    category: str = "Overall",
    state1: str = "Karnataka",
    state2: str = "Tamil Nadu",
    year_range: Optional[Tuple[int, int]] = (2017, 2025),
    plot_type: str = "Box Plot",
    breakdown: str = "Overall Period",
) -> Tuple[go.Figure, str]:
    """4B. Comparative Box Plot or Violin Plot of institutional rankings with inverted Y-axis."""
    sub = df[(df["category"] == category) & (df["state"].isin([state1, state2]))].copy()

    if year_range:
        sub = sub[(sub["year"] >= year_range[0]) & (sub["year"] <= year_range[1])]

    if sub.empty:
        fig = go.Figure()
        fig = apply_theme(fig, title=f"No comparison data for {state1} and {state2} in {category}")
        return fig, f"No ranking records found for {state1} or {state2} in category '{category}'."

    s1_data = sub[sub["state"] == state1]
    s2_data = sub[sub["state"] == state2]

    med1 = s1_data["rank"].median() if not s1_data.empty else float("nan")
    med2 = s2_data["rank"].median() if not s2_data.empty else float("nan")
    cnt1 = len(s1_data)
    cnt2 = len(s2_data)

    takeaway = (
        f"{state1} median rank is #{med1:.1f} (n={cnt1}) vs {state2} median rank of #{med2:.1f} (n={cnt2}) "
        f"in {category} ({year_range[0] if year_range else 2017}–{year_range[1] if year_range else 2025})."
    )

    fig = go.Figure()

    if breakdown == "Year-wise Comparison":
        years = sorted(sub["year"].unique())
        for st_name, color in [(state1, PALETTE[0]), (state2, PALETTE[4])]:
            st_sub = sub[sub["state"] == st_name]
            if plot_type == "Violin Plot":
                fig.add_trace(
                    go.Violin(
                        x=st_sub["year"],
                        y=st_sub["rank"],
                        name=st_name,
                        marker_color=color,
                        box_visible=True,
                        meanline_visible=True,
                    )
                )
            else:
                fig.add_trace(
                    go.Box(
                        x=st_sub["year"],
                        y=st_sub["rank"],
                        name=st_name,
                        marker_color=color,
                        boxpoints="outliers",
                    )
                )
        fig.update_layout(boxmode="group")
        fig.update_xaxes(type="category")
    else:
        for st_name, color, med in [(state1, PALETTE[0], med1), (state2, PALETTE[4], med2)]:
            st_sub = sub[sub["state"] == st_name]
            if plot_type == "Violin Plot":
                fig.add_trace(
                    go.Violin(
                        y=st_sub["rank"],
                        name=f"{st_name} (Med: #{med:.1f})",
                        marker_color=color,
                        box_visible=True,
                        meanline_visible=True,
                        points="all",
                        jitter=0.05,
                        hovertemplate=f"<b>{st_name}</b><br>Rank: #%{{y}}<extra></extra>",
                    )
                )
            else:
                fig.add_trace(
                    go.Box(
                        y=st_sub["rank"],
                        name=f"{st_name} (Med: #{med:.1f})",
                        marker_color=color,
                        boxpoints="all",
                        jitter=0.2,
                        pointpos=-1.8,
                        hovertemplate=f"<b>{st_name}</b><br>Rank: #%{{y}}<extra></extra>",
                    )
                )

    fig = apply_theme(
        fig,
        title=f"Rank Distribution: {state1} vs {state2} ({category})",
        xaxis_title="State" if breakdown == "Overall Period" else "Year",
        yaxis_title="NIRF Rank (Inverted: #1 at Top)",
        invert_yaxis=True,
        show_legend=True,
    )
    fig.update_layout(height=650)
    return fig, takeaway


# ---------------------------------------------------------
# 5. Newly Appearing Institutions: Density Map & Bar Chart
# ---------------------------------------------------------
def plot_new_entrant_map(
    df: pd.DataFrame,
    category: str = "Overall",
    base_year: int = 2017,
    selected_entry_year: Union[str, int] = "All Years (2018–2025)",
    color_palette: str = "Viridis",
) -> go.Figure:
    """5A. Choropleth Map: Shaded by number of newly appearing institutions by state."""
    sub_cat = df[df["category"] == category].copy()
    geojson = _load_india_geojson()
    if geojson is None:
        fig = go.Figure()
        return apply_theme(fig, title="Map data unavailable")

    feature_id_key, state_prop_key, reverse_mapping = _get_geo_key_and_mappings(geojson)
    first_appearances = sub_cat.groupby("name")["year"].min().reset_index()
    first_appearances.columns = ["name", "first_year"]

    name_state = sub_cat[["name", "state"]].drop_duplicates(subset=["name"])
    new_ent = first_appearances.merge(name_state, on="name")
    new_ent = new_ent[new_ent["first_year"] > base_year]

    is_all_years = str(selected_entry_year).startswith("All")
    if not is_all_years:
        new_ent = new_ent[new_ent["first_year"] == int(selected_entry_year)]

    counts_dict = new_ent["state"].value_counts().to_dict()

    geo_state_names = []
    for feat in geojson.get("features", []):
        st_name = feat.get("properties", {}).get(state_prop_key)
        if st_name and st_name not in geo_state_names:
            geo_state_names.append(st_name)

    records = []
    for geo_st in geo_state_names:
        csv_st = reverse_mapping.get(geo_st, geo_st)
        cnt = int(counts_dict.get(csv_st, 0))
        records.append({
            "geo_state": geo_st,
            "display_state": csv_st,
            "count": cnt,
        })
    counts = pd.DataFrame(records)

    year_label = selected_entry_year if selected_entry_year != "All Years (2018–2025)" else f"{base_year+1}–2025"
    fig = go.Figure(
        go.Choropleth(
            geojson=geojson,
            featureidkey=feature_id_key,
            locations=counts["geo_state"],
            z=counts["count"],
            colorscale=color_palette,
            zmin=0,
            marker_line_color="#475569",
            marker_line_width=0.8,
            colorbar=dict(
                title=dict(text="New Entrants", font=dict(color=TEXT_COLOR, size=11)),
                tickfont=dict(color=MUTED_TEXT, size=10),
                bgcolor="rgba(23, 25, 30, 0.75)",
                bordercolor=GRID_COLOR,
                borderwidth=1,
            ),
            customdata=counts[["display_state"]],
            hovertemplate="<b>%{customdata[0]}</b><br>Newly Appearing Institutions: %{z}<extra></extra>",
        )
    )
    fig = apply_theme(
        fig,
        title=f"New Institutional Entrants Map: {category} ({year_label})",
    )
    fig.update_geos(
        visible=False,
        bgcolor=DARK_BG,
        projection_type="mercator",
        lataxis_range=[6.0, 38.5],
        lonaxis_range=[66.5, 100.5],
        showland=False,
        showocean=False,
        showlakes=False,
        showrivers=False,
        showcountries=False,
        showcoastlines=False,
        framewidth=0,
    )
    fig.update_layout(height=680, margin=dict(l=0, r=10, t=45, b=0))
    return fig


def plot_new_entrant_states(
    df: pd.DataFrame,
    category: str = "Overall",
    base_year: int = 2017,
    top_n: int = 12,
    selected_entry_year: Union[str, int] = "All Years (2018–2025)",
) -> Tuple[go.Figure, str]:
    """5B. Horizontal Bar Chart: States with largest number of newly appearing institutions."""
    sub_cat = df[df["category"] == category].copy()

    if sub_cat.empty:
        fig = go.Figure()
        fig = apply_theme(fig, title=f"No data for {category}")
        return fig, f"No records found for category '{category}'."

    first_appearances = sub_cat.groupby("name")["year"].min().reset_index()
    first_appearances.columns = ["name", "first_year"]

    name_state = sub_cat[["name", "state"]].drop_duplicates(subset=["name"])
    new_ent = first_appearances.merge(name_state, on="name")
    new_ent = new_ent[new_ent["first_year"] > base_year]

    is_all_years = str(selected_entry_year).startswith("All")
    if not is_all_years:
        new_ent = new_ent[new_ent["first_year"] == int(selected_entry_year)]

    if new_ent.empty:
        fig = go.Figure()
        fig = apply_theme(
            fig,
            title=f"No new entrants found in {category} for {selected_entry_year}",
        )
        return fig, f"No newly appearing institutions recorded in {category} for {selected_entry_year}."

    state_counts = new_ent["state"].value_counts().reset_index()
    state_counts.columns = ["state", "new_entrants_count"]
    top_states = state_counts.head(top_n).iloc[::-1]

    top_st = state_counts.iloc[0]["state"]
    top_cnt = int(state_counts.iloc[0]["new_entrants_count"])
    runner_st = state_counts.iloc[1]["state"] if len(state_counts) > 1 else None
    runner_cnt = int(state_counts.iloc[1]["new_entrants_count"]) if len(state_counts) > 1 else 0

    period_label = f"In {selected_entry_year}" if not is_all_years else f"Since {base_year}"
    if runner_st:
        takeaway = (
            f"{period_label} ({category}), {top_st} ({top_cnt}) and {runner_st} ({runner_cnt}) "
            f"produced the largest number of newly appearing institutions in NIRF rankings."
        )
    else:
        takeaway = (
            f"{period_label} ({category}), {top_st} produced the largest number of newly appearing "
            f"institutions ({top_cnt}) in NIRF rankings."
        )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=top_states["new_entrants_count"],
            y=top_states["state"],
            orientation="h",
            marker=dict(
                color=PALETTE[3],
                line=dict(color=GRID_COLOR, width=1),
            ),
            text=top_states["new_entrants_count"],
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>New Institutions: %{x}<extra></extra>",
        )
    )

    year_label = selected_entry_year if selected_entry_year != "All Years (2018–2025)" else f"{base_year+1}–2025"
    fig = apply_theme(
        fig,
        title=f"Newly Appearing Institutions Ranking: {category} ({year_label})",
        xaxis_title="Number of Newly Appearing Institutions",
        yaxis_title="State",
    )
    fig.update_layout(height=650)
    return fig, takeaway
