"""Question-aware Plotly renderer for NIRF analytics results.

Compatible with the AnalyticsResult dictionary produced by the current
NIRFAnalyticsEngine. It preserves the full result data and limits only the
visual marks shown in the figure to avoid clutter.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from charts.theme import (
    apply_theme, LINE_COLOR, ACCENT_GREEN, ACCENT_RED,
    MUTED_TEXT, TEXT_COLOR, PANEL_BG, GRID_COLOR, PALETTE,
)
from .chart_specs import DEFAULT_LIMITS, normalize_chart_type
from .chart_validator import validate_chart_input


class VisualizationError(Exception):
    """Raised when a result cannot be rendered as a meaningful chart."""


class NIRFVisualizationEngine:
    def __init__(self, max_categories: int = 15, max_line_series: int = 8):
        self.metrics = ["TLR", "RPC", "GO", "OI", "PR", "QNR", "QLR", "SFC"]
        self.max_categories = max(3, int(max_categories))
        self.max_line_series = max(2, int(max_line_series))

    @staticmethod
    def _get_data(result: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
        chart = result.get("chart") or {}
        records = chart.get("data") or result.get("data") or []
        frame = pd.DataFrame(records).copy()
        if frame.empty:
            raise VisualizationError("No analytical records available for visualization.")
        return frame, chart

    @staticmethod
    def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
        if column not in frame.columns:
            raise VisualizationError(f"Required numeric column '{column}' is missing.")
        return pd.to_numeric(frame[column], errors="coerce")

    @staticmethod
    def _display_name_column(df: pd.DataFrame) -> str:
        for col in ("display_name", "name", "institution", "institution_name"):
            if col in df.columns:
                return col
        raise VisualizationError("Institution/category label column is unavailable.")

    @staticmethod
    def _fit_line(fig: go.Figure, frame: pd.DataFrame, x: str, y: str) -> None:
        """Add a numpy OLS line without requiring statsmodels."""
        clean = frame[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(clean) < 3 or clean[x].nunique() < 2:
            return
        slope, intercept = np.polyfit(clean[x].to_numpy(), clean[y].to_numpy(), 1)
        xs = np.linspace(float(clean[x].min()), float(clean[x].max()), 100)
        fig.add_trace(go.Scatter(
            x=xs, y=slope * xs + intercept, mode="lines", name="Linear trend",
            line={"color": "#f59e0b", "width": 2, "dash": "dash"},
            hoverinfo="skip",
        ))

    def _ranking_figure(self, df: pd.DataFrame, chart: dict, title: str) -> tuple[go.Figure, int, str]:
        """Render a compact leaderboard; never label every institution point."""
        name_col = self._display_name_column(df)
        rank_col = "rank_num" if "rank_num" in df.columns else "rank"
        if rank_col not in df.columns:
            raise VisualizationError("Ranking chart requires rank_num or rank.")
        df[rank_col] = self._numeric(df, rank_col)
        df = df.dropna(subset=[rank_col]).sort_values(rank_col, ascending=True).head(self.max_categories).copy()
        if df.empty:
            raise VisualizationError("No valid ranked institutions are available.")
        df["rank_label"] = df[rank_col].map(lambda x: f"#{int(x)}")

        # Scores provide a meaningful magnitude comparison. Only use the score bar
        # when every displayed ranked record has a score; otherwise do not silently
        # drop institutions with missing scores.
        if "score" in df.columns and pd.to_numeric(df["score"], errors="coerce").notna().all():
            df["score"] = pd.to_numeric(df["score"], errors="coerce")
            df["institution_label"] = df.apply(
                lambda row: f"#{int(row[rank_col])}  {row[name_col]}", axis=1
            )
            labels = df["institution_label"].astype(str).tolist()
            hover_fields = {rank_col: True, "score": ":.2f"}
            for optional in ("state", "city"):
                if optional in df.columns:
                    hover_fields[optional] = True
            fig = px.bar(
                df, x="score", y="institution_label", orientation="h",
                category_orders={"institution_label": list(reversed(labels))},
                hover_data=hover_fields,
                color_discrete_sequence=[LINE_COLOR], title=title,
            )
            fig.update_xaxes(title="NIRF score", rangemode="tozero")
            fig.update_yaxes(title="Institution")
            subtitle = f"Showing the top {len(df)} institutions by rank, using score for visual comparison."
        else:
            names = df[name_col].astype(str).tolist()
            fig = px.scatter(
                df, x=rank_col, y=name_col,
                category_orders={name_col: list(reversed(names))},
                hover_data=[c for c in ["state", "city", "rank_band"] if c in df.columns],
                title=title,
            )
            fig.update_traces(marker={"size": 10, "color": LINE_COLOR})
            fig.update_xaxes(title="NIRF rank (lower is better)", autorange="reversed")
            fig.update_yaxes(title="Institution", autorange="reversed")
            subtitle = f"Showing {len(df)} ranked institutions. Lower rank number is better."
        return fig, len(df), subtitle

    def _horizontal_bar(self, df: pd.DataFrame, chart: dict, title: str) -> tuple[go.Figure, int, str]:
        x, y = chart.get("x"), chart.get("y")
        # AnalyticsResult convention is x=category, y=value. Some pre-aggregated
        # results may already use x=value, y=category. Detect both safely.
        if x in df.columns and y in df.columns:
            x_numeric = pd.to_numeric(df[x], errors="coerce").notna().sum()
            y_numeric = pd.to_numeric(df[y], errors="coerce").notna().sum()
            if y_numeric >= x_numeric:
                category_col, value_col = x, y
            else:
                category_col, value_col = y, x
        else:
            numeric = [c for c in df if pd.api.types.is_numeric_dtype(df[c])]
            categories = [c for c in df if c not in numeric]
            if not numeric or not categories:
                raise VisualizationError("Horizontal bar chart needs one categorical and one numeric field.")
            category_col, value_col = categories[0], numeric[0]
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=[value_col, category_col]).sort_values(value_col, ascending=True).tail(self.max_categories).copy()
        if df.empty:
            raise VisualizationError("No valid values for horizontal bar chart.")
        labels = df[category_col].astype(str).tolist()
        fig = px.bar(
            df, x=value_col, y=category_col, orientation="h",
            category_orders={category_col: labels}, color_discrete_sequence=[LINE_COLOR],
            hover_data=df.columns.tolist(), title=title,
        )
        fig.update_yaxes(title="", categoryorder="array", categoryarray=labels)
        fig.update_xaxes(title=value_col.replace("_", " ").title(), zeroline=True)
        return fig, len(df), f"Showing {len(df)} categories, sorted by {value_col.replace('_', ' ')}."

    def _diverging_bar(self, df: pd.DataFrame, chart: dict, title: str) -> tuple[go.Figure, int, str]:
        """Render signed differences for either institution or regional labels."""
        x, y = chart.get("x"), chart.get("y")
        if "change" in df.columns:
            value_col = "change"
        elif y in df.columns and pd.to_numeric(df[y], errors="coerce").notna().sum() > 0:
            value_col = y
        elif x in df.columns and pd.to_numeric(df[x], errors="coerce").notna().sum() > 0:
            value_col = x
        else:
            raise VisualizationError("Diverging bar requires a signed numeric change column.")

        label_candidates = [c for c in (y, x, "display_name", "state", "name", "institution")
                            if c in df.columns and c != value_col]
        if not label_candidates:
            raise VisualizationError("Diverging bar requires a category or institution label.")
        label_col = label_candidates[0]
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=[value_col, label_col]).copy()
        if df.empty:
            raise VisualizationError("No signed change values to plot.")

        positives = df[df[value_col] > 0].nlargest(10, value_col)
        negatives = df[df[value_col] < 0].nsmallest(10, value_col)
        zeros = df[df[value_col] == 0].head(2)
        shown = pd.concat([negatives, zeros, positives]).drop_duplicates(label_col)
        shown = shown.sort_values(value_col, ascending=True)
        colors = [ACCENT_GREEN if v > 0 else ACCENT_RED if v < 0 else MUTED_TEXT for v in shown[value_col]]
        custom_cols = [c for c in ["start_value", "end_value", "start_rank", "end_rank", "start_count", "end_count"] if c in shown.columns]
        custom = shown[custom_cols].to_numpy() if custom_cols else None
        hover = f"<b>%{{y}}</b><br>{value_col.replace('_', ' ').title()}: %{{x:+.2f}}<extra></extra>"
        fig = go.Figure(go.Bar(
            x=shown[value_col], y=shown[label_col], orientation="h",
            marker_color=colors, customdata=custom, hovertemplate=hover,
        ))
        fig.update_xaxes(title=value_col.replace("_", " ").title() + " (positive = increase)", zeroline=True,
                         zerolinewidth=2, zerolinecolor=TEXT_COLOR)
        fig.update_yaxes(title=label_col.replace("_", " ").title(), autorange="reversed")
        return fig, len(shown), "Signed values are centered on zero; positive and negative directions are visually distinguished."

    def _metric_dot_plot(self, df: pd.DataFrame, chart: dict, title: str) -> tuple[go.Figure, int, str]:
        """Plot a numeric metric against a manageable set of entity labels."""
        x, y = chart.get("x"), chart.get("y")
        if x in df.columns and y in df.columns:
            x_num = pd.to_numeric(df[x], errors="coerce").notna().sum()
            y_num = pd.to_numeric(df[y], errors="coerce").notna().sum()
            if x_num >= y_num:
                value_col, label_col = x, y
            else:
                value_col, label_col = y, x
        else:
            label_col = self._display_name_column(df)
            numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not numeric:
                raise VisualizationError("Dot plot requires a numerical metric.")
            value_col = next((c for c in numeric if c not in {"year"}), numeric[0])
        if label_col not in df.columns or value_col not in df.columns:
            raise VisualizationError("Dot plot requires a label and a numerical metric.")
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=[value_col, label_col]).sort_values(value_col, ascending=True).tail(self.max_categories).copy()
        if df.empty:
            raise VisualizationError("No valid values for dot plot.")
        fig = px.scatter(df, x=value_col, y=label_col,
                         hover_data=[c for c in ["rank_num", "rank", "score", "state", "start_value", "end_value"] if c in df.columns],
                         title=title)
        fig.update_traces(mode="markers", marker={"size": 10, "color": LINE_COLOR})
        fig.update_xaxes(title=value_col.replace("_", " ").title(), zeroline=value_col in {"change", "movement"})
        fig.update_yaxes(title="", autorange="reversed")
        return fig, len(df), f"Showing {len(df)} observations; exact values are available in the hover details."

    def _line(self, df: pd.DataFrame, chart: dict, title: str) -> tuple[go.Figure, int, str]:
        x, y = chart.get("x"), chart.get("y")
        if x not in df.columns or y not in df.columns:
            raise VisualizationError("Line chart requires valid x and y columns.")
        df[x] = pd.to_numeric(df[x], errors="coerce")
        df[y] = pd.to_numeric(df[y], errors="coerce")
        df = df.dropna(subset=[x, y]).copy().sort_values(x)
        if df.empty:
            raise VisualizationError("No valid time-series observations.")
        color = None
        note = "Chronological observations are connected in year order."
        series_col = next((c for c in ["correlation_type", "movement_status", "display_name"] if c in df.columns and df[c].nunique() > 1), None)
        if series_col:
            color = series_col
            n_series = df[series_col].nunique()
            if series_col == "display_name" and n_series > self.max_line_series:
                    # Keep the most relevant series, rather than connecting unrelated entities.
                    if "rank_num" in df.columns:
                        latest_year = df[x].max()
                        chosen = (df[df[x] == latest_year].sort_values("rank_num")[series_col].head(self.max_line_series).tolist())
                    else:
                        chosen = df.groupby(series_col)[y].last().sort_values(ascending=False).head(self.max_line_series).index.tolist()
                    df = df[df[series_col].isin(chosen)]
                    note = f"Showing {len(chosen)} selected institution series to avoid overplotting; full data remains in the table."
        if {"q1", "q3"}.issubset(df.columns) and y == "median":
            # Draw the IQR envelope explicitly; the median remains the focal line.
            lower = pd.to_numeric(df["q1"], errors="coerce")
            upper = pd.to_numeric(df["q3"], errors="coerce")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df[x], y=lower, mode="lines", line={"width": 0},
                name="Q1", showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=df[x], y=upper, mode="lines", line={"width": 0},
                fill="tonexty", fillcolor="rgba(59,130,246,0.18)",
                name="Interquartile range", hovertemplate="Year: %{x}<br>Q3: %{y:.2f}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=df[x], y=df[y], mode="lines+markers", name="Median",
                line={"color": LINE_COLOR, "width": 2.5},
                hovertemplate="Year: %{x}<br>Median: %{y:.2f}<extra></extra>",
            ))
        else:
            fig = px.line(df, x=x, y=y, color=color, markers=True,
                          hover_data=[c for c in ["display_name", "correlation_type", "rank_num", "score", "state", "sample_size"] if c in df.columns],
                          color_discrete_sequence=PALETTE, title=title)
        fig.update_xaxes(title=x.replace("_", " ").title(), dtick=1 if x == "year" else None)
        fig.update_yaxes(title=y.replace("_", " ").title())
        if y in {"rank_num", "rank"} or "rank" in y.lower():
            fig.update_yaxes(autorange="reversed")
        return fig, len(df), note

    def _scatter(self, df: pd.DataFrame, chart: dict, title: str,
                 quadrant: bool = False, invert_rank: bool = False,
                 add_trend: bool = True) -> tuple[go.Figure, int, str]:
        x, y = chart.get("x"), chart.get("y")
        if x not in df.columns or y not in df.columns:
            raise VisualizationError("Scatter plot requires valid x and y columns.")
        df[x] = pd.to_numeric(df[x], errors="coerce")
        df[y] = pd.to_numeric(df[y], errors="coerce")
        df = df.dropna(subset=[x, y]).copy()
        if len(df) < 2:
            raise VisualizationError("At least two complete observations are required for scatter.")
        hover_name = "display_name" if "display_name" in df.columns else ("name" if "name" in df.columns else None)
        hover_cols = [c for c in ["state", "rank_num", "score"] if c in df.columns]
        if "is_outlier" in df.columns:
            df["is_outlier"] = df["is_outlier"].fillna(False).astype(bool)
            fig = px.scatter(
                df, x=x, y=y, color="is_outlier", hover_name=hover_name,
                hover_data=hover_cols + [c for c in ["research_residual", "rank_residual", "score_change"] if c in df.columns],
                color_discrete_map={True: ACCENT_RED, False: LINE_COLOR},
                category_orders={"is_outlier": [False, True]}, opacity=0.78, title=title,
            )
        else:
            fig = px.scatter(df, x=x, y=y, hover_name=hover_name, hover_data=hover_cols,
                             opacity=0.72, title=title)
        fig.update_traces(marker={"size": 9, "line": {"width": 0.4, "color": PANEL_BG}})
        if add_trend:
            self._fit_line(fig, df, x, y)
        if quadrant:
            x_med, y_med = float(df[x].median()), float(df[y].median())
            fig.add_vline(x=x_med, line_dash="dot", line_color=MUTED_TEXT)
            fig.add_hline(y=y_med, line_dash="dot", line_color=MUTED_TEXT)
        fig.update_xaxes(title=x.replace("_", " ").title())
        fig.update_yaxes(title=y.replace("_", " ").title(), autorange="reversed" if invert_rank or y in {"rank_num", "rank"} else True)
        stats = chart.get("statistics") or {}
        return fig, len(df), "Points represent pairwise-complete observations; correlation does not imply causation."

    def _histogram(self, df: pd.DataFrame, chart: dict, title: str) -> tuple[go.Figure, int, str]:
        if {"bin_start", "bin_end", "count"}.issubset(df.columns):
            for c in ("bin_start", "bin_end", "count"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["bin_start", "bin_end", "count"])
            mid = (df["bin_start"] + df["bin_end"]) / 2
            widths = df["bin_end"] - df["bin_start"]
            fig = go.Figure(go.Bar(x=mid, y=df["count"], width=widths,
                                   marker_color=LINE_COLOR,
                                   customdata=np.column_stack([df["bin_start"], df["bin_end"]]),
                                   hovertemplate="Range: %{customdata[0]:.2f}–%{customdata[1]:.2f}<br>Count: %{y}<extra></extra>"))
            fig.update_xaxes(title="Metric value")
            fig.update_yaxes(title="Number of institutions")
            return fig, int(df["count"].sum()), "Histogram bins are calculated by the analytics engine."
        x = chart.get("x")
        if x not in df.columns:
            raise VisualizationError("Histogram requires raw values or bin_start/bin_end/count columns.")
        df[x] = pd.to_numeric(df[x], errors="coerce")
        df = df.dropna(subset=[x])
        # Plotly's nbinsx property is an integer; the string "auto" is not valid
        # across supported Plotly versions. Use the analytics plan's validated bin count.
        try:
            nbins = int(chart.get("bins", 12))
        except (TypeError, ValueError):
            nbins = 12
        nbins = min(40, max(5, nbins))
        fig = px.histogram(df, x=x, nbins=nbins, title=title, color_discrete_sequence=[LINE_COLOR])
        fig.update_xaxes(title=x.replace("_", " ").title())
        fig.update_yaxes(title="Frequency")
        return fig, len(df), "Histogram shows the observed frequency distribution."

    def _box(self, df: pd.DataFrame, chart: dict, title: str) -> tuple[go.Figure, int, str]:
        x, y = chart.get("x"), chart.get("y")
        if y not in df.columns:
            # Choose a single numeric attribute only if the result schema is unambiguous.
            numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(numeric) != 1:
                raise VisualizationError("Box plot needs a specified numerical metric.")
            y = numeric[0]
        df[y] = pd.to_numeric(df[y], errors="coerce")
        df = df.dropna(subset=[y])
        if df.empty:
            raise VisualizationError("No numeric observations for box plot.")
        fig = px.box(df, x=x if x in df.columns else None, y=y,
                     points="outliers", color=x if x in df.columns and df[x].nunique() <= 8 else None,
                     color_discrete_sequence=PALETTE, title=title)
        fig.update_yaxes(title=y.replace("_", " ").title())
        if x in df.columns:
            fig.update_xaxes(title=x.replace("_", " ").title())
        if y in {"rank_num", "rank"}:
            fig.update_yaxes(autorange="reversed", title="Institution rank (lower is better)")
        return fig, len(df), "Boxes show median and interquartile range; lower numerical ranks are better." if y in {"rank_num", "rank"} else "Boxes show median and interquartile range; points outside whiskers are potential outliers."

    def _heatmap(self, df: pd.DataFrame, title: str) -> tuple[go.Figure, int, str]:
        """Render NIRF parameter, year-by-group, membership or change matrices."""
        shown = df.copy()
        label_col = next((c for c in ["display_name", "state", "name", "institution"] if c in shown.columns), None)
        if label_col is None:
            raise VisualizationError("Heatmap requires an institution, state or category row label.")

        year_cols = sorted([c for c in shown.columns if str(c).isdigit() and len(str(c)) == 4])
        delta_cols = [c for c in shown.columns if str(c).startswith("delta_") and
                      pd.to_numeric(shown[c], errors="coerce").notna().any()]
        metric_cols = [m for m in self.metrics if m in shown.columns and
                       pd.to_numeric(shown[m], errors="coerce").notna().any()]

        if year_cols:
            value_cols = year_cols
            shown[value_cols] = shown[value_cols].apply(pd.to_numeric, errors="coerce")
            # Binary year columns are membership matrices; count matrices retain their native scale.
            observed = shown[value_cols].stack().dropna()
            is_binary = not observed.empty and set(observed.unique()).issubset({0, 1})
            colorscale = "Blues"
            zmin, zmax = 0, (1 if is_binary else None)
            color_title = "Top-10 membership" if is_binary else "Institutions"
            x_labels = value_cols
            note = ("Columns represent observed ranking years; 1 means top-10 membership and 0 means no top-10 membership."
                    if is_binary else "Rows are states and columns are observed years; cell values count institutions.")
        elif delta_cols:
            value_cols = delta_cols
            shown[value_cols] = shown[value_cols].apply(pd.to_numeric, errors="coerce")
            colorscale = "RdBu"
            finite = shown[value_cols].to_numpy(dtype=float)
            finite = finite[np.isfinite(finite)]
            bound = float(np.max(np.abs(finite))) if finite.size else 1.0
            zmin, zmax = -bound, bound
            color_title = "Parameter score change"
            x_labels = [c.replace("delta_", "") for c in value_cols]
            note = "Cells show endpoint change in each available parameter; missing parameter changes remain blank."
        elif metric_cols:
            value_cols = metric_cols
            shown[value_cols] = shown[value_cols].apply(pd.to_numeric, errors="coerce")
            colorscale = "Blues"
            zmin, zmax = 0, 100
            color_title = "Parameter score"
            x_labels = value_cols
            note = "Rows compare institutions across the available NIRF parameter scores; missing values remain blank."
        else:
            raise VisualizationError("No supported matrix columns are available for this heatmap.")

        shown = shown.head(DEFAULT_LIMITS["heatmap_rows"])
        z = shown[value_cols].to_numpy(dtype=float)
        row_labels = shown[label_col].astype(str).tolist()
        fig = go.Figure(go.Heatmap(
            z=z, x=x_labels, y=row_labels, colorscale=colorscale,
            zmin=zmin, zmax=zmax, colorbar={"title": color_title},
            hovertemplate="%{y}<br>%{x}: %{z:.2f}<extra></extra>", hoverongaps=False,
        ))
        fig.update_xaxes(title="Year" if year_cols else ("NIRF parameter" if not delta_cols else "Parameter"), side="bottom")
        fig.update_yaxes(title=label_col.replace("_", " ").title(), autorange="reversed")
        return fig, len(shown), f"Showing {len(shown)} of {len(df)} rows. {note}"

    @staticmethod
    def _rank_band_key(value: Any) -> tuple[int, str]:
        text = str(value)
        digits = ''.join(ch for ch in text if ch.isdigit())
        return (int(digits) if digits else 999999, text)

    def _transition_heatmap(self, df: pd.DataFrame, title: str) -> tuple[go.Figure, int, str]:
        if not {"start_band", "end_band"}.issubset(df.columns):
            raise VisualizationError("Transition heatmap requires start_band and end_band columns.")
        clean = df.dropna(subset=["start_band", "end_band"])
        if clean.empty:
            raise VisualizationError("No valid rank-band transitions are available.")
        if "transition_pct" in clean.columns:
            pivot = clean.pivot_table(index="start_band", columns="end_band", values="transition_pct", aggfunc="first", fill_value=0)
            color_title = "Within-band share (%)"
            hover = "From %{y} to %{x}<br>Share of start band: %{z:.1f}%<extra></extra>"
            zmax = 100
        elif "institution_count" in clean.columns:
            pivot = clean.pivot_table(index="start_band", columns="end_band", values="institution_count", aggfunc="sum", fill_value=0)
            color_title = "Institutions"
            hover = "From %{y} to %{x}<br>Institutions: %{z}<extra></extra>"
            zmax = None
        else:
            pivot = pd.crosstab(clean["start_band"].astype(str), clean["end_band"].astype(str))
            color_title = "Institutions"
            hover = "From %{y} to %{x}<br>Institutions: %{z}<extra></extra>"
            zmax = None
        rows = sorted(pivot.index.astype(str), key=self._rank_band_key)
        cols = sorted(pivot.columns.astype(str), key=self._rank_band_key)
        pivot.index, pivot.columns = pivot.index.astype(str), pivot.columns.astype(str)
        pivot = pivot.reindex(index=rows, columns=cols, fill_value=0)
        fig = go.Figure(go.Heatmap(z=pivot.to_numpy(), x=cols, y=rows,
                                   colorscale="Blues", zmin=0, zmax=zmax,
                                   colorbar={"title": color_title}, hovertemplate=hover))
        fig.update_xaxes(title="End-year rank band")
        fig.update_yaxes(title="Start-year rank band", autorange="reversed")
        return fig, len(clean), "Cells count institutions moving from each start-year band to each end-year band."

    def _sankey(self, df: pd.DataFrame, title: str) -> tuple[go.Figure, int, str]:
        if not {"start_band", "end_band"}.issubset(df.columns):
            raise VisualizationError("Sankey requires start_band and end_band columns.")
        flows = df.dropna(subset=["start_band", "end_band"]).groupby(["start_band", "end_band"]).size().reset_index(name="count")
        labels = sorted(set(flows["start_band"].astype(str)) | set(flows["end_band"].astype(str)), key=self._rank_band_key)
        label_map = {v: i for i, v in enumerate(labels)}
        fig = go.Figure(go.Sankey(
            arrangement="snap",
            node={"label": labels, "pad": 16, "thickness": 18, "color": LINE_COLOR},
            link={"source": [label_map[str(v)] for v in flows["start_band"]],
                  "target": [label_map[str(v)] for v in flows["end_band"]],
                  "value": flows["count"].tolist()},
        ))
        return fig, int(flows["count"].sum()), "Flow width represents the number of institutions in each transition."

    def _stacked_bar(self, df: pd.DataFrame, chart: dict, title: str) -> tuple[go.Figure, int, str]:
        x, y = chart.get("x"), chart.get("y")
        group = next((c for c in ["movement_status", "status", "series"] if c in df.columns), None)
        if x not in df.columns or y not in df.columns or group is None:
            raise VisualizationError("Stacked bar requires x, y and a category/series column.")
        fig = px.bar(df, x=x, y=y, color=group, barmode="stack", color_discrete_sequence=PALETTE,
                     hover_data=df.columns.tolist(), title=title)
        return fig, len(df), "Bars are stacked by the returned movement/category groups."

    def _table(self, df: pd.DataFrame, title: str) -> go.Figure:
        # Keep table as a readable, explicit visual when no quantitative chart is justified.
        show = df.head(100).copy()
        for c in show.columns:
            if pd.api.types.is_numeric_dtype(show[c]):
                show[c] = show[c].round(2)
        fig = go.Figure(go.Table(
            header={"values": list(show.columns), "fill_color": PANEL_BG,
                     "font": {"color": TEXT_COLOR, "size": 11}, "align": "left"},
            cells={"values": [show[c].tolist() for c in show.columns],
                   "fill_color": PANEL_BG,
                   "font": {"color": TEXT_COLOR, "size": 10}, "align": "left", "height": 27},
        ))
        fig.update_layout(title=title)
        return fig

    def create_figure(self, result: dict[str, Any], chart_type: str | None = None) -> dict[str, Any]:
        if result.get("status") != "success":
            raise VisualizationError("Cannot visualize an unsuccessful analytical result.")
        df, chart = self._get_data(result)
        requested = chart_type or chart.get("type") or result.get("visualization") or "table"
        x, y = chart.get("x"), chart.get("y")
        normalized = normalize_chart_type(requested, result.get("intent"), x, y)
        validation = validate_chart_input(result, normalized)
        if not validation["valid"]:
            raise VisualizationError("; ".join(validation["errors"]))
        title = result.get("title") or "NIRF Analytical Visualization"
        notes: list[str] = []
        visible_count = len(df)

        # A single-record query is normally an answer card. If the caller explicitly
        # asks for a figure, show a compact table rather than manufacturing a chart.
        if len(df) == 1 and normalized not in {"table"}:
            normalized = "table"
            notes.append("Single-record result: shown as a table instead of a chart.")

        is_ranking = result.get("intent") == "ranking"
        if is_ranking and normalized in {"bar", "dot_plot", "horizontal_bar"}:
            fig, visible_count, note = self._ranking_figure(df.copy(), chart, title)
            notes.append(note)
            normalized = "ranking_leaderboard"
        elif normalized in {"bar", "horizontal_bar"}:
            fig, visible_count, note = self._horizontal_bar(df.copy(), chart, title)
            notes.append(note)
        elif normalized == "diverging_bar":
            fig, visible_count, note = self._diverging_bar(df.copy(), chart, title)
            notes.append(note)
        elif normalized == "line":
            fig, visible_count, note = self._line(df.copy(), chart, title)
            notes.append(note)
        elif normalized in {"scatter", "scatter_quadrant", "scatter_rank"}:
            fig, visible_count, note = self._scatter(
                df.copy(), chart, title,
                quadrant=normalized == "scatter_quadrant",
                invert_rank=normalized == "scatter_rank",
                add_trend=result.get("intent") == "correlation_analysis",
            )
            notes.append(note)
            stats = result.get("statistics") or {}
            coeff = []
            for label, key in (("Pearson", "pearson_correlation"), ("Spearman", "spearman_correlation")):
                val = stats.get(key)
                if isinstance(val, (int, float)) and np.isfinite(val):
                    coeff.append(f"{label}: {val:.3f}")
            if stats.get("sample_size") is not None:
                coeff.append(f"n = {stats['sample_size']}")
            if coeff:
                fig.add_annotation(text="<br>".join(coeff), xref="paper", yref="paper",
                                   x=0.02, y=0.98, showarrow=False, align="left",
                                   bgcolor=PANEL_BG, bordercolor=GRID_COLOR,
                                   font={"color": TEXT_COLOR, "size": 11}, borderwidth=1)
        elif normalized == "histogram":
            fig, visible_count, note = self._histogram(df.copy(), chart, title)
            notes.append(note)
        elif normalized == "box":
            fig, visible_count, note = self._box(df.copy(), chart, title)
            notes.append(note)
        elif normalized == "map":
            # Until an India state GeoJSON is explicitly wired into this renderer,
            # a sorted horizontal bar is more honest than a fake map.
            fig, visible_count, note = self._horizontal_bar(df.copy(), chart, title)
            notes.append("Map view unavailable; shown as a sorted state comparison bar chart.")
            notes.append(note)
            normalized = "horizontal_bar_fallback"
        elif normalized == "heatmap":
            fig, visible_count, note = self._heatmap(df.copy(), title)
            notes.append(note)
        elif normalized == "transition_heatmap":
            fig, visible_count, note = self._transition_heatmap(df.copy(), title)
            notes.append(note)
        elif normalized == "sankey":
            fig, visible_count, note = self._sankey(df.copy(), title)
            notes.append(note)
        elif normalized == "stacked_bar":
            fig, visible_count, note = self._stacked_bar(df.copy(), chart, title)
            notes.append(note)
        elif normalized == "difference_plot":
            fig, visible_count, note = self._diverging_bar(df.copy(), chart, title)
            notes.append(note)
        elif normalized == "dot_plot":
            fig, visible_count, note = self._metric_dot_plot(df.copy(), chart, title)
            notes.append(note)
        elif normalized == "table":
            fig = self._table(df.copy(), title)
        else:
            raise VisualizationError(f"Unsupported chart type: {normalized}")

        # Consistent dark theme. Preserve axis titles created by each builder.
        is_rank_line = normalized == "line" and y in {"rank_num", "rank"}
        invert_y = (
            is_rank_line
            or normalized in {"ranking_leaderboard", "horizontal_bar", "horizontal_bar_fallback", "diverging_bar", "dot_plot", "heatmap", "transition_heatmap"}
            or normalized == "scatter_rank"
        )
        x_title = getattr(getattr(fig.layout, "xaxis", None), "title", None)
        y_title = getattr(getattr(fig.layout, "yaxis", None), "title", None)
        x_title_text = getattr(x_title, "text", "") or ""
        y_title_text = getattr(y_title, "text", "") or ""
        fig = apply_theme(fig, title=title, xaxis_title=x_title_text,
                          yaxis_title=y_title_text, invert_yaxis=invert_y,
                          show_legend=len(fig.data) > 1)
        # apply_theme defaults normal axes to autorange=True; restore the
        # inverse scale for numerical ranks, where smaller values are better.
        if normalized == "box" and y in {"rank_num", "rank"}:
            fig.update_yaxes(autorange="reversed")
        # Compact chart heights are a key part of reducing clutter.
        if normalized in {"heatmap", "transition_heatmap"}:
            height = max(420, min(760, 250 + visible_count * 24))
        elif normalized in {"table"}:
            height = max(350, min(780, 100 + visible_count * 28))
        elif normalized in {"horizontal_bar", "horizontal_bar_fallback", "diverging_bar", "dot_plot", "ranking_leaderboard"}:
            height = max(380, min(760, 210 + visible_count * 30))
        else:
            height = 500
        fig.update_layout(height=height, margin={"l": 220 if normalized in {"horizontal_bar", "horizontal_bar_fallback", "diverging_bar", "dot_plot"} else 70,
                                                 "r": 45, "t": 75, "b": 65})
        return {
            "figure": fig,
            "requested_type": requested,
            "rendered_type": normalized,
            "title": title,
            "record_count": len(df),
            "visible_record_count": visible_count,
            "notes": notes,
            "validation": validation,
        }

    def save_html(self, result: dict[str, Any], filepath: str | Path) -> str:
        output = self.create_figure(result)
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        output["figure"].write_html(str(path), include_plotlyjs=True, full_html=True)
        return str(path)

    def save_png(self, result: dict[str, Any], filepath: str | Path,
                 width: int = 1400, height: int = 850, scale: int = 2) -> str:
        output = self.create_figure(result)
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        output["figure"].write_image(str(path), format="png", width=width, height=height, scale=scale)
        return str(path)