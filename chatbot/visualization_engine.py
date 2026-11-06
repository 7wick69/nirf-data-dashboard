
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path

from charts.theme import (
    apply_theme,
    LINE_COLOR,
    ACCENT_GREEN,
    ACCENT_RED,
    MUTED_TEXT,
    PALETTE,
)


class VisualizationError(Exception):
    pass


class NIRFVisualizationEngine:

    def __init__(self):
        self.metrics = [
            "TLR", "RPC", "GO", "OI", "PR",
            "QNR", "QLR", "SFC"
        ]

    def _get_data(self, result):

        chart = result.get("chart", {})

        records = (
            chart.get("data")
            or result.get("data")
            or []
        )

        df = pd.DataFrame(records)

        if df.empty:
            raise VisualizationError(
                "No analytical records available for visualization."
            )

        return df, chart

    def create_figure(self, result):

        if result.get("status") != "success":
            raise VisualizationError(
                "Cannot visualize an unsuccessful analytical result."
            )

        df, chart = self._get_data(result)

        chart_type = (
            chart.get("type")
            or result.get("visualization")
            or "table"
        ).lower()

        x = chart.get("x")
        y = chart.get("y")

        title = result.get(
            "title",
            "NIRF Analytical Visualization"
        )

        fig = None
        rendered_as = chart_type

        # ---------------------------------
        # 1. BAR CHART
        # ---------------------------------

        if chart_type == "bar":

            if x not in df.columns or y not in df.columns:
                raise VisualizationError(
                    f"Bar chart requires valid X={x} and Y={y}."
                )

            if (
                result.get("intent") == "trend_analysis"
                and {"display_name", "change"}.issubset(df.columns)
            ):
                improvements = (
                    df[df["change"] > 0]
                    .nlargest(10, "change")
                )
                declines = (
                    df[df["change"] < 0]
                    .nsmallest(10, "change")
                )
                df = pd.concat([
                    improvements,
                    declines
                ]).drop_duplicates(
                    subset=["display_name"]
                )
                df = df.sort_values(
                    "change",
                    ascending=True
                )
                df["movement"] = df["change"].apply(
                    lambda value:
                        "Improved" if value > 0
                        else "Declined" if value < 0
                        else "Unchanged"
                )
                fig = px.bar(
                    df,
                    x="change",
                    y="display_name",
                    orientation="h",
                    color="movement",
                    color_discrete_map={
                        "Improved": ACCENT_GREEN,
                        "Declined": ACCENT_RED,
                        "Unchanged": MUTED_TEXT,
                    },
                    text="change",
                    hover_data=df.columns.tolist(),
                    title=title
                )
                fig.update_xaxes(
                    title="Rank improvement (positive is better)",
                    zeroline=True
                )
                fig.update_yaxes(title="Institution")
                fig.update_traces(textposition="outside")

            elif y == "rank_num":

                df = df.sort_values(
                    "rank_num",
                    ascending=True
                )
                institution_order = df["display_name"].tolist()
                fig = px.scatter(
                    df,
                    x="rank_num",
                    y="display_name",
                    text="rank_num",
                    hover_data=[
                        column for column in [
                            "state", "city", "score", "rank_band"
                        ] if column in df.columns
                    ],
                    category_orders={
                        "display_name": list(reversed(institution_order))
                    },
                    title=title
                )
                fig.update_traces(
                    mode="markers+text",
                    marker=dict(
                        size=13,
                        color=LINE_COLOR
                    ),
                    text=df["rank_num"].astype(int).astype(str),
                    textposition="middle right",
                )
                fig.update_xaxes(title="Institution Rank (lower is better)")
                fig.update_yaxes(title="Institution")

            else:

                fig = px.bar(
                    df,
                    x=x,
                    y=y,
                    color=(
                        "state"
                        if "state" in df.columns
                        and x != "state"
                        else None
                    ),
                    color_discrete_sequence=PALETTE,
                    hover_data=df.columns.tolist(),
                    title=title
                )

                fig.update_xaxes(title=x)
                fig.update_yaxes(title=y)

        # ---------------------------------
        # 2. LINE CHART
        # ---------------------------------

        elif chart_type == "line":

            if x not in df.columns or y not in df.columns:
                raise VisualizationError(
                    "Line chart requires valid X and Y columns."
                )

            color = (
                "display_name"
                if "display_name" in df.columns
                and df["display_name"].nunique() <= 15
                else None
            )

            fig = px.line(
                df.sort_values(x),
                x=x,
                y=y,
                color=color,
                markers=True,
                hover_data=df.columns.tolist(),
                title=title
            )

            fig.update_xaxes(title=x)
            fig.update_yaxes(title=y)

            if y == "rank_num":
                fig.update_yaxes(autorange="reversed")

        # ---------------------------------
        # 3. SCATTER PLOT
        # ---------------------------------

        elif chart_type == "scatter":

            if x not in df.columns or y not in df.columns:
                raise VisualizationError(
                    "Scatter plot requires valid X and Y."
                )

            df[x] = pd.to_numeric(df[x], errors="coerce")
            df[y] = pd.to_numeric(df[y], errors="coerce")
            df = df.dropna(subset=[x, y])

            if len(df) < 2:
                raise VisualizationError(
                    "Insufficient data for scatter plot."
                )

            hover_fields = [
                field for field in ["state", "rank_num"]
                if field in df.columns
            ]

            fig = px.scatter(
                df,
                x=x,
                y=y,
                hover_name=(
                    "display_name"
                    if "display_name" in df.columns
                    else None
                ),
                hover_data=hover_fields,
                trendline="ols",
                title=title
            )

            fig.update_xaxes(title=x)
            fig.update_yaxes(title=y)

            statistics = result.get("statistics", {})
            coefficient_text = "<br>".join(
                f"{label}: {statistics[key]:.3f}"
                for label, key in [
                    ("Pearson", "pearson_correlation"),
                    ("Spearman", "spearman_correlation")
                ]
                if isinstance(statistics.get(key), (int, float))
            )
            if coefficient_text:
                fig.add_annotation(
                    text=coefficient_text,
                    xref="paper",
                    yref="paper",
                    x=0.02,
                    y=0.98,
                    showarrow=False,
                    align="left",
                    bgcolor="white",
                    bordercolor="#cccccc",
                    borderwidth=1
                )

        # ---------------------------------
        # 4. HISTOGRAM
        # ---------------------------------

        elif chart_type == "histogram":

            if {
                "bin_start",
                "bin_end",
                "count"
            }.issubset(df.columns):

                df["bin_midpoint"] = (
                    df["bin_start"] + df["bin_end"]
                ) / 2

                fig = px.bar(
                    df,
                    x="bin_midpoint",
                    y="count",
                    hover_data=[
                        "bin_start",
                        "bin_end",
                        "count"
                    ],
                    title=title
                )

                fig.update_xaxes(title="Score")
                fig.update_yaxes(title="Number of institutions")

            elif x in df.columns:

                fig = px.histogram(
                    df,
                    x=x,
                    nbins=10,
                    title=title
                )

                fig.update_xaxes(title=x)
                fig.update_yaxes(title="Frequency")

            else:

                raise VisualizationError(
                    "Histogram data is unavailable."
                )

        # ---------------------------------
        # 5. BOX PLOT
        # ---------------------------------

        elif chart_type == "box":

            if y not in df.columns:
                raise VisualizationError(
                    "Box plot metric is unavailable."
                )

            fig = px.box(
                df,
                y=y,
                x=x if x in df.columns else None,
                points="outliers",
                title=title
            )

        # ---------------------------------
        # 6. GEOGRAPHICAL ANALYSIS
        # ---------------------------------

        elif chart_type == "map":

            if not {
                "state",
                "institution_count"
            }.issubset(df.columns):

                raise VisualizationError(
                    "State-wise map data is unavailable."
                )

            df = df.sort_values(
                "institution_count",
                ascending=True
            )

            # Bar chart fallback until India GeoJSON
            # is configured for state-level mapping.

            fig = px.bar(
                df,
                x="institution_count",
                y="state",
                orientation="h",
                text="institution_count",
                title=title,
                hover_data=df.columns.tolist()
            )

            fig.update_xaxes(
                title="Number of institutions"
            )

            fig.update_yaxes(title="State")

            rendered_as = "bar_fallback"

        # ---------------------------------
        # 7. RADAR CHART
        # ---------------------------------

        elif chart_type == "radar":

            available = [
                metric for metric in self.metrics
                if metric in df.columns
            ]

            if len(available) < 3:
                raise VisualizationError(
                    "Insufficient parameters for radar chart."
                )

            if "display_name" not in df.columns:
                raise VisualizationError(
                    "Institution names are missing."
                )

            fig = go.Figure()

            for _, row in df.iterrows():

                values = [
                    row[m] for m in available
                ]

                if pd.isna(values).any():
                    continue

                fig.add_trace(
                    go.Scatterpolar(
                        r=values + [values[0]],
                        theta=available + [available[0]],
                        fill="toself",
                        name=str(row["display_name"])
                    )
                )

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                title=title
            )

        # ---------------------------------
        # 8. SANKEY DIAGRAM
        # ---------------------------------

        elif chart_type == "sankey":

            if not {
                "start_band",
                "end_band"
            }.issubset(df.columns):

                raise VisualizationError(
                    "Ranking transition bands are unavailable."
                )

            flows = (
                df.groupby(
                    ["start_band", "end_band"]
                )
                .size()
                .reset_index(name="count")
            )

            labels = sorted(
                set(flows["start_band"])
                | set(flows["end_band"])
            )

            label_map = {
                label: i
                for i, label in enumerate(labels)
            }

            fig = go.Figure(
                go.Sankey(
                    node=dict(
                        label=labels,
                        pad=15,
                        thickness=20
                    ),
                    link=dict(
                        source=[
                            label_map[v]
                            for v in flows["start_band"]
                        ],
                        target=[
                            label_map[v]
                            for v in flows["end_band"]
                        ],
                        value=flows["count"].tolist()
                    )
                )
            )

            fig.update_layout(title=title)

        # ---------------------------------
        # 9. TABLE
        # ---------------------------------

        elif chart_type == "table":

            fig = go.Figure(
                data=[
                    go.Table(
                        header=dict(
                            values=list(df.columns)
                        ),
                        cells=dict(
                            values=[
                                df[column].tolist()
                                for column in df.columns
                            ]
                        )
                    )
                ]
            )

            fig.update_layout(title=title)

        else:

            raise VisualizationError(
                f"Unsupported chart type: {chart_type}"
            )

        fig = apply_theme(
            fig,
            title=title,
            xaxis_title=fig.layout.xaxis.title.text or "",
            yaxis_title=fig.layout.yaxis.title.text or "",
            invert_yaxis=(
                chart_type == "line"
                and y in ["rank_num", "rank"]
            ),
            show_legend=len(fig.data) > 1
        )
        fig.update_layout(
            height=max(550, min(1100, len(df) * 38 + 180))
        )

        return {
            "figure": fig,
            "requested_type": chart_type,
            "rendered_type": rendered_as,
            "title": title,
            "record_count": len(df)
        }

    # ---------------------------------
    # SAVE CHART AS INTERACTIVE HTML
    # ---------------------------------

    def save_html(self, result, filepath):

        output = self.create_figure(result)

        filepath = Path(filepath)

        filepath.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        output["figure"].write_html(
            str(filepath),
            include_plotlyjs=True,
            full_html=True
        )

        return str(filepath)

    def save_png(
        self,
        result,
        filepath,
        width=1400,
        height=850,
        scale=2
    ):
        output = self.create_figure(result)
        filepath = Path(filepath)
        filepath.parent.mkdir(
            parents=True,
            exist_ok=True
        )
        output["figure"].write_image(
            str(filepath),
            format="png",
            width=width,
            height=height,
            scale=scale
        )
        return str(filepath)
