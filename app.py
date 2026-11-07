"""
app.py

NIRF Interactive Analytics Dashboard
Minimal, monocolor dark theme matching the design wireframe.
"""

import streamlit as st
import pandas as pd
from core.data import (
    load_nirf_data,
    get_categories,
    get_states,
    get_year_range,
    get_institutions_list,
)
from charts.trends import (
    plot_institution_rank_trend,
    plot_greatest_improvers,
    plot_largest_decliners,
    plot_average_score_trend,
    plot_consistently_top10,
)
from charts.geo import (
    plot_top_states_map,
    plot_top_states_top100,
    plot_geo_distribution_over_time,
    plot_state_representation_map,
    plot_state_representation_improvement,
    plot_state_rank_comparison_map,
    plot_state_rank_comparison,
    plot_new_entrant_map,
    plot_new_entrant_states,
)
from charts.explore import render_explore_tab
from charts.univariate import (
    plot_score_distribution,
    plot_rank_distribution,
    plot_summary_statistics,
    plot_rank_band_breakdown,
    plot_score_spread_by_category,
    plot_parameter_distribution,
)

# ---------------------------------------------------------
# Page Configuration & Custom Monocolor Dark Theme CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="NIRF Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
/* Main Dark Theme & Fonts */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Backgrounds */
.stApp {
    background-color: #0f1013;
    color: #e6edf3;
}

section[data-testid="stSidebar"] {
    background-color: #14161a;
    border-right: 1px solid #23272f;
}

/* Sidebar Headings & Labels */
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #f0f3f6;
    font-weight: 600;
    letter-spacing: -0.02em;
}

/* Input widgets styling */
div[data-baseweb="select"] > div {
    background-color: #1a1d24 !important;
    border: 1px solid #2d323b !important;
    border-radius: 6px !important;
    color: #e6edf3 !important;
}

div[data-baseweb="select"] svg {
    fill: #8b949e !important;
}

/* Scoped Disabled State Filter styling (Geospatial Section) */
div[data-baseweb="select"][aria-disabled="true"],
div[data-baseweb="select"][aria-disabled="true"] > div,
div[data-testid="stSelectbox"]:has(div[aria-disabled="true"]) > div {
    background-color: #13151a !important;
    border: 1px solid #21252d !important;
    opacity: 0.52 !important;
    cursor: not-allowed !important;
    pointer-events: none !important;
}

div[data-baseweb="select"][aria-disabled="true"] *,
div[data-baseweb="select"][aria-disabled="true"] span,
div[data-baseweb="select"][aria-disabled="true"] div {
    color: #64748b !important;
    cursor: not-allowed !important;
}

div[data-baseweb="select"][aria-disabled="true"] svg {
    fill: #475569 !important;
    cursor: not-allowed !important;
}

div[data-testid="stSelectbox"]:has(div[aria-disabled="true"]) label p {
    color: #6e7681 !important;
    cursor: not-allowed !important;
}

/* Takeaway Card */
.takeaway-card {
    background-color: #16181e;
    border: 1px solid #262a33;
    border-radius: 8px;
    padding: 12px 18px;
    margin-top: 14px;
    margin-bottom: 20px;
    color: #c9d1d9;
    font-size: 13.5px;
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
}

.takeaway-highlight {
    color: #58a6ff;
    font-weight: 600;
}

/* Tab Bar Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent;
    border-bottom: 1px solid #23272f;
    padding-bottom: 4px;
}

.stTabs [data-baseweb="tab"] {
    height: 38px;
    background-color: transparent;
    border: none;
    color: #8b949e;
    font-size: 13.5px;
    font-weight: 500;
    padding: 0 14px;
    border-radius: 6px;
    transition: all 0.15s ease-in-out;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #f0f3f6;
    background-color: #1a1d24;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #58a6ff;
    background-color: #1a1d24;
    border-bottom: 2px solid #388bfd;
}

/* Metric / KPI Containers */
div[data-testid="stMetric"] {
    background-color: #16181e;
    border: 1px solid #262a33;
    padding: 12px 16px;
    border-radius: 8px;
}

/* Chat Mockup Container */
.chat-wireframe {
    background-color: #14161a;
    border: 1px solid #262a33;
    border-radius: 8px;
    padding: 16px;
    margin-top: 15px;
}
.chat-bubble-user {
    background-color: #1f3554;
    color: #c8e1ff;
    padding: 10px 14px;
    border-radius: 12px 12px 2px 12px;
    max-width: 65%;
    margin-left: auto;
    margin-bottom: 12px;
    font-size: 13.5px;
}
.chat-bubble-bot {
    background-color: #1a1d24;
    color: #e6edf3;
    padding: 10px 14px;
    border-radius: 12px 12px 12px 2px;
    max-width: 75%;
    margin-bottom: 12px;
    font-size: 13.5px;
    border: 1px solid #262a33;
}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Data Loader with Caching
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_dataset(custom_file=None):
    if custom_file is not None:
        if custom_file.name.endswith(".parquet"):
            df = pd.read_parquet(custom_file)
        else:
            df = pd.read_csv(custom_file)
        from core.metrics import compute_derived_metrics
        return compute_derived_metrics(df)
    return load_nirf_data()


# ---------------------------------------------------------
# Sidebar: Global Filters
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### NIRF explorer")

    # Optional Custom Dataset Uploader
    with st.expander("📁 Data Source", expanded=False):
        uploaded_file = st.file_uploader(
            "Upload NIRF Parquet / CSV",
            type=["parquet", "csv"],
            help="Upload your own NIRF dataset or use the clean default.",
        )
        if uploaded_file is not None:
            st.success(f"Loaded: {uploaded_file.name}")

    df_full = get_dataset(uploaded_file)

    # Years Filter
    min_yr, max_yr = get_year_range(df_full)
    year_range = st.slider(
        "Years",
        min_value=min_yr,
        max_value=max_yr,
        value=(min_yr, max_yr),
        step=1,
        help="Select year span",
    )

    # Category Filter
    categories = get_categories(df_full)
    selected_category = st.selectbox(
        "Category",
        options=categories,
        index=0,
        help="Select NIRF category",
    )

    # Detect if currently selected topic belongs to Geospatial & Regional Analysis
    active_topic = str(st.session_state.get("active_tab", "Trends"))
    is_geo_active = (
        "geo" in active_topic.lower()
        or active_topic in ["Geo", "Geospatial", "Geospatial & Regional", "Geospatial and Regional Analysis"]
    )

    # State Filter: dynamically disabled & greyed out ONLY for Geospatial topics/graphs
    all_states = ["All states"] + get_states(df_full)
    selected_state = st.selectbox(
        "States",
        options=all_states,
        index=0 if is_geo_active else 0,
        disabled=is_geo_active,
        help=(
            "State filter is disabled specifically for Geospatial and Regional Analysis because all states are analyzed simultaneously across India."
            if is_geo_active
            else "Filter by state or all states"
        ),
        key="global_state_selector",
    )


# ---------------------------------------------------------
# Filter Dataset Globally
# ---------------------------------------------------------
df_filtered = df_full[
    (df_full["year"] >= year_range[0])
    & (df_full["year"] <= year_range[1])
    & (df_full["category"] == selected_category)
]
if selected_state != "All states":
    df_filtered = df_filtered[df_filtered["state"] == selected_state]


# ---------------------------------------------------------
# Main App Header & Tabs Navigation
# ---------------------------------------------------------
tabs = st.tabs([
    "Trends",
    "Geospatial and Regional Analysis",
    "Univariate",
    "Bivariate",
    "Multivariate",
    "Distribution",
    "Transitions",
    "Outliers",
    "Explore",
    "Ask",
], default="Trends", key="active_tab", on_change="rerun")

with tabs[0]:
    col_q, col_param = st.columns([1.6, 1.1])

    with col_q:
        question = st.selectbox(
            "Question",
            options=[
                "How has the overall NIRF ranking of a selected institution changed from 2017 to 2025?",
                "Which institutions have shown the greatest improvement in rank over the nine years?",
                "Which institutions have experienced the largest decline in ranking?",
                "How has the average score of institutions changed over time?",
                "Which institutions have remained consistently in the top 10 across multiple years?",
            ],
            index=0,
            key="trends_question_select",
        )

    # 1. Institution Rank Over Time
    if question == "How has the overall NIRF ranking of a selected institution changed from 2017 to 2025?":
        inst_options = get_institutions_list(df_full, selected_category, selected_state)

        if not inst_options:
            st.info(f"No institutions found in {selected_state} for category '{selected_category}'.")
        else:
            default_inst = "Indian Institute of Technology Madras"
            default_idx = inst_options.index(default_inst) if default_inst in inst_options else 0

            with col_param:
                selected_inst = st.selectbox(
                    "Institution",
                    options=inst_options,
                    index=default_idx,
                    key="trends_institution_select",
                )

            fig, takeaway = plot_institution_rank_trend(
                df=df_full,
                institution_name=selected_inst,
                category=selected_category,
                year_range=year_range,
                state=selected_state,
            )

            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            st.markdown(
                f"""<div class="takeaway-card">
                <span class="takeaway-highlight">Takeaway line:</span> {takeaway}
                </div>""",
                unsafe_allow_html=True,
            )

    # 2. Greatest Improvers
    elif question == "Which institutions have shown the greatest improvement in rank over the nine years?":
        with col_param:
            top_n = st.slider("Top N improvers", min_value=1, max_value=15, value=5, key="top_improvers_slider")

        fig, takeaway = plot_greatest_improvers(
            df=df_full,
            category=selected_category,
            start_year=year_range[0],
            end_year=year_range[1],
            top_n=top_n,
            state=selected_state,
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            f"""<div class="takeaway-card">
            <span class="takeaway-highlight">Takeaway line:</span> {takeaway}
            </div>""",
            unsafe_allow_html=True,
        )

    # 3. Largest Decliners
    elif question == "Which institutions have experienced the largest decline in ranking?":
        with col_param:
            top_n = st.slider("Top N decliners", min_value=1, max_value=15, value=5, key="top_decliners_slider")

        fig, takeaway = plot_largest_decliners(
            df=df_full,
            category=selected_category,
            start_year=year_range[0],
            end_year=year_range[1],
            top_n=top_n,
            state=selected_state,
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            f"""<div class="takeaway-card">
            <span class="takeaway-highlight">Takeaway line:</span> {takeaway}
            </div>""",
            unsafe_allow_html=True,
        )

    # 4. Average Score Over Time
    elif question == "How has the average score of institutions changed over time?":
        fig, takeaway = plot_average_score_trend(
            df=df_full,
            category=selected_category,
            year_range=year_range,
            state=selected_state,
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            f"""<div class="takeaway-card">
            <span class="takeaway-highlight">Takeaway line:</span> {takeaway}
            </div>""",
            unsafe_allow_html=True,
        )

    # 5. Consistently Top 10 (or Top in State)
    elif question == "Which institutions have remained consistently in the top 10 across multiple years?":
        if selected_state != "All states":
            st_inst_count = df_full[
                (df_full["category"] == selected_category) & (df_full["state"] == selected_state)
            ]["name"].nunique()

            with col_param:
                if st_inst_count > 1:
                    top_n = st.slider(
                        f"Top institutions in {selected_state}",
                        min_value=1,
                        max_value=min(10, st_inst_count),
                        value=min(10, st_inst_count),
                        key="state_top_n_slider",
                    )
                else:
                    top_n = max(1, st_inst_count)
                    st.caption(f"Showing all {st_inst_count} ranked institutions in {selected_state}")

            fig, takeaway = plot_consistently_top10(
                df=df_full,
                category=selected_category,
                state=selected_state,
                top_n=top_n,
            )
        else:
            with col_param:
                min_years = st.slider(
                    "Minimum years in Top 10",
                    min_value=2,
                    max_value=max_yr - min_yr + 1,
                    value=min(5, max_yr - min_yr + 1),
                    key="consistently_top10_slider",
                )

            fig, takeaway = plot_consistently_top10(
                df=df_full,
                category=selected_category,
                min_years=min_years,
                state="All states",
            )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            f"""<div class="takeaway-card">
            <span class="takeaway-highlight">Takeaway line:</span> {takeaway}
            </div>""",
            unsafe_allow_html=True,
        )

    # Open in Explore Shortcut Button
    c_btn1, c_btn2 = st.columns([1.4, 3])
    with c_btn1:
        if st.button("🚀 Open this Trends query in Explore", key="open_trends_in_explore"):
            inst_param = selected_inst if "selected_inst" in locals() else None
            steps_list = [{"op": "filter", "where": [{"field": "category", "op": "eq", "value": selected_category}]}]
            if selected_state != "All states":
                steps_list[0]["where"].append({"field": "state", "op": "eq", "value": selected_state})
            if inst_param:
                steps_list.append({"op": "filter", "where": [{"field": "name", "op": "eq", "value": inst_param}]})
            steps_list.append({"op": "top", "n": 25})
            st.session_state["explore_spec"] = {
                "title": f"Explore: {question}",
                "takeaway": "Loaded from Trends tab",
                "steps": steps_list,
                "chart": {"type": "line", "x": "year", "y": "rank", "invert_yaxis": True},
            }
            st.toast("✅ Query loaded into Explore! Click on the 'Explore' tab to modify.", icon="🚀")

    # Details on Demand Expander
    with st.expander("🔍 View underlying data table", expanded=False):
        table_df = df_filtered.copy()
        if question == "How has the overall NIRF ranking of a selected institution changed from 2017 to 2025?" and 'selected_inst' in locals():
            table_df = table_df[table_df["name"] == selected_inst]
        
        display_cols = [c for c in ["year", "name", "state", "rank", "score", "rank_change", "TLR", "RPC", "GO", "OI", "PR"] if c in table_df.columns]
        result_sub_df = table_df[display_cols].sort_values(by=["year", "rank"] if "rank" in table_df.columns else "year")
        st.dataframe(result_sub_df, use_container_width=True)
        csv_bytes = result_sub_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_bytes, file_name="trends_data.csv", mime="text/csv")


# ---------------------------------------------------------
# TAB 2: Geospatial & Regional Analysis
# ---------------------------------------------------------
with tabs[1]:
    col_geo_q, col_geo_param = st.columns([1.5, 1.2])

    with col_geo_q:
        geo_question = st.selectbox(
            "Question",
            options=[
                "Which states have the highest number of institutions in the Top 100?",
                "How is the geographical distribution of top-ranked institutions changing from 2017 to 2025?",
                "Which states have improved their representation in the Top 100 over time?",
                "How do the rankings of institutions in Karnataka compare with those in Tamil Nadu?",
                "Which states have produced the largest number of newly appearing institutions in the rankings?",
                "All Geospatial & Regional Visualizations (Overview)",
            ],
            index=0,
            key="geo_question_selector",
        )

    # 1. Highest number of institutions in Top 100
    if geo_question == "Which states have the highest number of institutions in the Top 100?":
        with col_geo_param:
            c1, c2, c3 = st.columns([1.0, 1.0, 1.0])
            with c1:
                year_opts = [None] + list(range(year_range[1], year_range[0] - 1, -1))
                q1_year = st.selectbox(
                    "Year",
                    options=year_opts,
                    format_func=lambda y: "All Years in Range" if y is None else str(y),
                    index=1 if (year_range[1] - year_range[0] >= 0) else 0,
                    key="q1_year_select",
                )
            with c2:
                top_n_q1 = st.slider("Top N states", min_value=5, max_value=25, value=15, key="q1_top_n_slider")
            with c3:
                q1_palette = st.selectbox(
                    "Palette",
                    options=["Viridis", "Plasma", "Turbo", "YlGnBu", "Tealgrn", "Blues"],
                    index=0,
                    key="q1_palette_select",
                )

        fig_map = plot_top_states_map(
            df=df_full,
            category=selected_category,
            year=q1_year,
            color_palette=q1_palette,
        )
        fig_chart, takeaway = plot_top_states_top100(
            df=df_full,
            category=selected_category,
            year=q1_year,
            top_n=top_n_q1,
        )

        col_map, col_chart = st.columns([1.6, 1.0])
        with col_map:
            st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})
        with col_chart:
            st.plotly_chart(fig_chart, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            f"""<div class="takeaway-card">
            <span class="takeaway-highlight">Takeaway line:</span> {takeaway}
            </div>""",
            unsafe_allow_html=True,
        )

        with st.expander("🔍 View underlying data table", expanded=False):
            sub_q1 = df_full[(df_full["category"] == selected_category) & (df_full["rank"] <= 100)]
            if q1_year is not None:
                sub_q1 = sub_q1[sub_q1["year"] == q1_year]
            table_q1 = sub_q1["state"].value_counts().reset_index()
            table_q1.columns = ["State", "Institutions in Top 100"]
            table_q1["Share (%)"] = (table_q1["Institutions in Top 100"] / len(sub_q1) * 100).round(1)
            st.dataframe(table_q1, use_container_width=True)

    # 2. Geographical distribution changing 2017-2025
    elif geo_question == "How is the geographical distribution of top-ranked institutions changing from 2017 to 2025?":
        with col_geo_param:
            c1, c2, c3 = st.columns([1.0, 1.0, 1.0])
            with c1:
                q2_map_year = st.slider(
                    "Year",
                    min_value=min_yr,
                    max_value=max_yr,
                    value=max_yr,
                    step=1,
                    key="q2_map_year_slider",
                )
            with c2:
                q2_top_n = st.slider("Top N states", min_value=5, max_value=12, value=8, key="q2_top_n_slider")
            with c3:
                q2_palette = st.selectbox(
                    "Palette",
                    options=["Viridis", "Plasma", "Turbo", "YlGnBu", "Tealgrn", "Blues"],
                    index=0,
                    key="q2_palette_select",
                )

        fig_map, takeaway = plot_geo_distribution_over_time(
            df=df_full,
            category=selected_category,
            selected_year=q2_map_year,
            view_mode="India Choropleth Map",
            color_palette=q2_palette,
        )
        fig_chart, _ = plot_geo_distribution_over_time(
            df=df_full,
            category=selected_category,
            selected_year=q2_map_year,
            view_mode="State-wise Stacked Bar Chart",
            top_n_states=q2_top_n,
        )

        col_map, col_chart = st.columns([1.6, 1.0])
        with col_map:
            st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})
        with col_chart:
            st.plotly_chart(fig_chart, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            f"""<div class="takeaway-card">
            <span class="takeaway-highlight">Takeaway line:</span> {takeaway}
            </div>""",
            unsafe_allow_html=True,
        )

        with st.expander("🔍 View underlying data table", expanded=False):
            sub_q2 = df_full[(df_full["category"] == selected_category) & (df_full["rank"] <= 100)]
            piv_q2 = sub_q2.pivot_table(index="state", columns="year", values="name", aggfunc="count", fill_value=0)
            tot_col = f"Total ({min_yr}-{max_yr})"
            piv_q2[tot_col] = piv_q2.sum(axis=1)
            st.dataframe(piv_q2.sort_values(by=tot_col, ascending=False), use_container_width=True)

    # 3. Improved representation in Top 100
    elif geo_question == "Which states have improved their representation in the Top 100 over time?":
        with col_geo_param:
            c1, c2 = st.columns(2)
            with c1:
                q3_style = st.selectbox(
                    "Chart Style",
                    options=["Bar (Delta)", "Slope Chart"],
                    index=0,
                    key="q3_style_select",
                )
            with c2:
                q3_years = st.slider(
                    "Compare Years",
                    min_value=min_yr,
                    max_value=max_yr,
                    value=(min_yr, max_yr),
                    step=1,
                    key="q3_years_select",
                )

        fig_map = plot_state_representation_map(
            df=df_full,
            category=selected_category,
            start_year=q3_years[0],
            end_year=q3_years[1],
        )
        fig_chart, takeaway = plot_state_representation_improvement(
            df=df_full,
            category=selected_category,
            start_year=q3_years[0],
            end_year=q3_years[1],
            chart_type=q3_style,
        )

        if q3_style == "Slope Chart":
            # Slope chart needs wide horizontal space for state labels on both sides.
            # Use stacked layout: choropleth on top, slope chart full-width below.
            col_map_only, col_map_pad = st.columns([1.6, 1.0])
            with col_map_only:
                st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})
            st.plotly_chart(fig_chart, use_container_width=True, config={"displayModeBar": False})
        else:
            # Bar (Delta): compact, works fine side-by-side
            col_map, col_chart = st.columns([1.6, 1.0])
            with col_map:
                st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})
            with col_chart:
                st.plotly_chart(fig_chart, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            f"""<div class="takeaway-card">
            <span class="takeaway-highlight">Takeaway line:</span> {takeaway}
            </div>""",
            unsafe_allow_html=True,
        )

        with st.expander("🔍 View underlying data table", expanded=False):
            sub_q3 = df_full[(df_full["category"] == selected_category) & (df_full["rank"] <= 100)]
            c_s = sub_q3[sub_q3["year"] == q3_years[0]]["state"].value_counts()
            c_e = sub_q3[sub_q3["year"] == q3_years[1]]["state"].value_counts()
            all_st_q3 = sorted(set(c_s.index) | set(c_e.index))
            t_q3 = pd.DataFrame([
                {
                    "State": s,
                    f"Count ({q3_years[0]})": int(c_s.get(s, 0)),
                    f"Count ({q3_years[1]})": int(c_e.get(s, 0)),
                    "Net Change (Delta)": int(c_e.get(s, 0)) - int(c_s.get(s, 0)),
                }
                for s in all_st_q3
            ]).sort_values(by="Net Change (Delta)", ascending=False)
            st.dataframe(t_q3, use_container_width=True)

    # 4. Karnataka vs Tamil Nadu rankings comparison
    elif geo_question == "How do the rankings of institutions in Karnataka compare with those in Tamil Nadu?":
        with col_geo_param:
            c1, c2 = st.columns(2)
            with c1:
                q4_type = st.selectbox(
                    "Plot Type",
                    options=["Box Plot", "Violin Plot"],
                    index=0,
                    key="q4_type_select",
                )
            with c2:
                q4_mode = st.selectbox(
                    "Comparison View",
                    options=["Overall Period", "Year-wise Comparison"],
                    index=0,
                    key="q4_mode_select",
                )

        fig_map = plot_state_rank_comparison_map(
            df=df_full,
            category=selected_category,
            state1="Karnataka",
            state2="Tamil Nadu",
            year_range=year_range,
        )
        fig_chart, takeaway = plot_state_rank_comparison(
            df=df_full,
            category=selected_category,
            state1="Karnataka",
            state2="Tamil Nadu",
            year_range=year_range,
            plot_type=q4_type,
            breakdown=q4_mode,
        )

        col_map, col_chart = st.columns([1.6, 1.0])
        with col_map:
            st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})
        with col_chart:
            st.plotly_chart(fig_chart, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            f"""<div class="takeaway-card">
            <span class="takeaway-highlight">Takeaway line:</span> {takeaway}
            </div>""",
            unsafe_allow_html=True,
        )

        with st.expander("🔍 View underlying data table", expanded=False):
            sub_q4 = df_full[
                (df_full["category"] == selected_category)
                & (df_full["state"].isin(["Karnataka", "Tamil Nadu"]))
                & (df_full["year"] >= year_range[0])
                & (df_full["year"] <= year_range[1])
            ][["year", "name", "state", "rank", "score"]].sort_values(by=["year", "rank"])
            st.dataframe(sub_q4, use_container_width=True)

    # 5. Newly appearing institutions by state
    elif geo_question == "Which states have produced the largest number of newly appearing institutions in the rankings?":
        with col_geo_param:
            c1, c2, c3 = st.columns([1.0, 1.0, 1.0])
            with c1:
                entry_opts = ["All Years (2018–2025)"] + [str(y) for y in range(min_yr + 1, max_yr + 1)]
                q5_year = st.selectbox(
                    "Entry Year",
                    options=entry_opts,
                    index=0,
                    key="q5_year_select",
                )
            with c2:
                q5_top_n = st.slider("Top N states", min_value=5, max_value=20, value=12, key="q5_top_n_slider")
            with c3:
                q5_palette = st.selectbox(
                    "Palette",
                    options=["Viridis", "Plasma", "Turbo", "YlGnBu", "Tealgrn", "Blues"],
                    index=0,
                    key="q5_palette_select",
                )

        fig_map = plot_new_entrant_map(
            df=df_full,
            category=selected_category,
            base_year=min_yr,
            selected_entry_year=q5_year,
            color_palette=q5_palette,
        )
        fig_chart, takeaway = plot_new_entrant_states(
            df=df_full,
            category=selected_category,
            base_year=min_yr,
            top_n=q5_top_n,
            selected_entry_year=q5_year,
        )

        col_map, col_chart = st.columns([1.6, 1.0])
        with col_map:
            st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})
        with col_chart:
            st.plotly_chart(fig_chart, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            f"""<div class="takeaway-card">
            <span class="takeaway-highlight">Takeaway line:</span> {takeaway}
            </div>""",
            unsafe_allow_html=True,
        )

        with st.expander("🔍 View underlying data table", expanded=False):
            sub_cat = df_full[df_full["category"] == selected_category]
            first_apps = sub_cat.groupby("name")["year"].min().reset_index()
            first_apps.columns = ["name", "First Appearance Year"]
            st_map = sub_cat[["name", "state"]].drop_duplicates(subset=["name"])
            new_table = first_apps.merge(st_map, on="name")
            new_table = new_table[new_table["First Appearance Year"] > min_yr]
            if q5_year != "All Years (2018–2025)":
                new_table = new_table[new_table["First Appearance Year"] == int(q5_year)]
            st.dataframe(new_table.sort_values(by=["First Appearance Year", "state", "name"]), use_container_width=True)

    # Overview: All 5 Visualizations
    elif geo_question == "All Geospatial & Regional Visualizations (Overview)":
        st.markdown("##### 1. Top States in Top 100")
        f1_map = plot_top_states_map(df_full, selected_category, year_range[1])
        f1_chart, t1 = plot_top_states_top100(df_full, selected_category, year_range[1], 12)
        c1, c2 = st.columns([1.6, 1.0])
        with c1:
            st.plotly_chart(f1_map, use_container_width=True, config={"displayModeBar": False})
        with c2:
            st.plotly_chart(f1_chart, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f'<div class="takeaway-card"><span class="takeaway-highlight">Takeaway line:</span> {t1}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("##### 2. Geographical Distribution Over Time")
        f2_map, t2 = plot_geo_distribution_over_time(df_full, selected_category, year_range[1], "India Choropleth Map")
        f2_chart, _ = plot_geo_distribution_over_time(df_full, selected_category, year_range[1], "State-wise Stacked Bar Chart")
        c1, c2 = st.columns([1.6, 1.0])
        with c1:
            st.plotly_chart(f2_map, use_container_width=True, config={"displayModeBar": False})
        with c2:
            st.plotly_chart(f2_chart, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f'<div class="takeaway-card"><span class="takeaway-highlight">Takeaway line:</span> {t2}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("##### 3. Representation Improvement (2017 to 2025)")
        f3_map = plot_state_representation_map(df_full, selected_category, min_yr, max_yr)
        f3_chart, t3 = plot_state_representation_improvement(df_full, selected_category, min_yr, max_yr, "Bar (Delta)")
        c1, c2 = st.columns([1.6, 1.0])
        with c1:
            st.plotly_chart(f3_map, use_container_width=True, config={"displayModeBar": False})
        with c2:
            st.plotly_chart(f3_chart, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f'<div class="takeaway-card"><span class="takeaway-highlight">Takeaway line:</span> {t3}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("##### 4. Karnataka vs. Tamil Nadu Rankings Comparison")
        f4_map = plot_state_rank_comparison_map(df_full, selected_category, "Karnataka", "Tamil Nadu", year_range)
        f4_chart, t4 = plot_state_rank_comparison(df_full, selected_category, "Karnataka", "Tamil Nadu", year_range)
        c1, c2 = st.columns([1.6, 1.0])
        with c1:
            st.plotly_chart(f4_map, use_container_width=True, config={"displayModeBar": False})
        with c2:
            st.plotly_chart(f4_chart, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f'<div class="takeaway-card"><span class="takeaway-highlight">Takeaway line:</span> {t4}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("##### 5. Newly Appearing Institutions by State")
        f5_map = plot_new_entrant_map(df_full, selected_category, min_yr)
        f5_chart, t5 = plot_new_entrant_states(df_full, selected_category, min_yr, 10)
        c1, c2 = st.columns([1.6, 1.0])
        with c1:
            st.plotly_chart(f5_map, use_container_width=True, config={"displayModeBar": False})
        with c2:
            st.plotly_chart(f5_chart, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f'<div class="takeaway-card"><span class="takeaway-highlight">Takeaway line:</span> {t5}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# TAB 3: Univariate  — Annexure A Section 3
# ---------------------------------------------------------
with tabs[2]:
    UNIVARIATE_QUESTIONS = [
        "Q1 · Score distribution for a selected year",
        "Q2 · Rank distribution for a selected year",
        "Q3 · Summary statistics (mean / median / min / max) by year",
        "Q4 · Quartile / rank-band breakdown over time",
        "Q5 · Score spread across categories for a year",
        "Q6 · Parameter-level distribution (TLR, RPC, GO, OI, PR)",
    ]

    col_uq, col_up = st.columns([1.6, 1.2])
    with col_uq:
        u_question = st.selectbox(
            "Question",
            options=UNIVARIATE_QUESTIONS,
            index=0,
            key="uni_q",
        )

    # ── Q1: Score distribution ──────────────────────────────────
    if u_question == UNIVARIATE_QUESTIONS[0]:
        all_years = sorted(df_full["year"].unique().tolist())
        default_yr = all_years[-1] if all_years else 2024
        with col_up:
            u_year = st.selectbox("Year", options=all_years,
                                  index=len(all_years)-1, key="uni_q1_yr")
            u_bins = st.slider("Bins", 10, 40, 20, key="uni_q1_bins")

        fig, takeaway = plot_score_distribution(
            df=df_full, year=u_year, category=selected_category, bins=u_bins
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f'''<div class="takeaway-card">
            <span class="takeaway-highlight">Takeaway:</span> {takeaway}
            </div>''', unsafe_allow_html=True
        )
        with st.expander("🔍 View underlying data", expanded=False):
            sub = df_full[
                (df_full["year"] == u_year) & (df_full["category"] == selected_category)
            ][["name", "state", "rank", "score"]].dropna(subset=["score"])
            st.dataframe(sub.sort_values("score", ascending=False), use_container_width=True)

    # ── Q2: Rank distribution ───────────────────────────────────
    elif u_question == UNIVARIATE_QUESTIONS[1]:
        all_years = sorted(df_full["year"].unique().tolist())
        with col_up:
            u_year = st.selectbox("Year", options=all_years,
                                  index=len(all_years)-1, key="uni_q2_yr")
            u_bins2 = st.slider("Bins", 10, 40, 20, key="uni_q2_bins")

        fig, takeaway = plot_rank_distribution(
            df=df_full, year=u_year, category=selected_category, bins=u_bins2
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f'''<div class="takeaway-card">
            <span class="takeaway-highlight">Takeaway:</span> {takeaway}
            </div>''', unsafe_allow_html=True
        )

    # ── Q3: Summary statistics ──────────────────────────────────
    elif u_question == UNIVARIATE_QUESTIONS[2]:
        fig, takeaway = plot_summary_statistics(
            df=df_full, category=selected_category, year_range=year_range
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f'''<div class="takeaway-card">
            <span class="takeaway-highlight">Takeaway:</span> {takeaway}
            </div>''', unsafe_allow_html=True
        )

    # ── Q4: Rank-band breakdown ─────────────────────────────────
    elif u_question == UNIVARIATE_QUESTIONS[3]:
        fig, takeaway = plot_rank_band_breakdown(
            df=df_full, category=selected_category, year_range=year_range
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f'''<div class="takeaway-card">
            <span class="takeaway-highlight">Takeaway:</span> {takeaway}
            </div>''', unsafe_allow_html=True
        )

    # ── Q5: Score spread across categories ─────────────────────
    elif u_question == UNIVARIATE_QUESTIONS[4]:
        all_years = sorted(df_full["year"].unique().tolist())
        all_cats = sorted(df_full["category"].dropna().unique().tolist())
        with col_up:
            u_year5 = st.selectbox("Year", options=all_years,
                                   index=len(all_years)-1, key="uni_q5_yr")
        sel_cats = st.multiselect(
            "Categories to compare",
            options=all_cats,
            default=all_cats[:6] if len(all_cats) > 6 else all_cats,
            key="uni_q5_cats",
        )

        fig, takeaway = plot_score_spread_by_category(
            df=df_full, year=u_year5,
            categories=sel_cats if sel_cats else None
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f'''<div class="takeaway-card">
            <span class="takeaway-highlight">Takeaway:</span> {takeaway}
            </div>''', unsafe_allow_html=True
        )

    # ── Q6: Parameter distribution ──────────────────────────────
    elif u_question == UNIVARIATE_QUESTIONS[5]:
        all_years = sorted(df_full["year"].unique().tolist())
        PARAMS = ["TLR", "RPC", "GO", "OI", "PR"]
        with col_up:
            u_year6 = st.selectbox("Year", options=all_years,
                                   index=len(all_years)-1, key="uni_q6_yr")
            u_style = st.radio("Chart style", ["violin", "box"],
                               horizontal=True, key="uni_q6_style")
        sel_params = st.multiselect(
            "Parameters",
            options=PARAMS,
            default=PARAMS,
            key="uni_q6_params",
        )

        fig, takeaway = plot_parameter_distribution(
            df=df_full,
            year=u_year6,
            category=selected_category,
            params=sel_params if sel_params else PARAMS,
            chart_style=u_style,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f'''<div class="takeaway-card">
            <span class="takeaway-highlight">Takeaway:</span> {takeaway}
            </div>''', unsafe_allow_html=True
        )

# ---------------------------------------------------------
# TAB 4: Bivariate (Placeholder)
# ---------------------------------------------------------
with tabs[3]:
    st.markdown("#### Bivariate Visualizations")
    st.info("Bivariate tab (Section 4 questions) will be activated in the next phase.")

# ---------------------------------------------------------
# TAB 5: Multivariate (Placeholder)
# ---------------------------------------------------------
with tabs[4]:
    st.markdown("#### Multivariate Visualizations")
    st.info("Multivariate tab (Section 5 questions) will be activated in the next phase.")

# ---------------------------------------------------------
# TAB 6: Distribution (Placeholder)
# ---------------------------------------------------------
with tabs[5]:
    st.markdown("#### Distribution / Statistical Comparison")
    st.info("Distribution tab (Section 6 questions) will be activated in the next phase.")

# ---------------------------------------------------------
# TAB 7: Transitions (Placeholder)
# ---------------------------------------------------------
with tabs[6]:
    st.markdown("#### Ranking Transitions")
    st.info("Transitions tab (Section 7 questions) will be activated in the next phase.")

# ---------------------------------------------------------
# TAB 8: Outliers (Placeholder)
# ---------------------------------------------------------
with tabs[7]:
    st.markdown("#### Outliers & Gaps")
    st.info("Outliers tab (Section 8 questions) will be activated in the next phase.")

with tabs[8]:
    render_explore_tab(df_full, df_filtered)

with tabs[9]:
    st.markdown("#### Ask (Level 3 - Natural Language Query)")
    st.info("The Ask tab (Level 3 natural language query interface) is staged for Phase 3.")



