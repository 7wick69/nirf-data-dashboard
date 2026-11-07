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
from core.transitions import (
    DEFAULT_RANK_BANDS,
    prepare_ranked_frame,
    compare_two_years,
    get_category_valid_years,
    get_category_consecutive_pairs,
    q1_band_transition,
    q2_top10_mobility,
    q3_band_retention,
    q4_biggest_movers,
    q5_annual_mobility_summary,
)
from charts.transitions import (
    plot_band_destination_bar,
    plot_mover_dumbbell,
    plot_band_transition_heatmap,
    plot_band_transition_sankey,
    plot_top10_mobility_slope,
    plot_top10_turnover_bars,
    plot_band_retention_trend,
    plot_band_retention_stacked_bar,
    plot_diverging_movers,
    plot_mobility_composition_bar,
)
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
            if "trends_institution_select" in st.session_state and st.session_state["trends_institution_select"] not in inst_options:
                del st.session_state["trends_institution_select"]
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
# TAB 7: Section 7 - Ranking Transitions & Institutional Mobility
# ---------------------------------------------------------
with tabs[6]:
    # -----------------------------------------------------------------
    # State Synchronization: Read Category & State directly from sidebar
    # -----------------------------------------------------------------
    s7_category = selected_category
    s7_state = selected_state

    # Invalidate dependent caches if Category or State changed in sidebar
    if st.session_state.get("s7_prev_cat") != s7_category:
        st.session_state["s7_prev_cat"] = s7_category
        for k in ["s7_q1_y1", "s7_q1_y2", "s7_q2_pair", "s7_q2_y1", "s7_q2_y2", "s7_q4_y1", "s7_q4_y2"]:
            if k in st.session_state:
                del st.session_state[k]

    if st.session_state.get("s7_prev_st") != s7_state:
        st.session_state["s7_prev_st"] = s7_state
        for k in ["s7_q1_y1", "s7_q1_y2", "s7_q2_pair", "s7_q2_y1", "s7_q2_y2", "s7_q4_y1", "s7_q4_y2"]:
            if k in st.session_state:
                del st.session_state[k]

    # Question Selection Bar
    col_q7, col_ctl = st.columns([1.6, 1.1])
    with col_q7:
        trans_question = st.selectbox(
            "Mobility Question",
            options=[
                "Q1: How many institutions moved from 51-100 band into the top 50?",
                "Q2: Which institutions entered or exited the top 10 between consecutive years?",
                "Q3: What percentage of institutions remained in the same rank band from one year to the next?",
                "Q4: Which institutions had the largest upward and downward rank movements?",
                "Q5: How many institutions moved up, moved down, or retained their positions each year?",
            ],
            index=0,
            key="s7_question_select",
        )

    # Rank Basis: show only when a specific state is selected
    if s7_state != "All states":
        col_basis_row, col_info = st.columns([1.5, 2.5])
        with col_basis_row:
            s7_basis = st.selectbox(
                f"Rank Basis ({s7_state})",
                options=["Within selected state", "National rank"],
                index=0,
                key="s7_basis_select",
                help=(
                    "Within selected state: re-ranks institutions 1..N inside the state before assigning bands. "
                    "National rank: uses official All-India NIRF ranks and bands."
                ),
            )
        with col_info:
            st.caption(
                f"ℹ️ **State Filter Active:** Evaluating **{s7_category}** institutions in **{s7_state}**. "
                f"Rank basis: *{s7_basis}*."
            )
    else:
        s7_basis = "National rank"

    # Validate category years
    s7_valid_years = get_category_valid_years(df_full, s7_category)
    s7_pairs = get_category_consecutive_pairs(df_full, s7_category)

    if len(s7_valid_years) < 2:
        st.warning(
            f"Category **'{s7_category}'** only has {len(s7_valid_years)} year(s) of records "
            f"({s7_valid_years}). Transition and mobility analysis requires at least two ranking years."
        )
    else:
        # -----------------------------------------------------------------
        # Q1: Band Transition (51-100 -> Top 50 / Custom Bands)
        # -----------------------------------------------------------------
        if trans_question.startswith("Q1"):
            with col_ctl:
                chart_style = st.radio(
                    "Chart View",
                    [
                        "Destination Breakdown (100% Bar)",
                        "Mover Dumbbell Chart",
                        "Transition Matrix Heatmap",
                        "Sankey Flow Diagram",
                    ],
                    horizontal=True,
                    key="s7_q1_chart_style",
                )

            # Row 2 Controls: Year 1, Year 2, From Band, To Target, Rank Range Mode
            col_y1, col_y2, col_fb, col_tb, col_meth = st.columns([1, 1, 1, 1, 1])
            with col_y1:
                default_y1_idx = max(0, len(s7_valid_years) - 2)
                q1_y1 = st.selectbox(
                    "Year 1 (Origin)",
                    options=s7_valid_years[:-1],
                    index=default_y1_idx,
                    key="s7_q1_y1",
                )
            with col_y2:
                valid_y2_opts = [y for y in s7_valid_years if y > q1_y1]
                q1_y2 = st.selectbox(
                    "Year 2 (Destination)",
                    options=valid_y2_opts,
                    index=0 if valid_y2_opts else 0,
                    key="s7_q1_y2",
                )
            with col_fb:
                from_band_choice = st.selectbox(
                    "From Band (in Y1)",
                    ["51-100", "Top 10", "11-50", "101-150", "151-200", "201+"],
                    index=0,
                    key="s7_q1_from_band",
                )
            with col_tb:
                to_band_choice = st.selectbox(
                    "To Target (in Y2)",
                    ["Top 50", "Top 10", "11-50", "51-100", "101-150", "151-200", "201+"],
                    index=0,
                    key="s7_q1_to_band",
                )
            with col_meth:
                band_method = st.selectbox(
                    "Rank Range Mode",
                    ["midpoint", "lower"],
                    index=0,
                    help="For banded ranks (e.g. 101-150): use midpoint (125.5) or lower bound (101.0)",
                    key="s7_q1_band_method",
                )

            # Data & Chart Computation
            q1_res = q1_band_transition(
                df=df_full,
                category=s7_category,
                y1=q1_y1,
                y2=q1_y2,
                from_band=from_band_choice,
                to_band=to_band_choice,
                state=s7_state,
                basis=s7_basis,
                band_rank_method=band_method,
            )

            # Active Selection Status Caption
            comp_q1 = compare_two_years(df_full, s7_category, q1_y1, q1_y2, state=s7_state, basis=s7_basis)
            st.caption(
                f"**Active Selection:** {s7_category} | {s7_state} | {s7_basis} | "
                f"{q1_y1} → {q1_y2} | {len(comp_q1)} institutions tracked"
            )

            # Metric KPIs
            kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
            mover_cnt = q1_res["kpi_count"]
            from_tot = q1_res["kpi_total_from"]

            # Card 1: Movers
            kpi_c1.metric(
                label=f"Movers ({from_band_choice} → {to_band_choice})",
                value=f"{mover_cnt}",
                help=f"Institutions transitioning from {from_band_choice} to {to_band_choice}",
            )
            kpi_c1.caption(f"{mover_cnt} of {from_tot} moved to {to_band_choice}")

            # Card 2: Total in Origin Band
            kpi_c2.metric(
                label=f"Total in {from_band_choice} ({q1_y1})",
                value=f"{from_tot}",
                help=f"Total institutions ranked in {from_band_choice} in base year ({s7_basis})",
            )
            kpi_c2.caption(f"Cohort base in {q1_y1}")

            # Card 3: Transition Success Rate
            kpi_c3.metric(
                label="Transition Success Rate",
                value=f"{q1_res['kpi_rate']}%",
                delta=f"{mover_cnt} of {from_tot}" if from_tot > 0 else None,
                delta_color="normal" if mover_cnt > 0 else "off",
                help="Percentage of origin band institutions reaching target",
            )
            kpi_c3.caption(f"Success rate: {q1_res['kpi_rate_formatted']}")

            # Card 4: Avg Rank Gain
            if mover_cnt > 0 and q1_res["avg_rank_jump"] is not None:
                avg_j = q1_res["avg_rank_jump"]
                sign = "+" if avg_j > 0 else ""
                d_color = "normal" if avg_j > 0 else ("inverse" if avg_j < 0 else "off")
                kpi_c4.metric(
                    label="Avg Rank Gain",
                    value=f"{sign}{avg_j:.1f} places",
                    delta=f"{sign}{avg_j:.1f} average jump",
                    delta_color=d_color,
                    help="Average rank improvement for institutions that transitioned",
                )
                kpi_c4.caption("Positive = rank improved")
            else:
                kpi_c4.metric(
                    label="Avg Rank Gain",
                    value="N/A",
                    help="No institutions completed this transition",
                )
                kpi_c4.caption("No movers to average")

            # Small cohort notice if origin band has < 5 institutions
            if q1_res["is_small_sample"]:
                st.info(
                    f"ℹ️ **Small Cohort Notice ({from_tot} institution(s) in {from_band_choice}):** "
                    f"Sample size in origin band is small (< 5 institutions); percentage transition rates "
                    f"may be volatile. Refer to raw counts ({mover_cnt} of {from_tot}) for evaluation."
                )

            # Chart Visualization
            if chart_style == "Destination Breakdown (100% Bar)":
                fig_q1, takeaway_q1 = plot_band_destination_bar(
                    df=df_full,
                    category=s7_category,
                    y1=q1_y1,
                    y2=q1_y2,
                    from_band=from_band_choice,
                    to_band=to_band_choice,
                    state=s7_state,
                    basis=s7_basis,
                    band_rank_method=band_method,
                )
            elif chart_style == "Mover Dumbbell Chart":
                fig_q1, takeaway_q1 = plot_mover_dumbbell(
                    df=df_full,
                    category=s7_category,
                    y1=q1_y1,
                    y2=q1_y2,
                    from_band=from_band_choice,
                    to_band=to_band_choice,
                    state=s7_state,
                    basis=s7_basis,
                    band_rank_method=band_method,
                )
            elif chart_style == "Transition Matrix Heatmap":
                fig_q1, takeaway_q1 = plot_band_transition_heatmap(
                    df=df_full,
                    category=s7_category,
                    y1=q1_y1,
                    y2=q1_y2,
                    from_band=from_band_choice,
                    to_band=to_band_choice,
                    state=s7_state,
                    basis=s7_basis,
                    band_rank_method=band_method,
                )
            else:
                fig_q1, takeaway_q1 = plot_band_transition_sankey(
                    df=df_full,
                    category=s7_category,
                    y1=q1_y1,
                    y2=q1_y2,
                    from_band=from_band_choice,
                    to_band=to_band_choice,
                    state=s7_state,
                    basis=s7_basis,
                    band_rank_method=band_method,
                )

            st.plotly_chart(fig_q1, use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                f"""<div class="takeaway-card">
                <span class="takeaway-highlight">Takeaway line:</span> {takeaway_q1}
                </div>""",
                unsafe_allow_html=True,
            )

            # Expandable Table: Institutions that moved
            with st.expander(f"📋 Institutions that moved ({from_band_choice} → {to_band_choice})", expanded=True):
                movers_df = q1_res["movers_df"]
                if movers_df.empty:
                    st.info(
                        f"No institutions moved from **{from_band_choice}** into **{to_band_choice}** "
                        f"in '{s7_category}' ({s7_state}) between {q1_y1} and {q1_y2} ({s7_basis})."
                    )
                else:
                    rank_col_y1 = f"Rank in {q1_y1}"
                    rank_col_y2 = f"Rank in {q1_y2}"
                    band_col_y1 = f"Band {q1_y1}"
                    band_col_y2 = f"Band {q1_y2}"
                    
                    display_movers = movers_df[[
                        "name", "state", "rank_y1", "rank_y2", "band_y1", "band_y2", "rank_change"
                    ]].rename(columns={
                        "name": "Institute name",
                        "state": "State",
                        "rank_y1": rank_col_y1,
                        "rank_y2": rank_col_y2,
                        "band_y1": band_col_y1,
                        "band_y2": band_col_y2,
                        "rank_change": "Rank change (positive = improved)",
                    })
                    st.dataframe(display_movers, use_container_width=True)
                    
                    csv_data = display_movers.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Movers (CSV)",
                        data=csv_data,
                        file_name=f"nirf_movers_{s7_category}_{q1_y1}_{q1_y2}.csv",
                        mime="text/csv",
                        key="s7_q1_download_movers_csv",
                    )

            # Methodology Expander
            with st.expander("ℹ️ How this is calculated", expanded=False):
                st.markdown(r"""
**Methodology & Transition Logic:**
- **Rank Basis:** 
  - *Within selected state:* Institutions belonging to the state are filtered first for both Year 1 and Year 2. They are then independently re-ranked from 1 to N inside that state per year (1 = highest national rank). Rank bands are assigned from these state-level ranks.
  - *National rank:* Keeps official NIRF All-India ranks and bands; the state filter restricts display to that state's institutions.
- **Rank Bands:** Standard NIRF brackets: Top 10 (1–10), 11–50 (11–50), 51–100 (51–100), 101–150 (101–150), 151–200 (151–200), 201+ (201+), and Unranked.
- **Rank Range Mode:** For institutions assigned to rank intervals (e.g. 101–150), the *midpoint* option uses 125.5 while *lower bound* uses 101.0.
- **Rank Change Formula:** $\Delta = 	ext{Rank}_{Y1} - 	ext{Rank}_{Y2}$. Positive values represent upward mobility (e.g. #76 to #43 is +33).
- **New Entries & Dropouts:** Institutions present only in Y2 are tagged *New Entry*; those present only in Y1 are tagged *Dropped Out*. Neither are classified as rank movers.
                """)

        # -----------------------------------------------------------------
        # Q2: Top 10 Entries and Exits
        # -----------------------------------------------------------------
        elif trans_question.startswith("Q2"):
            with col_ctl:
                q2_chart_view = st.radio(
                    "Chart View",
                    ["Slope Chart (Primary)", "Turnover Across Cycles (Bar)"],
                    horizontal=True,
                    key="s7_q2_chart_view",
                )
                consec_only = st.checkbox("Consecutive Years Mode", value=True, key="s7_q2_consec_toggle")

            if consec_only and s7_pairs:
                col_p, col_empty = st.columns([1.5, 2.5])
                with col_p:
                    pair_options = [f"{p[0]} → {p[1]}" for p in s7_pairs]
                    default_pair_idx = len(pair_options) - 1
                    chosen_pair_str = st.selectbox(
                        "Select Year Cycle",
                        pair_options,
                        index=default_pair_idx,
                        key="s7_q2_pair",
                    )
                    q2_y1, q2_y2 = int(chosen_pair_str.split(" → ")[0]), int(chosen_pair_str.split(" → ")[1])
            else:
                col_y1, col_y2 = st.columns(2)
                with col_y1:
                    default_y1_idx = max(0, len(s7_valid_years) - 2)
                    q2_y1 = st.selectbox("Year 1", s7_valid_years[:-1], index=default_y1_idx, key="s7_q2_y1")
                with col_y2:
                    valid_y2_opts = [y for y in s7_valid_years if y > q2_y1]
                    q2_y2 = st.selectbox("Year 2", valid_y2_opts, index=0 if valid_y2_opts else 0, key="s7_q2_y2")

            q2_data = q2_top10_mobility(df_full, s7_category, q2_y1, q2_y2, state=s7_state, basis=s7_basis)

            # Active Selection Status Caption
            comp_q2 = compare_two_years(df_full, s7_category, q2_y1, q2_y2, state=s7_state, basis=s7_basis)
            st.caption(
                f"**Active Selection:** {s7_category} | {s7_state} | {s7_basis} | "
                f"{q2_y1} → {q2_y2} | {len(comp_q2)} institutions tracked"
            )

            # KPIs with institution name badges (Entry = green, Exit = red)
            k1, k2, k3 = st.columns(3)
            k1.metric(
                "Entered Top 10",
                f"{q2_data['num_entered']} institutions",
                delta=f"+{q2_data['num_entered']}" if q2_data['num_entered'] > 0 else None,
                delta_color="normal",
            )
            if not q2_data["entered_df"].empty:
                entry_badges = " ".join([
                    f'<span style="display:inline-block; background:rgba(46,160,67,0.18); color:#3fb950; border:1px solid #2ea043; border-radius:4px; padding:2px 6px; font-size:11px; margin:2px 2px; font-weight:600;">{row["name"]} (#{int(row["rank_y2"]) if pd.notna(row["rank_y2"]) else "NR"})</span>'
                    for _, row in q2_data["entered_df"].iterrows()
                ])
                k1.markdown(entry_badges, unsafe_allow_html=True)
            else:
                k1.caption("No new entrants in Top 10")

            k2.metric(
                "Exited Top 10",
                f"{q2_data['num_exited']} institutions",
                delta=f"-{q2_data['num_exited']}" if q2_data['num_exited'] > 0 else None,
                delta_color="inverse",
            )
            if not q2_data["exited_df"].empty:
                exit_badges = " ".join([
                    f'<span style="display:inline-block; background:rgba(248,81,73,0.18); color:#f85149; border:1px solid #f85149; border-radius:4px; padding:2px 6px; font-size:11px; margin:2px 2px; font-weight:600;">{row["name"]} ({f"#{int(row.rank_y2)}" if pd.notna(row.rank_y2) else "Dropped"})</span>'
                    for _, row in q2_data["exited_df"].iterrows()
                ])
                k2.markdown(exit_badges, unsafe_allow_html=True)
            else:
                k2.caption("No institutions exited Top 10")

            k3.metric("Retained Top 10", f"{q2_data['num_retained']} institutions")
            k3.caption(f"{q2_data['num_retained']} of 10 held rank in {q2_y2}")

            # State context banner when National Rank basis is active
            if q2_data.get("is_state_active") and q2_data.get("basis") == "National rank":
                st_name = q2_data["state_name"]
                st_cnt = q2_data["state_top10_count"]
                if st_cnt > 0:
                    st.success(
                        f"🌟 **{st_name}** has **{st_cnt}** institution(s) in the National Top 10 "
                        f"({q2_y1}→{q2_y2}): {st_cnt} retained/entered. Highlighted in gold below."
                    )
                else:
                    best_name = q2_data.get("state_best_name", "—")
                    best_rank = q2_data.get("state_best_rank_y2")
                    best_rank_str = f"#{int(best_rank)}" if best_rank and not (best_rank != best_rank) else "unranked"
                    st.info(
                        f"ℹ️ **{st_name}** has **0** institutions in the National Top 10 "
                        f"({q2_y1}→{q2_y2}). Best contender in {q2_y2}: **{best_name}** at {best_rank_str}. "
                        f"To see Top 10 institutions inside {st_name}, switch Rank Basis to 'Within selected state'."
                    )

            # Chart Visualization
            if q2_chart_view == "Slope Chart (Primary)":
                fig_q2, takeaway_q2 = plot_top10_mobility_slope(df_full, s7_category, q2_y1, q2_y2, state=s7_state, basis=s7_basis)
            else:
                fig_q2, takeaway_q2 = plot_top10_turnover_bars(df_full, s7_category, state=s7_state, basis=s7_basis)

            st.plotly_chart(fig_q2, use_container_width=True, config={"displayModeBar": False})

            st.markdown(
                f"""<div class="takeaway-card">
                <span class="takeaway-highlight">Takeaway line:</span> {takeaway_q2}
                </div>""",
                unsafe_allow_html=True,
            )

            # Side-by-side Roster Tables
            col_ent, col_exit, col_ret = st.columns(3)
            with col_ent:
                st.markdown(f"**Newly Entered Top 10 in {q2_y2}**")
                if q2_data["entered_df"].empty:
                    st.caption("No new entrants in Top 10.")
                else:
                    st.dataframe(q2_data["entered_df"][["name", "rank_y1", "rank_y2"]].rename(columns={"name": "Institution", "rank_y1": f"{q2_y1}", "rank_y2": f"{q2_y2}"}), use_container_width=True)
            with col_exit:
                st.markdown(f"**Dropped Out of Top 10 in {q2_y2}**")
                if q2_data["exited_df"].empty:
                    st.caption("No institutions exited Top 10.")
                else:
                    st.dataframe(q2_data["exited_df"][["name", "rank_y1", "rank_y2"]].rename(columns={"name": "Institution", "rank_y1": f"{q2_y1}", "rank_y2": f"{q2_y2}"}), use_container_width=True)
            with col_ret:
                st.markdown(f"**Maintained Top 10 ({q2_y1} & {q2_y2})**")
                st.dataframe(q2_data["retained_df"][["name", "rank_y1", "rank_y2"]].rename(columns={"name": "Institution", "rank_y1": f"{q2_y1}", "rank_y2": f"{q2_y2}"}), use_container_width=True)

        # -----------------------------------------------------------------
        # Q3: Band Retention Percentage Across Consecutive Years
        # -----------------------------------------------------------------
        elif trans_question.startswith("Q3"):
            with col_ctl:
                chart_style_q3 = st.radio(
                    "Chart View",
                    ["Retention Trendline (Line)", "Band Movement Composition (Stacked Bar)"],
                    horizontal=True,
                    key="s7_q3_chart_style",
                )

            ret_df = q3_band_retention(df_full, s7_category, state=s7_state, basis=s7_basis)

            # Active Selection Status Caption
            st.caption(
                f"**Active Selection:** {s7_category} | {s7_state} | {s7_basis} | "
                f"Evaluating across all {len(s7_pairs)} consecutive year cycles"
            )

            if ret_df.empty:
                st.info(f"Insufficient multi-year records for '{s7_category}' ({s7_state}).")
            else:
                avg_ret = ret_df["retention_rate_pct"].mean()
                best_pair = ret_df.loc[ret_df["retention_rate_pct"].idxmax()]
                worst_pair = ret_df.loc[ret_df["retention_rate_pct"].idxmin()]

                k1, k2, k3 = st.columns(3)
                k1.metric("Historical Avg Band Retention", f"{avg_ret:.1f}%")
                k2.metric("Highest Band Stability", f"{best_pair['pair']} ({best_pair['retention_rate_pct']}%)")
                k3.metric("Highest Mobility Cycle", f"{worst_pair['pair']} ({worst_pair['retention_rate_pct']}% retained)")

                if chart_style_q3 == "Retention Trendline (Line)":
                    fig_q3, takeaway_q3 = plot_band_retention_trend(df_full, s7_category, state=s7_state, basis=s7_basis)
                else:
                    fig_q3, takeaway_q3 = plot_band_retention_stacked_bar(df_full, s7_category, state=s7_state, basis=s7_basis)

                st.plotly_chart(fig_q3, use_container_width=True, config={"displayModeBar": False})

                st.markdown(
                    f"""<div class="takeaway-card">
                    <span class="takeaway-highlight">Takeaway line:</span> {takeaway_q3}
                    </div>""",
                    unsafe_allow_html=True,
                )

                st.markdown("##### Consecutive Year Retention Breakdown Table")
                display_ret = ret_df[["pair", "continuing_total", "same_band_count", "retention_rate_pct", "promoted_count", "demoted_count"]].rename(columns={
                    "pair": "Year Pair",
                    "continuing_total": "Continuing Cohort",
                    "same_band_count": "Retained in Same Band",
                    "retention_rate_pct": "Retention Rate (%)",
                    "promoted_count": "Promoted to Higher Tier",
                    "demoted_count": "Demoted to Lower Tier",
                })
                st.dataframe(display_ret, use_container_width=True)

        # -----------------------------------------------------------------
        # Q4: Diverging Biggest Risers & Fallers
        # -----------------------------------------------------------------
        elif trans_question.startswith("Q4"):
            with col_ctl:
                top_n = st.slider("Top N Risers & Fallers", min_value=3, max_value=20, value=10, step=1, key="s7_q4_top_n")

            col_y1, col_y2, col_meth4 = st.columns([1, 1, 1])
            with col_y1:
                default_y1_idx = max(0, len(s7_valid_years) - 2)
                q4_y1 = st.selectbox("Year 1", s7_valid_years[:-1], index=default_y1_idx, key="s7_q4_y1")
            with col_y2:
                valid_y2_opts = [y for y in s7_valid_years if y > q4_y1]
                q4_y2 = st.selectbox("Year 2", valid_y2_opts, index=0 if valid_y2_opts else 0, key="s7_q4_y2")
            with col_meth4:
                band_method_q4 = st.selectbox(
                    "Rank Range Mode",
                    ["midpoint", "lower"],
                    index=0,
                    help="For banded ranks (e.g. 101-150): use midpoint (125.5) or lower bound (101.0)",
                    key="s7_q4_band_method",
                )

            q4_data = q4_biggest_movers(
                df_full, s7_category, q4_y1, q4_y2, top_n=top_n, state=s7_state, basis=s7_basis, band_rank_method=band_method_q4
            )

            # Active Selection Status Caption
            comp_q4 = compare_two_years(df_full, s7_category, q4_y1, q4_y2, state=s7_state, basis=s7_basis)
            st.caption(
                f"**Active Selection:** {s7_category} | {s7_state} | {s7_basis} | "
                f"{q4_y1} → {q4_y2} | {len(comp_q4)} institutions tracked"
            )

            if not q4_data["risers_df"].empty and not q4_data["fallers_df"].empty:
                best_riser = q4_data["risers_df"].iloc[0]
                worst_faller = q4_data["fallers_df"].iloc[0]
                k1, k2 = st.columns(2)
                k1.metric("Top Climber", f"{best_riser['name']}", f"+{int(best_riser['rank_change'])} places")
                k2.metric("Largest Decline", f"{worst_faller['name']}", f"{int(worst_faller['rank_change'])} places", delta_color="inverse")

            fig_q4, takeaway_q4 = plot_diverging_movers(
                df_full, s7_category, q4_y1, q4_y2, top_n=top_n, state=s7_state, basis=s7_basis, band_rank_method=band_method_q4
            )
            st.plotly_chart(fig_q4, use_container_width=True, config={"displayModeBar": False})
            st.caption("ℹ️ *Note: Ranks within banded brackets (e.g. 101–150) are mapped using the selected **midpoint** (125.5) or **lower bound** (101.0) convention.*")

            st.markdown(
                f"""<div class="takeaway-card">
                <span class="takeaway-highlight">Takeaway line:</span> {takeaway_q4}
                </div>""",
                unsafe_allow_html=True,
            )

            col_r, col_f = st.columns(2)
            with col_r:
                st.markdown(f"**Top {top_n} Risers ({q4_y1} → {q4_y2})**")
                risers_table = q4_data["risers_df"].copy()
                if not risers_table.empty:
                    risers_table["Climb (+)"] = risers_table["rank_change"].apply(lambda x: f"▲ +{int(x)}")
                    st.dataframe(
                        risers_table[["name", "rank_y1", "rank_y2", "Climb (+)"]].rename(
                            columns={"name": "Institution", "rank_y1": f"Rank {q4_y1}", "rank_y2": f"Rank {q4_y2}"}
                        ),
                        use_container_width=True,
                    )
            with col_f:
                st.markdown(f"**Top {top_n} Decliners ({q4_y1} → {q4_y2})**")
                fallers_table = q4_data["fallers_df"].copy()
                if not fallers_table.empty:
                    fallers_table["Drop (-)"] = fallers_table["rank_change"].apply(lambda x: f"▼ {int(x)}")
                    st.dataframe(
                        fallers_table[["name", "rank_y1", "rank_y2", "Drop (-)"]].rename(
                            columns={"name": "Institution", "rank_y1": f"Rank {q4_y1}", "rank_y2": f"Rank {q4_y2}"}
                        ),
                        use_container_width=True,
                    )

        # -----------------------------------------------------------------
        # Q5: Annual Mobility Breakdown (Up / Down / Same / New / Exited)
        # -----------------------------------------------------------------
        elif trans_question.startswith("Q5"):
            with col_ctl:
                chart_mode = st.radio("Display Metric", ["Number of Institutions", "Percentage Share (%)"], horizontal=True, key="s7_q5_mode_radio")
                mode_str = "percent" if "Percentage" in chart_mode else "count"

            mob_df = q5_annual_mobility_summary(df_full, s7_category, state=s7_state, basis=s7_basis)

            # Active Selection Status Caption
            st.caption(
                f"**Active Selection:** {s7_category} | {s7_state} | {s7_basis} | "
                f"Breakdown across all {len(s7_pairs)} consecutive cycles"
            )

            if mob_df.empty:
                st.info(f"No annual mobility records for '{s7_category}' ({s7_state}).")
            else:
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Avg Climbers Share", f"{mob_df['Moved Up (%)'].mean():.1f}%")
                k2.metric("Avg Decliners Share", f"{mob_df['Moved Down (%)'].mean():.1f}%")
                k3.metric("Avg Exact Stability", f"{mob_df['Retained (%)'].mean():.1f}%")
                avg_turnover = (mob_df["New Entry"] + mob_df["Dropped Out"]).mean()
                k4.metric("Avg Annual Turnover", f"{avg_turnover:.1f} inst/yr")

                fig_q5, takeaway_q5 = plot_mobility_composition_bar(df_full, s7_category, state=s7_state, basis=s7_basis, mode=mode_str)
                st.plotly_chart(fig_q5, use_container_width=True, config={"displayModeBar": False})
                st.caption(
                    "ℹ️ *Denominator note: 100% share is computed over the full tracked institutional cohort for that consecutive cycle "
                    "(Continuing institutions that moved up, moved down, or retained position, plus new entrants and dropouts).*"
                )

                st.markdown(
                    f"""<div class="takeaway-card">
                    <span class="takeaway-highlight">Takeaway line:</span> {takeaway_q5}
                    </div>""",
                    unsafe_allow_html=True,
                )

                st.markdown("##### Complete Annual Mobility Breakdown Data")
                st.dataframe(mob_df, use_container_width=True)

    # -----------------------------------------------------------------
    # Global Details on Demand Expander for Section 7
    # -----------------------------------------------------------------
    with st.expander("🔍 Search & Inspect Institution Mobility Data", expanded=False):
        search_query = st.text_input("Filter institutions by keyword / city / state", key="s7_search_input")
        if len(s7_valid_years) >= 2:
            raw_comp = compare_two_years(
                df_full,
                s7_category,
                s7_valid_years[0],
                s7_valid_years[-1],
                state=s7_state,
                basis=s7_basis,
            )
            if search_query:
                raw_comp = raw_comp[
                    raw_comp["name"].str.contains(search_query, case=False, na=False)
                    | raw_comp["city"].str.contains(search_query, case=False, na=False)
                    | raw_comp["state"].str.contains(search_query, case=False, na=False)
                ]
            st.dataframe(
                raw_comp[[
                    "name", "state", "city", "rank_y1", "rank_y2", "rank_change", "mobility_status", "band_y1", "band_y2"
                ]].rename(columns={
                    "name": "Institution",
                    "rank_y1": f"Rank ({s7_valid_years[0]})",
                    "rank_y2": f"Rank ({s7_valid_years[-1]})",
                    "rank_change": "Total Rank Change",
                    "mobility_status": "Status",
                    "band_y1": f"Band ({s7_valid_years[0]})",
                    "band_y2": f"Band ({s7_valid_years[-1]})",
                }),
                use_container_width=True,
            )

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



