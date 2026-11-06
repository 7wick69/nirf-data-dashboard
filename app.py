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
from charts.explore import render_explore_tab

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

    # State Filter
    all_states = ["All states"] + get_states(df_full)
    selected_state = st.selectbox(
        "States",
        options=all_states,
        index=0,
        help="Filter by state or all states",
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
    "Geo",
    "Univariate",
    "Bivariate",
    "Multivariate",
    "Distribution",
    "Transitions",
    "Outliers",
    "Explore",
    "Ask",
])

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
# Placeholder Tabs (Next Phases / Teammate Modules)
# ---------------------------------------------------------
for idx, tab_name in enumerate(["Geo", "Univariate", "Bivariate", "Multivariate", "Distribution", "Transitions", "Outliers"], start=1):
    with tabs[idx]:
        st.markdown(f"#### {tab_name} Visualizations")
        st.info(f"Phase 1 focused on `trends.py` validation. The {tab_name} questions from Annexure A will be activated in the next step.")

with tabs[8]:
    render_explore_tab(df_full, df_filtered)

with tabs[9]:
    st.markdown("#### Ask (Level 3 - Natural Language Query)")
    st.info("The Ask tab (Level 3 natural language query interface) is staged for Phase 3.")



