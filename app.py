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
    plot_biggest_improvers_decliners,
    plot_average_score_trend,
    plot_consistently_top10,
)
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
        help="Global filter: select year span",
    )

    # Category Filter
    categories = get_categories(df_full)
    selected_category = st.selectbox(
        "Category",
        options=categories,
        index=0,
        help="Global filter: NIRF category",
    )

    # State Filter
    all_states = ["All states"] + get_states(df_full)
    selected_state = st.selectbox(
        "States",
        options=all_states,
        index=0,
        help="Global filter: filter by state or all states",
    )

    st.markdown(
        "<p style='color: #8b949e; font-size: 12px; margin-top: 25px;'>Global filters:<br>every tab reads these</p>",
        unsafe_allow_html=True,
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


# ---------------------------------------------------------
# TAB 1: Trends (P0 Focus)
# ---------------------------------------------------------
with tabs[0]:
    col_q, col_param = st.columns([1.5, 1.2])

    with col_q:
        question = st.selectbox(
            "Question",
            options=[
                "How has the rank of a selected institution changed over time?",
                "Who are the biggest improvers and decliners?",
                "How has the average score changed over time?",
                "Which institutions are consistently in the top 10?",
            ],
            index=0,
        )

    # 1. Institution Rank Over Time
    if question == "How has the rank of a selected institution changed over time?":
        inst_options = get_institutions_list(df_full, selected_category)
        default_inst = "Indian Institute of Technology Madras"
        default_idx = inst_options.index(default_inst) if default_inst in inst_options else 0

        with col_param:
            selected_inst = st.selectbox(
                "Institution",
                options=inst_options,
                index=default_idx,
            )

        fig, takeaway = plot_institution_rank_trend(
            df=df_full,
            institution_name=selected_inst,
            category=selected_category,
            year_range=year_range,
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            f"""<div class="takeaway-card">
            <span class="takeaway-highlight">Takeaway line:</span> {takeaway}
            </div>""",
            unsafe_allow_html=True,
        )

    # 2. Biggest Improvers & Decliners
    elif question == "Who are the biggest improvers and decliners?":
        with col_param:
            top_n = st.slider("Top N institutions", min_value=3, max_value=10, value=5)

        fig, takeaway = plot_biggest_improvers_decliners(
            df=df_full,
            category=selected_category,
            start_year=year_range[0],
            end_year=year_range[1],
            top_n=top_n,
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            f"""<div class="takeaway-card">
            <span class="takeaway-highlight">Takeaway line:</span> {takeaway}
            </div>""",
            unsafe_allow_html=True,
        )

    # 3. Average Score Over Time
    elif question == "How has the average score changed over time?":
        fig, takeaway = plot_average_score_trend(
            df=df_full,
            category=selected_category,
            year_range=year_range,
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            f"""<div class="takeaway-card">
            <span class="takeaway-highlight">Takeaway line:</span> {takeaway}
            </div>""",
            unsafe_allow_html=True,
        )

    # 4. Consistently Top 10
    elif question == "Which institutions are consistently in the top 10?":
        with col_param:
            min_years = st.slider("Minimum years in Top 10", min_value=2, max_value=max_yr - min_yr + 1, value=5)

        fig, takeaway = plot_consistently_top10(
            df=df_full,
            category=selected_category,
            min_years=min_years,
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            f"""<div class="takeaway-card">
            <span class="takeaway-highlight">Takeaway line:</span> {takeaway}
            </div>""",
            unsafe_allow_html=True,
        )

    # Details on Demand Expander
    with st.expander("🔍 View underlying data table", expanded=False):
        if question == "How has the rank of a selected institution changed over time?":
            inst_df = df_full[
                (df_full["name"] == selected_inst)
                & (df_full["category"] == selected_category)
                & (df_full["year"] >= year_range[0])
                & (df_full["year"] <= year_range[1])
            ][["year", "rank", "score", "rank_change", "TLR", "RPC", "GO", "OI", "PR"]]
            st.dataframe(inst_df.sort_values(by="year"), use_container_width=True)


# ---------------------------------------------------------
# TAB 2: Geo (Placeholder)
# ---------------------------------------------------------
with tabs[1]:
    st.markdown("#### Geo Visualizations")
    st.info("Geo tab (Section 2 questions) will be activated in the next phase.")

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
    st.markdown("#### Explore (Level 2)")
    st.info("Interactive query and custom chart builder will be connected in Phase 2.")

with tabs[9]:
    st.markdown("#### Ask (Level 3 - Natural Language Query)")
    st.markdown(
        """
        <div class="chat-wireframe">
            <div class="chat-bubble-user">
                Which states improved their top 100 count since 2020?
            </div>
            <div class="chat-bubble-bot">
                Here's the change in top-100 count by state:
                <div style="border: 1px dashed #30363d; border-radius: 6px; padding: 20px; text-align: center; margin-top: 10px; color: #8b949e;">
                    Plotly chart will render here via NL query parser
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Natural language query executor is staged for Level 3 implementation.")

