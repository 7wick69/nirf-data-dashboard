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
import importlib
import charts.bivariate
import charts.outliers
importlib.reload(charts.bivariate)
importlib.reload(charts.outliers)
from charts.bivariate import (
    plot_scatter_marginals,
    plot_correlation_vs_weight,
    plot_param_correlation_heatmap,
    plot_tlr_rpc_quadrants,
    plot_tlr_rpc_by_category,
    plot_param_corr_over_years,
    plot_param_year_heatmap,
    plot_explorer,
    plot_rpc_vs_overall,
    plot_pr_vs_overall,
    plot_parameter_correlations,
    plot_tlr_vs_rpc,
    plot_rpc_overall_correlation_over_years,
)
from charts.outliers import (
    plot_connected_scatter_research_outliers,
    plot_research_vs_teaching_gap,
    plot_score_volatility_trends,
    plot_high_param_low_rank,
    plot_ranking_volatility,
)
from charts.explore import render_explore_tab
from charts.multivariate import render_multivariate_tab
from charts.distribution_analysis import render_distribution_tab

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
df_state = df_full[df_full["state"] == selected_state].copy() if selected_state != "All states" else df_full.copy()

df_filtered = df_state[
    (df_state["year"] >= year_range[0])
    & (df_state["year"] <= year_range[1])
    & (df_state["category"] == selected_category)
].copy()


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
        inst_options = get_institutions_list(df_state, selected_category)

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

            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

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

        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

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

        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        st.markdown(
            f"""<div class="takeaway-card">
            <span class="takeaway-highlight">Takeaway line:</span> {takeaway}
            </div>""",
            unsafe_allow_html=True,
        )

    # 4. Average Score Over Time
    elif question == "How has the average score of institutions changed over time?":
        fig, takeaway = plot_average_score_trend(
            df=df_state,
            category=selected_category,
            year_range=year_range,
            state=selected_state,
        )

        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

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

        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

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
        st.dataframe(result_sub_df, width="stretch")
        csv_bytes = result_sub_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_bytes, file_name="trends_data.csv", mime="text/csv")


# ---------------------------------------------------------
# Placeholder Tabs (Next Phases / Teammate Modules)
# ---------------------------------------------------------
# Geo, Univariate placeholders (phases yet to come)
for idx, tab_name in enumerate(["Geo", "Univariate"], start=1):
    with tabs[idx]:
        st.markdown(f"#### {tab_name} Visualizations")
        st.info(f"The {tab_name} questions from Annexure A will be activated in the next step.")


# ---------------------------------------------------------
# TAB 4: Bivariate Analysis & Correlation (Annexure A §4)
# ---------------------------------------------------------
with tabs[3]:  # "Bivariate"

    BIVARIATE_QUESTIONS = [
        "Q1 · How strongly are RPC scores associated with overall scores?",
        "Q2 · How does Perception (PR) relate to overall scores?",
        "Q3 · Which NIRF parameter has the strongest correlation with overall score?",
        "Q4 · How are TLR and RPC scores related across institutions?",
        "Q5 · Has the correlation between research and overall scores changed (2017–2025)?",
        "⚡ Interactive Two-Variable Explorer (Custom attributes & regression)",
    ]

    col_bq, col_bp = st.columns([1.6, 1.1])

    with col_bq:
        biv_question = st.selectbox(
            "Select Bivariate Question / Feature",
            options=BIVARIATE_QUESTIONS,
            index=0,
            key="biv_question_select",
        )

    # ── Q1: RPC vs Overall Score ──────────────────────────────────────────────
    if biv_question == BIVARIATE_QUESTIONS[0]:
        with col_bp:
            years_available = sorted(df_full["year"].dropna().unique().tolist())
            biv_year = st.selectbox(
                "Year",
                options=years_available,
                index=len(years_available) - 1,
                key="biv_q1_year",
            )
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                corr_choice = st.radio(
                    "Method",
                    options=["Pearson", "Spearman"],
                    horizontal=True,
                    key="biv_q1_corr",
                )
            with col_c2:
                top_outliers = st.slider("Residual outliers to label", 2, 8, 4, key="biv_q1_outliers")

        fig, takeaway = plot_scatter_marginals(
            df=df_full,
            x_param="RPC",
            year=biv_year,
            category=selected_category,
            state=None if selected_state == "All states" else selected_state,
            corr_type=corr_choice.lower(),
            top_n_outliers=top_outliers,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f"""<div class="takeaway-card">
            <span class="takeaway-highlight">Takeaway:</span> {takeaway}
            </div>""",
            unsafe_allow_html=True,
        )
        with st.expander("🔍 View underlying data & residual details", expanded=False):
            show_cols = ["name", "state", "rank", "score", "RPC", "TLR", "GO", "OI", "PR", "rank_band"]
            show_cols = [c for c in show_cols if c in df_full.columns]
            sub = df_full[(df_full["year"] == biv_year) & (df_full["category"] == selected_category)]
            if selected_state != "All states":
                sub = sub[sub["state"] == selected_state]
            st.dataframe(sub[show_cols].dropna(subset=["RPC", "score"]).sort_values("rank"), use_container_width=True)

    # ── Q2: Perception (PR) vs Overall Score ──────────────────────────────────
    elif biv_question == BIVARIATE_QUESTIONS[1]:
        with col_bp:
            years_available = sorted(df_full["year"].dropna().unique().tolist())
            biv_year = st.selectbox(
                "Year",
                options=years_available,
                index=len(years_available) - 1,
                key="biv_q2_year",
            )
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                corr_choice = st.radio(
                    "Method",
                    options=["Pearson", "Spearman"],
                    horizontal=True,
                    key="biv_q2_corr",
                )
            with col_c2:
                top_outliers = st.slider("Residual outliers to label", 2, 8, 4, key="biv_q2_outliers")

        fig, takeaway = plot_scatter_marginals(
            df=df_full,
            x_param="PR",
            year=biv_year,
            category=selected_category,
            state=None if selected_state == "All states" else selected_state,
            corr_type=corr_choice.lower(),
            top_n_outliers=top_outliers,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f"""<div class="takeaway-card">
            <span class="takeaway-highlight">Takeaway:</span> {takeaway}
            </div>""",
            unsafe_allow_html=True,
        )
        with st.expander("💡 Statistical Insight: Pearson vs Spearman Divergence", expanded=False):
            st.markdown("""
**Why do Pearson and Spearman diverge significantly for Perception (PR)?**
- **Perception (PR)** has an extremely skewed distribution: most institutions receive very low PR scores (near 0 to 10), while a select few elite institutions (IITs, IISc) receive near-perfect 100s.
- **Pearson r** is sensitive to extreme values and measures purely linear association.
- **Spearman ρ** assesses monotonic ranking order and is robust to extreme skew.
- The gap between linear and monotonic correlation clearly demonstrates the non-linear "superstar effect" of institutional perception in NIRF.
""")

    # ── Q3: Strongest parameter correlation with overall score ────────────────
    elif biv_question == BIVARIATE_QUESTIONS[2]:
        with col_bp:
            q3_view = st.radio(
                "Display Mode",
                options=["Correlation vs Official Weight", "5×5 Cross-Correlation Heatmap", "Show Both Side-by-Side"],
                horizontal=False,
                key="biv_q3_view",
            )

        if q3_view == "Correlation vs Official Weight":
            fig_bar, takeaway_bar = plot_correlation_vs_weight(
                df=df_full,
                category=selected_category,
                year_range=year_range,
                state=None if selected_state == "All states" else selected_state,
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                f"""<div class="takeaway-card">
                <span class="takeaway-highlight">Takeaway:</span> {takeaway_bar}
                </div>""",
                unsafe_allow_html=True,
            )
        elif q3_view == "5×5 Cross-Correlation Heatmap":
            fig_heat, takeaway_heat = plot_param_correlation_heatmap(
                df=df_full,
                category=selected_category,
                year_range=year_range,
                state=None if selected_state == "All states" else selected_state,
            )
            st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                f"""<div class="takeaway-card">
                <span class="takeaway-highlight">Takeaway:</span> {takeaway_heat}
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                fig_bar, takeaway_bar = plot_correlation_vs_weight(
                    df=df_full,
                    category=selected_category,
                    year_range=year_range,
                    state=None if selected_state == "All states" else selected_state,
                )
                st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
            with col_g2:
                fig_heat, takeaway_heat = plot_param_correlation_heatmap(
                    df=df_full,
                    category=selected_category,
                    year_range=year_range,
                    state=None if selected_state == "All states" else selected_state,
                )
                st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

            st.markdown(
                f"""<div class="takeaway-card">
                <span class="takeaway-highlight">Key Insights:</span> {takeaway_bar} {takeaway_heat}
                </div>""",
                unsafe_allow_html=True,
            )

        with st.expander("📐 Weights & Correlation Reference Table", expanded=False):
            st.markdown("""
| NIRF Parameter | Full Name | Official Weight | Role in Overall Score |
|---|---|---|---|
| **TLR** | Teaching, Learning & Resources | **30%** | Faculty-student ratio, financial resources |
| **RPC** | Research and Professional Practice | **30%** | Publications, citations, patent metrics |
| **GO** | Graduation Outcomes | **20%** | Placement rate, median salary, PhD graduates |
| **OI** | Outreach & Inclusivity | **10%** | Diversity, women percentage, regional diversity |
| **PR** | Perception | **10%** | Academic and employer peer survey |

*Notice that PR (10% weight) and RPC (30% weight) exhibit outsized predictive correlation with the overall score, while OI operates nearly independently.*
""")

    # ── Q4: TLR vs RPC (Quadrant Map + Category Lollipop) ─────────────────────
    elif biv_question == BIVARIATE_QUESTIONS[3]:
        with col_bp:
            q4_mode = st.radio(
                "Analysis Perspective",
                options=["Quadrant Map (Median Crosshairs)", "Discipline Breakdown (Lollipop by Category)", "Show Both"],
                horizontal=False,
                key="biv_q4_mode",
            )
            years_available = sorted(df_full["year"].dropna().unique().tolist())
            biv_year = st.selectbox(
                "Year",
                options=years_available,
                index=len(years_available) - 1,
                key="biv_q4_year",
            )

        if q4_mode == "Quadrant Map (Median Crosshairs)":
            fig_quad, takeaway_quad = plot_tlr_rpc_quadrants(
                df=df_full,
                year=biv_year,
                category=selected_category,
                state=None if selected_state == "All states" else selected_state,
                top_n_labels=6,
            )
            st.plotly_chart(fig_quad, use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                f"""<div class="takeaway-card">
                <span class="takeaway-highlight">Takeaway:</span> {takeaway_quad}
                </div>""",
                unsafe_allow_html=True,
            )
        elif q4_mode == "Discipline Breakdown (Lollipop by Category)":
            fig_lolli, takeaway_lolli = plot_tlr_rpc_by_category(
                df=df_full,
                year=biv_year,
            )
            st.plotly_chart(fig_lolli, use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                f"""<div class="takeaway-card">
                <span class="takeaway-highlight">Takeaway:</span> {takeaway_lolli}
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            fig_quad, takeaway_quad = plot_tlr_rpc_quadrants(
                df=df_full,
                year=biv_year,
                category=selected_category,
                state=None if selected_state == "All states" else selected_state,
                top_n_labels=5,
            )
            st.plotly_chart(fig_quad, use_container_width=True, config={"displayModeBar": False})

            fig_lolli, takeaway_lolli = plot_tlr_rpc_by_category(
                df=df_full,
                year=biv_year,
            )
            st.plotly_chart(fig_lolli, use_container_width=True, config={"displayModeBar": False})

            st.markdown(
                f"""<div class="takeaway-card">
                <span class="takeaway-highlight">Viva Synthesis:</span> {takeaway_quad}<br><br>{takeaway_lolli}
                </div>""",
                unsafe_allow_html=True,
            )

    # ── Q5: Multi-parameter Correlation Trends Over Time ──────────────────────
    elif biv_question == BIVARIATE_QUESTIONS[4]:
        with col_bp:
            q5_view = st.radio(
                "Visualization Type",
                options=["Trajectory Lines with 95% CI Band", "Parameter × Year Heatmap Strip", "Show Both"],
                horizontal=False,
                key="biv_q5_view",
            )

        if q5_view == "Trajectory Lines with 95% CI Band":
            fig_lines, takeaway_lines = plot_param_corr_over_years(
                df=df_full,
                category=selected_category,
                state=None if selected_state == "All states" else selected_state,
            )
            st.plotly_chart(fig_lines, use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                f"""<div class="takeaway-card">
                <span class="takeaway-highlight">Takeaway:</span> {takeaway_lines}
                </div>""",
                unsafe_allow_html=True,
            )
        elif q5_view == "Parameter × Year Heatmap Strip":
            fig_strip, _ = plot_param_year_heatmap(
                df=df_full,
                category=selected_category,
                state=None if selected_state == "All states" else selected_state,
            )
            st.plotly_chart(fig_strip, use_container_width=True, config={"displayModeBar": False})
        else:
            fig_lines, takeaway_lines = plot_param_corr_over_years(
                df=df_full,
                category=selected_category,
                state=None if selected_state == "All states" else selected_state,
            )
            st.plotly_chart(fig_lines, use_container_width=True, config={"displayModeBar": False})

            fig_strip, _ = plot_param_year_heatmap(
                df=df_full,
                category=selected_category,
                state=None if selected_state == "All states" else selected_state,
            )
            st.plotly_chart(fig_strip, use_container_width=True, config={"displayModeBar": False})

            st.markdown(
                f"""<div class="takeaway-card">
                <span class="takeaway-highlight">Takeaway:</span> {takeaway_lines}
                </div>""",
                unsafe_allow_html=True,
            )

        with st.expander("📐 Methodology: Fisher z-transformation & CI", expanded=False):
            st.markdown("""
- Each annual correlation $r$ is computed over the institutions participating in that year.
- The 95% Confidence Interval is calculated using the **Fisher $z$-transformation**:
  $$z = \\operatorname{arctanh}(r), \\quad \\operatorname{SE} = \\frac{1}{\\sqrt{n - 3}}$$
  $$z_{\\text{CI}} = z \\pm 1.96 \\cdot \\operatorname{SE}, \\quad r_{\\text{CI}} = \\tanh(z_{\\text{CI}})$$
- When confidence bands are non-overlapping between years, the shift in correlation represents a statistically significant change in ranking dynamics.
""")

    # ── Interactive Two-Variable Explorer ────────────────────────────────────
    elif biv_question == BIVARIATE_QUESTIONS[5]:
        numeric_options = [
            "score", "rank", "TLR", "RPC", "GO", "OI", "PR",
            "rank_change", "score_change_yoy", "param_gap", "volatility",
        ]
        available_num = [c for c in numeric_options if c in df_full.columns]

        with col_bp:
            col_x, col_y = st.columns(2)
            with col_x:
                var_x = st.selectbox("X Variable", options=available_num, index=available_num.index("RPC") if "RPC" in available_num else 0, key="exp_var_x")
            with col_y:
                var_y = st.selectbox("Y Variable", options=available_num, index=available_num.index("score") if "score" in available_num else 1, key="exp_var_y")

            years_available = ["All Years"] + sorted(df_full["year"].dropna().unique().tolist())
            selected_yr_exp = st.selectbox("Year", options=years_available, index=len(years_available) - 1, key="exp_year_sel")
            yr_filter = None if selected_yr_exp == "All Years" else int(selected_yr_exp)

            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                show_lowess = st.checkbox("Include LOWESS curve", value=True, key="exp_lowess")
            with col_opt2:
                all_inst_names = ["None"] + sorted(df_full["name"].dropna().unique().tolist())
                hi_inst = st.selectbox("Highlight institution", options=all_inst_names, index=0, key="exp_hi_inst")
                highlight_query = None if hi_inst == "None" else hi_inst

        fig_exp, stats_dict, auto_sentence = plot_explorer(
            df=df_full,
            x_var=var_x,
            y_var=var_y,
            year=yr_filter,
            category=selected_category,
            state=None if selected_state == "All states" else selected_state,
            show_lowess=show_lowess,
            highlight_inst=highlight_query,
        )

        # Statistical Metrics Summary Bar
        m_c1, m_c2, m_c3, m_c4, m_c5 = st.columns(5)
        with m_c1:
            st.metric("Pearson r", f"{stats_dict['pearson_r']:.3f}" if stats_dict['pearson_r'] is not None else "N/A")
        with m_c2:
            st.metric("Spearman ρ", f"{stats_dict['spearman_r']:.3f}" if stats_dict['spearman_r'] is not None else "N/A")
        with m_c3:
            st.metric("R² (Variance)", f"{stats_dict['r_squared']:.3f}" if stats_dict['r_squared'] is not None else "N/A")
        with m_c4:
            st.metric("OLS Slope", f"{stats_dict['slope']:.4f}" if stats_dict['slope'] is not None else "N/A")
        with m_c5:
            st.metric("Sample Size (n)", f"{stats_dict['n']}")

        st.plotly_chart(fig_exp, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            f"""<div class="takeaway-card">
            <span class="takeaway-highlight">Automated Deduction:</span> {auto_sentence}
            </div>""",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# TAB 5: Multivariate Analysis (Annexure A §5)
# ---------------------------------------------------------
with tabs[4]:
    render_multivariate_tab(df_state, df_full=df_full)

# ---------------------------------------------------------
# TAB 6: Distribution Analysis (Annexure A §6)
# ---------------------------------------------------------
with tabs[5]:
    render_distribution_tab(df_state, df_full=df_full)

# ---------------------------------------------------------
# TAB 7: Transitions Visualizations (Annexure A §7)
# ---------------------------------------------------------
with tabs[6]:
    st.markdown("#### Transitions Visualizations")
    st.info("The Transitions questions from Annexure A will be activated in the next step.")

# ---------------------------------------------------------
# TAB 8: Outlier Detection & Performance Gaps (Annexure A §8)
# ---------------------------------------------------------
with tabs[7]:
    OUTLIER_VIEWS = [
        "1. Connected Trajectory: High Research vs. Overall Score",
        "2. Research vs. Teaching Gap (TLR vs. RPC Parity)",
        "3. Score Volatility (Steep Vertical Drops & Climbs)",
        "4. High Parameter, Low Rank Anomaly",
        "5. Ranking Volatility (Box Plot Spread & Sequence Trajectories)",
    ]

    col_oq, col_op = st.columns([1.6, 1.1])
    with col_oq:
        outlier_selection = st.radio(
            "Select Outlier Analysis Method",
            options=OUTLIER_VIEWS,
            index=0,
            horizontal=False,
            key="outlier_view_radio",
        )

    # 1. Connected Scatter Plot (Research vs Overall Trajectory)
    if outlier_selection == OUTLIER_VIEWS[0]:
        with col_op:
            top_n_insts = st.slider("Number of outlier institutions to track", min_value=3, max_value=10, value=5, key="outlier_q1_topn")
            inst_options = get_institutions_list(df_full, selected_category)
            custom_inst_pick = st.multiselect(
                "Optional: Override with specific institutions",
                options=inst_options,
                default=[],
                key="outlier_q1_custom",
                help="Leave empty to automatically track the institutions with the highest research-to-overall score residuals.",
            )

        fig, takeaway = plot_connected_scatter_research_outliers(
            df=df_full,
            category=selected_category,
            state=None if selected_state == "All states" else selected_state,
            top_n=top_n_insts,
            selected_institutions=custom_inst_pick if custom_inst_pick else None,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f"""<div class="takeaway-card">
            <span class="takeaway-highlight">Takeaway:</span> {takeaway}
            </div>""",
            unsafe_allow_html=True,
        )
        with st.expander("🔍 View underlying trajectory data", expanded=False):
            sub_q1 = df_full[df_full["category"] == selected_category]
            if selected_state != "All states":
                sub_q1 = sub_q1[sub_q1["state"] == selected_state]
            show_cols = ["year", "name", "state", "rank", "score", "RPC", "TLR"]
            st.dataframe(sub_q1[show_cols].sort_values(["name", "year"]), use_container_width=True)

    # 2. Research vs Teaching Gap
    elif outlier_selection == OUTLIER_VIEWS[1]:
        with col_op:
            years_available = sorted(df_full["year"].dropna().unique().tolist())
            q2_year = st.selectbox(
                "Select Year",
                options=years_available,
                index=len(years_available) - 1,
                key="outlier_q2_year",
            )
            q2_top_n = st.slider("Outliers to annotate", 3, 12, 6, key="outlier_q2_topn")

        fig, takeaway = plot_research_vs_teaching_gap(
            df=df_full,
            year=q2_year,
            category=selected_category,
            state=None if selected_state == "All states" else selected_state,
            top_n_outliers=q2_top_n,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f"""<div class="takeaway-card">
            <span class="takeaway-highlight">Takeaway:</span> {takeaway}
            </div>""",
            unsafe_allow_html=True,
        )
        with st.expander("🔍 View Research vs. Teaching comparison table", expanded=False):
            sub_q2 = df_full[(df_full["year"] == q2_year) & (df_full["category"] == selected_category)].copy()
            if selected_state != "All states":
                sub_q2 = sub_q2[sub_q2["state"] == selected_state]
            sub_q2["RPC_minus_TLR"] = (sub_q2["RPC"] - sub_q2["TLR"]).round(2)
            show_cols = ["name", "state", "rank", "score", "RPC", "TLR", "RPC_minus_TLR"]
            st.dataframe(sub_q2[show_cols].sort_values("RPC_minus_TLR", ascending=False), use_container_width=True)

    # 3. Score Volatility (Line Chart)
    elif outlier_selection == OUTLIER_VIEWS[2]:
        with col_op:
            q3_top_n = st.slider("Top volatile institutions", 3, 10, 6, key="outlier_q3_topn")
            inst_options = get_institutions_list(df_full, selected_category)
            q3_custom_inst = st.multiselect(
                "Optional: Select specific institutions",
                options=inst_options,
                default=[],
                key="outlier_q3_custom",
            )

        fig, takeaway = plot_score_volatility_trends(
            df=df_full,
            category=selected_category,
            state=None if selected_state == "All states" else selected_state,
            top_n=q3_top_n,
            selected_institutions=q3_custom_inst if q3_custom_inst else None,
            year_range=year_range,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f"""<div class="takeaway-card">
            <span class="takeaway-highlight">Takeaway:</span> {takeaway}
            </div>""",
            unsafe_allow_html=True,
        )
        with st.expander("🔍 View largest YoY score shifts", expanded=False):
            sub_q3 = df_full[df_full["category"] == selected_category]
            if selected_state != "All states":
                sub_q3 = sub_q3[sub_q3["state"] == selected_state]
            show_cols = ["year", "name", "state", "rank", "score", "score_change_yoy"]
            st.dataframe(sub_q3[show_cols].dropna(subset=["score_change_yoy"]).sort_values("score_change_yoy", ascending=False), use_container_width=True)

    # 4. High Parameter, Low Rank Anomaly
    elif outlier_selection == OUTLIER_VIEWS[3]:
        with col_op:
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                selected_param = st.selectbox(
                    "Parameter to inspect",
                    options=["RPC", "TLR", "GO", "OI", "PR"],
                    index=0,
                    key="outlier_q4_param",
                )
            with col_p2:
                years_available = sorted(df_full["year"].dropna().unique().tolist())
                q4_year = st.selectbox(
                    "Year",
                    options=years_available,
                    index=len(years_available) - 1,
                    key="outlier_q4_year",
                )
            q4_top_n = st.slider("Anomaly callouts to label", 3, 10, 6, key="outlier_q4_topn")

        fig, takeaway = plot_high_param_low_rank(
            df=df_full,
            year=q4_year,
            category=selected_category,
            state=None if selected_state == "All states" else selected_state,
            param=selected_param,
            top_n_outliers=q4_top_n,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f"""<div class="takeaway-card">
            <span class="takeaway-highlight">Takeaway:</span> {takeaway}
            </div>""",
            unsafe_allow_html=True,
        )
        with st.expander("🔍 View high-parameter anomaly data", expanded=False):
            sub_q4 = df_full[(df_full["year"] == q4_year) & (df_full["category"] == selected_category)]
            if selected_state != "All states":
                sub_q4 = sub_q4[sub_q4["state"] == selected_state]
            candidate_cols = ["name", "state", "rank", "score", selected_param, "TLR", "RPC", "GO", "OI", "PR"]
            show_cols = list(dict.fromkeys([c for c in candidate_cols if c in df_full.columns]))
            st.dataframe(sub_q4[show_cols].sort_values("rank", ascending=False), use_container_width=True)

    # 5. Ranking Volatility (Dual Options)
    elif outlier_selection == OUTLIER_VIEWS[4]:
        with col_op:
            vol_mode_choice = st.radio(
                "Volatility Representation",
                options=["Box Plot (Rank Distribution Spread)", "Line Chart (Trajectory Fluctuations)"],
                horizontal=False,
                key="outlier_q5_mode",
            )
            q5_top_n = st.slider("Top volatile institutions to display", 4, 15, 8, key="outlier_q5_topn")

        chosen_mode = "boxplot" if "Box Plot" in vol_mode_choice else "line"
        fig, takeaway = plot_ranking_volatility(
            df=df_full,
            category=selected_category,
            state=None if selected_state == "All states" else selected_state,
            top_n=q5_top_n,
            mode=chosen_mode,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f"""<div class="takeaway-card">
            <span class="takeaway-highlight">Takeaway:</span> {takeaway}
            </div>""",
            unsafe_allow_html=True,
        )
        with st.expander("🔍 View ranking standard deviation & historical range", expanded=False):
            sub_q5 = df_full[df_full["category"] == selected_category]
            if selected_state != "All states":
                sub_q5 = sub_q5[sub_q5["state"] == selected_state]
            vol_summary = (
                sub_q5.groupby(["institution_id", "name", "state"])["rank"]
                .agg(std_dev="std", best_rank="min", worst_rank="max", years_ranked="count")
                .reset_index()
                .sort_values(by="std_dev", ascending=False)
            )
            vol_summary["std_dev"] = vol_summary["std_dev"].round(2)
            st.dataframe(vol_summary, use_container_width=True)

with tabs[8]:
    render_explore_tab(df_full, df_filtered)

with tabs[9]:
    st.markdown("#### Ask (Level 3 - Natural Language Query)")
    st.info("The Ask tab (Level 3 natural language query interface) is staged for Phase 3.")



