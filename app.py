"""NIRF Interactive Analytics Dashboard.

Release-polished Streamlit UI. Analytics and chat processing remain in their
existing modules; this file handles presentation, loading feedback and session UI.
"""

from html import escape

import pandas as pd
import streamlit as st

from chatbot.chat_service import process_chat_message
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


# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NIRF Explorer | Analytics Workspace",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Theme and small UI details
# -----------------------------------------------------------------------------
DARK_CSS = """
<style>
:root {
    --nirf-bg: #0f1013;
    --nirf-panel: #15171d;
    --nirf-panel-2: #1a1d24;
    --nirf-border: #282d37;
    --nirf-text: #e6edf3;
    --nirf-muted: #9aa4b2;
    --nirf-blue: #58a6ff;
    --nirf-blue-strong: #388bfd;
    --nirf-green: #3fb950;
    --nirf-red: #f85149;
}

html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 Helvetica, Arial, sans-serif;
}
.stApp {
    background: var(--nirf-bg);
    color: var(--nirf-text);
}
.block-container {
    max-width: 1500px;
    padding-top: 1.7rem;
    padding-bottom: 3rem;
}
section[data-testid="stSidebar"] {
    background: #14161a;
    border-right: 1px solid #23272f;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #f0f3f6;
    font-weight: 650;
    letter-spacing: -0.02em;
}

/* Inputs */
div[data-baseweb="select"] > div,
.stTextInput input,
.stNumberInput input {
    background: var(--nirf-panel-2) !important;
    border: 1px solid #2d323b !important;
    border-radius: 7px !important;
    color: var(--nirf-text) !important;
}
div[data-baseweb="select"] svg { fill: var(--nirf-muted) !important; }

/* App header */
.nirf-hero {
    padding: 4px 0 16px 0;
    margin-bottom: 12px;
    border-bottom: 1px solid var(--nirf-border);
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

.nirf-eyebrow {
    color: var(--nirf-blue) !important;
    font-size: .68rem;
    font-weight: 800;
    letter-spacing: .16em;
    text-transform: uppercase;
    margin-bottom: 7px;
}
.nirf-subtitle {
    color: var(--nirf-muted) !important;
    font-size: .88rem;
    line-height: 1.55;
    margin-top: 5px;
}
.nirf-takeaway {
    background: #16181e;
    border: 1px solid #262a33;
    border-left: 3px solid var(--nirf-blue-strong);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 14px 0 20px;
    color: #c9d1d9;
    font-size: .84rem;
    line-height: 1.55;
}
.nirf-takeaway-label {
    color: var(--nirf-blue);
    font-weight: 700;
    margin-right: 5px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 7px;
    background: transparent;
    border-bottom: 1px solid #23272f;
    padding-bottom: 4px;
}
.stTabs [data-baseweb="tab"] {
    height: 39px;
    background: transparent;
    border: none;
    color: #8b949e;
    font-size: .82rem;
    font-weight: 550;
    padding: 0 13px;
    border-radius: 7px 7px 0 0;
    transition: background .15s ease, color .15s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #f0f3f6;
    background: #1a1d24;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: var(--nirf-blue);
    background: #1a1d24;
    border-bottom: 2px solid var(--nirf-blue-strong);
}

/* Metrics and data tables */
div[data-testid="stMetric"] {
    background: var(--nirf-panel);
    border: 1px solid var(--nirf-border);
    padding: 12px 15px;
    border-radius: 9px;
}
[data-testid="stDataFrame"] {
    border: 1px solid var(--nirf-border);
    border-radius: 8px;
    overflow: hidden;
}
div[data-testid="stPlotlyChart"] {
    border-radius: 9px;
    overflow: hidden;
}

/* Chat surface */
[data-testid="stChatMessage"] {
    border: 1px solid transparent;
    border-radius: 12px;
    padding: 13px 15px;
    margin-bottom: 11px;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: #171d28;
    border-color: #253247;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: #14161b;
    border-color: #262a33;
}
.nirf-empty {
    background: linear-gradient(145deg, #171a21, #121419);
    border: 1px solid #2a303b;
    border-radius: 13px;
    padding: 24px;
    margin: 12px 0 18px;
}
.nirf-empty-title {
    color: var(--nirf-text) !important;
    font-size: 1.12rem;
    font-weight: 700;
    margin-bottom: 5px;
}
.nirf-empty-copy {
    color: var(--nirf-muted) !important;
    font-size: .84rem;
    line-height: 1.6;
}
.nirf-suggestion {
    background: #191d25;
    border: 1px solid #2b3340;
    border-radius: 8px;
    padding: 10px 12px;
    min-height: 70px;
    color: #c9d1d9;
    font-size: .78rem;
    line-height: 1.45;
}

/* Single native Streamlit loading indicator. */
[data-testid="stSpinner"] * { color: var(--nirf-muted) !important; }
[data-testid="stAlert"] { border-radius: 9px !important; }

/* Buttons */
.stButton > button {
    border: 1px solid #303744;
    border-radius: 8px;
    background: #1a1d24;
    color: var(--nirf-text);
    font-weight: 600;
    transition: border-color .15s ease, background .15s ease;
}
.stButton > button:hover {
    border-color: #5277a8;
    background: #202633;
    color: white;
}

/* Responsive layout and motion accessibility */
@media (max-width: 760px) {
    .block-container { padding-top: 1.1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; overflow-x: auto; }
    .stTabs [data-baseweb="tab"] { padding: 0 9px; font-size: .75rem; }
    [data-testid="stChatMessage"] { padding: 10px; }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: .01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: .01ms !important;
        scroll-behavior: auto !important;
    }
}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def render_takeaway(takeaway: str | None) -> None:
    if not takeaway:
        return
    st.markdown(
        f'<div class="nirf-takeaway"><span class="nirf-takeaway-label">Takeaway</span>{escape(str(takeaway))}</div>',
        unsafe_allow_html=True,
    )


def render_assistant_message(message: dict) -> None:
    content = message.get("content") or "The request finished without a response. Please try again."
    st.markdown(content)

    if message.get("figure") is not None:
        st.plotly_chart(
            message["figure"],
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )

    render_takeaway(message.get("takeaway"))

    result = message.get("result")
    if result:
        with st.expander("🔍 View underlying data and statistics", expanded=False):
            records = result.get("data") or []
            if records:
                st.dataframe(
                    pd.DataFrame(records),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("The analysis returned no matching records.")
            statistics = result.get("statistics") or {}
            if statistics:
                st.markdown("**Statistics**")
                st.json(statistics)
            notes = result.get("notes") or []
            if notes:
                st.markdown("**Method notes**")
                for note in notes:
                    st.caption(f"• {note}")


def render_trend_result(fig, takeaway) -> None:
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )
    render_takeaway(takeaway)


def render_empty_chat_state() -> None:
    st.markdown(
        """
        <div class="nirf-empty">
            <div class="nirf-empty-title">Explore India's NIRF rankings with natural language</div>
            <div class="nirf-empty-copy">
                Ask about institutional rankings, yearly trends, state representation,
                score distributions, parameter relationships or ranking movements.
                Your question is analysed locally; Python performs the calculations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    suggestions = [
        "Show the top 10 Engineering institutions in 2025.",
        "Compare RPC and TLR across Engineering institutions in 2025.",
        "Which states have the most institutions in the Overall top 100 in 2025?",
    ]
    cols = st.columns(3)
    for col, suggestion in zip(cols, suggestions):
        with col:
            st.markdown(
                f'<div class="nirf-suggestion">{escape(suggestion)}</div>',
                unsafe_allow_html=True,
            )


# -----------------------------------------------------------------------------
# Dataset loading with caching
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_dataset(custom_file=None):
    if custom_file is not None:
        if custom_file.name.lower().endswith(".parquet"):
            frame = pd.read_parquet(custom_file)
        else:
            frame = pd.read_csv(custom_file)
        from core.metrics import compute_derived_metrics
        return compute_derived_metrics(frame)
    return load_nirf_data()


# -----------------------------------------------------------------------------
# Sidebar and global filters
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📊 NIRF Explorer")
    st.caption("Interactive ranking and performance analysis")

    with st.expander("📁 Data source", expanded=False):
        uploaded_file = st.file_uploader(
            "Upload NIRF Parquet / CSV",
            type=["parquet", "csv"],
            help="Upload a compatible NIRF dataset or use the cleaned default dataset.",
        )
        if uploaded_file is not None:
            st.success(f"Loaded: {uploaded_file.name}")
        else:
            st.caption("Using the default cleaned NIRF dataset.")

    with st.spinner("Loading NIRF dataset..."):
        df_full = get_dataset(uploaded_file)

    if df_full.empty:
        st.error("The selected dataset contains no records.")
        st.stop()

    min_yr, max_yr = get_year_range(df_full)
    year_range = st.slider(
        "Years",
        min_value=int(min_yr),
        max_value=int(max_yr),
        value=(int(min_yr), int(max_yr)),
        step=1,
        help="Global year range for the dashboard views.",
    )

    categories = get_categories(df_full)
    selected_category = st.selectbox(
        "Category",
        options=categories,
        index=0,
        help="Global NIRF category filter for dashboard views.",
    )

    # Detect if currently selected topic belongs to Geospatial & Regional Analysis
    active_topic = str(st.session_state.get("active_tab", "Trends"))
    is_geo_active = (
        "geo" in active_topic.lower()
        or active_topic in ["Geo", "Geospatial", "Geospatial & Regional", "Geospatial and Regional Analysis"]
    )

    all_states = ["All states"] + get_states(df_full)
    selected_state = st.selectbox(
        "State",
        options=all_states,
        index=0 if is_geo_active else 0,
        disabled=is_geo_active,
        help=(
            "State filter is disabled specifically for Geospatial and Regional Analysis because all states are analyzed simultaneously across India."
            if is_geo_active
            else "Global state filter for dashboard views."
        ),
        key="global_state_selector",
    )

    st.divider()
    st.caption("Dataset snapshot")
    metric_cols = st.columns(2)
    metric_cols[0].metric("Records", f"{len(df_full):,}")
    metric_cols[1].metric("Categories", f"{len(categories):,}")
    st.caption(f"Available years: {min_yr}–{max_yr} · {len(get_states(df_full))} states/UTs")
    st.caption("Global filters apply to the analytical dashboard tabs. Ask queries use the category/year stated in your question.")

# -----------------------------------------------------------------------------
# Apply global filters
# -----------------------------------------------------------------------------
df_state = df_full[df_full["state"] == selected_state].copy() if selected_state != "All states" else df_full.copy()

df_filtered = df_state[
    (df_state["year"] >= year_range[0])
    & (df_state["year"] <= year_range[1])
    & (df_state["category"] == selected_category)
].copy()


# -----------------------------------------------------------------------------
# Main header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="nirf-hero">
        <div class="nirf-eyebrow">National Institutional Ranking Framework</div>
        <h1 style="margin:0; color:#f0f3f6; letter-spacing:-.035em;">NIRF Explorer</h1>
        <div class="nirf-subtitle">
            Explore institutional performance, ranking movements and regional trends
            across NIRF categories and years.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Navigation
# -----------------------------------------------------------------------------
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
        inst_options = get_institutions_list(df_state, selected_category)

        if not inst_options:
            st.info(f"No institutions found in {selected_state} for category '{selected_category}'.")
        else:
            default_inst = "Indian Institute of Technology Madras"
            if "trends_institution_select" in st.session_state and st.session_state["trends_institution_select"] not in inst_options:
                del st.session_state["trends_institution_select"]
            default_idx = inst_options.index(default_inst) if default_inst in inst_options else 0

            if "trends_institution_select" in st.session_state and st.session_state["trends_institution_select"] not in inst_options:
                st.session_state["trends_institution_select"] = inst_options[default_idx]

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
    render_multivariate_tab(df_state, df_full=df_full, category=selected_category)

# ---------------------------------------------------------
# TAB 6: Distribution Analysis (Annexure A §6)
# ---------------------------------------------------------
with tabs[5]:
    render_distribution_tab(df_state, df_full=df_full, category=selected_category)


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

# -----------------------------------------------------------------------------
# TAB 10: Ask — local NLP chatbot
# -----------------------------------------------------------------------------
with tabs[9]:
    st.markdown("#### 🎓 NIRF AI Analyst")
    st.caption("Ask questions in natural language. Local Qwen interprets the query; Python performs the analysis.")

    if "nirf_ask_messages" not in st.session_state:
        st.session_state.nirf_ask_messages = []
    if "nirf_ask_context" not in st.session_state:
        st.session_state.nirf_ask_context = None

    col_title, col_reset = st.columns([5, 1])
    with col_title:
        st.caption("Questions can cover rankings, trends, comparisons, distributions, correlations, geography and outliers.")
    with col_reset:
        if st.button("Clear chat", key="nirf_clear_chat", use_container_width=True):
            st.session_state.nirf_ask_messages = []
            st.session_state.nirf_ask_context = None
            st.rerun()

    for message in st.session_state.nirf_ask_messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_assistant_message(message)
            else:
                st.markdown(message.get("content", ""))

    question = st.chat_input("Ask a question about NIRF rankings...")
    if not st.session_state.nirf_ask_messages and not question:
        render_empty_chat_state()

    if question:
        st.session_state.nirf_ask_messages.append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.markdown(question)

        # One native Streamlit spinner: no duplicate custom loading animation.
        try:
            with st.spinner("NIRF AI Analyst is processing your question..."):
                response = process_chat_message(
                    question,
                    st.session_state.nirf_ask_context,
                )
        except Exception as exc:  # Keep the UI usable if a backend exception escapes.
            response = {
                "status": "error",
                "answer": "The analysis could not be completed. Please try again or simplify the question.",
                "technical_error": str(exc),
                "context": st.session_state.nirf_ask_context,
                "result": None,
                "figure": None,
                "takeaway": None,
            }

        st.session_state.nirf_ask_context = response.get(
            "context", st.session_state.nirf_ask_context
        )
        assistant_message = {
            "role": "assistant",
            "content": response.get("answer") or "No response was returned. Please try again.",
            "status": response.get("status", "error"),
            "figure": response.get("figure"),
            "takeaway": response.get("takeaway"),
            "result": response.get("result"),
            "technical_error": response.get("technical_error"),
        }
        st.session_state.nirf_ask_messages.append(assistant_message)

        with st.chat_message("assistant"):
            render_assistant_message(assistant_message)
            if assistant_message.get("technical_error"):
                with st.expander("Technical details", expanded=False):
                    st.code(assistant_message["technical_error"])
