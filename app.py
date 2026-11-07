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
    plot_biggest_improvers_decliners,
    plot_average_score_trend,
    plot_consistently_top10,
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

    all_states = ["All states"] + get_states(df_full)
    selected_state = st.selectbox(
        "State",
        options=all_states,
        index=0,
        help="Global state filter for dashboard views.",
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
df_filtered = df_full[
    (df_full["year"] >= year_range[0])
    & (df_full["year"] <= year_range[1])
    & (df_full["category"] == selected_category)
]
if selected_state != "All states":
    df_filtered = df_filtered[df_filtered["state"] == selected_state]


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


# -----------------------------------------------------------------------------
# TAB 1: Trends
# -----------------------------------------------------------------------------
with tabs[0]:
    st.markdown("#### Trend analysis")
    st.caption(f"{selected_category} · {year_range[0]}–{year_range[1]}")
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
            key="trend_question",
        )

    if question == "How has the rank of a selected institution changed over time?":
        inst_options = get_institutions_list(df_full, selected_category)
        if not inst_options:
            st.info("No institutions are available for the selected category.")
        else:
            default_inst = "Indian Institute of Technology Madras"
            default_idx = inst_options.index(default_inst) if default_inst in inst_options else 0
            with col_param:
                selected_inst = st.selectbox(
                    "Institution",
                    options=inst_options,
                    index=default_idx,
                    key="trend_institution",
                )
            fig, takeaway = plot_institution_rank_trend(
                df=df_full,
                institution_name=selected_inst,
                category=selected_category,
                year_range=year_range,
            )
            render_trend_result(fig, takeaway)

            with st.expander("🔍 View underlying data table", expanded=False):
                name_col = "name" if "name" in df_full.columns else "display_name"
                rank_col = "rank" if "rank" in df_full.columns else "rank_num"
                available_cols = [
                    col for col in ["year", rank_col, "score", "rank_change", "TLR", "RPC", "GO", "OI", "PR"]
                    if col in df_full.columns
                ]
                inst_df = df_full[
                    (df_full[name_col] == selected_inst)
                    & (df_full["category"] == selected_category)
                    & (df_full["year"] >= year_range[0])
                    & (df_full["year"] <= year_range[1])
                ][available_cols]
                st.dataframe(inst_df.sort_values("year"), use_container_width=True, hide_index=True)

    elif question == "Who are the biggest improvers and decliners?":
        with col_param:
            top_n = st.slider("Top N institutions", min_value=3, max_value=10, value=5, key="trend_top_n")
        fig, takeaway = plot_biggest_improvers_decliners(
            df=df_full,
            category=selected_category,
            start_year=year_range[0],
            end_year=year_range[1],
            top_n=top_n,
        )
        render_trend_result(fig, takeaway)

    elif question == "How has the average score changed over time?":
        fig, takeaway = plot_average_score_trend(
            df=df_full,
            category=selected_category,
            year_range=year_range,
        )
        render_trend_result(fig, takeaway)

    elif question == "Which institutions are consistently in the top 10?":
        with col_param:
            min_years = st.slider(
                "Minimum years in Top 10",
                min_value=2,
                max_value=max(2, max_yr - min_yr + 1),
                value=min(5, max(2, max_yr - min_yr + 1)),
                key="trend_min_years",
            )
        fig, takeaway = plot_consistently_top10(
            df=df_full,
            category=selected_category,
            min_years=min_years,
        )
        render_trend_result(fig, takeaway)


# -----------------------------------------------------------------------------
# Guided analysis tabs: point users to the complete Ask/query engine
# -----------------------------------------------------------------------------
GUIDED_TABS = [
    (1, "Geographical analysis", "Explore state representation, top-100 institution counts and regional changes.", "Which states have the most institutions in the Overall top 100 in 2025?"),
    (2, "Univariate analysis", "Explore score and rank distributions, quartiles and descriptive statistics.", "Show the distribution of Engineering scores in 2025."),
    (3, "Bivariate analysis", "Measure relationships between NIRF parameters and institutional scores.", "How are RPC and TLR related across Engineering institutions in 2025?"),
    (4, "Multivariate analysis", "Compare institutions across multiple NIRF parameters and identify performance gaps.", "Which institutions have high research scores but relatively low teaching scores?"),
    (5, "Distribution analysis", "Compare score distributions across years, states and institution groups.", "Show the median and interquartile range of Engineering scores from 2017 to 2025."),
    (6, "Ranking transitions", "Investigate rank-band movement, top-10 entries and ranking retention.", "What percentage of Engineering institutions remained in the same rank band between 2020 and 2025?"),
    (7, "Outlier analysis", "Identify unusual score patterns, rank volatility and parameter gaps.", "Which Engineering institutions have unusually high ranking volatility?") ,
]

for tab_index, title, description, example in GUIDED_TABS:
    with tabs[tab_index]:
        st.markdown(f"#### {title}")
        st.markdown(f"_{description}_")
        st.info("These analyses are available through the Ask tab. A dedicated guided control panel for this section is a future UI enhancement.")
        st.markdown("**Try asking:**")
        st.code(example, language=None)


with tabs[8]:
    st.markdown("#### Explore (Level 2)")
    st.info("Interactive query and custom chart builder will be connected in Phase 2.")


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
