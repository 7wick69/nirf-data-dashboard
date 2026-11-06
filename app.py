"""
app.py

NIRF Interactive Analytics Dashboard
Minimal, monocolor dark theme matching the design wireframe.
"""

from html import escape

import streamlit as st
import pandas as pd
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


def render_takeaway(takeaway):
    if not takeaway:
        return

    safe_text = escape(takeaway)
    st.markdown(
        f"""
        <div class="takeaway-card">
            <span class="takeaway-highlight">
                Takeaway line:
            </span>
            {safe_text}
        </div>
        """,
        unsafe_allow_html=True
    )


def render_assistant_message(message):
    st.markdown(message["content"])

    if message.get("figure") is not None:
        st.plotly_chart(
            message["figure"],
            use_container_width=True,
            config={"displayModeBar": False}
        )

    render_takeaway(message.get("takeaway"))

    result = message.get("result")
    if result:
        with st.expander(
            "🔍 View underlying data table",
            expanded=False
        ):
            st.dataframe(
                pd.DataFrame(result.get("data", [])),
                use_container_width=True,
                hide_index=True
            )
            st.json(result.get("statistics", {}))


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
# Placeholder Tabs (Next Phases)
# ---------------------------------------------------------
for idx, tab_name in enumerate(["Geo", "Univariate", "Bivariate", "Multivariate", "Distribution", "Transitions", "Outliers"], start=1):
    with tabs[idx]:
        st.markdown(f"#### {tab_name} Visualizations")
        st.info(f"Phase 1 focused on `trends.py` validation. The {tab_name} questions from Annexure A will be activated in the next step.")

with tabs[8]:
    st.markdown("#### Explore (Level 2)")
    st.info("Interactive query and custom chart builder will be connected in Phase 2.")

with tabs[9]:
    st.markdown("#### 🎓 NIRF AI Analyst")
    st.caption(
        "Ask questions about NIRF rankings using "
        "natural language. Powered by local Qwen."
    )

    if "nirf_ask_messages" not in st.session_state:
        st.session_state.nirf_ask_messages = []
    if "nirf_ask_context" not in st.session_state:
        st.session_state.nirf_ask_context = None

    col_title, col_reset = st.columns([5, 1])
    with col_reset:
        if st.button("Clear chat", key="nirf_clear_chat"):
            st.session_state.nirf_ask_messages = []
            st.session_state.nirf_ask_context = None
            st.rerun()

    for message in st.session_state.nirf_ask_messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_assistant_message(message)
            else:
                st.markdown(message["content"])

    if question := st.chat_input("Ask a question about NIRF rankings"):
        st.session_state.nirf_ask_messages.append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.markdown(question)

        response = process_chat_message(
            question,
            st.session_state.nirf_ask_context,
        )
        st.session_state.nirf_ask_context = response.get("context")
        assistant_message = {
            "role": "assistant",
            "content": response["answer"],
            "figure": response.get("figure"),
            "takeaway": response.get("takeaway"),
            "result": response.get("result"),
        }
        st.session_state.nirf_ask_messages.append(assistant_message)

        with st.chat_message("assistant"):
            render_assistant_message(assistant_message)
