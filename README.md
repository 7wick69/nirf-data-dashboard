# NIRF Interactive Analytics Dashboard

A modern, high-performance interactive analytics dashboard and intelligence platform for the **National Institutional Ranking Framework (NIRF)** rankings (2017–2025), built with **Streamlit**, **Plotly**, and **Pandas/NumPy**, with an integrated local LLM natural language analytics assistant.

Designed with a sleek, minimal, monochromatic dark theme optimized for publication-grade visualization and exploratory analysis.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.11+
- (Optional for AI Chatbot) [Ollama](https://ollama.com/) with `qwen2.5` installed locally

### 2. Installation
```bash
git clone git@github.com:7wick69/nirf-data-dashboard.git
cd nirf-data-dashboard

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Dashboard
```bash
streamlit run app.py
```
Opens in your browser at **http://localhost:8501**.

### 4. Run the Test Suite
```bash
PYTHONPATH=. pytest                    # Full test suite (332 tests)
PYTHONPATH=. pytest tests/test_transitions.py -v   # Transitions module
PYTHONPATH=. pytest tests/test_univariate.py -v    # Univariate module
```

### 5. Run Data Cleaning Pipeline
```bash
python data_prep.py                        # Uses data/clean/data_nirf_clean2.csv
python data_prep.py path/to/custom.csv     # Custom dataset
```

---

## 📊 Dataset Overview

| Property | Value |
|:---|:---|
| **Source** | Official NIRF data (nirfindia.org, all categories 2017–2025) |
| **Location** | `data/clean/data_nirf_clean2.csv` |
| **Records** | 6,866 ranked institution-year records |
| **Institutions** | 923 unique higher education institutions |
| **Years** | 2017 – 2025 (9 consecutive annual ranking cycles) |
| **Categories** | 15 categories (Overall, Engineering, University, College, Management, Medical, Pharmacy, Law, Architecture, Dental, Research, Agriculture and Allied Sectors, Innovation, Open University, Skill University) |
| **Core Parameters** | TLR, RPC, GO, OI, PR (each scored 0–100) |

### NIRF Parameter Glossary
| Code | Parameter Name | Description |
|:---|:---|:---|
| **TLR** | Teaching, Learning & Resources | Faculty-student ratio, faculty qualifications, financial resources & utilization |
| **RPC** | Research and Professional Practice | Publications, citations, patent quality, research funding |
| **GO**  | Graduation Outcomes | Metric for university examinations and median graduate salary/placements |
| **OI**  | Outreach and Inclusivity | Regional diversity, women representation, economically/socially challenged students |
| **PR**  | Perception | Peer perception among employers, academics, and public survey |

---

## 📐 Project Architecture

```
nirf-data-dashboard/
├── app.py                       # Main Streamlit dashboard (10 integrated tabs)
├── univariate_app.py            # Standalone entry-point for univariate module
├── data_prep.py                 # Standalone data cleaning & normalization pipeline
│
├── core/
│   ├── data.py                  # Dataset loading, caching, and state/category filtering
│   ├── metrics.py               # Derived metrics engine (rank shifts, z-scores, bands)
│   ├── transitions.py           # Transition matrices, cohort tracking, and mobility logic
│   ├── schema.py                # Pydantic chart query specification schema
│   ├── safe_expr.py             # AST-constrained safe arithmetic expression evaluator
│   └── executor.py              # Query plan execution engine
│
├── charts/
│   ├── theme.py                 # Minimalist Plotly dark theme (#0f1013 canvas, zero-overlap legends)
│   ├── trends.py                # Section 1: Time-series trajectories, improvers, decliners
│   ├── geo.py                   # Section 2: India Choropleths & regional distribution
│   ├── univariate.py            # Section 3: Score & rank distributions, rolling KDE, donuts
│   ├── bivariate.py             # Section 4: Correlation scatter plots, regression lines
│   ├── multivariate.py          # Section 5: Radar charts, quadrant scatter, disparity dumbbells
│   ├── distribution_analysis.py # Section 6: Comprehensive parameter spread & skewness
│   ├── transitions.py           # Section 7: Mover slope graphs, transition matrices, Sankey
│   ├── outliers.py              # Section 8: Statistical anomaly detection (IQR, Z-score)
│   ├── explore.py               # Level 2 Explore workbench (Tableau-style shelf builder)
│   └── registry.py              # Central chart registry and metadata catalog
│
├── chatbot/
│   ├── chat_service.py          # Conversational assistant orchestrator & prompt builder
│   ├── dynamic_query.py         # Schema-constrained query planner with pandas execution
│   ├── query_parser.py          # Intent classification and entity/slot extraction
│   ├── visualization_engine.py  # Conversational Plotly chart generator
│   └── analytics_engine.py      # Natural language analytical computations
│
├── data/
│   └── clean/
│       ├── data_nirf_clean2.csv # Normalized primary NIRF dataset
│       └── institutions.parquet # Canonical institution master registry
│
└── tests/                       # Automated test suite (332 test cases)
    ├── test_trends.py           # Trend visualization tests
    ├── test_geo.py              # Geospatial analysis tests
    ├── test_univariate.py       # Univariate distribution tests
    ├── test_bivariate.py        # Bivariate correlation tests
    ├── test_multivariate_q5_3.py# Multivariate quadrant & dumbbell tests
    ├── test_transitions.py      # Transitions, slope charts, and mobility tests
    ├── test_outliers.py         # Anomaly detection tests
    ├── test_explore.py          # Explore workbench tests
    ├── test_dynamic_query.py    # Chatbot query planning tests
    └── ...
```

---

## 🖥️ Modules & Analytics Capabilities

The dashboard is structured into 10 dedicated analytical tabs:

### 1. Trends (`charts/trends.py`)
- **Rank Trajectories:** Multi-year line plots with inverted Y-axis (Rank 1 at the top).
- **Greatest Improvers & Decliners:** Top climbing and declining institutions across custom time horizons.
- **Score Progression:** Multi-year score trends benchmarked against All-India averages.
- **Consistently Top-Ranked:** Multi-line bump chart tracking perennial national Top-10 contenders with redundant geometric marker encoding.

### 2. Geospatial & Regional Analysis (`charts/geo.py`)
- Synchronized **side-by-side layout** pairing an interactive India Choropleth map with companion statistical visualizations:
  - **Top 100 Density:** State-wise concentration of India's elite institutions.
  - **Geographic Evolution:** Year-by-year expansion of higher education hubs.
  - **Net Representation Delta:** Diverging maps and slope charts showing emerging states.
  - **Regional Head-to-Head:** Direct comparative violin and box plots (e.g. Karnataka vs. Tamil Nadu).
  - **New Entrant Incubators:** Geographic distribution of newly appearing institutions ($Y_{first} > 2017$).

### 3. Univariate Distributions (`charts/univariate.py`)
- **Score Distribution:** Histograms with rolling-window smoothed density curve overlay, median, and mean reference lines.
- **Rank Distributions:** Histogram paired with 100% stacked tier breakdown (`1–10`, `11–50`, `51–100`, `101+`).
- **Summary Statistics:** Multi-year interactive data tables with mean, standard deviation, median, and IQR.
- **Quartile Share Donut:** Interactive donut chart displaying score boundaries and slice percentages.
- **Category Spreads & Violins:** Comparative box plots with standard deviation diamonds and parameter violin plots.

### 4. Bivariate Correlations (`charts/bivariate.py`)
- Parameter vs. Score and Parameter vs. Rank relationships across all 5 NIRF parameters.
- Regression trend lines, Pearson correlation coefficients, and state-level scatter breakdowns.

### 5. Multivariate Analysis (`charts/multivariate.py`)
- **Radar & Grouped Bar Charts:** Multi-dimensional parameter profiles for Top-5 institutions.
- **Teaching vs. Perception Quadrant Scatter ($X = \text{TLR}, Y = \text{PR}$):**
  - Isolates institutions with strong teaching fundamentals lagging in national perception.
  - Percentile-based threshold crosshairs to address the extreme right-skewness of NIRF Perception scores.
  - Shaded target quadrant with arrow callouts for top disparity outliers.
- **Ranked Disparity Companion:** Horizontal dumbbell charts and diverging gap bars sorted by percentile disparity ($\text{gap} = \text{TLR \%ile} - \text{PR \%ile}$).

### 6. Distribution Analysis (`charts/distribution_analysis.py`)
- Detailed skewness, kurtosis, and dispersion measurements across disciplines and cycles.
- Multi-parameter distribution comparisons highlighting evaluation asymmetries.

### 7. Transitions & Mobility (`charts/transitions.py`)
- **Mover Slope Graph:** Dedicated slope chart tracking institutions breaking across tiers (e.g., 51–100 into the Top 50):
  - Inverted Y-axis where upward slope indicates rank improvement.
  - Labeled threshold line (Rank 50.5) and shaded destination tier.
  - Redundant encoding (distinct qualitative hues + distinct geometric marker symbols).
  - Faint background cohort context lines displaying non-moving peers.
- **Destination Breakdown (100% Stacked Bar):** Cohort destination shares (Moved to Target, Stayed in Band, Moved to Other Bands, Dropped Out).
- **Transition Matrix Heatmap:** Cross-tabulation of origin vs. destination bands with zero cells blank.
- **Sankey Flow Diagram:** Multi-tier institutional mobility flows across consecutive years.

### 8. Outliers & Anomalies (`charts/outliers.py`)
- **IQR & Z-Score Detection:** Identifies performance outliers within each discipline and parameter.
- **Parameter Disparity:** Flags institutions with extreme gaps between their highest and lowest parameters (e.g. high research but near-zero perception).

### 9. Explore Workbench (`charts/explore.py`)
- Tableau-style self-service visualization builder:
  - **Shelves:** Columns (X), Rows (Y), Color, and Facet.
  - **Mark Types:** Automatic, Bar, Line, Scatter, Area, Box Plot, Histogram, and Table.
  - **Aggregation Controls:** `None (raw rows)`, `Mean`, `Median`, `Sum`, `Max`, `Min`, `Count`, `Std`.
  - **Built-in Presets:** Quick one-click configurations for common analytical inquiries.
  - **Live Data Inspection:** Expandable table showing the exact transformed DataFrame and Plotly figure specification.

### 10. Ask AI Assistant (`chatbot/`)
- Natural-language query interface powered by local Ollama models (e.g., `qwen2.5`) with automatic fallback to deterministic parsing.
- Safe execution architecture: LLM only plans a schema-bounded JSON query; all calculations, filtering, and aggregations are executed deterministically via pandas.
- Renders conversational summaries accompanied by interactive Plotly visualizations and data tables.

---

## 📌 Data Cleaning & Methodology

The data pipeline in `data_prep.py` and `core/data.py` enforces rigorous standardization:

1. **Rank-Band Handling:**
   NIRF publishes non-top ranks in bands (e.g., `"101–150"`, `"151–200"`).
   - `Rank_Original`: Preserves the exact published string.
   - `Rank_Numeric`: Parsed using the band **midpoint** (e.g., `125.0` for `101–150`) to avoid systematic bias toward lower or upper boundaries. Open-ended bands like `"301+"` use the lower bound (`301.0`).
   - `Is_Band_Rank`: Boolean flag marking banded ranks.
2. **Missing Values Policy:**
   Missing parameters (`TLR`, `RPC`, `GO`, `OI`, `PR`) are preserved as `NaN` rather than imputed with zero, preventing artificial skewing of statistical means, medians, and correlations.
3. **State & Name Normalization:**
   Institution and state names are normalized via dictionary mappings (`Orissa → Odisha`, standardized IIT/NIT naming conventions) to maintain historical continuity across all 9 cycles.
4. **Safe Arithmetic Derivations:**
   Custom expressions and metric calculations use Python's `ast` module (`core/safe_expr.py`), restricting execution strictly to safe arithmetic without `eval()` or arbitrary execution.

---

## 🧪 Testing & Verification

The test suite covers **332 test cases** across all modules with zero failures:

```bash
PYTHONPATH=. pytest
============================== 332 passed in 34.5s ==============================
```

| Test File | Focus Area |
|:---|:---|
| `test_trends.py` | Trajectory logic, rank delta calculations, bump charts |
| `test_geo.py` | Choropleth mapping, state filtering, new entrant detection |
| `test_univariate.py` | Distribution metrics, KDE smoothing, quartile donuts |
| `test_bivariate.py` | Correlation matrices, scatter traces, regression fit |
| `test_multivariate_q5_3.py` | TLR vs. PR quadrant scatter, percentile ranking, dumbbell charts |
| `test_transitions.py` | Mover slope charts, transition matrices, Sankey flows |
| `test_outliers.py` | Statistical outlier formulas (IQR, Z-score) |
| `test_explore.py` & `test_explore_fuzz.py` | Tableau workbench shelf mapping, aggregations, fuzz testing |
| `test_dynamic_query.py` | Chatbot schema validation, query planning, fallback mechanisms |
| `test_metrics.py` | Derived metrics engine, year-over-year deltas, volatility |
