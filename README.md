# NIRF Interactive Analytics Dashboard

A modern, high-performance interactive analytics dashboard for NIRF (National Institutional Ranking Framework) rankings (2017–2025), built with **Streamlit**, **Plotly**, and **Pandas/PyArrow**.

Designed with a sleek, minimal, monochromatic dark theme matching the project wireframe.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.11+
- Git

### 2. Installation
```bash
# Clone the repository
git clone git@github.com:7wick69/nirf-data-dashboard.git
cd nirf-data-dashboard

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 3. Launch the Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Dataset Overview

The dashboard uses the curated, multi-year NIRF dataset:
- **Location:** `data/clean/data_nirf_clean2.csv`
- **Coverage:** 9 Ranking Years (**2017 to 2025**)
- **Records:** **6,866 ranked rows** across **923 institutions**
- **15 Ranking Categories:** `Overall`, `Engineering`, `Management`, `University`, `College`, `Medical`, `Pharmacy`, `Law`, `Architecture`, `Dental`, `Research`, `Agriculture and Allied Sectors`, `Innovation`, `Open University`, `Skill University`.
- **Core Parameters:**
  - `TLR`: Teaching, Learning & Resources
  - `RPC`: Research and Professional Practice
  - `GO`: Graduation Outcomes
  - `OI`: Outreach and Inclusivity
  - `PR`: Perception

---

## 📐 Project Architecture

```
nirf-data-dashboard/
├── app.py                     # Streamlit frontend with minimal dark UI & reactive filters
├── core/
│   ├── data.py                # Dataset loading, caching, column standardizer
│   ├── metrics.py             # Derived metrics engine (YoY delta, bands, volatility, z-scores)
│   ├── schema.py              # Pydantic v2 chart query spec schema
│   └── safe_expr.py           # Safe AST arithmetic evaluator (blocks arbitrary execution)
├── charts/
│   ├── theme.py               # Minimalist Plotly dark theme (charcoal canvas, electric blue)
│   ├── trends.py              # Trend visualizations (Rank trajectory, Delta, Scores, Top 10)
│   └── geo.py                 # Geospatial & Regional visualizations (Top 100 states, Choropleth, Delta, Boxplot, New Entrants)
├── data/
│   └── clean/
│       ├── data_nirf_clean2.csv   # Clean primary NIRF dataset
│       └── institutions.parquet   # Canonical institutions registry
├── pipeline/
│   └── validate.py            # Data integrity and quality validator
├── tests/
│   ├── test_metrics.py        # Derived metric tests
│   └── test_trends.py         # Trend chart & takeaway calculation tests
├── requirements.txt           # Project dependencies
├── .env.example               # Configuration template for future LLM integration
└── README.md
```

---

## 📈 Implemented Visualizations (`charts/trends.py`)

1. **Rank Trajectory Over Time:**
   - Plots an institution's historical ranking with an **inverted Y-axis** (Rank 1 at the top).
   - Dynamic hovertooltips displaying Rank, Score, and Year-over-Year movement.
   - **Automated Takeaway:** e.g., *"rank improved from 2 to 1 (+1 places) over 9 years"* for IIT Madras.
2. **Biggest Improvers & Decliners:**
   - Horizontal delta comparison identifying the largest climbers and decliners between user-selected years.
3. **Average & Median Score Trajectory:**
   - Multi-metric trendline comparing mean and median scores over time.
4. **Consistently Top-10 Institutions:**
   - Multi-institution trajectory tracking perennial contenders in the top tier.

---

## 🗺️ Geospatial & Regional Visualizations (`charts/geo.py`)

Each research question features a synchronized **side-by-side dashboard layout** pairing an interactive India Choropleth map with a complementary statistical chart:

1. **Top States in Top 100:**
   - **Left:** India Choropleth map showing state-wise Top 100 density.
   - **Right:** Horizontal bar chart sorted by institution count with counts and percentage shares.
2. **Geographical Distribution Evolution:**
   - **Left:** Year-by-year India Choropleth map with year slider (2017–2025).
   - **Right:** State-wise stacked bar chart showing multi-year evolution (2017–2025).
3. **State Representation Improvement:**
   - **Left:** Diverging Choropleth map (Emerald green for growth, Red for contraction).
   - **Right:** Net representation delta bar chart or multi-state slope chart.
4. **Regional Comparison (Karnataka vs. Tamil Nadu):**
   - **Left:** Geographic focus map highlighting Karnataka and Tamil Nadu with median ranks.
   - **Right:** Comparative Boxplot or Violin Plot with inverted Y-axis (#1 at top).
5. **Newly Appearing Institutions by State:**
   - **Left:** New institutional entrants density Choropleth map ($Y_{first} > 2017$).
   - **Right:** State-wise new entrant incubator bar chart.

---

- **`rank_change`**: $Rank_{prev} - Rank_{curr}$ (Positive indicates an improved rank; e.g. from 8 to 2 is `+6`).
- **`score_change_yoy`**: Score difference compared to previous available year.
- **`rank_band`**: Standardized ranking brackets (`1-10`, `11-50`, `51-100`, `101+`).
- **`years_in_top10`**: Number of years ranked within the Top 10 for that category.
- **`volatility`**: Standard deviation ($\sigma$) of rank across available years.
- **`param_gap`**: $\max - \min$ across the 5 parameter scores.
- **`best_param` / `worst_param`**: Relative strengths and weaknesses per institution.
- **`zscore` / `pct_rank`**: Category-year normalized metrics.

---

## 🧪 Testing & Validation

Run unit tests:
```bash
PYTHONPATH=. pytest
```

Run dataset validation:
```bash
python pipeline/validate.py
```

---

## 🎓 Viva Preparation Notes

1. **Why Inverted Y-Axis?** In ranking visualizations, rank 1 represents the highest achievement. Inverting the axis ensures that upward slopes visually correspond to improvement.
2. **Missing Values Policy:** Missing parameters are preserved as `NaN` rather than imputed with `0`, preventing artificial skewing of statistical means and correlations.
3. **Safe Evaluation:** Custom derivations (`core/safe_expr.py`) parse syntax trees using Python's `ast` module, whitelisting only basic arithmetic operators and column identifiers to prevent arbitrary code execution vulnerabilities.
