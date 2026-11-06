"""
charts/bivariate.py — v2 (full revision)

Bivariate Analysis & Correlation — Annexure A §4 (all 5 questions) + Explorer.

Q1  RPC vs Overall Score      — scatter + marginal histograms, rank-band colours,
                                 OLS trendline, residual-outlier auto-labels,
                                 Pearson / Spearman toggle, r & R² in title.
Q2  PR vs Overall Score        — same layout (Pearson vs Spearman diverge ~0.15 for PR).
Q3  Strongest parameter        — correlation-vs-weight grouped bar chart
                                 + annotated 5×5 cross-correlation heatmap.
Q4  TLR vs RPC                 — median-crosshair quadrant scatter with named quadrants
                                 + lollipop of TLR–RPC r by discipline category.
Q5  Correlations over time     — multi-line r per year (one line/param) with 95 % CI bands
                                 + alternative parameter × year heatmap strip.
Explorer  free-form scatter    — any two columns, OLS + optional LOWESS, Pearson/Spearman,
                                 R², slope, n, auto sentence, institution highlight.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess_fn
    _HAS_LOWESS = True
except ImportError:
    _HAS_LOWESS = False

from charts.theme import (
    ACCENT_GREEN,
    ACCENT_RED,
    DARK_BG,
    GRID_COLOR,
    LINE_COLOR,
    MUTED_TEXT,
    PANEL_BG,
    TEXT_COLOR,
    apply_theme,
)

# ── Official NIRF parameter weights (out of 100) ──────────────────────────────
NIRF_WEIGHTS: Dict[str, int] = {"TLR": 30, "RPC": 30, "GO": 20, "OI": 10, "PR": 10}

PARAM_LABELS: Dict[str, str] = {
    "TLR": "Teaching, Learning & Resources (TLR)",
    "RPC": "Research & Professional Practice (RPC)",
    "GO":  "Graduation Outcomes (GO)",
    "OI":  "Outreach & Inclusivity (OI)",
    "PR":  "Perception (PR)",
}

RANK_BAND_ORDER = ["1-10", "11-50", "51-100", "101+"]
RANK_BAND_COLORS: Dict[str, str] = {
    "1-10":    "#60a5fa",   # sky-blue
    "11-50":   "#34d399",   # emerald
    "51-100":  "#fbbf24",   # amber
    "101+":    "#94a3b8",   # slate
    "Unranked":"#475569",
}

# One colour per parameter (multi-line charts, lollipops)
PARAM_COLORS: Dict[str, str] = {
    "TLR": "#60a5fa",
    "RPC": "#34d399",
    "GO":  "#a78bfa",
    "OI":  "#fb923c",
    "PR":  "#f472b6",
}

# ── Private helpers ────────────────────────────────────────────────────────────

def _shorten(name: str) -> str:
    """Abbreviate verbose institution names for compact chart labels."""
    for long, short in [
        ("Indian Institute of Technology",          "IIT"),
        ("Indian Institute of Science",             "IISc"),
        ("National Institute of Technology",        "NIT"),
        ("Indian Institute of Management",          "IIM"),
        ("Indian Institute of Information Technology", "IIIT"),
        ("All India Institute of Medical Sciences", "AIIMS"),
        ("Jawaharlal Nehru University",             "JNU"),
        ("University of",                           "Univ of"),
        ("Institute of Technology",                 "Inst. of Tech."),
    ]:
        name = name.replace(long, short)
    return name


def _hex_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Convert '#rrggbb' to 'rgba(r,g,b,alpha)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _ols(x: np.ndarray, y: np.ndarray, n_pts: int = 120) -> Tuple:
    """Return (xs, ys, slope, intercept) of fitted OLS line, or empties on failure."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.array([]), np.array([]), np.nan, np.nan
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    xs = np.linspace(float(x[mask].min()), float(x[mask].max()), n_pts)
    return xs, slope * xs + intercept, float(slope), float(intercept)


def _pearson(x: pd.Series, y: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    """NaN-safe Pearson r. Returns (r, p-value) or (None, None)."""
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 4:
        return None, None
    r, p = scipy_stats.pearsonr(df["x"].to_numpy(), df["y"].to_numpy())
    return round(float(r), 4), round(float(p), 4)


def _spearman(x: pd.Series, y: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    """NaN-safe Spearman ρ. Returns (rho, p-value) or (None, None)."""
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 4:
        return None, None
    r, p = scipy_stats.spearmanr(df["x"].to_numpy(), df["y"].to_numpy())
    return round(float(r), 4), round(float(p), 4)


def _fisher_ci(r: float, n: int) -> Tuple[float, float]:
    """95 % CI for Pearson r via Fisher z-transform."""
    if n < 4 or not np.isfinite(r):
        return float("nan"), float("nan")
    z  = np.arctanh(np.clip(r, -0.9999, 0.9999))
    se = 1.0 / np.sqrt(max(n - 3, 1))
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def _residual_outliers(
    x: np.ndarray,
    y: np.ndarray,
    names: np.ndarray,
    top_n: int = 4,
) -> List[Tuple[int, float, str]]:
    """
    Fit OLS, compute residuals = actual − predicted.
    Returns [(row_index, residual, name)] for top_n overperformers (pos)
    and top_n underperformers (neg).
    """
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4:
        return []
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    residuals = y - (slope * x + intercept)
    valid_idx   = np.where(mask)[0]
    valid_res   = residuals[mask]
    valid_names = names[mask]

    out: List[Tuple[int, float, str]] = []
    for li in np.argsort(valid_res)[::-1][:top_n]:       # top positive
        out.append((int(valid_idx[li]), round(float(valid_res[li]), 2), str(valid_names[li])))
    for li in np.argsort(valid_res)[:top_n]:              # top negative
        out.append((int(valid_idx[li]), round(float(valid_res[li]), 2), str(valid_names[li])))
    return out


def _base_layout(
    fig: go.Figure,
    title: str,
    xl: str = "",
    yl: str = "",
    h: int = 640,
    legend: bool = True,
) -> go.Figure:
    """Apply the shared dark layout to a simple (non-subplots) figure."""
    fig = apply_theme(fig, title=title, xaxis_title=xl, yaxis_title=yl, show_legend=legend)
    fig.update_layout(
        height=h,
        margin=dict(l=72, r=52, t=80, b=72),
        font=dict(family="Inter, system-ui, sans-serif"),
    )
    return fig


def _subplots_dark(fig: go.Figure, title_html: str, h: int = 700) -> go.Figure:
    """Apply the shared dark layout to a make_subplots figure."""
    fig.update_layout(
        title=dict(
            text=title_html,
            x=0.04, y=0.98, xanchor="left", yanchor="top",
            font=dict(size=15, color=TEXT_COLOR, family="Inter, system-ui, sans-serif"),
        ),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT_COLOR, family="Inter, system-ui, sans-serif", size=12),
        height=h + 35,
        margin=dict(l=72, r=52, t=90, b=110),
        barmode="overlay",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.16,
            xanchor="center",
            x=0.42,
            bgcolor="rgba(23, 25, 30, 0.92)",
            bordercolor=GRID_COLOR,
            borderwidth=1,
            font=dict(color=TEXT_COLOR, size=11),
            itemsizing="constant",
        ),
        hoverlabel=dict(
            bgcolor=PANEL_BG, bordercolor=GRID_COLOR,
            font=dict(color=TEXT_COLOR, size=12),
        ),
    )
    # Style all axes uniformly
    fig.update_xaxes(
        gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
        showline=True, linecolor=GRID_COLOR,
        tickfont=dict(size=11, color=MUTED_TEXT),
    )
    fig.update_yaxes(
        gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
        showline=True, linecolor=GRID_COLOR,
        tickfont=dict(size=11, color=MUTED_TEXT),
    )
    return fig


# ── Q1 & Q2: Scatter with marginal histograms ──────────────────────────────────

def plot_scatter_marginals(
    df: pd.DataFrame,
    x_param: str,
    year: int,
    category: str = "Overall",
    state: Optional[str] = None,
    corr_type: str = "pearson",   # "pearson" | "spearman"
    top_n_outliers: int = 4,
) -> Tuple[go.Figure, str]:
    """
    Scatter of x_param vs Overall Score with:
      - Marginal histograms on both axes (stacked by rank band)
      - OLS trendline (amber dashed)
      - Residual-outlier text labels (top_n_outliers above & below the OLS line)
      - Pearson r, Spearman ρ, R² displayed in subtitle
    Returns (figure, takeaway_string).
    """
    sub = df[(df["year"] == year) & (df["category"] == category)].copy()
    if state:
        sub = sub[sub["state"] == state]
    sub = sub.dropna(subset=[x_param, "score", "rank_band"]).reset_index(drop=True)

    if sub.empty or len(sub) < 3:
        fig = go.Figure()
        return _base_layout(fig, f"No data — {category} · {year}"), "Insufficient data."

    x_arr     = sub[x_param].to_numpy(dtype=float)
    y_arr     = sub["score"].to_numpy(dtype=float)
    names_arr = sub["name"].to_numpy()

    # Correlations
    pr, _  = _pearson(sub[x_param],  sub["score"])
    sr, _  = _spearman(sub[x_param], sub["score"])
    r2 = round(pr ** 2, 3) if pr is not None else None

    # OLS trendline
    xs_ols, ys_ols, slope, intercept = _ols(x_arr, y_arr)

    # Residual outliers
    outliers = _residual_outliers(x_arr, y_arr, names_arr, top_n_outliers)

    # ── Build 2 × 2 subplot grid ───────────────────────────────────────────────
    # Layout:
    #   (1,1) = top histogram (x-distribution)   | (1,2) empty
    #   (2,1) = main scatter                      | (2,2) right histogram (y-distribution)
    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[0.20, 0.80],
        column_widths=[0.80, 0.20],
        shared_xaxes=True,   # col-1 panels share x (top hist ↔ main scatter)
        shared_yaxes=True,   # row-2 panels share y (main scatter ↔ right hist)
        horizontal_spacing=0.012,
        vertical_spacing=0.010,
    )

    # ── Scatter traces — one per rank band (row=2, col=1) ─────────────────────
    for band in RANK_BAND_ORDER:
        bdf = sub[sub["rank_band"] == band]
        if bdf.empty:
            continue
        fig.add_trace(go.Scatter(
            x=bdf[x_param],
            y=bdf["score"],
            mode="markers",
            name=f"Rank {band}",
            marker=dict(
                color=RANK_BAND_COLORS.get(band, MUTED_TEXT),
                size=9, opacity=0.80,
                line=dict(color="#0a0c10", width=0.6),
            ),
            customdata=list(zip(
                bdf["name"].apply(_shorten),
                bdf["rank"].fillna(-1).astype(int),
            )),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                f"{x_param}: %{{x:.2f}}<br>"
                "Overall Score: %{y:.2f}<br>"
                "Rank: #%{customdata[1]}"
                "<extra></extra>"
            ),
        ), row=2, col=1)

    # ── OLS trendline (row=2, col=1) ──────────────────────────────────────────
    if len(xs_ols):
        fig.add_trace(go.Scatter(
            x=xs_ols, y=ys_ols,
            mode="lines",
            name="OLS fit",
            line=dict(color="#f59e0b", width=2.2, dash="dash"),
        ), row=2, col=1)

    # ── Residual-outlier labels as a text trace (row=2, col=1) ───────────────
    if outliers:
        seen = set()
        ol_x, ol_y, ol_text, ol_colors = [], [], [], []
        for (gi, resid, nm) in outliers:
            if gi in seen or gi >= len(sub):
                continue
            seen.add(gi)
            ol_x.append(float(sub.loc[gi, x_param]))
            ol_y.append(float(sub.loc[gi, "score"]))
            sign = f"+{resid:.1f}" if resid >= 0 else f"{resid:.1f}"
            ol_text.append(f"{_shorten(nm)} ({sign})")
            ol_colors.append(ACCENT_GREEN if resid >= 0 else ACCENT_RED)

        fig.add_trace(go.Scatter(
            x=ol_x, y=ol_y,
            mode="text",
            text=ol_text,
            textposition="top center",
            textfont=dict(size=8.5, color=TEXT_COLOR),
            showlegend=False,
            hoverinfo="skip",
        ), row=2, col=1)

    # ── Top histogram — x distribution (row=1, col=1) ────────────────────────
    for band in RANK_BAND_ORDER:
        bdf = sub[sub["rank_band"] == band]
        if bdf.empty:
            continue
        fig.add_trace(go.Histogram(
            x=bdf[x_param],
            name=f"Rank {band}",
            marker_color=RANK_BAND_COLORS.get(band, MUTED_TEXT),
            opacity=0.72, nbinsx=22, showlegend=False,
        ), row=1, col=1)

    # ── Right histogram — y distribution (row=2, col=2) ──────────────────────
    for band in RANK_BAND_ORDER:
        bdf = sub[sub["rank_band"] == band]
        if bdf.empty:
            continue
        fig.add_trace(go.Histogram(
            y=bdf["score"],
            name=f"Rank {band}",
            marker_color=RANK_BAND_COLORS.get(band, MUTED_TEXT),
            opacity=0.72, nbinsy=22, showlegend=False,
        ), row=2, col=2)

    # ── Title with r, ρ, R² ───────────────────────────────────────────────────
    r_str  = f"r = {pr:.3f}"  if pr is not None else "r = N/A"
    sp_str = f"ρ = {sr:.3f}"  if sr is not None else "ρ = N/A"
    r2_str = f"R² = {r2:.3f}" if r2 is not None else ""
    state_sfx = f" · {state}" if state else ""
    param_full = PARAM_LABELS.get(x_param, x_param)

    title_html = (
        f"{x_param} vs Overall Score — {category} · {year}{state_sfx}"
        f"<br><sup style='font-size:12px;color:{MUTED_TEXT}'>"
        f"Pearson {r_str} &nbsp;·&nbsp; Spearman {sp_str} &nbsp;·&nbsp; {r2_str} &nbsp;·&nbsp; n = {len(sub)}"
        f"</sup>"
    )

    fig = _subplots_dark(fig, title_html, h=720)

    # Axis labels — only on main scatter
    fig.update_xaxes(
        title_text=param_full,
        title_font=dict(color=MUTED_TEXT, size=12),
        row=2, col=1,
    )
    fig.update_yaxes(
        title_text="Overall NIRF Score",
        title_font=dict(color=MUTED_TEXT, size=12),
        row=2, col=1,
    )
    # Suppress marginal axis labels/titles
    fig.update_xaxes(showticklabels=False, title_text="", row=1, col=1)
    fig.update_yaxes(showticklabels=False, title_text="", row=1, col=1)
    fig.update_xaxes(showticklabels=False, title_text="", row=2, col=2)
    fig.update_yaxes(showticklabels=False, title_text="", row=2, col=2)

    # ── Auto takeaway ─────────────────────────────────────────────────────────
    if pr is not None and sr is not None:
        strength = (
            "very strong" if abs(pr) > 0.85 else
            "strong"      if abs(pr) > 0.65 else
            "moderate"    if abs(pr) > 0.40 else "weak"
        )
        direction = "positive" if pr > 0 else "negative"
        diff_note = ""
        if abs(pr - sr) > 0.08:
            diff_note = (
                f" Pearson and Spearman diverge ({pr:.2f} vs {sr:.2f}), "
                "suggesting a non-linear relationship — a few institutions near extremes "
                "are pulling the linear estimate up."
            )
        takeaway = (
            f"{x_param} shows a {strength} {direction} correlation with overall score "
            f"({r_str}, {sp_str}, {r2_str}) across {len(sub)} institutions "
            f"in {category} {year}.{diff_note}"
        )
    else:
        takeaway = "Insufficient data to compute correlations."

    return fig, takeaway


# ── Q3a: Correlation vs official weight (bar + diamond marker) ─────────────────

def plot_correlation_vs_weight(
    df: pd.DataFrame,
    category: str = "Overall",
    year_range: Optional[Tuple[int, int]] = None,
    state: Optional[str] = None,
) -> Tuple[go.Figure, str]:
    """
    Grouped horizontal chart showing, for each NIRF parameter:
      - Bar  → Pearson r with overall score (pooled across years)
      - ◆    → Official NIRF weight (normalised to 0–1 scale)
    Bars coloured per parameter. Reference lines at r = 0.4 (Moderate), 0.7 (Strong).
    """
    sub = df[df["category"] == category].copy()
    if year_range:
        sub = sub[(sub["year"] >= year_range[0]) & (sub["year"] <= year_range[1])]
    if state:
        sub = sub[sub["state"] == state]

    params = [p for p in ["TLR", "RPC", "GO", "OI", "PR"] if p in sub.columns]
    if sub.empty or not params:
        fig = go.Figure()
        return _base_layout(fig, "No data"), "Insufficient data."

    rows = []
    for p in params:
        r, _ = _pearson(sub[p], sub["score"])
        rows.append({
            "param": p,
            "label": PARAM_LABELS.get(p, p),
            "r": r if r is not None else 0.0,
            "weight_norm": NIRF_WEIGHTS.get(p, 0) / 100.0,
        })
    cdf = pd.DataFrame(rows).sort_values("r", ascending=True)

    fig = go.Figure()

    # Bars: correlation
    fig.add_trace(go.Bar(
        y=cdf["label"],
        x=cdf["r"],
        orientation="h",
        name="Pearson r (↔ overall score)",
        marker=dict(
            color=[PARAM_COLORS.get(row["param"], LINE_COLOR) for _, row in cdf.iterrows()],
            opacity=0.86,
            line=dict(color="#0a0c10", width=0.5),
        ),
        hovertemplate="<b>%{y}</b><br>Pearson r: %{x:.3f}<extra></extra>",
    ))

    # Diamonds: official weight
    fig.add_trace(go.Scatter(
        x=cdf["weight_norm"],
        y=cdf["label"],
        mode="markers",
        name="Official NIRF weight (normalised)",
        marker=dict(
            symbol="diamond", size=16, color="#f8fafc",
            line=dict(color="#374151", width=1.5),
        ),
        hovertemplate="<b>%{y}</b><br>NIRF weight: %{x:.0%}<extra></extra>",
    ))

    # Reference lines
    for xv, lab in [(0.4, "Moderate"), (0.7, "Strong"), (0.9, "Very strong")]:
        fig.add_vline(
            x=xv,
            line=dict(color=GRID_COLOR, width=1, dash="dot"),
            annotation_text=lab,
            annotation_font=dict(color=MUTED_TEXT, size=10),
            annotation_position="top",
        )

    yr_label = f"{year_range[0]}–{year_range[1]}" if year_range else "All Years"
    state_sfx = f" · {state}" if state else ""
    fig = _base_layout(
        fig,
        title=f"Parameter Correlation vs Official Weight — {category} · {yr_label}{state_sfx}",
        xl="Pearson r  /  Normalised weight (◆)",
        yl="",
        h=460,
        legend=True,
    )
    fig.update_xaxes(range=[0, 1.08])
    fig.update_layout(margin=dict(l=320, r=52, t=80, b=60))

    best  = cdf.iloc[-1]
    worst = cdf.iloc[0]
    has_pr  = "PR"  in cdf["param"].values
    has_rpc = "RPC" in cdf["param"].values
    note = (
        " PR and RPC both outrun their official 10 % and 30 % weights respectively — "
        "research output and institutional perception are the dominant predictors of NIRF rank."
        if has_pr and has_rpc else ""
    )
    takeaway = (
        f"'{best['param']}' has the highest correlation with overall score "
        f"(r = {best['r']:.3f}), while '{worst['param']}' is weakest (r = {worst['r']:.3f}).{note}"
    )
    return fig, takeaway


# ── Q3b: 5×5 parameter cross-correlation heatmap ─────────────────────────────

def plot_param_correlation_heatmap(
    df: pd.DataFrame,
    category: str = "Overall",
    year_range: Optional[Tuple[int, int]] = None,
    state: Optional[str] = None,
) -> Tuple[go.Figure, str]:
    """
    Annotated 5×5 heatmap of Pearson correlations among TLR, RPC, GO, OI, PR.
    Diverging blue–white–red colorscale centred at 0.
    """
    sub = df[df["category"] == category].copy()
    if year_range:
        sub = sub[(sub["year"] >= year_range[0]) & (sub["year"] <= year_range[1])]
    if state:
        sub = sub[sub["state"] == state]

    params = [p for p in ["TLR", "RPC", "GO", "OI", "PR"] if p in sub.columns]
    if len(params) < 2 or sub.empty:
        fig = go.Figure()
        return _base_layout(fig, "Insufficient data"), ""

    mat = sub[params].dropna(how="all").corr(method="pearson")
    z   = mat.values.tolist()
    n   = len(params)
    text = [[f"{mat.iat[i, j]:.2f}" for j in range(n)] for i in range(n)]

    fig = go.Figure(go.Heatmap(
        z=z, x=params, y=params,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=13, color="#f8fafc", family="Inter, monospace"),
        colorscale=[
            [0.0,  "#1e3a5f"],
            [0.35, "#2563eb"],
            [0.5,  "#f8fafc"],
            [0.65, "#ef4444"],
            [1.0,  "#7f1d1d"],
        ],
        zmid=0, zmin=-1, zmax=1,
        colorbar=dict(
            title=dict(text="r", font=dict(color=MUTED_TEXT, size=12)),
            thickness=14, len=0.82,
            tickfont=dict(color=MUTED_TEXT, size=10),
        ),
        hovertemplate="<b>%{y} ↔ %{x}</b><br>Pearson r = %{z:.3f}<extra></extra>",
    ))

    yr_label  = f"{year_range[0]}–{year_range[1]}" if year_range else "All Years"
    state_sfx = f" · {state}" if state else ""
    fig.update_layout(
        title=dict(
            text=f"Parameter Cross-Correlation Matrix — {category} · {yr_label}{state_sfx}",
            x=0.04, font=dict(size=14, color=TEXT_COLOR, family="Inter, system-ui, sans-serif"),
        ),
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font=dict(color=TEXT_COLOR, size=12),
        height=440,
        margin=dict(l=72, r=52, t=72, b=60),
        xaxis=dict(tickfont=dict(color=MUTED_TEXT, size=12)),
        yaxis=dict(tickfont=dict(color=MUTED_TEXT, size=12)),
    )

    # Takeaway from notable off-diagonal pairs
    rpc_pr  = round(float(mat.loc["RPC", "PR"]),  3) if "RPC" in mat.index and "PR"  in mat.columns else None
    rpc_tlr = round(float(mat.loc["RPC", "TLR"]), 3) if "RPC" in mat.index and "TLR" in mat.columns else None
    oi_avg  = round(float(mat.loc["OI"].drop("OI").mean()), 3) if "OI" in mat.index else None

    parts = []
    if rpc_pr  is not None: parts.append(f"RPC ↔ PR r = {rpc_pr:.2f} — research & reputation largely co-move.")
    if rpc_tlr is not None: parts.append(f"RPC ↔ TLR r = {rpc_tlr:.2f}.")
    if oi_avg  is not None: parts.append(f"OI is nearly independent of all others (avg r ≈ {oi_avg:.2f}).")
    msg = "  ".join(parts)
    return fig, msg


# ── Q4a: TLR vs RPC — median-crosshair quadrant scatter ──────────────────────

def plot_tlr_rpc_quadrants(
    df: pd.DataFrame,
    year: int,
    category: str = "Overall",
    state: Optional[str] = None,
    top_n_labels: int = 5,
) -> Tuple[go.Figure, str]:
    """
    Scatter TLR (x) vs RPC (y) with median crosshairs that create four named quadrants:
      Top-left   = Research-led  |  Top-right    = All-rounders
      Bottom-left = Developing   |  Bottom-right  = Teaching-led
    Off-diagonal institutions (top_n_labels each) are labelled automatically.
    """
    sub = df[(df["year"] == year) & (df["category"] == category)].copy()
    if state:
        sub = sub[sub["state"] == state]
    sub = sub.dropna(subset=["TLR", "RPC", "rank_band"]).reset_index(drop=True)

    if sub.empty:
        fig = go.Figure()
        return _base_layout(fig, f"No data — {category} · {year}"), "Insufficient data."

    med_tlr = float(sub["TLR"].median())
    med_rpc = float(sub["RPC"].median())
    pr_val, _ = _pearson(sub["TLR"], sub["RPC"])
    r_str = f"r = {pr_val:.3f}" if pr_val is not None else ""

    # Ranges for crosshair lines
    x_min = float(sub["TLR"].min()) - 1.5
    x_max = float(sub["TLR"].max()) + 1.5
    y_min = float(sub["RPC"].min()) - 1.5
    y_max = float(sub["RPC"].max()) + 1.5

    fig = go.Figure()

    # Scatter by rank band
    for band in RANK_BAND_ORDER:
        bdf = sub[sub["rank_band"] == band]
        if bdf.empty:
            continue
        fig.add_trace(go.Scatter(
            x=bdf["TLR"], y=bdf["RPC"],
            mode="markers",
            name=f"Rank {band}",
            marker=dict(
                color=RANK_BAND_COLORS.get(band, MUTED_TEXT),
                size=9, opacity=0.82,
                line=dict(color="#0a0c10", width=0.6),
            ),
            customdata=list(zip(
                bdf["name"].apply(_shorten),
                bdf["rank"].fillna(-1).astype(int),
                bdf["score"].round(2),
            )),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "TLR: %{x:.2f}<br>RPC: %{y:.2f}<br>"
                "Rank: #%{customdata[1]}<br>Score: %{customdata[2]:.2f}"
                "<extra></extra>"
            ),
        ))

    # OLS trendline
    xs_ols, ys_ols, _, _ = _ols(sub["TLR"].to_numpy(float), sub["RPC"].to_numpy(float))
    if len(xs_ols):
        fig.add_trace(go.Scatter(
            x=xs_ols, y=ys_ols,
            mode="lines",
            name=f"OLS fit ({r_str})",
            line=dict(color="#f59e0b", width=2.0, dash="dash"),
        ))

    # Median crosshairs
    fig.add_shape(type="line", x0=med_tlr, x1=med_tlr, y0=y_min, y1=y_max,
                  line=dict(color="#f8fafc", width=1.3, dash="dot"))
    fig.add_shape(type="line", x0=x_min, x1=x_max, y0=med_rpc, y1=med_rpc,
                  line=dict(color="#f8fafc", width=1.3, dash="dot"))

    # Median labels
    fig.add_annotation(x=med_tlr, y=y_max, text=f"TLR median = {med_tlr:.1f}",
                       showarrow=False, font=dict(size=9, color=MUTED_TEXT), yshift=6)
    fig.add_annotation(x=x_min, y=med_rpc, text=f"RPC median = {med_rpc:.1f}",
                       showarrow=False, font=dict(size=9, color=MUTED_TEXT),
                       xshift=-5, xanchor="right")

    # Quadrant labels
    _qpad = 2.5
    for (qx, qy, qlabel, col) in [
        (x_min + _qpad, y_max - _qpad, "Research-led",  "#a78bfa"),
        (x_max - _qpad, y_max - _qpad, "All-rounders",  ACCENT_GREEN),
        (x_min + _qpad, y_min + _qpad, "Developing",    MUTED_TEXT),
        (x_max - _qpad, y_min + _qpad, "Teaching-led",  "#fbbf24"),
    ]:
        fig.add_annotation(
            x=qx, y=qy, text=f"<b>{qlabel}</b>",
            showarrow=False, font=dict(size=11, color=col), opacity=0.70,
        )

    # Label off-diagonal institutions (research-led + teaching-led), by rank
    off_diag = sub[
        ((sub["TLR"] < med_tlr) & (sub["RPC"] > med_rpc)) |   # research-led
        ((sub["TLR"] > med_tlr) & (sub["RPC"] < med_rpc))      # teaching-led
    ].nsmallest(top_n_labels, "rank")

    for _, row in off_diag.iterrows():
        fig.add_annotation(
            x=float(row["TLR"]), y=float(row["RPC"]),
            text=_shorten(str(row["name"])),
            showarrow=True, arrowhead=2, arrowsize=0.8,
            arrowcolor=MUTED_TEXT, ax=22, ay=-22,
            font=dict(size=8.5, color=TEXT_COLOR),
            bgcolor=PANEL_BG, borderpad=2,
        )

    state_sfx = f" · {state}" if state else ""
    fig = _base_layout(
        fig,
        title=f"TLR vs RPC Quadrant Map — {category} · {year}{state_sfx}",
        xl="Teaching, Learning & Resources (TLR)",
        yl="Research & Professional Practice (RPC)",
        h=640,
    )
    fig.update_layout(
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[y_min, y_max]),
    )

    takeaway = (
        f"Median crosshairs (TLR = {med_tlr:.1f}, RPC = {med_rpc:.1f}) split institutions "
        f"into four quadrants. {len(off_diag)} off-diagonal institutions are labelled — these "
        f"show the most pronounced teaching–research imbalance. TLR–RPC correlation: {r_str}."
    )
    return fig, takeaway


# ── Q4b: TLR–RPC correlation by category — lollipop ──────────────────────────

def plot_tlr_rpc_by_category(
    df: pd.DataFrame,
    year: Optional[int] = None,
) -> Tuple[go.Figure, str]:
    """
    Lollipop chart: Pearson r(TLR, RPC) computed per NIRF discipline category.
    Colour-maps r value from red (low) through amber to green (high).
    Shows how the teaching–research coupling varies by field.
    """
    sub = df.copy()
    if year:
        sub = sub[sub["year"] == year]
    sub = sub.dropna(subset=["TLR", "RPC"])

    cats = sorted(sub["category"].dropna().unique().tolist())
    rows = []
    for cat in cats:
        cdf = sub[sub["category"] == cat]
        r, _ = _pearson(cdf["TLR"], cdf["RPC"])
        if r is not None and len(cdf) >= 4:
            rows.append({"category": cat, "r": r, "n": int(len(cdf))})

    if not rows:
        fig = go.Figure()
        return _base_layout(fig, "Insufficient data per category"), ""

    cdf_r = pd.DataFrame(rows).sort_values("r", ascending=True)

    fig = go.Figure()

    # Lollipop stems
    for _, row in cdf_r.iterrows():
        fig.add_shape(
            type="line",
            x0=0, x1=row["r"],
            y0=row["category"], y1=row["category"],
            line=dict(color=MUTED_TEXT, width=1.8),
        )

    # Lollipop heads — colour-coded by r
    fig.add_trace(go.Scatter(
        x=cdf_r["r"],
        y=cdf_r["category"],
        mode="markers",
        marker=dict(
            size=14,
            color=cdf_r["r"].tolist(),
            colorscale=[[0, ACCENT_RED], [0.5, "#fbbf24"], [1, ACCENT_GREEN]],
            cmin=-1, cmax=1,
            line=dict(color="#0a0c10", width=1),
            showscale=True,
            colorbar=dict(
                title=dict(text="r", font=dict(color=MUTED_TEXT, size=12)),
                thickness=12,
                tickfont=dict(color=MUTED_TEXT, size=10),
            ),
        ),
        customdata=list(zip(cdf_r["r"].round(3), cdf_r["n"])),
        hovertemplate="<b>%{y}</b><br>TLR–RPC r: %{customdata[0]:.3f}<br>n: %{customdata[1]}<extra></extra>",
        showlegend=False,
    ))

    fig.add_vline(x=0, line=dict(color=GRID_COLOR, width=1))

    yr_label = str(year) if year else "All Years"
    h = max(380, len(cdf_r) * 46 + 90)
    fig = _base_layout(
        fig,
        title=f"TLR–RPC Correlation by Discipline — {yr_label}",
        xl="Pearson r  (TLR ↔ RPC)",
        yl="",
        h=h,
        legend=False,
    )
    fig.update_xaxes(range=[-1.08, 1.08])
    fig.update_layout(margin=dict(l=160, r=52, t=80, b=60))

    best  = cdf_r.iloc[-1]
    worst = cdf_r.iloc[0]
    takeaway = (
        f"TLR–RPC coupling varies sharply by discipline — strongest in "
        f"'{best['category']}' (r = {best['r']:.3f}) and weakest in "
        f"'{worst['category']}' (r = {worst['r']:.3f}). "
        "The 'it depends on the field' story is the viva-winning insight here."
    )
    return fig, takeaway


# ── Q5a: All-parameter correlations with score over years ─────────────────────

def plot_param_corr_over_years(
    df: pd.DataFrame,
    category: str = "Overall",
    state: Optional[str] = None,
) -> Tuple[go.Figure, str]:
    """
    Multi-line chart: for each of the 5 NIRF parameters, Pearson r with overall score
    is plotted per year. Each parameter gets its own line + shaded 95 % CI band
    (Fisher z-transform). Reference lines at r = 0.4, 0.7, 0.9.
    """
    sub = df[df["category"] == category].copy()
    if state:
        sub = sub[sub["state"] == state]

    params = [p for p in ["TLR", "RPC", "GO", "OI", "PR"] if p in sub.columns]
    years  = sorted(sub["year"].dropna().unique().tolist())

    if not params or len(years) < 2:
        fig = go.Figure()
        return _base_layout(fig, "Insufficient data"), ""

    fig = go.Figure()

    for param in params:
        r_arr, lo_arr, hi_arr, n_arr = [], [], [], []
        for yr in years:
            ydf = sub[sub["year"] == yr].dropna(subset=[param, "score"])
            n   = len(ydf)
            if n >= 4:
                r, _  = _pearson(ydf[param], ydf["score"])
                lo, hi = _fisher_ci(r, n) if r is not None else (float("nan"), float("nan"))
            else:
                r, lo, hi = None, float("nan"), float("nan")
            r_arr.append(r if r is not None else float("nan"))
            lo_arr.append(lo); hi_arr.append(hi); n_arr.append(n)

        yr_np = np.array(years, dtype=float)
        r_np  = np.array(r_arr)
        lo_np = np.array(lo_arr)
        hi_np = np.array(hi_arr)
        col   = PARAM_COLORS.get(param, LINE_COLOR)

        # CI band (shaded)
        valid = np.isfinite(lo_np) & np.isfinite(hi_np)
        if valid.sum() >= 2:
            fig.add_trace(go.Scatter(
                x=np.concatenate([yr_np[valid], yr_np[valid][::-1]]),
                y=np.concatenate([hi_np[valid], lo_np[valid][::-1]]),
                fill="toself",
                fillcolor=_hex_rgba(col, 0.14),
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip",
            ))

        # r line
        fig.add_trace(go.Scatter(
            x=years, y=r_np,
            mode="lines+markers",
            name=param,
            line=dict(color=col, width=2.6),
            marker=dict(size=7, color=col),
            customdata=list(zip(r_np, n_arr)),
            hovertemplate=(
                f"<b>{param} · %{{x}}</b><br>"
                "r: %{customdata[0]:.3f}<br>"
                "n: %{customdata[1]}"
                "<extra></extra>"
            ),
        ))

    # Reference lines
    for yv, lab in [(0.9, "Very strong (0.9)"), (0.7, "Strong (0.7)"), (0.4, "Moderate (0.4)")]:
        fig.add_hline(
            y=yv, line=dict(color=GRID_COLOR, width=0.8, dash="dot"),
            annotation_text=lab,
            annotation_font=dict(color=MUTED_TEXT, size=9),
            annotation_position="right",
        )

    state_sfx = f" · {state}" if state else ""
    fig = _base_layout(
        fig,
        title=f"Parameter–Score Correlation Over Time — {category}{state_sfx}",
        xl="Year",
        yl="Pearson r  (↔ overall score)",
        h=580,
    )
    fig.update_xaxes(tickmode="linear", tick0=min(years), dtick=1)
    fig.update_yaxes(range=[-0.05, 1.08])

    # Targeted takeaway for TLR and RPC
    tlr_vals, rpc_vals = [], []
    for yr in years:
        ydf = sub[sub["year"] == yr].dropna(subset=["score"])
        if "TLR" in sub.columns:
            ydf_t = ydf.dropna(subset=["TLR"])
            r_t, _ = _pearson(ydf_t["TLR"], ydf_t["score"]) if len(ydf_t) >= 4 else (None, None)
            if r_t: tlr_vals.append(r_t)
        if "RPC" in sub.columns:
            ydf_r = ydf.dropna(subset=["RPC"])
            r_r, _ = _pearson(ydf_r["RPC"], ydf_r["score"]) if len(ydf_r) >= 4 else (None, None)
            if r_r: rpc_vals.append(r_r)

    if tlr_vals and rpc_vals:
        takeaway = (
            f"RPC holds the most stable correlation with overall score "
            f"(≈ {np.mean(rpc_vals):.2f} across years), "
            f"while TLR shows more variation ({tlr_vals[0]:.2f} → {tlr_vals[-1]:.2f}). "
            "Shaded bands are 95 % CI — movements inside the band are likely sampling noise."
        )
    else:
        takeaway = "Parameter correlation trends across the selected years."

    return fig, takeaway


# ── Q5b: Parameter × Year heatmap strip ───────────────────────────────────────

def plot_param_year_heatmap(
    df: pd.DataFrame,
    category: str = "Overall",
    state: Optional[str] = None,
) -> Tuple[go.Figure, str]:
    """
    Compact heatmap: rows = parameters (5), columns = years.
    Cell colour = Pearson r with overall score.
    Blue = strong positive, white = zero, (empty if data insufficient).
    """
    sub = df[df["category"] == category].copy()
    if state:
        sub = sub[sub["state"] == state]

    params = [p for p in ["TLR", "RPC", "GO", "OI", "PR"] if p in sub.columns]
    years  = sorted(sub["year"].dropna().unique().tolist())

    z_data, text_data = [], []
    for param in params:
        row_z, row_t = [], []
        for yr in years:
            ydf = sub[sub["year"] == yr].dropna(subset=[param, "score"])
            r, _ = _pearson(ydf[param], ydf["score"]) if len(ydf) >= 4 else (None, None)
            row_z.append(r if r is not None else float("nan"))
            row_t.append(f"{r:.2f}" if r is not None else "—")
        z_data.append(row_z)
        text_data.append(row_t)

    fig = go.Figure(go.Heatmap(
        z=z_data, x=years, y=params,
        text=text_data, texttemplate="%{text}",
        textfont=dict(size=11, color="#f8fafc", family="Inter, monospace"),
        colorscale=[[0, "#1e3a5f"], [0.4, "#2563eb"], [0.7, "#34d399"], [1, "#22c55e"]],
        zmin=0, zmax=1,
        colorbar=dict(
            title=dict(text="r", font=dict(color=MUTED_TEXT, size=12)),
            thickness=12, tickfont=dict(color=MUTED_TEXT, size=10),
        ),
        hovertemplate="<b>%{y} · %{x}</b><br>r = %{z:.3f}<extra></extra>",
    ))

    state_sfx = f" · {state}" if state else ""
    fig.update_layout(
        title=dict(
            text=f"Parameter × Year Correlation Strip — {category}{state_sfx}",
            x=0.04, font=dict(size=14, color=TEXT_COLOR, family="Inter, system-ui, sans-serif"),
        ),
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font=dict(color=TEXT_COLOR, size=12),
        height=320,
        margin=dict(l=72, r=52, t=72, b=60),
        xaxis=dict(tickmode="linear", tick0=min(years) if years else 2017, dtick=1,
                   tickfont=dict(color=MUTED_TEXT, size=11)),
        yaxis=dict(tickfont=dict(color=MUTED_TEXT, size=12)),
    )
    return fig, ""


# ── Explorer: free-form bivariate scatter ─────────────────────────────────────

def plot_explorer(
    df: pd.DataFrame,
    x_var: str,
    y_var: str,
    year: Optional[int] = None,
    category: str = "Overall",
    state: Optional[str] = None,
    corr_type: str = "pearson",
    show_lowess: bool = False,
    highlight_inst: Optional[str] = None,
) -> Tuple[go.Figure, Dict[str, Any], str]:
    """
    Free-form bivariate scatter for the Explorer panel.
    - Any two numeric columns on x and y.
    - OLS trendline always shown.
    - Optional LOWESS smooth (requires statsmodels).
    - Institution highlight: matching names are enlarged + labelled.
    Returns (figure, stats_dict, auto_sentence).
    stats_dict keys: pearson_r, spearman_r, r_squared, slope, intercept, n.
    """
    sub = df[df["category"] == category].copy()
    if year:
        sub = sub[sub["year"] == year]
    if state:
        sub = sub[sub["state"] == state]
    sub = sub.dropna(subset=[x_var, y_var]).reset_index(drop=True)

    _empty_stats: Dict[str, Any] = {
        "pearson_r": None, "spearman_r": None, "r_squared": None,
        "slope": None, "intercept": None, "n": 0,
    }

    if sub.empty or len(sub) < 3:
        fig = go.Figure()
        return _base_layout(fig, "No data for selected filters"), _empty_stats, "Insufficient data."

    x_arr = sub[x_var].to_numpy(dtype=float)
    y_arr = sub[y_var].to_numpy(dtype=float)

    pr, _ = _pearson(sub[x_var],  sub[y_var])
    sr, _ = _spearman(sub[x_var], sub[y_var])
    xs_ols, ys_ols, slope, intercept = _ols(x_arr, y_arr)
    r2 = round(pr ** 2, 4) if pr is not None else None

    # Build figure
    fig = go.Figure()

    # Scatter (coloured by rank band if available, else single colour)
    if "rank_band" in sub.columns:
        for band in RANK_BAND_ORDER:
            bdf = sub[sub["rank_band"] == band]
            if bdf.empty:
                continue
            # Determine highlight sizes
            if highlight_inst and "name" in bdf.columns:
                is_hi = bdf["name"].str.contains(highlight_inst, case=False, na=False).tolist()
            else:
                is_hi = [False] * len(bdf)
            sizes   = [15 if h else 8 for h in is_hi]
            borders = ["#f8fafc" if h else "#0a0c10" for h in is_hi]
            bwidths = [2.5 if h else 0.6 for h in is_hi]

            fig.add_trace(go.Scatter(
                x=bdf[x_var], y=bdf[y_var],
                mode="markers",
                name=f"Rank {band}",
                marker=dict(
                    color=RANK_BAND_COLORS.get(band, MUTED_TEXT),
                    size=sizes, opacity=0.82,
                    line=dict(color=borders, width=bwidths),
                ),
                customdata=list(zip(
                    bdf["name"].apply(_shorten),
                    bdf.get("rank", pd.Series([-1] * len(bdf))).fillna(-1).astype(int),
                )),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    f"{x_var}: %{{x:.2f}}<br>{y_var}: %{{y:.2f}}<br>"
                    "Rank: #%{customdata[1]}"
                    "<extra></extra>"
                ),
            ))
    else:
        fig.add_trace(go.Scatter(
            x=sub[x_var], y=sub[y_var],
            mode="markers",
            name="Institutions",
            marker=dict(color=LINE_COLOR, size=8, opacity=0.80),
            hovertemplate=f"{x_var}: %{{x:.2f}}<br>{y_var}: %{{y:.2f}}<extra></extra>",
        ))

    # OLS trendline
    if len(xs_ols):
        r_lbl = f"r = {pr:.3f}" if pr is not None else ""
        fig.add_trace(go.Scatter(
            x=xs_ols, y=ys_ols,
            mode="lines",
            name=f"OLS ({r_lbl})",
            line=dict(color="#f59e0b", width=2.2, dash="dash"),
        ))

    # Optional LOWESS
    if show_lowess and _HAS_LOWESS:
        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        if mask.sum() >= 10:
            smoothed = _lowess_fn(y_arr[mask], x_arr[mask], frac=0.4, it=3)
            # sort by x for a clean line
            order = np.argsort(smoothed[:, 0])
            fig.add_trace(go.Scatter(
                x=smoothed[order, 0], y=smoothed[order, 1],
                mode="lines",
                name="LOWESS",
                line=dict(color="#a78bfa", width=2.5),
            ))

    # Institution highlight label
    if highlight_inst and "name" in sub.columns:
        hdf = sub[sub["name"].str.contains(highlight_inst, case=False, na=False)]
        for _, hrow in hdf.iterrows():
            fig.add_annotation(
                x=float(hrow[x_var]), y=float(hrow[y_var]),
                text=f"<b>{_shorten(str(hrow['name']))}</b>",
                showarrow=True, arrowhead=2, arrowsize=0.9,
                arrowcolor="#f8fafc", ax=32, ay=-28,
                font=dict(size=10, color="#f8fafc"),
                bgcolor=PANEL_BG, borderpad=3,
            )

    # Title
    xl     = PARAM_LABELS.get(x_var, x_var)
    yl     = PARAM_LABELS.get(y_var, y_var)
    yr_lbl = str(year) if year else "All Years"
    st_sfx = f" · {state}" if state else ""
    r_str  = f"r = {pr:.3f}"  if pr is not None else "r = N/A"
    sp_str = f"ρ = {sr:.3f}"  if sr is not None else "ρ = N/A"
    r2_str = f"R² = {r2:.3f}" if r2 is not None else ""

    title_html = (
        f"{x_var} vs {y_var} — {category} · {yr_lbl}{st_sfx}"
        f"<br><sup style='font-size:12px;color:{MUTED_TEXT}'>"
        f"Pearson {r_str} &nbsp;·&nbsp; Spearman {sp_str} &nbsp;·&nbsp; {r2_str} &nbsp;·&nbsp; n = {len(sub)}"
        f"</sup>"
    )

    fig = _base_layout(fig, title=title_html, xl=xl, yl=yl, h=620)

    # Stats dict
    stats: Dict[str, Any] = {
        "pearson_r":  pr,
        "spearman_r": sr,
        "r_squared":  r2,
        "slope":      round(slope, 4)     if np.isfinite(slope)     else None,
        "intercept":  round(intercept, 4) if np.isfinite(intercept) else None,
        "n":          int(len(sub)),
    }

    # Auto interpretation sentence
    if pr is not None and sr is not None:
        strength = (
            "Very strong" if abs(pr) > 0.85 else
            "Strong"      if abs(pr) > 0.65 else
            "Moderate"    if abs(pr) > 0.40 else
            "Weak"        if abs(pr) > 0.20 else "Negligible"
        )
        direction = "positive" if pr > 0 else "negative"
        verb      = "tend to score higher" if pr > 0 else "tend to score lower"
        x_short   = PARAM_LABELS.get(x_var, x_var).split(" (")[0]
        y_short   = PARAM_LABELS.get(y_var, y_var).split(" (")[0]
        sentence  = (
            f"{strength} {direction} relationship (r = {pr:.3f}, ρ = {sr:.3f}): "
            f"institutions with higher {x_short} {verb} on {y_short}."
        )
    else:
        sentence = "Insufficient data to characterise the relationship."

    return fig, stats, sentence


# ── Backwards-compatibility aliases ──────────────────────────────────────────

def plot_rpc_vs_overall(
    df: pd.DataFrame,
    year: int,
    category: str = "Overall",
    state: Optional[str] = None,
    show_labels: bool = True,
    top_n_labels: int = 10,
) -> Tuple[go.Figure, str]:
    return plot_scatter_marginals(
        df=df, x_param="RPC", year=year, category=category, state=state,
        corr_type="pearson", top_n_outliers=top_n_labels
    )


def plot_pr_vs_overall(
    df: pd.DataFrame,
    year: int,
    category: str = "Overall",
    state: Optional[str] = None,
    show_labels: bool = True,
    top_n_labels: int = 10,
) -> Tuple[go.Figure, str]:
    return plot_scatter_marginals(
        df=df, x_param="PR", year=year, category=category, state=state,
        corr_type="pearson", top_n_outliers=top_n_labels
    )


def plot_parameter_correlations(
    df: pd.DataFrame,
    category: str = "Overall",
    year_range: Optional[Tuple[int, int]] = None,
    state: Optional[str] = None,
) -> Tuple[go.Figure, str]:
    return plot_correlation_vs_weight(df=df, category=category, year_range=year_range, state=state)


def plot_tlr_vs_rpc(
    df: pd.DataFrame,
    year: int,
    category: str = "Overall",
    state: Optional[str] = None,
    show_labels: bool = True,
    top_n_labels: int = 8,
) -> Tuple[go.Figure, str]:
    return plot_tlr_rpc_quadrants(df=df, year=year, category=category, state=state, top_n_labels=top_n_labels)


def plot_rpc_overall_correlation_over_years(
    df: pd.DataFrame,
    category: str = "Overall",
    state: Optional[str] = None,
    param: str = "RPC",
) -> Tuple[go.Figure, str]:
    return plot_param_corr_over_years(df=df, category=category, state=state)

