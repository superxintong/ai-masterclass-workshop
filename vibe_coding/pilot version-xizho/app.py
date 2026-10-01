"""
E-commerce Anomaly Detection Dashboard
---------------------------------------
Streamlit app: loads the e-commerce metrics dataset, computes a rolling
baseline per metric, flags statistically unusual days, and visualizes it.

Run with:
    streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from anomaly import METRIC_LABELS, METRICS, anomaly_summary, compute_baseline, load_data

# ---------------------------------------------------------------------------
# Palette (validated categorical / status colors — kept consistent across
# every chart in the app rather than picked per-chart).
# ---------------------------------------------------------------------------
COLOR_LINE = "#2a78d6"       # primary series (blue)
COLOR_BAND = "rgba(42, 120, 214, 0.12)"   # baseline band fill
COLOR_BAND_LINE = "rgba(42, 120, 214, 0.35)"
COLOR_ANOMALY = "#d03b3b"    # status: critical (red)
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"
COLOR_TEXT_SECONDARY = "#52514e"
COLOR_MUTED = "#898781"
SURFACE = "#fcfcfb"

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(
    page_title="E-commerce Anomaly Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Light, professional styling on top of Streamlit's defaults.
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }}
        h1 {{
            font-weight: 700;
            letter-spacing: -0.01em;
        }}
        .app-subtitle {{
            color: {COLOR_TEXT_SECONDARY};
            font-size: 0.95rem;
            margin-top: -0.6rem;
            margin-bottom: 1.6rem;
        }}
        div[data-testid="stMetric"] {{
            background: {SURFACE};
            border: 1px solid {COLOR_GRID};
            border-radius: 10px;
            padding: 14px 18px 10px 18px;
        }}
        div[data-testid="stMetricLabel"] {{
            color: {COLOR_TEXT_SECONDARY};
        }}
        .section-label {{
            color: {COLOR_MUTED};
            text-transform: uppercase;
            font-size: 0.72rem;
            letter-spacing: 0.06em;
            font-weight: 600;
            margin-top: 1.6rem;
            margin-bottom: 0.4rem;
        }}
        .anomaly-badge {{
            display: inline-block;
            background: rgba(208, 59, 59, 0.10);
            color: {COLOR_ANOMALY};
            border-radius: 999px;
            padding: 1px 10px;
            font-size: 0.78rem;
            font-weight: 600;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — data source & detection controls
# ---------------------------------------------------------------------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader(
    "Upload a dataset (optional)", type=["csv", "xlsx"], help="Defaults to the bundled sample dataset."
)

if uploaded is not None:
    filetype = "csv" if uploaded.name.lower().endswith(".csv") else "excel"
    df = load_data(uploaded, filetype=filetype)
    st.sidebar.caption(f"Using uploaded file: {uploaded.name}")
else:
    default_csv = DATA_DIR / "ecommerce_dataset.csv"
    default_xlsx = DATA_DIR / "ecommerce_dataset.xlsx"
    if default_csv.exists():
        df = load_data(default_csv, filetype="csv")
    elif default_xlsx.exists():
        df = load_data(default_xlsx, filetype="excel")
    else:
        st.error("No dataset found. Upload a CSV or Excel file to get started.")
        st.stop()
    st.sidebar.caption("Using bundled sample dataset")

st.sidebar.header("Detection settings")
window = st.sidebar.slider(
    "Rolling baseline window (days)",
    min_value=3,
    max_value=21,
    value=7,
    help="How many prior days feed the rolling mean/std baseline for each metric.",
)
z_threshold = st.sidebar.slider(
    "Sensitivity (z-score threshold)",
    min_value=1.5,
    max_value=4.0,
    value=2.5,
    step=0.1,
    help="A day is flagged when it's this many standard deviations from its rolling baseline. Lower = more sensitive.",
)

available_metrics = [m for m in METRICS if m in df.columns]
selected_metrics = st.sidebar.multiselect(
    "Metrics to show",
    options=available_metrics,
    default=available_metrics,
    format_func=lambda m: METRIC_LABELS.get(m, m),
)

min_date, max_date = df["date"].min().date(), df["date"].max().date()
date_range = st.sidebar.slider(
    "Date range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
)

st.sidebar.divider()
st.sidebar.caption(
    "Method: trailing rolling mean/std per metric → z-score vs. baseline → "
    "flagged when |z| exceeds the threshold above."
)

if not selected_metrics:
    st.warning("Select at least one metric in the sidebar to see results.")
    st.stop()

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
long_df = compute_baseline(df, window=window, z_threshold=z_threshold, metrics=selected_metrics)
mask = (long_df["date"].dt.date >= date_range[0]) & (long_df["date"].dt.date <= date_range[1])
long_df = long_df[mask]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("E-commerce Anomaly Dashboard")
st.markdown(
    f'<div class="app-subtitle">Rolling-baseline anomaly detection across '
    f"{len(selected_metrics)} metric(s) · {date_range[0]:%b %d, %Y} – {date_range[1]:%b %d, %Y}</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
total_anomalies = int(long_df["is_anomaly"].sum())
days_covered = df[(df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])].shape[0]
latest_date = long_df["date"].max()
latest_day_anomalies = int(long_df[(long_df["date"] == latest_date) & (long_df["is_anomaly"])].shape[0])
pct_flagged = (total_anomalies / max(len(long_df), 1)) * 100

kpi_cols = st.columns(4)
kpi_cols[0].metric("Days covered", f"{days_covered}")
kpi_cols[1].metric("Metrics monitored", f"{len(selected_metrics)}")
kpi_cols[2].metric("Anomalies flagged", f"{total_anomalies}", help="Total flagged (date, metric) pairs in the selected range.")
kpi_cols[3].metric(
    f"On {latest_date:%b %d}",
    f"{latest_day_anomalies} flagged" if latest_day_anomalies else "Nominal",
)

# ---------------------------------------------------------------------------
# Charts — one per metric, baseline band + line + anomaly markers
# ---------------------------------------------------------------------------
st.markdown('<div class="section-label">Metric detail</div>', unsafe_allow_html=True)

def render_metric_chart(metric_df: pd.DataFrame, metric: str) -> go.Figure:
    metric_df = metric_df.sort_values("date")
    fig = go.Figure()

    # Baseline band (upper/lower), drawn first so it sits behind the line.
    fig.add_trace(
        go.Scatter(
            x=metric_df["date"], y=metric_df["upper_band"],
            line=dict(width=0), mode="lines",
            showlegend=False, hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=metric_df["date"], y=metric_df["lower_band"],
            line=dict(width=0), mode="lines", fill="tonexty",
            fillcolor=COLOR_BAND, name="Expected range",
            hoverinfo="skip",
        )
    )
    # Baseline (rolling mean) line.
    fig.add_trace(
        go.Scatter(
            x=metric_df["date"], y=metric_df["baseline"],
            line=dict(width=1, color=COLOR_BAND_LINE, dash="dot"),
            mode="lines", name="Rolling baseline", hoverinfo="skip",
        )
    )
    # Actual value.
    fig.add_trace(
        go.Scatter(
            x=metric_df["date"], y=metric_df["value"],
            line=dict(width=2, color=COLOR_LINE), mode="lines",
            name=METRIC_LABELS.get(metric, metric),
            hovertemplate="%{x|%b %d, %Y}<br>Value: %{y:,.3g}<extra></extra>",
        )
    )
    # Anomalies.
    anomalies = metric_df[metric_df["is_anomaly"]]
    if not anomalies.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies["date"], y=anomalies["value"],
                mode="markers", name="Anomaly",
                marker=dict(size=11, color=COLOR_ANOMALY, symbol="diamond", line=dict(width=1, color="white")),
                customdata=anomalies[["z_score", "direction"]],
                hovertemplate=(
                    "<b>Anomaly</b> (%{customdata[1]})<br>%{x|%b %d, %Y}<br>"
                    "Value: %{y:,.3g}<br>z-score: %{customdata[0]:.2f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
        font=dict(size=12, color="#0b0b0b"),
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor=COLOR_AXIS, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=COLOR_GRID, showline=False, zeroline=False)
    return fig


cols = st.columns(2)
for i, metric in enumerate(selected_metrics):
    metric_df = long_df[long_df["metric"] == metric]
    n_anom = int(metric_df["is_anomaly"].sum())
    with cols[i % 2]:
        header_cols = st.columns([3, 1])
        header_cols[0].markdown(f"**{METRIC_LABELS.get(metric, metric)}**")
        if n_anom:
            header_cols[1].markdown(f'<span class="anomaly-badge">{n_anom} flagged</span>', unsafe_allow_html=True)
        st.plotly_chart(render_metric_chart(metric_df, metric), width="stretch", config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# Anomaly table + download
# ---------------------------------------------------------------------------
st.markdown('<div class="section-label">Flagged anomalies</div>', unsafe_allow_html=True)

summary = anomaly_summary(long_df)
if summary.empty:
    st.info("No anomalies flagged for the current settings and date range.")
else:
    display = summary.copy()
    display["date"] = display["date"].dt.strftime("%Y-%m-%d")
    display["value"] = display["value"].round(3)
    display["baseline"] = display["baseline"].round(3)
    display["z_score"] = display["z_score"].round(2)
    display = display.rename(
        columns={
            "date": "Date",
            "metric_label": "Metric",
            "value": "Value",
            "baseline": "Expected (baseline)",
            "z_score": "Z-score",
            "direction": "Direction",
        }
    )
    st.dataframe(display, width="stretch", hide_index=True)

    st.download_button(
        "Download flagged anomalies (CSV)",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="flagged_anomalies.csv",
        mime="text/csv",
    )

st.caption(
    "Baseline = trailing rolling mean of the prior window; a day is flagged when it falls "
    "more than the chosen z-score threshold away from that baseline's standard deviation."
)
