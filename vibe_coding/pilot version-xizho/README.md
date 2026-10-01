# E-commerce Anomaly Dashboard

A Streamlit dashboard that flags unusual days in your e-commerce metrics
using a simple rolling-baseline / z-score method — no ML training step,
easy to explain to a non-technical audience.

## What it does

1. **Rolling baseline** — for each metric, computes a trailing rolling
   mean and standard deviation (default: 7-day window).
2. **Flags anomalies** — any day whose value is more than N standard
   deviations (default: 2.5) from that baseline is flagged, in either
   direction (spike or drop).
3. **Dashboard** — one chart per metric (actual value, baseline, expected
   range band, and highlighted anomalies), KPI summary cards, and a
   sortable/downloadable table of every flagged anomaly.

Metrics covered out of the box: revenue, order count, average order
value, customer churn rate, support tickets, and AI feature usage rate.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

The bundled sample dataset (`data/ecommerce_dataset.csv`) loads
automatically. To use your own data, upload a CSV or XLSX file from the
sidebar — it needs a `date` column plus any of the metric columns above.

## Tuning detection

Use the sidebar to adjust:

- **Rolling baseline window** — how many prior days set the "normal"
  range. Shorter windows react faster but flag more noise; longer windows
  are more stable but slower to adapt to real shifts.
- **Sensitivity (z-score threshold)** — how far from baseline counts as
  "unusual." Lower = more anomalies flagged.
- **Metrics to show** and **date range** — scope the view.

## Files

- `app.py` — the Streamlit UI
- `anomaly.py` — the baseline/anomaly-detection logic (kept separate so it
  can be reused or unit-tested independently of the UI)
- `data/ecommerce_dataset.csv` / `.xlsx` — the bundled sample dataset
