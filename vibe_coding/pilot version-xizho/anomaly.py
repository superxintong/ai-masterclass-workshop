"""
Anomaly-detection logic, kept separate from the Streamlit UI so it can be
tested or reused on its own.

Method: trailing rolling mean/std baseline per metric, z-score vs. that
baseline, flag anything beyond a configurable threshold. This is a
classic, easy-to-explain approach — no ML model, no training step — which
is what "simple anomaly detection" calls for.
"""

import numpy as np
import pandas as pd

METRICS = [
    "revenue_usd",
    "orders_count",
    "avg_order_value_usd",
    "customer_churn_rate",
    "support_tickets",
    "ai_feature_usage_rate",
]

METRIC_LABELS = {
    "revenue_usd": "Revenue (USD)",
    "orders_count": "Orders",
    "avg_order_value_usd": "Avg. Order Value (USD)",
    "customer_churn_rate": "Customer Churn Rate",
    "support_tickets": "Support Tickets",
    "ai_feature_usage_rate": "AI Feature Usage Rate",
}

def load_data(path_or_buffer, filetype: str | None = None) -> pd.DataFrame:
    """Load the e-commerce dataset from a CSV or Excel file/buffer."""
    if filetype is None:
        name = getattr(path_or_buffer, "name", str(path_or_buffer))
        filetype = "csv" if str(name).lower().endswith(".csv") else "excel"

    if filetype == "csv":
        df = pd.read_csv(path_or_buffer)
    else:
        df = pd.read_excel(path_or_buffer)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def compute_baseline(
    df: pd.DataFrame,
    window: int = 7,
    z_threshold: float = 2.5,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """
    For each metric, compute a trailing rolling-window baseline (mean + std,
    excluding the current day so the baseline never peeks at the value it's
    judging) and flag points whose z-score vs. that baseline exceeds
    z_threshold in either direction.
    """
    df = df.copy()
    metrics = metrics or METRICS
    min_periods = max(3, window // 2)

    rows = []
    for metric in metrics:
        series = df[metric]
        roll_mean = series.rolling(window=window, min_periods=min_periods).mean().shift(1)
        roll_std = series.rolling(window=window, min_periods=min_periods).std().shift(1)

        baseline = roll_mean
        safe_std = roll_std.replace(0, np.nan)
        z = (series - baseline) / safe_std

        is_anomaly = z.abs() > z_threshold
        is_anomaly = is_anomaly.fillna(False)
        direction = np.where(z > 0, "spike", "drop")

        rows.append(
            pd.DataFrame(
                {
                    "date": df["date"],
                    "metric": metric,
                    "value": series,
                    "baseline": baseline,
                    "baseline_std": roll_std,
                    "upper_band": baseline + z_threshold * roll_std,
                    "lower_band": baseline - z_threshold * roll_std,
                    "z_score": z,
                    "is_anomaly": is_anomaly,
                    "direction": direction,
                }
            )
        )

    return pd.concat(rows, ignore_index=True)


def anomaly_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    """Flat table of just the flagged anomalies, newest first."""
    flagged = long_df[long_df["is_anomaly"]].copy()
    flagged["metric_label"] = flagged["metric"].map(METRIC_LABELS)
    flagged = flagged.sort_values("date", ascending=False)
    return flagged[["date", "metric_label", "value", "baseline", "z_score", "direction"]]
