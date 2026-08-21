"""
core/thresholds.py

Turns a quantile forecast into an operator decision: is this station going to
exceed its capacity, when, and how seriously.

Pure functions over (forecast, capacity). No I/O, no model. That keeps the
alerting logic testable and lets real capacity ratings replace the derived
proxy without touching anything else.

Severity uses the quantiles for what they mean operationally:

  warning   P90 crosses capacity — the reserve requirement no longer fits
  critical  P50 crosses capacity — expected demand alone exceeds the rating

Capacity in this deployment is derived from the upper tail of training demand,
not a nameplate rating, so every surface that shows a threshold must say so.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core import config

SEVERITY_ORDER = {"critical": 0, "warning": 1, "none": 2}

ACTIONS = {
    "critical": "Curtail or shift load before this hour",
    "warning": "Hold reserve; stagger scheduled sessions",
    "none": "No action",
}


@dataclass(frozen=True)
class Breach:
    station_id: str
    horizon_h: int
    timestamp: object
    p50: float
    p90: float
    capacity: float
    severity: str

    @property
    def headroom_pct(self):
        if not self.capacity:
            return float("nan")
        return (self.capacity - self.p90) / self.capacity * 100.0

    @property
    def action(self):
        return ACTIONS[self.severity]


def severity_for(p50, p90, capacity):
    """Critical when expected demand exceeds capacity, warning when P90 does."""
    if not capacity or not np.isfinite(capacity):
        return "none"
    if p50 > capacity:
        return "critical"
    if p90 > capacity:
        return "warning"
    return "none"


def find_breaches(forecast, station_id, capacity):
    """Every hour in the horizon where the forecast crosses capacity."""
    frame = forecast.frame
    p50 = frame[f"q{config.Q_MEDIAN:.2f}"].to_numpy()
    p90 = frame[f"q{config.Q_HIGH:.2f}"].to_numpy()

    breaches = []
    for position in range(len(frame)):
        severity = severity_for(p50[position], p90[position], capacity)
        if severity == "none":
            continue
        breaches.append(
            Breach(
                station_id=station_id,
                horizon_h=int(frame["horizon_h"].iloc[position]),
                timestamp=frame["timestamp"].iloc[position],
                p50=float(p50[position]),
                p90=float(p90[position]),
                capacity=float(capacity),
                severity=severity,
            )
        )
    return breaches


def utilisation(forecast, capacity, quantile=config.Q_HIGH):
    """Forecast as a fraction of capacity, for the hour-grid heatmap."""
    if not capacity or not np.isfinite(capacity):
        return np.full(len(forecast.frame), np.nan)
    values = forecast.frame[f"q{quantile:.2f}"].to_numpy()
    return values / capacity


def headroom(forecast, capacity):
    """Tightest margin across the horizon, as a percentage of capacity."""
    if not capacity or not np.isfinite(capacity):
        return float("nan")
    peak_p90 = float(forecast.frame[f"q{config.Q_HIGH:.2f}"].max())
    return (capacity - peak_p90) / capacity * 100.0


def breaches_to_frame(breaches):
    """Ranked table for the Alerts page: critical first, then by hour."""
    if not breaches:
        return pd.DataFrame(
            columns=[
                "station_id",
                "horizon_h",
                "timestamp",
                "p50",
                "p90",
                "capacity",
                "headroom_pct",
                "severity",
                "action",
            ]
        )

    frame = pd.DataFrame(
        {
            "station_id": [b.station_id for b in breaches],
            "horizon_h": [b.horizon_h for b in breaches],
            "timestamp": [b.timestamp for b in breaches],
            "p50": [b.p50 for b in breaches],
            "p90": [b.p90 for b in breaches],
            "capacity": [b.capacity for b in breaches],
            "headroom_pct": [b.headroom_pct for b in breaches],
            "severity": [b.severity for b in breaches],
            "action": [b.action for b in breaches],
        }
    )
    frame["_rank"] = frame["severity"].map(SEVERITY_ORDER)
    return (
        frame.sort_values(["_rank", "horizon_h"])
        .drop(columns="_rank")
        .reset_index(drop=True)
    )