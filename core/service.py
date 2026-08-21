"""
core/service.py

The seam between the model and the pages. Everything a page needs is one call
here, so no page assembles tensors or touches the ONNX session directly.

Results are cached on (station_id, origin), which makes scrubbing the rolling
origin slider instant on revisit and keeps the network view cheap: twelve
stations at roughly 1.5 ms each is under 20 ms for a full refresh.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from core import config, engine, loader, postprocess, thresholds, window


@dataclass(frozen=True)
class StationForecast:
    station_id: str
    display_name: str
    group: str
    capacity_kwh: float
    capacity_source: str
    confidence_tier: str
    forecast: object
    breaches: list
    origin_time_index: int
    origin_timestamp: object

    @property
    def peak(self):
        return self.forecast.peak()

    @property
    def expected_energy(self):
        return self.forecast.expected_energy()

    @property
    def headroom_pct(self):
        return thresholds.headroom(self.forecast, self.capacity_kwh)

    @property
    def worst_severity(self):
        if not self.breaches:
            return "none"
        return min(self.breaches, key=lambda b: thresholds.SEVERITY_ORDER[b.severity]).severity


@lru_cache(maxsize=512)
def forecast_station(station_id, origin_time_index):
    """Full forecast for one station at one origin."""
    meta = loader.station_index().get(station_id)
    if meta is None:
        raise KeyError(f"Unknown station: {station_id}")

    inputs = window.build_inputs(station_id, int(origin_time_index))
    raw, latency = engine.run(inputs.as_feed())
    forecast = postprocess.build_forecast(
        raw,
        inputs.horizon_timestamps,
        latency_ms=latency,
        origin_timestamp=inputs.origin_timestamp,
    )

    capacity = float(meta.get("capacity_kwh") or 0.0)
    breaches = thresholds.find_breaches(forecast, station_id, capacity)

    return StationForecast(
        station_id=station_id,
        display_name=meta.get("display_name", station_id),
        group=meta.get("group", "ungrouped"),
        capacity_kwh=capacity,
        capacity_source=meta.get("capacity_source", "derived"),
        confidence_tier=meta.get("confidence_tier", "unknown"),
        forecast=forecast,
        breaches=breaches,
        origin_time_index=int(origin_time_index),
        origin_timestamp=inputs.origin_timestamp,
    )


def common_origins():
    """Origins valid at every active station, so the network view lines up."""
    active = loader.active_stations()["station_id"].tolist()
    shared = None
    for station_id in active:
        origins = set(loader.valid_origins(station_id))
        shared = origins if shared is None else shared & origins
    return sorted(shared or [])


def forecast_network(origin_time_index):
    """Every active station at one origin, plus the aggregate curve."""
    active = loader.active_stations()["station_id"].tolist()

    stations = []
    for station_id in active:
        try:
            stations.append(forecast_station(station_id, origin_time_index))
        except (window.WindowError, KeyError):
            # A station without a full window at this origin is skipped rather
            # than zero-filled, so the aggregate never includes invented demand.
            continue

    if not stations:
        raise ValueError(f"No station has a valid window at origin {origin_time_index}")

    timestamps = stations[0].forecast.frame["timestamp"].tolist()
    aggregate = pd.DataFrame({"timestamp": timestamps})
    for quantile in (config.Q_LOW, config.Q_MEDIAN, config.Q_HIGH):
        column = f"q{quantile:.2f}"
        aggregate[column] = np.sum(
            [s.forecast.frame[column].to_numpy() for s in stations], axis=0
        )
    aggregate.insert(0, "horizon_h", np.arange(1, len(aggregate) + 1))

    total_capacity = float(sum(s.capacity_kwh for s in stations))
    all_breaches = [b for s in stations for b in s.breaches]

    peak_position = int(np.argmax(aggregate[f"q{config.Q_MEDIAN:.2f}"].to_numpy()))
    peak_p90 = float(aggregate[f"q{config.Q_HIGH:.2f}"].iloc[peak_position])

    return {
        "stations": stations,
        "aggregate": aggregate,
        "origin_time_index": int(origin_time_index),
        "origin_timestamp": stations[0].origin_timestamp,
        "total_capacity_kwh": total_capacity,
        "peak": {
            "horizon_h": peak_position + 1,
            "timestamp": aggregate["timestamp"].iloc[peak_position],
            "p50": float(aggregate[f"q{config.Q_MEDIAN:.2f}"].iloc[peak_position]),
            "p90": peak_p90,
        },
        "headroom_pct": (
            (total_capacity - peak_p90) / total_capacity * 100.0
            if total_capacity
            else float("nan")
        ),
        "expected_energy": {
            "p50_total": float(aggregate[f"q{config.Q_MEDIAN:.2f}"].sum()),
            "p10_total": float(aggregate[f"q{config.Q_LOW:.2f}"].sum()),
            "p90_total": float(aggregate[f"q{config.Q_HIGH:.2f}"].sum()),
            "basis": "hourly bounds summed",
        },
        "breaches": thresholds.breaches_to_frame(all_breaches),
        "stations_at_risk": sum(1 for s in stations if s.breaches),
        "mean_latency_ms": float(np.mean([s.forecast.latency_ms for s in stations])),
    }


def utilisation_matrix(network):
    """Station x hour utilisation against capacity, for the Overview heatmap."""
    rows = []
    for station in network["stations"]:
        rows.append(
            {
                "station_id": station.station_id,
                "display_name": station.display_name,
                "group": station.group,
                "values": thresholds.utilisation(station.forecast, station.capacity_kwh),
            }
        )
    return rows


def rescale_params():
    """(center, scale) of the fitted target normalizer — the rescale step's
    own parameters, for the System page's decision trace."""
    return postprocess.target_scale()


def decision_trace(station_id, origin_time_index):
    """The forecast pipeline as a step-by-step trace: window, inference,
    rescale, quantile check, threshold check, result. Every field comes from
    data forecast_station() already computed — this just narrates it.
    """
    station = forecast_station(station_id, int(origin_time_index))
    center, scale = rescale_params()
    crossings = station.forecast.crossing_pairs
    breaches = station.breaches
    critical = sum(1 for b in breaches if b.severity == "critical")
    warning = sum(1 for b in breaches if b.severity == "warning")
    peak = station.peak

    if critical:
        threshold_state, threshold_icon = "CRITICAL", "🔴"
    elif warning:
        threshold_state, threshold_icon = "WARNING", "🟡"
    else:
        threshold_state, threshold_icon = "CLEAR", "🟢"

    return [
        {
            "step": 1, "stage": "Window", "state": "✅ COMPLETE",
            "action": "Assemble 6 input tensors (168h encoder, 24h decoder)",
            "reason": f"{config.ENCODER_LENGTH}h history + {config.HORIZON}h "
                      f"horizon available at origin {station.origin_time_index}",
        },
        {
            "step": 2, "stage": "Inference", "state": "✅ COMPLETE",
            "action": "Run ONNX graph, single-threaded CPU",
            "reason": f"{station.forecast.latency_ms:.2f} ms",
        },
        {
            "step": 3, "stage": "Rescale", "state": "✅ COMPLETE",
            "action": "z = center + scale·raw, then expm1, floor at 0",
            "reason": f"center={center:.4f}, scale={scale:.4f}",
        },
        {
            "step": 4, "stage": "Quantile check",
            "state": "⚠️ CORRECTED" if crossings else "✅ COMPLETE",
            "action": "Sort quantiles ascending" if crossings else "No correction needed",
            "reason": f"{crossings} crossing pair(s) in the raw output" if crossings
                      else "quantiles already monotonic",
        },
        {
            "step": 5, "stage": "Threshold check", "state": f"{threshold_icon} {threshold_state}",
            "action": f"Compare P50/P90 against {station.capacity_kwh:,.0f} kWh capacity",
            "reason": f"{len(breaches)} breach-hour(s): {critical} critical, {warning} warning"
                      if breaches else "no hour crosses capacity",
        },
        {
            "step": 6, "stage": "Result", "state": "✅ COMPLETE",
            "action": "24-hour forecast ready",
            "reason": f"peak {peak['p50']:,.0f} kWh at H+{peak['horizon_h']}",
        },
    ]


def graph_signature():
    """ONNX input names and shapes, for the System page."""
    return engine.input_signature()


def benchmark(passes=200, warmup=10, station_id=None, origin_time_index=None):
    """Latency profile under the paper's 200-pass protocol, on a real window.

    Defaults to a representative station/origin (first active station, its
    middle valid origin) so the System page can call this with no arguments.
    """
    if station_id is None:
        station_id = loader.active_stations()["station_id"].iloc[0]
    if origin_time_index is None:
        origins = loader.valid_origins(station_id)
        origin_time_index = origins[len(origins) // 2]

    inputs = window.build_inputs(station_id, int(origin_time_index))
    profile = engine.benchmark(inputs.as_feed(), passes=passes, warmup=warmup)
    profile["station_id"] = station_id
    profile["origin_time_index"] = int(origin_time_index)
    return profile


def model_manifest():
    """Param counts, epoch, and the paper's external record. See loader.load_model_manifest."""
    return loader.load_model_manifest()


@lru_cache(maxsize=1)
def baseline_accuracy():
    """Baseline-student WMAPE/MAE over the full replay window, all origins.

    Not the same protocol as the paper's Table 5 figure or station_scores.csv
    (see core/service.model_manifest and loader.load_station_scores) — this
    is measured directly against comparison_medians.parquet, so it is
    comparable to nothing else in the app except itself. Labelled as such
    wherever it's shown.
    """
    table = loader.load_comparison_medians()
    if table is None:
        return None

    panel = loader.load_panel()[["station_id", "Time_Index", "Energy_kWh"]]
    long = table.melt(
        id_vars=["station_id", "origin_time_index", "model"],
        value_vars=[f"h{i}" for i in range(1, config.HORIZON + 1)],
        var_name="h", value_name="predicted",
    )
    long["h"] = long["h"].str[1:].astype(int)
    long["Time_Index"] = long["origin_time_index"] + long["h"]
    merged = long.merge(panel, on=["station_id", "Time_Index"], how="inner")

    error = (merged["Energy_kWh"] - merged["predicted"]).abs()
    denom = merged["Energy_kWh"].abs().sum()
    return {
        "wmape": float(error.sum() / denom * 100) if denom > 0 else float("nan"),
        "mae": float(error.mean()),
        "n": int(len(merged)),
    }


def comparison_available():
    """Whether scripts/06_export_comparison.py has been run."""
    return loader.load_comparison_medians() is not None


def baseline_median(station_id, origin_time_index):
    """Baseline-student (no distillation) median forecast, or None if unavailable.

    Precomputed by scripts/06_export_comparison.py — no torch import here, so
    this stays cheap even when the comparison file has never been generated.
    Teacher is deliberately absent: it was trained on a 336-hour encoder with
    its own vocabulary, a contract core/window.py does not build.
    """
    table = loader.load_comparison_medians()
    if table is None:
        return None
    match = table[
        (table["station_id"] == station_id)
        & (table["origin_time_index"] == int(origin_time_index))
        & (table["model"] == "baseline")
    ]
    if match.empty:
        return None
    columns = [f"h{i}" for i in range(1, config.HORIZON + 1)]
    return match.iloc[0][columns].to_numpy(dtype=float)


def prior_week_actual(station_id, origin_time_index):
    """Same 24 hours one week earlier — the comparison for expected energy."""
    frame = loader.station_frame(station_id)
    start = int(origin_time_index) + 1 - 168
    idx = [i for i in range(start, start + config.HORIZON) if i in frame.index]
    if len(idx) < config.HORIZON:
        return None
    return float(frame.loc[idx, "Energy_kWh"].sum())


def station_history(station_id, origin_time_index, hours=48):
    """Observed demand leading up to the origin, for the left half of a fan chart."""
    return window.observed_history(station_id, origin_time_index, hours=hours)


def station_actuals(station_id, origin_time_index):
    """Ground truth over the forecast horizon, where the replay data has it."""
    return window.ground_truth(station_id, origin_time_index)


def rolling_accuracy(station_id, origin_time_index, days=7):
    """WMAPE/MAE of the P50 forecast over the `days` most recent daily origins.

    Each of the last `days` days (24h apart, ending at origin) gets its own
    24-hour forecast, scored against what actually happened. Cheap — the
    ONNX graph runs in ~1.5ms — and it reflects this station's forecasts
    under real conditions leading up to now, not a global validation figure.
    """
    origin = int(origin_time_index)
    errors = []
    actuals = []
    n_days = 0

    for offset in range(1, days + 1):
        eval_origin = origin - offset * 24
        try:
            forecast = forecast_station(station_id, eval_origin)
        except (window.WindowError, KeyError):
            continue
        actual = window.ground_truth(station_id, eval_origin)
        if actual is None or len(actual) < config.HORIZON:
            continue

        predicted = forecast.forecast.median()
        observed = actual["Energy_kWh"].to_numpy()[: config.HORIZON]
        errors.append(np.abs(observed - predicted))
        actuals.append(np.abs(observed))
        n_days += 1

    if n_days == 0:
        return None

    error = np.concatenate(errors)
    actual_abs = np.concatenate(actuals)
    denom = actual_abs.sum()

    return {
        "wmape": float(error.sum() / denom * 100) if denom > 0 else float("nan"),
        "mae": float(error.mean()),
        "n_days": n_days,
    }


def idle_rate(station_id, origin_time_index, hours=168):
    """Share of the trailing `hours` observed as idle, ending at the origin.

    Reads Was_Idle_LastHour straight off the panel — no model needed.
    """
    frame = loader.station_frame(station_id)
    start = int(origin_time_index) - hours + 1
    idx = [i for i in range(start, int(origin_time_index) + 1) if i in frame.index]
    if not idx:
        return None
    return float(frame.loc[idx, "Was_Idle_LastHour"].mean())