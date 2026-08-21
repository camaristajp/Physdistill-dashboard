"""
core/postprocess.py

Turns raw graph output into kilowatt-hours an operator can act on.

Four steps, in order:

  1. target rescaling — GroupNormalizer(center=False, scale_by_group=False)
     reads like an identity transform and is not one. center=False disables
     centering; a global scale is still fitted. The ONNX wrapper passed
     target_scale=[0, 1], so the graph does not apply it and it is applied
     here: z = center + scale * raw. Skipping this gives forecasts with the
     right shape and the wrong magnitude.
  2. expm1 — the target was trained as log1p(Energy_kWh).
  3. floor at zero — demand is non-negative.
  4. monotone rearrangement — pinball loss constrains each quantile head
     independently, so nothing enforces q02 <= q10 <= ... The paper measures
     34.58% crossing on Jiaxing for this model. Sorting cannot increase pinball
     loss and costs one sort per forecast.

The rearrangement is reported rather than applied silently. An operator reading
a capacity bound deserves to know the ordering was corrected.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core import config, loader


@dataclass(frozen=True)
class Forecast:
    frame: pd.DataFrame
    quantiles: list
    rearranged: bool
    crossing_pairs: int
    latency_ms: float
    origin_timestamp: object

    def median(self):
        return self.frame[f"q{config.Q_MEDIAN:.2f}"].to_numpy()

    def band(self, low=config.Q_LOW, high=config.Q_HIGH):
        return (
            self.frame[f"q{low:.2f}"].to_numpy(),
            self.frame[f"q{high:.2f}"].to_numpy(),
        )

    def expected_energy(self):
        """Total kWh over the horizon, with bounds from the outer quantiles.

        Summing quantiles pointwise is not the quantile of the sum, so the
        range is labelled as summed hourly bounds rather than an interval on
        the total.
        """
        return {
            "p50_total": float(self.median().sum()),
            "p10_total": float(self.frame[f"q{config.Q_LOW:.2f}"].sum()),
            "p90_total": float(self.frame[f"q{config.Q_HIGH:.2f}"].sum()),
            "basis": "hourly bounds summed",
        }

    def peak(self):
        median = self.median()
        position = int(np.argmax(median))
        return {
            "horizon_h": position + 1,
            "timestamp": self.frame["timestamp"].iloc[position],
            "p50": float(median[position]),
            "p90": float(self.frame[f"q{config.Q_HIGH:.2f}"].iloc[position]),
        }


def target_scale():
    """(center, scale) from the fitted normalizer, via preprocessing.json."""
    target = loader.load_preprocessing().get("target", {})
    pair = target.get("target_scale", [0.0, 1.0])
    center = float(pair[0]) if len(pair) > 0 else 0.0
    scale = float(pair[1]) if len(pair) > 1 else 1.0
    if scale == 0:
        scale = 1.0
    return center, scale


def to_kwh(raw_output, center=None, scale=None):
    """Undo normalization and log1p, then clamp. raw is [horizon, n_quantiles]."""
    if center is None or scale is None:
        center, scale = target_scale()
    values = np.asarray(raw_output, dtype=np.float64)
    log_space = center + scale * values
    return np.maximum(np.expm1(log_space), 0.0)


def count_crossings(values):
    """Adjacent-pair violations across quantile levels, before sorting."""
    if values.ndim != 2:
        raise ValueError("expected [horizon, n_quantiles]")
    return int((np.diff(values, axis=1) < 0).sum())


def build_forecast(raw_output, timestamps, latency_ms=0.0, origin_timestamp=None,
                   rearrange=True):
    """Assemble a tidy forecast frame from one graph output."""
    kwh = to_kwh(raw_output)
    horizon, n_quantiles = kwh.shape

    if n_quantiles != len(config.QUANTILES):
        raise ValueError(
            f"model returned {n_quantiles} quantiles, expected {len(config.QUANTILES)}"
        )
    if len(timestamps) != horizon:
        raise ValueError(f"{len(timestamps)} timestamps for a {horizon}-step horizon")

    crossings = count_crossings(kwh)
    if rearrange and crossings:
        kwh = np.sort(kwh, axis=1)

    frame = pd.DataFrame(
        {f"q{q:.2f}": kwh[:, i] for i, q in enumerate(config.QUANTILES)}
    )
    frame.insert(0, "horizon_h", np.arange(1, horizon + 1))
    frame.insert(1, "timestamp", list(timestamps))

    return Forecast(
        frame=frame,
        quantiles=list(config.QUANTILES),
        rearranged=bool(rearrange and crossings),
        crossing_pairs=crossings,
        latency_ms=float(latency_ms),
        origin_timestamp=origin_timestamp,
    )


def attach_actuals(forecast, actuals):
    """Join ground truth onto a forecast frame for the accuracy overlay."""
    if actuals is None or actuals.empty:
        return forecast.frame.assign(actual=np.nan)
    merged = forecast.frame.copy()
    merged["actual"] = actuals["Energy_kWh"].to_numpy()[: len(merged)]
    return merged