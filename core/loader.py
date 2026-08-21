"""
core/loader.py

Cached reads of the serving artifacts. Pure Python caching so the module works
outside Streamlit too — the pages wrap these in st.cache_data where useful.
"""

import json
from functools import lru_cache

import pandas as pd

from core import config


@lru_cache(maxsize=1)
def load_preprocessing():
    config.require(
        config.PREPROCESSING_PATH,
        "Run scripts/02_export_preprocessing.py and 03b_extract_dataset_scalers.py",
    )
    with open(config.PREPROCESSING_PATH, encoding="utf-8") as handle:
        payload = json.load(handle)

    for key in ("x_reals", "x_categoricals", "vocab", "derived_scalers",
                "panel_column_transforms"):
        if key not in payload:
            raise KeyError(f"preprocessing.json is missing '{key}'")
    return payload


@lru_cache(maxsize=1)
def load_panel():
    config.require(config.PANEL_PATH, "Run scripts/01_build_panel.py")
    panel = pd.read_parquet(config.PANEL_PATH)
    return panel.sort_values(["station_id", "Time_Index"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def load_station_meta():
    config.require(config.STATION_META_PATH, "Run scripts/05_seed_capacity.py")
    return pd.read_csv(config.STATION_META_PATH)


@lru_cache(maxsize=1)
def load_station_scores():
    config.require(config.STATION_SCORES_PATH, "Run scripts/04_score_stations.py")
    return pd.read_csv(config.STATION_SCORES_PATH)


@lru_cache(maxsize=1)
def load_model_manifest():
    """Parameter counts and epoch per checkpoint, plus the paper's external
    record (KD alpha, Table 5 WMAPE, reduction factor, Pi 5 latency) — none
    of which a checkpoint can prove about itself. See core/config.require's
    hint and scripts/07_export_model_manifest.py for provenance.
    """
    config.require(
        config.MODEL_MANIFEST_PATH,
        "Run scripts/07_export_model_manifest.py",
    )
    with open(config.MODEL_MANIFEST_PATH, encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_comparison_medians():
    """Baseline-student median forecasts, keyed by (station_id, origin_time_index).

    Optional: unlike the other serving files, the dashboard runs fine without
    this one. It only backs the Stations page's compare toggle, so a missing
    file disables that toggle rather than failing the app.
    """
    if not config.COMPARISON_MEDIANS_PATH.exists():
        return None
    return pd.read_parquet(config.COMPARISON_MEDIANS_PATH)


@lru_cache(maxsize=1)
def station_index():
    """station_id -> {encoder_key, display_name, group, capacity_kwh, ...}"""
    meta = load_station_meta()
    return {row.station_id: row._asdict() for row in meta.itertuples(index=False)}


@lru_cache(maxsize=None)
def station_frame(station_id):
    """One station's rows, time-ordered, indexed by Time_Index for fast slicing."""
    panel = load_panel()
    frame = panel[panel["station_id"] == station_id]
    if frame.empty:
        raise KeyError(f"Unknown station_id: {station_id}")
    return frame.set_index("Time_Index", drop=False).sort_index()


@lru_cache(maxsize=None)
def valid_origins(station_id):
    """Time_Index values that can serve as a forecast origin, ascending."""
    frame = station_frame(station_id)
    return tuple(frame.loc[frame["is_valid_origin"], "Time_Index"].tolist())


def origin_calendar(station_id):
    """Timestamp for every valid origin — what the date picker binds to."""
    frame = station_frame(station_id)
    origins = list(valid_origins(station_id))
    return frame.loc[origins, ["Time_Index", "Timestamp"]].reset_index(drop=True)


def active_stations():
    """Stations with usable forecasts, ordered by group then name."""
    meta = load_station_meta()
    active = meta[meta["active"]].copy()
    return active.sort_values(["group", "display_name"]).reset_index(drop=True)
