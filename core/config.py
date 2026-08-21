"""
core/config.py

Paths and constants. Everything is relative to the repo root and overridable
by environment variable, so the same code runs on Windows and on the Pi.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = Path(os.getenv("PHYSDISTILL_ARTIFACTS", REPO_ROOT / "artifacts"))
SERVING_DIR = Path(os.getenv("PHYSDISTILL_SERVING", REPO_ROOT / "data" / "serving"))

ONNX_PATH = ARTIFACTS_DIR / "physdistill_ev_jiaxing.onnx"
PANEL_PATH = SERVING_DIR / "jiaxing_panel.parquet"
PREPROCESSING_PATH = SERVING_DIR / "preprocessing.json"
STATION_META_PATH = SERVING_DIR / "station_meta.csv"
STATION_SCORES_PATH = SERVING_DIR / "station_scores.csv"
COMPARISON_MEDIANS_PATH = SERVING_DIR / "comparison_medians.parquet"
MODEL_MANIFEST_PATH = SERVING_DIR / "model_manifest.json"

ENCODER_LENGTH = 168
HORIZON = 24
QUANTILES = [0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98]

Q_LOW = 0.10
Q_MEDIAN = 0.50
Q_HIGH = 0.90

# Severity tiers. Warning when the reserve requirement crosses capacity,
# critical when expected demand does.
SEVERITY_WARNING_QUANTILE = Q_HIGH
SEVERITY_CRITICAL_QUANTILE = Q_MEDIAN

# Single-thread inference, matching the paper's Raspberry Pi benchmark.
ORT_INTRA_OP_THREADS = 1
ORT_INTER_OP_THREADS = 1

CONFIDENCE_TIERS = {
    "high": "Forecasts track demand closely at this station.",
    "moderate": "Usable, but expect wider errors than the network average.",
    "low": "Weak temporal structure — treat forecasts as indicative only.",
    "unavailable": "No validation data. No forecast is offered.",
}


def require(path, hint=""):
    """Fail loudly at import rather than mysteriously at first use."""
    if not Path(path).exists():
        message = f"Required file missing: {path}"
        if hint:
            message += f"\n  {hint}"
        raise FileNotFoundError(message)
    return Path(path)
