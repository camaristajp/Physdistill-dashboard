"""
04_score_stations.py

Writes data/serving/station_scores.csv — per-station accuracy used for the
confidence badges on the station rail and the About tab.

    python scripts\\04_score_stations.py

Source is actualvspred_physdistill_ev.csv (694,824 rows: 13 stations x 2,227
sequences x 24 horizons). Metrics are recomputed here rather than trusting the
file's seq_r2 column, so the definition matches the paper's evaluation.

Note on absolute values: this file was generated under what the paper calls
the original evaluation protocol, so global WMAPE lands near 15.7% rather than
Table 5's 13.28%. Relative ordering across stations is what the badges use.
"""

import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "data" / "tft_artifacts_jiaxing" / "actualvspred_physdistill_ev.csv"
OUT = REPO_ROOT / "data" / "serving" / "station_scores.csv"

# Tiers follow the per-station R2 spread reported in the paper's heatmap.
TIER_HIGH = 0.90
TIER_MODERATE = 0.70


def banner(text):
    print(f"\n{'=' * 62}\n  {text}\n{'=' * 62}")


def slugify(name):
    """'Industrial Park\\t' -> 'industrial_park'. Matches 01_build_panel.py."""
    cleaned = unicodedata.normalize("NFKD", str(name)).strip()
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in cleaned)
    return "_".join(cleaned.lower().split())


def tier_for(r2, n_valid):
    if n_valid == 0 or not np.isfinite(r2):
        return "unavailable"
    if r2 >= TIER_HIGH:
        return "high"
    if r2 >= TIER_MODERATE:
        return "moderate"
    return "low"


# --------------------------------------------------------------------------
banner("1. Read predictions")

if not SRC.exists():
    print(f"  FAILED: {SRC} not found")
    sys.exit(1)

df = pd.read_csv(SRC)
required = {"station", "horizon_h", "actual", "prediction"}
missing = sorted(required.difference(df.columns))
if missing:
    print(f"  FAILED: missing columns {missing}")
    sys.exit(1)

df["station_id"] = df["station"].map(slugify)
df["encoder_key"] = df["station"]

print(f"  rows     : {len(df):,}")
print(f"  stations : {df.station_id.nunique()}")
print(f"  horizons : {df.horizon_h.min()}-{df.horizon_h.max()}")


# --------------------------------------------------------------------------
banner("2. Score each station")


def metrics(group):
    actual = group["actual"].to_numpy(dtype=float)
    pred = group["prediction"].to_numpy(dtype=float)
    mask = np.isfinite(actual) & np.isfinite(pred)
    actual, pred = actual[mask], pred[mask]

    if actual.size == 0:
        return pd.Series(
            {
                "n_points": 0,
                "r2": np.nan,
                "wmape": np.nan,
                "mae": np.nan,
                "rmse": np.nan,
                "mean_actual": np.nan,
            }
        )

    error = actual - pred
    denom = np.abs(actual).sum()
    variance = ((actual - actual.mean()) ** 2).sum()

    return pd.Series(
        {
            "n_points": int(actual.size),
            # Degenerate when a station has no demand variance in the window.
            "r2": float(1 - (error**2).sum() / variance) if variance > 0 else np.nan,
            "wmape": float(np.abs(error).sum() / denom * 100) if denom > 0 else np.nan,
            "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt((error**2).mean())),
            "mean_actual": float(actual.mean()),
        }
    )


scores = df.groupby("station_id").apply(metrics).reset_index()

keys = df.groupby("station_id")["encoder_key"].first().reset_index()
scores = scores.merge(keys, on="station_id")

scores["n_sequences"] = (scores["n_points"] // 24).astype(int)
scores["confidence_tier"] = [
    tier_for(row.r2, row.n_points) for row in scores.itertuples()
]
scores["active"] = scores["confidence_tier"] != "unavailable"

scores = scores.sort_values("r2", ascending=False, na_position="last").reset_index(
    drop=True
)

for row in scores.itertuples():
    r2 = "     n/a" if not np.isfinite(row.r2) else f"{row.r2:8.3f}"
    wmape = "    n/a" if not np.isfinite(row.wmape) else f"{row.wmape:6.2f}%"
    print(f"  {row.station_id:32} R2={r2}  WMAPE={wmape}  [{row.confidence_tier}]")


# --------------------------------------------------------------------------
banner("3. Global figures")

actual = df["actual"].to_numpy(dtype=float)
pred = df["prediction"].to_numpy(dtype=float)
mask = np.isfinite(actual) & np.isfinite(pred)
actual, pred = actual[mask], pred[mask]
error = actual - pred

print(f"  WMAPE : {np.abs(error).sum() / np.abs(actual).sum() * 100:6.2f}%")
print(f"  MAE   : {np.abs(error).mean():6.2f} kWh")
print(f"  RMSE  : {np.sqrt((error ** 2).mean()):6.2f} kWh")
print(f"  R2    : {1 - (error ** 2).sum() / ((actual - actual.mean()) ** 2).sum():6.3f}")
print("\n  These reflect the original evaluation protocol, not Table 5.")
print("  Use them for station ordering, not for the About tab headline.")


# --------------------------------------------------------------------------
banner("4. Write station_scores.csv")

columns = [
    "station_id",
    "encoder_key",
    "r2",
    "wmape",
    "mae",
    "rmse",
    "mean_actual",
    "n_points",
    "n_sequences",
    "confidence_tier",
    "active",
]

OUT.parent.mkdir(parents=True, exist_ok=True)
scores[columns].to_csv(OUT, index=False)

print(f"  wrote : {OUT.relative_to(REPO_ROOT)}")
print(f"  rows  : {len(scores)}")
print(f"  tiers : {scores.confidence_tier.value_counts().to_dict()}")
print("\n  Next: 05_seed_capacity.py")