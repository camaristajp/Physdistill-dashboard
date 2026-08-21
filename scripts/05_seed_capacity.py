"""
05_seed_capacity.py

Writes data/serving/station_meta.csv — display names, groupings, and the
capacity threshold every alert on the dashboard is measured against.

    python scripts\\05_seed_capacity.py
    python scripts\\05_seed_capacity.py --force --percentile 0.99 --headroom 1.10

The Jiaxing dataset carries no rated capacity, so this derives a proxy from
the upper tail of each station's training-period hourly demand, with a small
headroom factor and rounding to a plausible increment.

Calibration matters more than it looks. At p95 roughly one training hour in
twenty already exceeds the threshold, so the Alerts page fills with permanent
breaches and stops carrying information. A real rating sits above the observed
envelope. p99 x 1.10 puts it near the top of historical demand, so a forecast
crossing it is a signal rather than the norm. The training exceedance rate is
printed for every station — that is the floor on your alert volume.

To use real ratings, edit capacity_kwh and set capacity_source to 'rated'.
Reruns will not overwrite an existing file unless you pass --force.
"""

import argparse
import math
import sys
import unicodedata
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = REPO_ROOT / "data" / "tft_artifacts_jiaxing" / "train_data.csv"
SCORES = REPO_ROOT / "data" / "serving" / "station_scores.csv"
OUT = REPO_ROOT / "data" / "serving" / "station_meta.csv"

DEFAULT_PERCENTILE = 0.99
DEFAULT_HEADROOM = 1.10
ROUND_TO_KWH = 10

# Grouping reflects demand regime, which is how an operator reads these
# stations: a fleet depot and a tourist site fail in different ways.
GROUPS = {
    "bus_station": "schedule_driven",
    "industrial_park": "schedule_driven",
    "government_agency": "schedule_driven",
    "shopping_mall": "retail_mixed",
    "wholesale_market": "retail_mixed",
    "financial_industrial_park": "retail_mixed",
    "expressway_service_district_a": "corridor",
    "expressway_service_district_b": "corridor",
    "expressway_service_district_c": "corridor",
    "park_a": "recreation",
    "park_b": "recreation",
    "tourist_attraction": "recreation",
    "technology_park": "recreation",
}


def banner(text):
    print(f"\n{'=' * 62}\n  {text}\n{'=' * 62}")


def slugify(name):
    """'Industrial Park\\t' -> 'industrial_park'. Matches 01_build_panel.py."""
    cleaned = unicodedata.normalize("NFKD", str(name)).strip()
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in cleaned)
    return "_".join(cleaned.lower().split())


def display_name(encoder_key):
    """Trim the stray tab; keep the operator-facing spelling otherwise."""
    return " ".join(str(encoder_key).split())


def read_csv_any(path):
    for sep in (",", "\t"):
        try:
            frame = pd.read_csv(path, sep=sep)
        except pd.errors.ParserError:
            continue
        if frame.shape[1] > 1:
            return frame
    return None


def round_up(value, increment):
    """A rating is a round number on a form, not a percentile to one decimal."""
    if not math.isfinite(value) or value <= 0:
        return float(increment)
    return float(math.ceil(value / increment) * increment)


parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true", help="overwrite an existing file")
parser.add_argument(
    "--percentile",
    type=float,
    default=DEFAULT_PERCENTILE,
    help=f"upper-tail percentile of training demand (default {DEFAULT_PERCENTILE})",
)
parser.add_argument(
    "--headroom",
    type=float,
    default=DEFAULT_HEADROOM,
    help=f"multiplier applied above the percentile (default {DEFAULT_HEADROOM})",
)
args = parser.parse_args()


# --------------------------------------------------------------------------
banner("1. Guard against clobbering edits")

if OUT.exists() and not args.force:
    existing = pd.read_csv(OUT)
    rated = int((existing.get("capacity_source") == "rated").sum())
    print(f"  {OUT.name} already exists ({len(existing)} rows, {rated} rated)")
    print("  Not overwriting. Pass --force to regenerate.")
    sys.exit(0)

print("  clear to write")


# --------------------------------------------------------------------------
banner("2. Derive capacity from training demand")

if not TRAIN_CSV.exists():
    print(f"  FAILED: {TRAIN_CSV} not found")
    sys.exit(1)

train = read_csv_any(TRAIN_CSV)
if train is None:
    print("  FAILED: could not parse train_data.csv")
    sys.exit(1)

if "Energy_kWh" not in train.columns:
    print("  FAILED: Energy_kWh not in train_data.csv")
    sys.exit(1)

train["station_id"] = train["Location_Information"].map(slugify)

agg = (
    train.groupby("station_id")
    .agg(
        encoder_key=("Location_Information", "first"),
        p_tail=("Energy_kWh", lambda s: float(s.quantile(args.percentile))),
        peak_observed=("Energy_kWh", "max"),
        mean_demand=("Energy_kWh", "mean"),
        n_hours=("Energy_kWh", "size"),
    )
    .reset_index()
)

agg["capacity_kwh"] = [
    round_up(row.p_tail * args.headroom, ROUND_TO_KWH) for row in agg.itertuples()
]

# Exceedance rate in training is the floor on how often alerts will fire.
exceed = {}
for station_id, group in train.groupby("station_id"):
    threshold = float(agg.loc[agg.station_id == station_id, "capacity_kwh"].iloc[0])
    exceed[station_id] = float((group["Energy_kWh"] > threshold).mean() * 100)
agg["train_exceedance_pct"] = agg["station_id"].map(exceed).round(2)

print(f"  stations   : {len(agg)}")
print(f"  basis      : p{args.percentile:.2%} x {args.headroom:.2f}, rounded up to {ROUND_TO_KWH} kWh")


# --------------------------------------------------------------------------
banner("3. Assemble metadata")

agg["display_name"] = agg["encoder_key"].map(display_name)
agg["group"] = agg["station_id"].map(GROUPS).fillna("ungrouped")
agg["capacity_source"] = "derived"

ungrouped = agg.loc[agg["group"] == "ungrouped", "station_id"].tolist()
if ungrouped:
    print(f"  WARNING: no group assigned for {ungrouped}")

if SCORES.exists():
    scores = pd.read_csv(SCORES)[["station_id", "confidence_tier", "active"]]
    agg = agg.merge(scores, on="station_id", how="left")
    agg["active"] = agg["active"].fillna(False)
    agg["confidence_tier"] = agg["confidence_tier"].fillna("unavailable")
    print("  merged confidence tiers from station_scores.csv")
else:
    agg["confidence_tier"] = "unknown"
    agg["active"] = True
    print("  WARNING: station_scores.csv absent — run 04 first for tiers")

agg["p_tail"] = agg["p_tail"].round(1)
agg["peak_observed"] = agg["peak_observed"].round(1)
agg["mean_demand"] = agg["mean_demand"].round(2)

agg = agg.sort_values(["group", "station_id"]).reset_index(drop=True)

current_group = None
for row in agg.itertuples():
    if row.group != current_group:
        current_group = row.group
        print(f"\n  {current_group}")
    ratio = row.peak_observed / row.capacity_kwh if row.capacity_kwh else float("nan")
    print(
        f"    {row.display_name:32} cap={row.capacity_kwh:>6.0f}"
        f"  peak={row.peak_observed:>7.1f} (x{ratio:.2f})"
        f"  exceeded {row.train_exceedance_pct:>5.2f}% of hours"
    )


# --------------------------------------------------------------------------
banner("4. Calibration check")

high_load = agg.loc[agg["train_exceedance_pct"] > 2.0, "display_name"].tolist()
never_hit = agg.loc[agg["train_exceedance_pct"] == 0.0, "display_name"].tolist()

print(f"  median exceedance : {agg['train_exceedance_pct'].median():.2f}% of hours")
if high_load:
    print(f"  above 2% (noisy)  : {high_load}")
    print("    consider --percentile 0.995 or --headroom 1.20")
if never_hit:
    print(f"  never exceeded    : {never_hit}")
    print("    thresholds here may be too high to ever alert")
if not high_load and not never_hit:
    print("  every station lands in a usable band")


# --------------------------------------------------------------------------
banner("5. Write station_meta.csv")

columns = [
    "station_id",
    "encoder_key",
    "display_name",
    "group",
    "capacity_kwh",
    "capacity_source",
    "p_tail",
    "peak_observed",
    "mean_demand",
    "train_exceedance_pct",
    "n_hours",
    "confidence_tier",
    "active",
]

OUT.parent.mkdir(parents=True, exist_ok=True)
agg[columns].to_csv(OUT, index=False)

print(f"  wrote : {OUT.relative_to(REPO_ROOT)}")
print(f"  rows  : {len(agg)}")
print("\n  capacity_source is 'derived'. The dashboard must label the")
print("  threshold line as a derived bound, not a rated capacity.")
print("\n  Next: core/window.py")