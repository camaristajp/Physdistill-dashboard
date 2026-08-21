"""
01_build_panel.py

Converts val_data.csv into data/serving/jiaxing_panel.parquet, the single
table every dashboard page reads from.

    python scripts\\01_build_panel.py

What it does:
  - reads the source without tripping over the literal tab in a station name
  - parses Timestamp with an explicit format (never infer on mixed US dates)
  - keeps only the columns the model consumes, plus display fields
  - drops the duplicate non-_f IsCharging columns
  - adds station_id, a URL-safe slug, alongside the exact encoder key
  - marks which hours are valid forecast origins
"""

import sys
import unicodedata
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "data" / "tft_artifacts_jiaxing" / "val_data.csv"
OUT = REPO_ROOT / "data" / "serving" / "jiaxing_panel.parquet"

# Origins before this are inside the training window and must not be selectable.
VAL_LOSS_START = pd.Timestamp("2021-10-01 00:00")
ENCODER_LENGTH = 168
HORIZON = 24

# The 13 model reals that come from the dataframe. encoder_length and
# relative_time_idx are the other two in x_reals; both are derived per window
# in core/window.py and are deliberately absent here.
MODEL_REALS = [
    "Fee",
    "Lagged_Energy_168h",
    "Rolling_Mean_Energy_24h",
    "Lagged_Energy_24h",
    "Rolling_Mean_Energy_168h",
    "IsCharging_Lag1h_f",
    "IsCharging_Lag24h_f",
    "Was_Idle_LastHour",
    "Temperature_C",
    "Relative_Humidity",
    "Precipitation_mm",
    "Is_Abnormal",
    "Is_Covid_Lockdown",
]

MODEL_CATS = [
    "Location_Information",
    "District_Name",
    "Month",
    "Hour_of_Day",
    "Day_of_Week",
    "Is_Holiday",
    "Is_Weekend",
    "ToU_Period",
    "Hour_of_Week",
]

DISPLAY_COLS = ["Timestamp", "Time_Index", "Energy_kWh", "Energy_kWh_log"]

DROP_COLS = ["IsCharging_Lag1h", "IsCharging_Lag24h", "Split"]


def banner(text):
    print(f"\n{'=' * 62}\n  {text}\n{'=' * 62}")


def slugify(name):
    """'Industrial Park\\t' -> 'industrial_park'. URL-safe, stable, lossy."""
    cleaned = unicodedata.normalize("NFKD", str(name)).strip()
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in cleaned)
    return "_".join(cleaned.lower().split())


def read_source(path):
    """Comma first. A literal tab inside a station name breaks tab parsing."""
    for sep, label in ((",", "comma"), ("\t", "tab")):
        try:
            frame = pd.read_csv(path, sep=sep)
        except pd.errors.ParserError:
            continue
        if frame.shape[1] > 1:
            print(f"  delimiter : {label}")
            return frame
    print("  FAILED: could not parse as comma- or tab-separated")
    sys.exit(1)


# --------------------------------------------------------------------------
banner("1. Read source")

if not SRC.exists():
    print(f"  FAILED: {SRC} not found")
    sys.exit(1)

df = read_source(SRC)

print(f"  rows      : {len(df):,}")
print(f"  columns   : {df.shape[1]}")


# --------------------------------------------------------------------------
banner("2. Validate schema")

required = set(MODEL_REALS) | set(MODEL_CATS) | {"Timestamp", "Time_Index", "Energy_kWh"}
missing = sorted(required.difference(df.columns))
if missing:
    print(f"  FAILED: missing columns {missing}")
    sys.exit(1)

print(f"  model reals present : {len(MODEL_REALS)}/13")
print(f"  model cats present  : {len(MODEL_CATS)}/9")

dropped = [c for c in DROP_COLS if c in df.columns]
if dropped:
    print(f"  dropping            : {dropped}")


# --------------------------------------------------------------------------
banner("3. Parse timestamps")

# Source is Excel-style M/D/YYYY H:MM. Inferring on a file this size can swap
# day and month on ambiguous rows, so the format is stated.
ts = pd.to_datetime(df["Timestamp"], format="%m/%d/%Y %H:%M", errors="coerce")
if ts.isna().any():
    fallback = pd.to_datetime(df["Timestamp"], errors="coerce")
    if fallback.isna().any():
        bad = df.loc[fallback.isna(), "Timestamp"].head(3).tolist()
        print(f"  FAILED: unparseable timestamps, e.g. {bad}")
        sys.exit(1)
    print("  note: explicit format failed, fell back to inference — verify output")
    ts = fallback

df["Timestamp"] = ts
print(f"  range : {ts.min()}  ->  {ts.max()}")
print(f"  hours : {ts.nunique():,}")


# --------------------------------------------------------------------------
banner("4. Station identity")

# The encoder vocabulary contains 'Industrial Park\t' with a trailing tab.
# encoder_key preserves it verbatim; station_id is what the UI and URLs use.
df["encoder_key"] = df["Location_Information"]
df["station_id"] = df["Location_Information"].map(slugify)

stations = (
    df[["station_id", "encoder_key"]]
    .drop_duplicates()
    .sort_values("station_id")
    .reset_index(drop=True)
)
whitespace_keys = 0
for row in stations.itertuples():
    dirty = row.encoder_key != str(row.encoder_key).strip()
    whitespace_keys += int(dirty)
    flag = "  <- whitespace preserved" if dirty else ""
    print(f"  {row.station_id:32} {row.encoder_key!r}{flag}")

print(f"\n  stations: {len(stations)}")
if whitespace_keys == 0:
    print("  NOTE: no whitespace found in any encoder key.")
    print("  The label encoder expects 'Industrial Park\\t'. If the tab was")
    print("  stripped on read, that station will fall through to <UNSEEN>.")


# --------------------------------------------------------------------------
banner("5. Mark selectable forecast origins")

# A valid origin needs ENCODER_LENGTH hours of history behind it, HORIZON
# hours of ground truth ahead, and must sit at or past the validation start.
per_station_max = df.groupby("station_id")["Timestamp"].transform("max")
horizon_room = df["Timestamp"] <= per_station_max - pd.Timedelta(hours=HORIZON)
lookback_room = df["Timestamp"] >= ts.min() + pd.Timedelta(hours=ENCODER_LENGTH)

df["is_valid_origin"] = (
    (df["Timestamp"] >= VAL_LOSS_START) & horizon_room & lookback_room
)

valid = df.loc[df["is_valid_origin"], "Timestamp"]
if valid.empty:
    print("  FAILED: no valid forecast origins — check the date range")
    sys.exit(1)

print(f"  validation start : {VAL_LOSS_START.date()}")
print(f"  origin range     : {valid.min()}  ->  {valid.max()}")
print(f"  selectable hours : {valid.nunique():,}")
print(f"  selectable days  : {valid.dt.date.nunique()}")


# --------------------------------------------------------------------------
banner("6. Write parquet")

keep = (
    ["station_id", "encoder_key"]
    + DISPLAY_COLS
    + MODEL_CATS
    + MODEL_REALS
    + ["is_valid_origin"]
)
keep = [c for c in dict.fromkeys(keep) if c in df.columns]

panel = (
    df[keep]
    .sort_values(["station_id", "Time_Index"])
    .reset_index(drop=True)
)

OUT.parent.mkdir(parents=True, exist_ok=True)
panel.to_parquet(OUT, index=False, compression="snappy")

size_mb = OUT.stat().st_size / 1024 / 1024
print(f"  wrote   : {OUT.relative_to(REPO_ROOT)}")
print(f"  rows    : {len(panel):,}")
print(f"  columns : {len(panel.columns)}")
print(f"  size    : {size_mb:.1f} MB")


# --------------------------------------------------------------------------
banner("7. Sanity checks")

gaps = 0
for sid, group in panel.groupby("station_id"):
    diffs = group["Time_Index"].diff().dropna()
    if not (diffs == 1).all():
        gaps += 1
        print(f"  WARNING: {sid} has non-contiguous Time_Index")
if gaps == 0:
    print("  Time_Index contiguous for every station")

nulls = panel[MODEL_REALS + MODEL_CATS].isna().sum()
nulls = nulls[nulls > 0]
if len(nulls):
    print("  WARNING: nulls in model inputs —")
    print(nulls.to_string())
else:
    print("  no nulls in model inputs")

zero_share = (panel["Energy_kWh"] <= 0).mean() * 100
print(f"  zero-demand hours : {zero_share:.1f}%")
print("\n  Next: 02_export_preprocessing.py")