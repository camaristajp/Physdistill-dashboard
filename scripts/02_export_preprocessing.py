"""
02_export_preprocessing.py

Writes data/serving/preprocessing.json — everything core/window.py needs to
build model inputs, with no sklearn or torch dependency at serving time.

    python scripts\\02_export_preprocessing.py

Why JSON rather than the pickles: unpickling sklearn estimators across
library versions is the classic deployment failure, and the Raspberry Pi will
not run the same versions as this machine. The scalers hold nine numbers each.

Why not label_encoders.pkl: pytorch-forecasting builds its own NaNLabelEncoder
inside TimeSeriesDataSet. The sklearn encoders filtered the training
vocabulary; they are not the mapping the embedding tables were trained on.

Why not the validation panel: it spans June-December only, so rebuilding a
Month vocabulary from it assigns June index 0 when the model expects 5. Class
counts must equal embedding table sizes exactly, and this script refuses to
write a mapping that does not.
"""

import json
import pickle
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO_ROOT / "data" / "tft_artifacts_jiaxing"
SCRIPTS_DIR = REPO_ROOT / "scripts"

SCALER_PKL = ARTIFACTS / "feature_scaler.pkl"
ENCODER_PKL = ARTIFACTS / "label_encoders.pkl"
TRAIN_CSV = ARTIFACTS / "train_data.csv"
CKPT_PATH = REPO_ROOT / "artifacts" / "physdistill_ev.ckpt"
PANEL = REPO_ROOT / "data" / "serving" / "jiaxing_panel.parquet"
OUT = REPO_ROOT / "data" / "serving" / "preprocessing.json"

ENCODER_LENGTH = 168
HORIZON = 24
QUANTILES = [0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98]

X_REALS = [
    "encoder_length",
    "Fee",
    "relative_time_idx",
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

DERIVED_REALS = ["encoder_length", "relative_time_idx"]

X_CATEGORICALS = [
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

HPARAM_SOURCES = ["embedding_labels", "categorical_encoders"]


def banner(text):
    print(f"\n{'=' * 62}\n  {text}\n{'=' * 62}")


def read_csv_any(path):
    """Comma first. A literal tab inside a station name breaks tab parsing."""
    for sep in (",", "\t"):
        try:
            frame = pd.read_csv(path, sep=sep)
        except pd.errors.ParserError:
            continue
        if frame.shape[1] > 1:
            return frame
    return None


def as_mapping(obj):
    """Coerce an encoder or label container into {value: index}."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): int(v) for k, v in obj.items()}
    classes = getattr(obj, "classes_", None)
    if classes is None:
        return None
    if isinstance(classes, dict):
        return {str(k): int(v) for k, v in classes.items()}
    return {str(v): i for i, v in enumerate(classes)}


# --------------------------------------------------------------------------
banner("1. Read the scaler pickle")

if not SCALER_PKL.exists():
    print(f"  FAILED: {SCALER_PKL} not found")
    sys.exit(1)

with open(SCALER_PKL, "rb") as handle:
    scaler_bundle = pickle.load(handle)

scalers = {}
for key in ("fee_scaler", "lag_scaler", "weather_scaler"):
    obj = scaler_bundle.get(key)
    if obj is None:
        continue
    names = list(getattr(obj, "feature_names_in_", []))
    for name, center, scale in zip(names, obj.center_, obj.scale_):
        scalers[name] = {
            "center": float(center),
            "scale": float(scale),
            "source": key,
        }
    print(f"  {key:16} {len(names)} column(s)")

# The bare 'scaler' key is an identity transform (center 0, scale 1) left over
# from an earlier pipeline. Using it would leave lag features unscaled.
if "scaler" in scaler_bundle:
    print("  ignoring 'scaler' key — identity transform, superseded by lag_scaler")

print(f"\n  columns with parameters: {len(scalers)}")


# --------------------------------------------------------------------------
banner("2. Load the checkpoint")

import torch  # noqa: E402
import torchmetrics  # noqa: E402

torchmetrics.Metric.device = property(lambda self: torch.device("cpu"))

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(REPO_ROOT))

# The checkpoint pickles the loss with module '__main__', because the training
# script defined it while running as the entry point. Importing these names
# into this script's namespace (also '__main__') is what lets torch.load
# resolve them. Dropping any of them reintroduces an AttributeError.
try:
    from train_all_jiaxingalpha import (  # noqa: F401
        QUANTILES as _TRAIN_QUANTILES,
        RawLossTFT,
        SparseQuantileLoss,
        StudentKDTFT,
    )
except ImportError as exc:
    missing = getattr(exc, "name", None) or str(exc).split()[-1].strip("'\"")
    print(f"  FAILED: {exc}")
    if missing and missing != "train_all_jiaxingalpha":
        print(f"\n      pip install {missing}\n")
    sys.exit(1)

if list(_TRAIN_QUANTILES) != QUANTILES:
    print(f"  WARNING: training quantiles {list(_TRAIN_QUANTILES)}")
    print(f"           differ from this script's {QUANTILES}")

model = StudentKDTFT.load_from_checkpoint(str(CKPT_PATH), map_location="cpu")
model.eval()

embedding_sizes = model.hparams.get("embedding_sizes", {})
cardinalities = {name: int(spec[0]) for name, spec in embedding_sizes.items()}
print(f"  embedding tables : {len(cardinalities)}")


# --------------------------------------------------------------------------
banner("3. Recover categorical vocabularies")

vocab = {}
vocab_source = {}

for source_key in HPARAM_SOURCES:
    container = model.hparams.get(source_key) or {}
    if not container:
        continue
    for name in X_CATEGORICALS:
        if name in vocab:
            continue
        mapping = as_mapping(container.get(name))
        if mapping:
            vocab[name] = mapping
            vocab_source[name] = source_key
    if vocab:
        print(f"  {source_key}: recovered {len(vocab)} vocabularies")

still_missing = [n for n in X_CATEGORICALS if n not in vocab]

if still_missing:
    print(f"  not in hparams: {still_missing}")
    print("\n  hparams keys available for inspection:")
    print(f"    {sorted(model.hparams.keys())}")

    # Fall back to train_data.csv, never the validation panel. The panel spans
    # June-December, so its Month vocabulary would be off by five.
    if not TRAIN_CSV.exists():
        print(f"\n  FAILED: {TRAIN_CSV} not found and hparams lack the mapping.")
        print("  Cannot reconstruct a trustworthy vocabulary. Aborting.")
        sys.exit(1)

    print(f"\n  rebuilding from {TRAIN_CSV.name} (full training range)")
    train_df = read_csv_any(TRAIN_CSV)
    if train_df is None:
        print("  FAILED: could not parse train_data.csv")
        sys.exit(1)

    for name in still_missing:
        if name not in train_df.columns:
            print(f"  FAILED: {name} absent from train_data.csv")
            sys.exit(1)
        values = sorted(train_df[name].dropna().unique(), key=lambda v: str(v))
        vocab[name] = {str(v): i for i, v in enumerate(values)}
        vocab_source[name] = "train_data.csv"
        print(f"    {name:24} {len(values)} classes")


# --------------------------------------------------------------------------
banner("4. Validate against embedding tables")

failures = []
for name in X_CATEGORICALS:
    mapping = vocab[name]
    table = cardinalities.get(name)
    highest = max(mapping.values()) if mapping else -1
    source = vocab_source.get(name, "?")

    if table is None:
        verdict = "FAIL — no embedding table"
        failures.append(name)
    elif len(mapping) != table:
        verdict = f"FAIL — {len(mapping)} classes vs {table} rows"
        failures.append(name)
    elif highest != table - 1:
        verdict = f"FAIL — highest index {highest}, expected {table - 1}"
        failures.append(name)
    else:
        verdict = "ok"

    print(f"  {name:24} n={len(mapping):>4} table={table:>4} [{source}] {verdict}")

if failures:
    print(f"\n  FAILED on: {failures}")
    print("  A vocabulary that does not exactly cover its embedding table means")
    print("  the model would read the wrong row. Forecasts would look plausible")
    print("  and be wrong. Recover the real mapping before continuing.")
    sys.exit(1)

print("\n  all vocabularies cover their tables exactly")


# --------------------------------------------------------------------------
banner("5. Check the panel against the vocabularies")

if not PANEL.exists():
    print(f"  FAILED: {PANEL} not found — run 01_build_panel.py first")
    sys.exit(1)

panel = pd.read_parquet(PANEL)

unmapped = {}
for name in X_CATEGORICALS:
    present = {str(v) for v in panel[name].dropna().unique()}
    missing_values = sorted(present.difference(vocab[name].keys()))
    if missing_values:
        unmapped[name] = missing_values
        print(f"  {name}: {len(missing_values)} unmapped -> {missing_values[:5]}")

if unmapped:
    print("\n  These need an <UNSEEN> fallback in core/window.py.")
else:
    print("  every panel value maps cleanly")

# Informational: the panel legitimately covers fewer classes than training.
for name in ("Month",):
    used = panel[name].nunique()
    print(f"  note: panel uses {used}/{cardinalities.get(name)} {name} classes")


# --------------------------------------------------------------------------
banner("6. Provenance of the sklearn pickle")

sklearn_vocab_sizes = {}
if ENCODER_PKL.exists():
    with open(ENCODER_PKL, "rb") as handle:
        sk_encoders = pickle.load(handle)
    for name, enc in sk_encoders.items():
        sklearn_vocab_sizes[name] = len(getattr(enc, "classes_", []))
        model_count = cardinalities.get(name, "-")
        note = "" if sklearn_vocab_sizes[name] == model_count else "  <- not the model's mapping"
        print(f"  {name:24} {sklearn_vocab_sizes[name]}{note}")
else:
    print("  label_encoders.pkl not present")


# --------------------------------------------------------------------------
banner("7. Write preprocessing.json")

payload = {
    "model": "physdistill_ev",
    "dataset": "jiaxing",
    "encoder_length": ENCODER_LENGTH,
    "prediction_length": HORIZON,
    "quantiles": QUANTILES,
    "target": {
        "column": "Energy_kWh_log",
        "transform": "log1p",
        "inverse": "expm1",
        # GroupNormalizer(transformation=None, center=False,
        # scale_by_group=False) is an identity transform, so target_scale is
        # a constant [0, 1] per row and needs no per-window inversion.
        "normalizer": "identity",
        "target_scale": [0.0, 1.0],
        "floor_at_zero": True,
    },
    "x_reals": X_REALS,
    "derived_reals": DERIVED_REALS,
    "panel_reals": [r for r in X_REALS if r not in DERIVED_REALS],
    "x_categoricals": X_CATEGORICALS,
    "cardinalities": cardinalities,
    "vocab": vocab,
    "vocab_source": vocab_source,
    "scalers": scalers,
    # val_data.csv was written post-scaling, so window.py must NOT reapply
    # these. They are kept for inverse transforms when showing real units.
    "scalers_already_applied_to_panel": True,
    "sklearn_vocab_sizes": sklearn_vocab_sizes,
    "unmapped_panel_values": unmapped,
    "notes": [
        "encoder_cont and decoder_cont follow x_reals order exactly.",
        "encoder_length and relative_time_idx are derived per window.",
        "Station keys may contain trailing whitespace; match on encoder_key.",
        "Vocabularies cover the full training range, not just the panel.",
    ],
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, ensure_ascii=False)

print(f"  wrote : {OUT.relative_to(REPO_ROOT)}")
print(f"  size  : {OUT.stat().st_size / 1024:.1f} KB")
print(f"  reals : {len(X_REALS)} ({len(DERIVED_REALS)} derived)")
print(f"  cats  : {len(X_CATEGORICALS)}")
print("\n  Next: 04_score_stations.py")