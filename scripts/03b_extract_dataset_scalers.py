"""
03b_extract_dataset_scalers.py

Recovers how TimeSeriesDataSet transforms every continuous channel, then
merges the result into data/serving/preprocessing.json.

    python scripts\\03b_extract_dataset_scalers.py

Reading dataset.scalers alone is not enough. An unfitted scaler and an
identity scaler look the same from the outside, and it is not obvious whether
a given column is transformed at batch time at all.

So this pulls real batches and compares encoder_cont against the source
dataframe, column by column. Whatever the model actually receives is the
contract core/window.py has to reproduce.

Binary columns are often constant across a single window, which leaves no
spread to fit a line through. Those are checked point-by-point against the
declared scaler instead, across several batches, so every column is verified
rather than inherited by assumption.

This is the last torch-dependent step. Everything after it reads JSON.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
ARTIFACTS = REPO_ROOT / "data" / "tft_artifacts_jiaxing"

TRAIN_CSV = ARTIFACTS / "train_data.csv"
VAL_CSV = ARTIFACTS / "val_data.csv"
SCALER_PKL = ARTIFACTS / "feature_scaler.pkl"
ENCODER_PKL = ARTIFACTS / "label_encoders.pkl"
STUDENT_CFG = ARTIFACTS / "student_tft_config.json"
PREPROCESSING = REPO_ROOT / "data" / "serving" / "preprocessing.json"

DERIVED = ["encoder_length", "relative_time_idx"]
TOLERANCE = 1e-4
N_BATCHES = 8


def banner(text):
    print(f"\n{'=' * 62}\n  {text}\n{'=' * 62}")


def scaler_params(obj):
    """Extract center/scale, and say plainly whether the scaler was fitted."""
    if obj is None:
        return None
    for center_attr, scale_attr, kind in (
        ("mean_", "scale_", "StandardScaler"),
        ("center_", "scale_", "RobustScaler"),
        ("data_min_", "data_range_", "MinMaxScaler"),
    ):
        if hasattr(obj, center_attr) and hasattr(obj, scale_attr):
            return {
                "center": float(np.ravel(getattr(obj, center_attr))[0]),
                "scale": float(np.ravel(getattr(obj, scale_attr))[0]),
                "type": type(obj).__name__,
                "fitted": True,
                "matched_as": kind,
            }
    return {
        "center": 0.0,
        "scale": 1.0,
        "type": type(obj).__name__,
        "fitted": False,
        "matched_as": "unfitted — treated as identity",
    }


def fit_affine(raw, seen):
    """Solve seen = (raw - center) / scale from observed pairs."""
    raw = np.asarray(raw, dtype=float)
    seen = np.asarray(seen, dtype=float)
    if raw.size < 2 or np.ptp(raw) == 0 or np.ptp(seen) == 0:
        return None
    slope, intercept = np.polyfit(raw, seen, 1)
    if abs(slope) < 1e-12:
        return None
    scale = 1.0 / float(slope)
    center = -float(intercept) * scale
    residual = float(np.abs((raw - center) / scale - seen).max())
    return {"center": center, "scale": scale, "max_residual": residual}


# --------------------------------------------------------------------------
banner("1. Import the training pipeline")

import torch  # noqa: E402
import torchmetrics  # noqa: E402

torchmetrics.Metric.device = property(lambda self: torch.device("cpu"))

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(REPO_ROOT))

try:
    import train_all_jiaxingalpha as train_mod
    from train_all_jiaxingalpha import (  # noqa: F401
        QUANTILES,
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

print("  training module imported")


# --------------------------------------------------------------------------
banner("2. Rebuild the training dataset")

for path in (TRAIN_CSV, VAL_CSV, STUDENT_CFG):
    if not path.exists():
        print(f"  FAILED: {path} not found")
        sys.exit(1)

with open(STUDENT_CFG, encoding="utf-8") as handle:
    cfg = json.load(handle)

train_df, val_df = train_mod.load_dataframes(str(TRAIN_CSV), str(VAL_CSV))
print(f"  train rows : {len(train_df):,}")

if hasattr(train_mod, "filter_vocab") and hasattr(train_mod, "load_preprocessing_artifacts"):
    try:
        encoders = train_mod.load_preprocessing_artifacts(
            str(SCALER_PKL), str(ENCODER_PKL)
        )
        if isinstance(encoders, tuple):
            encoders = encoders[-1]
        train_df, val_df = train_mod.filter_vocab(train_df, val_df, encoders)
        print("  vocab filter applied")
    except Exception as exc:
        print(f"  vocab filter skipped ({type(exc).__name__})")

normalizer = train_mod.make_target_normalizer()
train_dataset, val_dataset = train_mod.build_timeseries_datasets(
    train_df, val_df, cfg, normalizer
)

reals = list(train_dataset.reals)
categoricals = list(train_dataset.categoricals)
idx = {name: i for i, name in enumerate(reals)}
panel_reals = [r for r in reals if r not in DERIVED]

print(f"  reals {len(reals)}  categoricals {len(categoricals)}")

with open(PREPROCESSING, encoding="utf-8") as handle:
    payload = json.load(handle)

if reals != payload["x_reals"]:
    print("\n  FAILED: rebuilt reals order differs from preprocessing.json")
    sys.exit(1)

print("  channel order matches preprocessing.json")


# --------------------------------------------------------------------------
banner("3. What dataset.scalers claims")

dataset_scalers = getattr(train_dataset, "scalers", {}) or {}
declared = {}
for name in reals:
    params = scaler_params(dataset_scalers.get(name))
    if params is None:
        continue
    declared[name] = params
    fitted = "fitted" if params["fitted"] else "UNFITTED"
    print(
        f"  {name:28} center={params['center']:>10.4f} "
        f"scale={params['scale']:>9.4f}  [{fitted}]"
    )


# --------------------------------------------------------------------------
banner("4. Collect several real batches")

group_col = cfg["group_ids"][0]
time_col = cfg["time_idx"]

loader = val_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

by_station = {
    name: frame.sort_values(time_col)
    for name, frame in val_df.groupby(group_col)
}

samples = []
loader_iter = iter(loader)
attempts = 0
while len(samples) < N_BATCHES and attempts < N_BATCHES * 6:
    attempts += 1
    try:
        x, _ = next(loader_iter)
    except StopIteration:
        break

    enc_len = int(x["encoder_lengths"][0])
    dec_start = int(x["decoder_time_idx"][0].numpy()[0])
    encoder_cont = x["encoder_cont"][0].numpy()

    for name, frame in by_station.items():
        rows = frame[
            (frame[time_col] >= dec_start - enc_len) & (frame[time_col] < dec_start)
        ]
        if len(rows) != enc_len:
            continue
        # Confirm alignment on a column with spread before trusting the match.
        probe = "Lagged_Energy_168h"
        if probe in rows.columns:
            fit = fit_affine(rows[probe].to_numpy(dtype=float), encoder_cont[:, idx[probe]])
            if fit is None or fit["max_residual"] > 1e-3:
                continue
        samples.append({"x": x, "rows": rows, "station": name, "enc_len": enc_len})
        break

if not samples:
    print("  FAILED: could not align any batch to its source rows")
    sys.exit(1)

print(f"  aligned {len(samples)} window(s): {sorted({s['station'] for s in samples})}")

first = samples[0]
enc0 = first["x"]["encoder_cont"][0].numpy()
dec0 = first["x"]["decoder_cont"][0].numpy()
enc_len0 = first["enc_len"]
dec_len0 = int(first["x"]["decoder_lengths"][0])

print(f"  encoder_cont {enc0.shape}   decoder_cont {dec0.shape}")


# --------------------------------------------------------------------------
banner("5. Solve for the derived scaling")

rel_enc = enc0[:, idx["relative_time_idx"]]
rel_dec = dec0[:, idx["relative_time_idx"]]
raw_ramp = np.concatenate([np.arange(-enc_len0, 0), np.arange(0, dec_len0)]).astype(float)
rel_fit = fit_affine(raw_ramp, np.concatenate([rel_enc, rel_dec]))

if rel_fit is None:
    print("  FAILED: relative_time_idx is constant, cannot solve")
    sys.exit(1)

print(
    f"  relative_time_idx  center={rel_fit['center']:>10.6f}  "
    f"scale={rel_fit['scale']:>10.6f}  residual={rel_fit['max_residual']:.2e}"
)

enc_channel = enc0[:, idx["encoder_length"]]
enc_observed = float(enc_channel[0])
enc_scale = float(enc_len0) / enc_observed if enc_observed else 1.0

print(f"  encoder_length     observed={enc_observed:.6f}  implies scale={enc_scale:.6f}")

# Both come out as division by max_encoder_length, which is what PTF does.
if abs(rel_fit["scale"] - enc_scale) < 1e-6:
    print(f"  both channels divide by {enc_scale:.0f} — consistent with max_encoder_length")

derived_scalers = {
    "relative_time_idx": {
        "center": rel_fit["center"],
        "scale": rel_fit["scale"],
        "source": "solved from batch",
    },
    "encoder_length": {
        "center": 0.0,
        "scale": enc_scale,
        "source": "solved from batch",
    },
}


# --------------------------------------------------------------------------
banner("6. Verify every panel column")

# Columns with spread get an affine fit. Binary columns are often constant
# across a window, so those are checked point-by-point against the declared
# scaler, pooled across every aligned window.
transforms = {}
unresolved = []

for col in panel_reals:
    fits = []
    constant_checks = []

    for sample in samples:
        rows, x = sample["rows"], sample["x"]
        if col not in rows.columns:
            continue
        raw = rows[col].to_numpy(dtype=float)
        seen = x["encoder_cont"][0].numpy()[:, idx[col]]

        if np.ptp(raw) > 0:
            fit = fit_affine(raw, seen)
            if fit and fit["max_residual"] < 1e-3:
                fits.append(fit)
        else:
            constant_checks.append((float(raw[0]), float(seen[0])))

    if fits:
        center = float(np.mean([f["center"] for f in fits]))
        scale = float(np.mean([f["scale"] for f in fits]))
        spread = max(abs(f["center"] - center) for f in fits) if len(fits) > 1 else 0.0
        transforms[col] = {
            "center": center,
            "scale": scale,
            "source": f"solved from {len(fits)} window(s)",
        }
        note = f"  (spread {spread:.2e})" if len(fits) > 1 else ""
        print(f"  {col:28} center={center:>9.4f} scale={scale:>8.4f}{note}")
        continue

    # No spread anywhere. Test whether the declared scaler reproduces the
    # constant values we did observe.
    params = declared.get(col)
    if params and constant_checks:
        errors = [
            abs((raw - params["center"]) / params["scale"] - seen)
            for raw, seen in constant_checks
        ]
        if max(errors) < TOLERANCE:
            transforms[col] = {
                "center": params["center"],
                "scale": params["scale"],
                "source": f"declared, confirmed on {len(constant_checks)} constant window(s)",
            }
            print(
                f"  {col:28} center={params['center']:>9.4f} "
                f"scale={params['scale']:>8.4f}  (constant, declared confirmed)"
            )
            continue
        print(
            f"  {col:28} CONSTANT and declared scaler is wrong "
            f"(max error {max(errors):.4f})"
        )
        unresolved.append(col)
        continue

    print(f"  {col:28} UNRESOLVED — no spread and no declared scaler")
    unresolved.append(col)

print(f"\n  resolved {len(transforms)}/{len(panel_reals)} panel columns")

if unresolved:
    print(f"  FAILED on: {unresolved}")
    print("  Every panel column needs a transform before window.py can be trusted.")
    print(f"  Try raising N_BATCHES above {N_BATCHES} to find a window with spread.")
    sys.exit(1)


# --------------------------------------------------------------------------
banner("7. Merge into preprocessing.json")

payload["derived_scalers"] = derived_scalers
payload["panel_column_transforms"] = transforms
payload["declared_dataset_scalers"] = declared
payload["static_reals"] = list(getattr(train_dataset, "static_reals", []))
payload["time_varying_known_reals"] = list(
    getattr(train_dataset, "time_varying_known_reals", [])
)
payload["time_varying_unknown_reals"] = list(
    getattr(train_dataset, "time_varying_unknown_reals", [])
)
payload["scalers_already_applied_to_panel"] = True
payload["notes"].append(
    "Panel values carry the training-time RobustScaler already. The dataset "
    "fits a SECOND scaler on top; panel_column_transforms is that layer and "
    "window.py must apply it. derived_scalers divide by max_encoder_length."
)

with open(PREPROCESSING, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, ensure_ascii=False)

print(f"  updated : {PREPROCESSING.relative_to(REPO_ROOT)}")
print(f"  size    : {PREPROCESSING.stat().st_size / 1024:.1f} KB")
print(f"  derived : {len(derived_scalers)}   panel: {len(transforms)}")
print("\n  Nothing after this needs torch. Next: core/window.py")