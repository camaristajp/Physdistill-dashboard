"""
03c_extract_target_scale.py

Recovers the target normalizer's fitted scale and merges it into
data/serving/preprocessing.json.

    python scripts\\03c_extract_target_scale.py

Why this exists: make_target_normalizer() returns

    GroupNormalizer(transformation=None, center=False, scale_by_group=False)

which reads like an identity transform and is not one. center=False disables
centering; the normalizer still fits a global scale during dataset
construction. The model's output is multiplied by that scale before the log1p
inverse, so treating it as 1.0 gives forecasts with the right shape and the
wrong magnitude — exactly what the parity check found (correlation 0.95-0.99,
relative error 0.92).

GroupNormalizer exposes this as norm_ = [{'center': ..., 'scale': ...}], a
list of dicts rather than an array, so the parser below handles dicts, lists
of dicts, and plain arrays alike.

The ONNX wrapper hardcoded target_scale = [0, 1], so the graph does not apply
the scale. core/postprocess.py applies it instead, which is arithmetically
identical and avoids re-exporting.
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


def banner(text):
    print(f"\n{'=' * 62}\n  {text}\n{'=' * 62}")


def pair_from(value):
    """Coerce whatever the normalizer exposes into (center, scale).

    GroupNormalizer.norm_ is a list of dicts. Other normalizers expose plain
    arrays. Both shapes, and a bare dict, are handled here.
    """
    if value is None:
        return None

    # Unwrap a one-element sequence wrapping a dict.
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]

    if isinstance(value, dict):
        center = value.get("center", 0.0)
        scale = value.get("scale", value.get("norm", 1.0))
        try:
            return float(np.ravel(center)[0]), float(np.ravel(scale)[0])
        except (TypeError, ValueError, IndexError):
            return None

    if isinstance(value, (list, tuple)) and value and isinstance(value[0], dict):
        return pair_from(value[0])

    try:
        array = np.ravel(np.asarray(value, dtype=np.float64))
    except (TypeError, ValueError):
        return None

    if array.size == 1:
        return 0.0, float(array[0])
    if array.size >= 2:
        return float(array[0]), float(array[1])
    return None


# --------------------------------------------------------------------------
banner("1. Rebuild the dataset")

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
    print(f"  FAILED: {exc}")
    sys.exit(1)

with open(STUDENT_CFG, encoding="utf-8") as handle:
    cfg = json.load(handle)

train_df, val_df = train_mod.load_dataframes(str(TRAIN_CSV), str(VAL_CSV))

if hasattr(train_mod, "filter_vocab") and hasattr(train_mod, "load_preprocessing_artifacts"):
    try:
        encoders = train_mod.load_preprocessing_artifacts(str(SCALER_PKL), str(ENCODER_PKL))
        if isinstance(encoders, tuple):
            encoders = encoders[-1]
        train_df, val_df = train_mod.filter_vocab(train_df, val_df, encoders)
    except Exception:
        pass

normalizer = train_mod.make_target_normalizer()
train_dataset, val_dataset = train_mod.build_timeseries_datasets(
    train_df, val_df, cfg, normalizer
)
print("  dataset rebuilt")


# --------------------------------------------------------------------------
banner("2. Inspect the fitted normalizer")

fitted = train_dataset.target_normalizer
print(f"  type    : {type(fitted).__name__}")
for attr in ("center", "scale_by_group", "transformation", "method"):
    if hasattr(fitted, attr):
        print(f"  {attr:15} = {getattr(fitted, attr)!r}")

for attr in ("norm_", "scale_", "center_", "norm"):
    if hasattr(fitted, attr):
        print(f"  {attr:15} = {getattr(fitted, attr)!r}")


# --------------------------------------------------------------------------
banner("3. Resolve center and scale")

center, scale, source = None, None, None

for attr in ("norm_", "scale_", "norm"):
    if not hasattr(fitted, attr):
        continue
    pair = pair_from(getattr(fitted, attr))
    if pair:
        center, scale = pair
        source = attr
        break

if scale is None and hasattr(fitted, "get_parameters"):
    try:
        pair = pair_from(fitted.get_parameters())
        if pair:
            center, scale = pair
            source = "get_parameters()"
    except Exception as exc:
        print(f"  get_parameters() unavailable ({type(exc).__name__})")

if scale is None or not np.isfinite(scale) or scale == 0:
    print("  FAILED: could not resolve a usable scale")
    sys.exit(1)

print(f"  center = {center:.6f}")
print(f"  scale  = {scale:.6f}   (from {source})")

if abs(scale - 1.0) < 1e-6:
    print("\n  scale is 1.0, so this is not the magnitude error. Look elsewhere.")


# --------------------------------------------------------------------------
banner("4. Confirm against a real batch")

# What the model receives is the ground truth, so a batch outranks the
# attribute if the two disagree.
loader = val_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
x, _ = next(iter(loader))

target_scale = x.get("target_scale")
if target_scale is not None:
    raw = target_scale[0] if hasattr(target_scale, "__getitem__") else target_scale
    observed = np.ravel(np.asarray(raw, dtype=np.float64))
    print(f"  batch target_scale = {observed}")
    if observed.size >= 2:
        batch_center, batch_scale = float(observed[0]), float(observed[1])
        agrees = abs(batch_scale - scale) < 1e-4 and abs(batch_center - center) < 1e-4
        print(f"  agrees with resolved values : {agrees}")
        if not agrees:
            print("  using the batch values — they are what the model receives")
            center, scale, source = batch_center, batch_scale, "batch target_scale"
else:
    print("  batch carries no target_scale")

print(f"\n  adopting center={center:.6f}  scale={scale:.6f}")


# --------------------------------------------------------------------------
banner("5. Sanity-check the magnitude")

# The parity check saw our medians roughly an order of magnitude low. Applying
# the scale in log space should close that gap.
sample_raw = 1.957  # a log-space median observed during the failing run
before = float(np.expm1(sample_raw))
after = float(np.expm1(center + scale * sample_raw))
print(f"  raw log-space value {sample_raw:.3f}")
print(f"    without scale : {before:8.2f} kWh")
print(f"    with scale    : {after:8.2f} kWh")
print(f"    ratio         : {after / before:8.2f}x")


# --------------------------------------------------------------------------
banner("6. Merge into preprocessing.json")

with open(PREPROCESSING, encoding="utf-8") as handle:
    payload = json.load(handle)

payload["target"]["normalizer"] = type(fitted).__name__
payload["target"]["target_scale"] = [center, scale]
payload["target"]["apply_scale_in_postprocess"] = True
payload["target"]["scale_source"] = source
payload["notes"].append(
    "The ONNX wrapper passed target_scale=[0,1], so the graph does not apply "
    "the normalizer. postprocess.to_kwh multiplies by target_scale before "
    "expm1: kWh = expm1(center + scale * raw)."
)

with open(PREPROCESSING, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, ensure_ascii=False)

print(f"  updated : {PREPROCESSING.relative_to(REPO_ROOT)}")
print(f"  target_scale = [{center:.6f}, {scale:.6f}]")
print("\n  Rerun scripts/03_parity_check.py")