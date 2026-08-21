"""
03a_probe_dataset_params.py

Diagnostic only — writes nothing. Reports what the checkpoint knows about the
two derived continuous channels, encoder_length and relative_time_idx.

    python scripts\\03a_probe_dataset_params.py

Both are added by TimeSeriesDataSet (add_encoder_length=True,
add_relative_time_idx=True) and scaled by StandardScalers fitted during
dataset construction. core/window.py has to reproduce those values exactly.
If the scaler parameters are in the checkpoint, we use them. If not, we
recover them empirically against actualvspred_physdistill_ev.csv instead of
guessing.
"""

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
CKPT_PATH = REPO_ROOT / "artifacts" / "physdistill_ev.ckpt"

DERIVED = ["encoder_length", "relative_time_idx"]


def banner(text):
    print(f"\n{'=' * 62}\n  {text}\n{'=' * 62}")


def describe(obj, indent="      "):
    """Print whatever a scaler-like object exposes, without assuming a type."""
    kind = type(obj).__name__
    fields = {}
    for attr in ("mean_", "scale_", "var_", "center_", "n_features_in_",
                 "feature_names_in_", "data_min_", "data_max_", "classes_"):
        if hasattr(obj, attr):
            value = getattr(obj, attr)
            if isinstance(value, np.ndarray):
                value = value.tolist()
            fields[attr] = value
    print(f"{indent}{kind}")
    if fields:
        for key, value in fields.items():
            print(f"{indent}  {key} = {value}")
    else:
        public = [a for a in dir(obj) if not a.startswith("_")][:12]
        print(f"{indent}  (no standard attrs) sample members: {public}")


# --------------------------------------------------------------------------
banner("1. Load checkpoint")

import torch  # noqa: E402
import torchmetrics  # noqa: E402

torchmetrics.Metric.device = property(lambda self: torch.device("cpu"))

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(REPO_ROOT))

try:
    from train_all_jiaxingalpha import (  # noqa: F401
        QUANTILES,
        RawLossTFT,
        SparseQuantileLoss,
        StudentKDTFT,
    )
except ImportError as exc:
    print(f"  FAILED: {exc}")
    sys.exit(1)

model = StudentKDTFT.load_from_checkpoint(str(CKPT_PATH), map_location="cpu")
model.eval()
print("  loaded")


# --------------------------------------------------------------------------
banner("2. Every hparam key")

keys = sorted(model.hparams.keys())
for key in keys:
    value = model.hparams[key]
    kind = type(value).__name__
    if isinstance(value, (int, float, str, bool)) or value is None:
        preview = repr(value)
    elif isinstance(value, (list, tuple)):
        preview = f"{kind}[{len(value)}]"
    elif isinstance(value, dict):
        preview = f"dict[{len(value)}] keys={list(value)[:6]}"
    else:
        preview = kind
    print(f"  {key:32} {preview}")


# --------------------------------------------------------------------------
banner("3. Anything scaler-shaped")

candidates = [
    k for k in keys
    if any(token in k.lower() for token in ("scaler", "normaliz", "dataset", "param"))
]

if not candidates:
    print("  no keys matching scaler / normalizer / dataset / param")
else:
    for key in candidates:
        value = model.hparams[key]
        print(f"\n  {key}  ({type(value).__name__})")
        if isinstance(value, dict):
            for name, obj in value.items():
                print(f"    {name}")
                describe(obj)
        else:
            describe(value, indent="    ")


# --------------------------------------------------------------------------
banner("4. The two derived channels")

found = {}
for key in keys:
    value = model.hparams[key]
    if isinstance(value, dict):
        for name in DERIVED:
            if name in value:
                found.setdefault(name, []).append(key)

for name in DERIVED:
    where = found.get(name)
    if where:
        print(f"  {name:20} referenced in: {where}")
        for key in where:
            print(f"    via {key}:")
            describe(model.hparams[key][name], indent="      ")
    else:
        print(f"  {name:20} not present in any hparam dict")

print("\n  If neither carries scaler parameters, window.py will recover them")
print("  empirically in the parity check rather than assuming a scaling.")