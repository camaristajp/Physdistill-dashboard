"""
07_export_model_manifest.py

Writes data/serving/model_manifest.json — parameter counts and training
epoch for all three checkpoints, read straight from each .ckpt's state_dict
and top-level metadata rather than retyped by hand, so the System and About
pages never need torch at runtime and never risk drifting from the actual
artifacts in artifacts/.

    python scripts\\07_export_model_manifest.py

Parameter count and epoch are the only facts a checkpoint can prove about
itself. KD alpha is not one of them: the training script sets it as a plain
instance attribute (self._kd_alpha) outside save_hyperparameters(), so it
never lands in hyper_parameters and cannot be recovered from the file — a
raw torch.load() of physdistill_ev.ckpt confirms no 'alpha' key exists
anywhere in it. The paper's WMAPE, reduction factor, and Pi 5 latency are
external measurements, not properties of the checkpoint at all. All four are
recorded below as the verified record from README.md, clearly separated from
the fields this script actually computed.
"""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

OUT = REPO_ROOT / "data" / "serving" / "model_manifest.json"

CHECKPOINTS = {
    "student": {
        "file": "physdistill_ev.ckpt",
        "label": "PhysDistill-EV (distilled student)",
    },
    "baseline": {
        "file": "baseline_student.ckpt",
        "label": "Baseline student (no distillation)",
    },
    "teacher": {
        "file": "teacher_tft.ckpt",
        "label": "Teacher TFT",
    },
}

# load_from_checkpoint needs the class each model was actually saved as (see
# README.md's checkpoint-loading gotcha) — StudentKDTFT and RawLossTFT are
# resolved below once train_all_jiaxingalpha is importable.

# Not recoverable from any checkpoint — see the module docstring. Sourced
# from README.md's verification record.
PAPER_RECORD = {
    "kd_alpha": 0.40,
    "wmape_table5_pct": 13.28,
    "reduction_x": 291,
    "pi5_latency_ms": 2.19,
    "desktop_latency_ms_mean": 1.49,
    "desktop_latency_ms_std": 0.14,
    "source": "README.md, Verified contracts / Verification record",
}


def banner(text):
    print(f"\n{'=' * 62}\n  {text}\n{'=' * 62}")


banner("1. Import the unpickling dependencies")

import torchmetrics  # noqa: E402

torchmetrics.Metric.device = property(lambda self: torch.device("cpu"))

try:
    from train_all_jiaxingalpha import (  # noqa: F401
        QUANTILES,
        RawLossTFT,
        SparseQuantileLoss,
        StudentBaseline,
        StudentKDTFT,
    )
except ImportError as exc:
    print(f"  FAILED: {exc}")
    sys.exit(1)

# Each checkpoint's own saving class (README.md, 03d_torch_vs_onnx.py,
# 06_export_comparison.py). Loading via the wrong class either fails outright
# or, worse, loads a differently-shaped model without complaint.
LOADERS = {
    "student": StudentKDTFT,
    "baseline": StudentBaseline,
    "teacher": RawLossTFT,
}


banner("2. Read each checkpoint's own metadata")

manifest = {}
for key, info in CHECKPOINTS.items():
    path = REPO_ROOT / "artifacts" / info["file"]
    if not path.exists():
        print(f"  FAILED: {path} not found")
        sys.exit(1)

    # Raw metadata (epoch, global_step) sits at the top of the checkpoint
    # dict regardless of which class saved it.
    raw = torch.load(str(path), map_location="cpu", weights_only=False)

    # Parameter count has to come from the instantiated module, not a plain
    # sum over state_dict tensors: state_dict also holds registered buffers
    # (e.g. the loss's quantile levels) that model.parameters() correctly
    # excludes. Summing state_dict directly overcounted the two student
    # checkpoints by 116 each against the verified 4,148 — caught by the
    # sanity check below before this manifest almost shipped that number.
    model = LOADERS[key].load_from_checkpoint(str(path), map_location="cpu")
    n_params = sum(p.numel() for p in model.parameters())

    manifest[key] = {
        "label": info["label"],
        "file": info["file"],
        "n_params": int(n_params),
        "epoch": int(raw.get("epoch", -1)),
        "global_step": int(raw.get("global_step", -1)),
    }
    print(f"  {info['label']:34} params={n_params:>9,}  epoch={manifest[key]['epoch']}")

manifest["paper_record"] = PAPER_RECORD


banner("3. Sanity check against README.md")

# teacher_tft.ckpt was originally documented as 1,331,761 — that figure came
# from summing state_dict() directly, which double-counts the tied
# encoder/decoder variable-selection weights (share_single_variable_networks
# links prescalers, post_lstm_gate_decoder, post_lstm_add_norm_decoder).
# model.parameters() below deduplicates them correctly, as it already does
# for student/baseline; README.md's teacher figure was corrected to match.
expected = {"student": 4148, "baseline": 4148, "teacher": 1296305}
mismatches = [
    key for key, count in expected.items() if manifest[key]["n_params"] != count
]
if mismatches:
    print(f"  WARNING: param counts differ from README.md for {mismatches}")
    print("  The checkpoint is the source of truth here — update README.md, not this script.")
else:
    print("  All three parameter counts match README.md exactly.")


banner("4. Write model_manifest.json")

import json  # noqa: E402

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)

print(f"  wrote : {OUT.relative_to(REPO_ROOT)}")
print("\n  Next: core/loader.load_model_manifest() serves this to System and About.")
