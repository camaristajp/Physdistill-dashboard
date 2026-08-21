"""
06_export_comparison.py

Writes data/serving/comparison_medians.parquet — precomputed median forecasts
from baseline_student.ckpt (the no-distillation ablation), one row per
(station, origin), so the Stations page can overlay it without ever loading
torch at runtime.

    python scripts\\06_export_comparison.py
    python scripts\\06_export_comparison.py --stations bus_station park_a
    python scripts\\06_export_comparison.py --limit-origins 50   # smoke test

Why baseline only, not teacher: baseline_student.ckpt was trained on
student_tft_config.json — the same 168-hour encoder, vocabulary, and feature
schema core/window.py already builds. teacher_tft.ckpt was trained on
tft_config.json, a 336-hour encoder with its own (unverified) vocabulary and
target scale. Feeding it the 168-hour window built here would either
shape-error or silently produce a plausible, wrong forecast — the exact
failure mode this pipeline exists to avoid. Supporting it means re-deriving
a second preprocessing contract and a second parity check, deliberately left
for later.

Inference is batched per station (window construction is cheap; the forward
pass is what benefits from batching) so this runs against all ~2,184 origins
per station in a couple of minutes on CPU rather than tens.
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from core import config, loader, postprocess, window  # noqa: E402

CKPT_PATH = REPO_ROOT / "artifacts" / "baseline_student.ckpt"
OUT = REPO_ROOT / "data" / "serving" / "comparison_medians.parquet"
MEDIAN_COLUMN = config.QUANTILES.index(config.Q_MEDIAN)
HORIZON_COLUMNS = [f"h{i}" for i in range(1, config.HORIZON + 1)]


def banner(text):
    print(f"\n{'=' * 62}\n  {text}\n{'=' * 62}")


parser = argparse.ArgumentParser()
parser.add_argument("--stations", nargs="*", default=None,
                     help="station_id(s) to export; default is all stations in station_meta.csv")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--limit-origins", type=int, default=None,
                     help="cap origins per station, for a fast smoke test")
parser.add_argument("--force", action="store_true", help="overwrite an existing file")
args = parser.parse_args()


# --------------------------------------------------------------------------
banner("1. Guard against clobbering a prior export")

if OUT.exists() and not args.force:
    existing = pd.read_parquet(OUT)
    print(f"  {OUT.name} already exists ({len(existing)} rows). Not overwriting.")
    print("  Pass --force to regenerate.")
    sys.exit(0)


# --------------------------------------------------------------------------
banner("2. Load the baseline checkpoint")

import torch  # noqa: E402
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

if not CKPT_PATH.exists():
    print(f"  FAILED: {CKPT_PATH} not found")
    sys.exit(1)

model = StudentBaseline.load_from_checkpoint(str(CKPT_PATH), map_location="cpu")
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"  parameters : {n_params:,}")

if list(QUANTILES) != list(config.QUANTILES):
    print(f"  FAILED: checkpoint QUANTILES {QUANTILES} != core.config.QUANTILES {config.QUANTILES}")
    sys.exit(1)


def torch_forward_batch(inputs_list):
    """Stack a list of ModelInputs (each batch=1) and run one forward pass."""
    batch = len(inputs_list)
    enc_len = inputs_list[0].encoder_cont.shape[1]
    horizon = inputs_list[0].decoder_cont.shape[1]

    x = {
        "encoder_cont": torch.from_numpy(np.concatenate([i.encoder_cont for i in inputs_list], axis=0)),
        "decoder_cont": torch.from_numpy(np.concatenate([i.decoder_cont for i in inputs_list], axis=0)),
        "encoder_cat": torch.from_numpy(np.concatenate([i.encoder_cat for i in inputs_list], axis=0)),
        "decoder_cat": torch.from_numpy(np.concatenate([i.decoder_cat for i in inputs_list], axis=0)),
        "encoder_lengths": torch.full((batch,), enc_len, dtype=torch.long),
        "decoder_lengths": torch.full((batch,), horizon, dtype=torch.long),
        "target_scale": torch.zeros(batch, 2).float().index_fill_(1, torch.tensor([1]), 1.0),
        "encoder_target": torch.zeros(batch, enc_len),
        "decoder_target": torch.zeros(batch, horizon),
        "decoder_time_idx": torch.arange(horizon, dtype=torch.long).unsqueeze(0).repeat(batch, 1),
        "groups": torch.zeros(batch, 1, dtype=torch.long),
    }
    with torch.no_grad():
        out = model(x)
    if isinstance(out, dict):
        out = out.get("prediction", next(iter(out.values())))
    elif hasattr(out, "prediction"):
        out = out.prediction
    elif isinstance(out, (tuple, list)):
        out = out[0]
    return out.numpy()  # [batch, horizon, n_quantiles]


def medians(raw_batch):
    """[batch, horizon, n_quantiles] raw -> [batch, horizon] kWh, rank-sorted.

    Mirrors postprocess.build_forecast: rescale, expm1, floor, sort. Sorting an
    already-monotone row is a no-op, so this is safe to apply unconditionally
    rather than re-deriving build_forecast's crossing check here.
    """
    kwh = postprocess.to_kwh(raw_batch)
    kwh_sorted = np.sort(kwh, axis=-1)
    return kwh_sorted[:, :, MEDIAN_COLUMN]


# --------------------------------------------------------------------------
banner("3. Select stations")

if args.stations:
    station_ids = args.stations
else:
    station_ids = loader.load_station_meta()["station_id"].tolist()
print(f"  stations : {len(station_ids)}")


# --------------------------------------------------------------------------
banner("4. Batched inference over every valid origin")

rows = []
skipped = 0
t0 = time.perf_counter()

for station_id in station_ids:
    try:
        origins = list(loader.valid_origins(station_id))
    except KeyError as exc:
        print(f"  {station_id}: {exc}")
        continue
    if args.limit_origins:
        origins = origins[: args.limit_origins]
    if not origins:
        print(f"  {station_id}: no valid origins, skipping")
        continue

    station_start = time.perf_counter()
    for chunk_start in range(0, len(origins), args.batch_size):
        chunk = origins[chunk_start: chunk_start + args.batch_size]

        inputs_list = []
        kept_origins = []
        for origin in chunk:
            try:
                inputs_list.append(window.build_inputs(station_id, origin))
                kept_origins.append(origin)
            except window.WindowError:
                skipped += 1
                continue
        if not inputs_list:
            continue

        raw_batch = torch_forward_batch(inputs_list)
        median_batch = medians(raw_batch)

        for origin, values in zip(kept_origins, median_batch):
            row = {"station_id": station_id, "origin_time_index": int(origin)}
            row.update(zip(HORIZON_COLUMNS, values.astype(np.float32)))
            rows.append(row)

    elapsed = time.perf_counter() - station_start
    print(f"  {station_id:32} {len(origins):5} origins  {elapsed:6.1f}s")

total_elapsed = time.perf_counter() - t0
print(f"\n  total rows    : {len(rows):,}")
print(f"  skipped       : {skipped}")
print(f"  elapsed       : {total_elapsed:.1f}s")


# --------------------------------------------------------------------------
banner("5. Write comparison_medians.parquet")

if not rows:
    print("  FAILED: nothing to write")
    sys.exit(1)

frame = pd.DataFrame(rows)
frame.insert(2, "model", "baseline")
frame = frame[["station_id", "origin_time_index", "model", *HORIZON_COLUMNS]]

OUT.parent.mkdir(parents=True, exist_ok=True)
frame.to_parquet(OUT, index=False)

print(f"  wrote : {OUT.relative_to(REPO_ROOT)}")
print(f"  rows  : {len(frame):,}")
print(f"  size  : {OUT.stat().st_size / 1024:.0f} KB")
print("\n  Schema carries a 'model' column so a verified teacher export can")
print("  append rows later without changing anything that reads this file.")
print("\n  Next: pages/2_Stations.py reads this via core/service.py's")
print("  baseline_comparison(station_id, origin) lookup.")
