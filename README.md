# PhysDistill-EV Operator Dashboard — project scaffold

A grid-operator forecasting dashboard over the Jiaxing EV charging dataset
(13 stations, hourly, 2020–2021), served by PhysDistill-EV: a 4,148-parameter
distilled Temporal Fusion Transformer running locally through ONNX Runtime.

**Scope.** This is a demonstrator, not a general product. The model carries 13
fixed station embeddings and a Jiaxing-specific feature schema, so it cannot
be pointed at another network without retraining. Say so on the About tab.

**Status.** Pipeline verified end to end. Overview, Stations, and Alerts
pages built. System and About remain.

---

## 1. Directory layout

```
physdistill-dashboard/
├── app.py                          Overview (Streamlit entry point)
├── requirements.txt                serving — 6 packages, ships to the Pi
├── requirements-dev.txt            adds torch, for scripts only
│
├── .streamlit/
│   └── config.toml                 theme
│
├── pages/
│   ├── 2_Stations.py               ✅ complete, verified
│   ├── 3_Alerts.py                 ✅ complete, verified
│   ├── 4_System.py                 ⬜ TO BUILD
│   └── 5_About.py                  ⬜ TO BUILD
│
├── core/                           ✅ complete, verified
│   ├── __init__.py
│   ├── config.py                   paths, quantiles, severity constants
│   ├── loader.py                   cached parquet + metadata reads
│   ├── window.py                   builds the six model input tensors
│   ├── engine.py                   ONNX session, warm-up, benchmarking
│   ├── postprocess.py              rescale → expm1 → floor → sort
│   ├── thresholds.py               breach detection, severity tiers
│   └── service.py                  one call per page-level question
│
├── components/                     ✅ complete, verified
│   ├── __init__.py
│   ├── charts.py                   band_chart, utilisation_heatmap, sparkline
│   └── ui.py                       tiles, station rail, origin control, CSS
│
├── scripts/                        ✅ all run, all passing
│   ├── train_all_jiaxingalpha.py   reference copy (needed for ckpt unpickling)
│   ├── convert_physdistill_ev_jiaxing.py
│   ├── 01_build_panel.py
│   ├── 02_export_preprocessing.py
│   ├── 03_parity_check.py
│   ├── 03a_probe_dataset_params.py
│   ├── 03b_extract_dataset_scalers.py
│   ├── 03c_extract_target_scale.py
│   ├── 03d_torch_vs_onnx.py
│   ├── 04_score_stations.py
│   ├── 05_seed_capacity.py
│   ├── 06_export_comparison.py     baseline-student medians, for the compare toggle
│   └── 07_export_model_manifest.py param counts + epoch, read from the checkpoints
│
├── artifacts/
│   ├── physdistill_ev_jiaxing.onnx  ← the serving graph (784 KB)
│   ├── physdistill_ev.ckpt          4,148 params, epoch 21, α=0.40
│   ├── baseline_student.ckpt        4,148 params, epoch 15
│   └── teacher_tft.ckpt             1,296,305 params, epoch 3
│
└── data/
    ├── tft_artifacts_jiaxing/       training outputs, read-only
    │   ├── train_data.csv           170,664 rows
    │   ├── val_data.csv             61,386 rows
    │   ├── feature_scaler.pkl
    │   ├── label_encoders.pkl
    │   ├── student_tft_config.json
    │   ├── tft_config.json
    │   └── actualvspred_physdistill_ev.csv
    └── serving/                     generated; this is what ships
        ├── jiaxing_panel.parquet    61,386 rows · 3.7 MB
        ├── preprocessing.json       14 KB
        ├── station_scores.csv
        ├── station_meta.csv
        ├── comparison_medians.parquet  baseline-student medians, 28,392 rows
        └── model_manifest.json         param counts + epoch per checkpoint
```

Only `artifacts/` and `data/serving/` are needed at runtime. Everything else
is development-time.

---

## 2. Verified contracts

These were established empirically, not assumed. Each one had a plausible
wrong answer that would have produced forecasts that looked fine.

### Channel order

`encoder_cont` and `decoder_cont` are `[batch, len, 15]` in exactly this order:

```python
['encoder_length', 'Fee', 'relative_time_idx',
 'Lagged_Energy_168h', 'Rolling_Mean_Energy_24h',
 'Lagged_Energy_24h', 'Rolling_Mean_Energy_168h',
 'IsCharging_Lag1h_f', 'IsCharging_Lag24h_f', 'Was_Idle_LastHour',
 'Temperature_C', 'Relative_Humidity', 'Precipitation_mm',
 'Is_Abnormal', 'Is_Covid_Lockdown']
```

`encoder_cat` and `decoder_cat` are `[batch, len, 9]`:

```python
['Location_Information', 'District_Name', 'Month', 'Hour_of_Day',
 'Day_of_Week', 'Is_Holiday', 'Is_Weekend', 'ToU_Period', 'Hour_of_Week']
```

### Two scaling layers, not one

Panel values already carry the training RobustScaler — `val_data.csv` was
written post-scaling. `TimeSeriesDataSet` then fitted a **second** scaler on
top, and that is what the model consumes. `preprocessing.json →
panel_column_transforms` holds the second layer and `window.py` applies it.
Skipping it was the single most dangerous available mistake.

### Derived channels

`encoder_length` and `relative_time_idx` never appear in the dataframe. Both
divide by `max_encoder_length = 168`:

- `encoder_length` → constant `1.0`
- `relative_time_idx` → `−1.0 … −1/168` across the encoder, `0 … 23/168` ahead

### Target scale

`GroupNormalizer(transformation=None, center=False, scale_by_group=False)`
reads like an identity transform and is not one. `center=False` disables
centring; a global scale of **2.297892** is still fitted. The ONNX wrapper
passed `target_scale=[0, 1]`, so the graph does not apply it and
`postprocess.to_kwh` does:

```
kWh = max(expm1(center + scale · raw), 0)
```

Omitting this gave correlation 0.976 with 92% relative error — right shape,
wrong magnitude by an order of magnitude.

### Origin convention

The origin is the **last observed hour**. Encoder covers `[origin−167, origin]`,
forecast covers `[origin+1, origin+24]`.

### Station keys

`Location_Information` for Industrial Park is `'Industrial Park\t'` — trailing
tab, present in all three source files. `encoder_key` preserves it verbatim for
the embedding lookup; `station_id` is the slug for URLs and display. Stripping
it anywhere routes the station to `<UNSEEN>`.

### Vocabularies

Come from the checkpoint's `embedding_labels`, not `label_encoders.pkl`. The
sklearn pickle holds 14 stations and 4 districts; the model has 13 and 3. It
was used for training-time vocabulary filtering, not as the model's mapping.

### Parameter counts: `.parameters()`, not a `state_dict()` sum

`sum(p.numel() for p in model.parameters())` is the number to trust; summing
every tensor in `state_dict()` is not. `TemporalFusionTransformer`'s
`share_single_variable_networks` ties several encoder/decoder layers
(`prescalers.*`, `post_lstm_gate_decoder`, `post_lstm_add_norm_decoder`)
together, and `state_dict()` emits one entry per registration path even for a
shared tensor, double-counting it. `.parameters()` deduplicates by object
identity and is correct.

This was caught, not assumed: an earlier pass of
`07_export_model_manifest.py` summed `state_dict()` directly and got
1,331,761 for `teacher_tft.ckpt` — the figure this file used to carry.
Switching to `.parameters()` gives **1,296,305**, matching student and
baseline (4,148 each, unaffected either way — their tied layers are tiny).
The manifest and this table now both report the deduplicated count.

---

## 3. Data contracts

### `jiaxing_panel.parquet`

61,386 rows = 13 stations × 4,722 hours, 18 Jun – 31 Dec 2021, contiguous
`Time_Index`, no nulls in model inputs.

| Column | Note |
|---|---|
| `station_id` | slug, e.g. `industrial_park` |
| `encoder_key` | exact vocabulary key, tab included |
| `Timestamp`, `Time_Index` | hourly, contiguous per station |
| `Energy_kWh`, `Energy_kWh_log` | ground truth and training target |
| 9 categoricals | model inputs |
| 13 reals | model inputs, pre-scaled once |
| `is_valid_origin` | precomputed; 2,184 hours across 91 days |

**Selectable origins: 1 Oct – 30 Dec 2021.** Earlier dates overlap the
training window (`val_loss_start = 2021-10-01`) and are not selectable.

### `preprocessing.json`

`x_reals`, `x_categoricals`, `vocab`, `cardinalities`, `derived_scalers`,
`panel_column_transforms`, `target.target_scale`, `scalers`.

### `station_meta.csv`

`station_id, encoder_key, display_name, group, capacity_kwh, capacity_source,
peak_observed, mean_demand, train_exceedance_pct, n_hours, confidence_tier,
active`

Capacity is **derived** — p99 of training demand × 1.10, rounded up to 10 kWh.
Median historical exceedance 0.39%. Every surface showing a threshold must say
it is derived, not rated. To use real ratings: edit the file, set
`capacity_source` to `rated`; script 05 will not overwrite without `--force`.

### `station_scores.csv`

`station_id, r2, wmape, mae, rmse, mean_actual, n_points, n_sequences,
confidence_tier, active`

Tiers: high ≥ 0.90 (7 stations), moderate 0.70–0.90 (4), low < 0.70
(Tourist Attraction), unavailable (Technology Park — no validation data).

### `comparison_medians.parquet`

`station_id, origin_time_index, model, h1..h24` — median kWh forecast per
horizon hour, from `baseline_student.ckpt` (no distillation), for every
active-and-inactive station across all 2,184 valid origins (28,392 rows).
Backs the Stations page's compare toggle.

Teacher is deliberately absent. `teacher_tft.ckpt` was trained on
`tft_config.json` — a 336-hour encoder with its own vocabulary and target
scale — a different contract than the 168-hour window `core/window.py`
builds. Feeding it that window would either shape-error or silently produce
a plausible, wrong forecast, which is the failure mode this whole verification
record exists to avoid. Supporting it means re-deriving a second
preprocessing pipeline and a second parity check; deliberately deferred.

### `model_manifest.json`

`{student, baseline, teacher, paper_record}`. The first three come straight
from each checkpoint: `n_params` via `.parameters()`, `epoch` and
`global_step` from the checkpoint's own top-level metadata. `paper_record`
holds the handful of facts no checkpoint can prove about itself — KD alpha
(0.40, a plain instance attribute set outside `save_hyperparameters()`, never
saved), the paper's Table 5 WMAPE (13.28%), the 291× reduction figure, and
the Pi 5 latency (2.19 ms) — each sourced from this document, not recomputed.

---

## 4. Module responsibilities

| Module | Does | Never does |
|---|---|---|
| `config` | paths, constants, thresholds | I/O |
| `loader` | cached reads, station index | computation |
| `window` | assembles 6 tensors; raises on gaps | zero-pads a short window |
| `engine` | one warm ONNX session, timing | rebuilds per request |
| `postprocess` | rescale, expm1, floor, sort | silently hides rearrangement |
| `thresholds` | pure `(forecast, capacity)` functions | I/O or model access |
| `service` | one call per page question | rendering |

Pages import `service` and `components`. No page touches `window` or `engine`.

---

## 5. Remaining work

### `pages/2_Stations.py` ✅

Fan chart with nested P10–P90 and P25–P75, capacity line, observed history to
the left of the origin, actual overlaid on the horizon. Rolling-origin slider
across all 2,184 origins with a working replay toggle (auto-advance +
speed). Station card: type, capacity, 7-day accuracy (a real rolling
backtest, not the global validation figure), confidence badge, idle-hour
rate. Compare toggle overlays the baseline (no-distillation) median, default
off — teacher deferred, see `comparison_medians.parquet` above.

Y-axis auto-scales per station. Station magnitudes span two orders of
magnitude; a shared axis flattens most of the network.

### `pages/3_Alerts.py` ✅

Ranked table from `network["breaches"]`: station, hour, P50, P90, capacity,
headroom, severity, suggested action. Critical first, then by hour.
Acknowledge state lives in `st.session_state`, keyed on `(station_id,
timestamp)` rather than `horizon_h` so it survives scrubbing the origin.

### `pages/4_System.py`

Per-station latency and last-forecast time, model version and parameter
count (from `model_manifest.json`), graph input signature from
`engine.input_signature()`, rolling accuracy from `station_scores.csv`.
`engine.benchmark()` gives the latency profile under the paper's 200-pass
protocol.

### `pages/5_About.py`

Headline figures (4,148 params, 291× reduction, 2.19 ms on Pi 5, 13.28%
WMAPE), a CC–CV explainer, the quantile vocabulary (P10 = minimum commitment,
P50 = expected, P90 = reserve requirement), known limitations, and the
three-model comparison table at the bottom framed as *why this model exists*.

Write it for operators; put parameter counts in a "for engineers" expander.

---

## 6. Gotchas

**Loading any checkpoint** requires importing four names from
`train_all_jiaxingalpha`, because the loss was pickled with module `__main__`:

```python
from train_all_jiaxingalpha import (
    QUANTILES, RawLossTFT, SparseQuantileLoss, StudentKDTFT
)
```

`baseline_student.ckpt` additionally needs `StudentBaseline` (same module) —
it's a `RawLossTFT` subclass, saved and loaded under its own class.

Also pin torchmetrics to CPU first, or a CUDA device recorded at training time
breaks the load on a CPU-only build:

```python
torchmetrics.Metric.device = property(lambda self: torch.device("cpu"))
```

**`onnx.checker` is not a validation gate.** It passed on a graph that could
not instantiate a session. Always construct an `InferenceSession`.

**Reading the source CSVs** needs comma parsing tried first — the literal tab
in a station name breaks tab-delimited parsing at line 28334.

**`actualvspred_physdistill_ev.csv` is from a different training run.** Its
global WMAPE is 15.71%, matching Table 8 (15.69%) rather than Table 5 (13.28%),
and its crossing incidence is 76% against the paper's 34.58%. Use it for
per-station ordering and confidence tiers, never as a parity reference. Script
`03d` compares ONNX against the checkpoint directly instead — max log-space
difference 1.4e-04.

**Network headroom is the tightest station, not the aggregate.** Summing
thresholds lets spare capacity at quiet stations mask an overloaded feeder.

**Parameter counts need `.parameters()`, not `state_dict()`.** See §2 — a
`state_dict()` sum double-counts the tied variable-selection layers and
overstated the teacher by 35,456 params (2.7%) in an earlier draft of
`model_manifest.json`.

**Streamlit specifics.**
- Cache the ONNX session at resource scope, never data scope.
- Avoid `st.metric` where the delta is a caption — it renders a directional
  arrow that reads as a change.
- A widget's `st.session_state[key]` cannot be written after that widget has
  been instantiated in the same script run (raises `StreamlitAPIException`).
  The Stations page's replay control seeds/advances the slider's session
  state *before* creating the slider each run, via a tick flag set on the
  previous run — never after.
- `itertuples()` renames any column starting with `_` to a positional name
  (`_1`, `_2`, …) rather than keeping it. The Alerts page's acknowledge
  columns are named `ack_key`/`acked`, not `_key`/`_acked`, for this reason.

---

## 7. Commands

```powershell
# environment
cd C:\Users\user\projects\physdistill-dashboard
.\physdistill\Scripts\Activate.ps1

# run
streamlit run app.py

# regenerate serving data from scratch
python scripts\convert_physdistill_ev_jiaxing.py
python scripts\01_build_panel.py
python scripts\02_export_preprocessing.py
python scripts\03b_extract_dataset_scalers.py
python scripts\03c_extract_target_scale.py
python scripts\04_score_stations.py
python scripts\05_seed_capacity.py --force
python scripts\06_export_comparison.py --force
python scripts\07_export_model_manifest.py

# verify
python scripts\03d_torch_vs_onnx.py     # authoritative
python scripts\03_parity_check.py       # against the reference file
```

Scripts 01–07 need `requirements-dev.txt`. The dashboard needs only
`requirements.txt`.

---

## 8. Verification record

| Check | Result |
|---|---|
| ONNX vs checkpoint, real windows | max log diff 1.4e-04 |
| Future unknowns zeroed | output unchanged (0.0) |
| Channel order vs rebuilt dataset | exact match |
| Vocabularies vs embedding tables | 9/9 exact |
| Panel transforms resolved | 13/13 |
| Inference latency, desktop CPU | 1.49 ms mean, 0.14 ms std |
| Paper reference, Pi 5 ARM | 2.19 ms mean |
| Checkpoint epoch vs documented | 21 / 15 / 3 — exact match, all three |
| Student/baseline param count | 4,148 / 4,148 — exact match |
| Teacher param count | 1,296,305 via `.parameters()` — corrected from 1,331,761 (`state_dict()` double-counted tied layers) |

Any change to `window.py`, `postprocess.py`, or the ONNX export invalidates
this record. Re-run `03d_torch_vs_onnx.py` before trusting output again.
