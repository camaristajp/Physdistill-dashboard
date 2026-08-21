"""
pages/5_About.py

Reference material, not a working view: what this model is, why it exists,
what the quantiles mean, and where it should not be trusted. No forecast is
built on this page, so it carries no origin control and no station rail.

    streamlit run app.py   (then open About from the top nav)

Routed through app.py's st.navigation — st.set_page_config() lives there
now, since it can only be called once per app.
"""

import streamlit as st

from components import ui
from core import service

ui.inject_css()
ui.replay_banner()
ui.settings_panel()

manifest = service.model_manifest()
student, baseline, teacher = manifest["student"], manifest["baseline"], manifest["teacher"]
paper = manifest["paper_record"]

ui.page_header(
    "About PhysDistill-EV",
    "What this model is, why it's this small, and what it can't do.",
)

st.info(
    "**This is a demonstrator, not a general product.** The model carries "
    "13 fixed station embeddings and a Jiaxing-specific feature schema — "
    "it cannot be pointed at another charging network without retraining."
)

ui.tiles(
    [
        ("Parameters", f"{student['n_params']:,}", None, "this build"),
        ("Reduction vs teacher", f"{paper['reduction_x']}", "×", "paper-reported"),
        ("Latency, Pi 5", f"{paper['pi5_latency_ms']:.2f}", "ms", "single-threaded"),
        ("Validation WMAPE", f"{paper['wmape_table5_pct']:.2f}", "%", "paper Table 5"),
    ]
)

st.divider()

# ---------------------------------------------------------------------------
st.markdown("### Why this is a hard forecasting problem")
st.markdown(
    """
An EV charging session doesn't draw constant power. Most chargers follow a
**CC–CV** profile — constant current, then constant voltage: power stays near
its peak while the battery is mostly empty, then tapers as it approaches
full. A site's total demand at any hour is the sum of however many sessions
are active, each at a different point on its own curve. Two stations with
identical session counts can draw very different power depending on where
those sessions happen to sit in their charge cycles.

That's why "average kWh per session × sessions expected" doesn't work as a
forecast, and why this is a genuinely temporal problem: the model has to
learn the shape of demand over the day, not just its volume, from 168 hours
of history per station.
"""
)

st.divider()

# ---------------------------------------------------------------------------
st.markdown("### Reading the quantiles")
q_cols = st.columns(3)
with q_cols[0]:
    st.markdown("**P10 — minimum commitment**")
    st.caption(
        "Demand is expected to be at or above this only 10% of the time. "
        "Sizing to this bound risks underestimating — treat it as a floor, "
        "not a plan."
    )
with q_cols[1]:
    st.markdown("**P50 — expected**")
    st.caption(
        "The median forecast. What the dashboard's headline numbers and the "
        "fan chart's centre line show. The single best guess, not a bound."
    )
with q_cols[2]:
    st.markdown("**P90 — reserve requirement**")
    st.caption(
        "Demand exceeds this only 10% of the time. This is the number "
        "capacity is checked against for a warning — the reserve an "
        "operator should hold to cover all but the worst hours."
    )

st.divider()

# ---------------------------------------------------------------------------
st.markdown("### Known limitations")
st.markdown(
    """
- **Jiaxing-only.** 13 fixed station embeddings and a feature schema built
  for this network. A new station or a new city needs retraining, not
  reconfiguration.
- **Capacity is derived, not rated.** Every threshold on this dashboard
  comes from the upper tail of training demand (p99 × 1.10), not a
  nameplate rating — median historical exceedance is 0.39%. Treat it as a
  reasonable proxy, not a certified limit.
- **Confidence varies by station.** One station (Tourist Attraction) has
  weak temporal structure and low confidence; one (Technology Park) has no
  validation data at all and is excluded from forecasts entirely.
- **Quantiles are sorted before display.** Pinball loss trains each quantile
  head independently, so nothing guarantees P10 ≤ P50 ≤ P90 in the raw
  output — the paper measures 34.58% crossing incidence on Jiaxing. This
  dashboard corrects it by sorting; a forecast's assumptions expander notes
  when that happened.
- **Expected energy is summed bounds, not a proper interval.** Adding up
  hourly P10s and P90s across a horizon is not the P10/P90 of the total —
  it's reported that way everywhere it appears.
- **Historical replay, not a live feed.** Every forecast on this dashboard
  is built from Jiaxing data, October–December 2021. Nothing here is
  real-time.
- **Teacher comparison isn't available yet.** The compare toggle on the
  Stations page overlays the undistilled baseline student only. The teacher
  was trained on a 336-hour encoder with its own vocabulary — a different
  contract than this pipeline builds — so supporting it means a second,
  independently verified preprocessing path. Deliberately deferred rather
  than approximated.
"""
)

st.divider()

# ---------------------------------------------------------------------------
st.markdown("### Three models, one question: how small can this get?")
st.markdown(
    "The teacher is accurate but too large for a Raspberry Pi. Shrinking the "
    "architecture alone (the baseline student) doesn't preserve accuracy. "
    "Knowledge distillation — training the small student against the "
    "teacher's soft targets, not just the raw labels — is what closes that "
    "gap. That's the case this table makes."
)

baseline_acc = service.baseline_accuracy()

rows = [
    {
        "model": teacher["label"],
        "parameters": f"{teacher['n_params']:,}",
        "epoch": teacher["epoch"],
        "wmape": "not evaluated in this build",
        "basis": "336h encoder, own vocabulary — deferred, see limitations",
    },
    {
        "model": baseline["label"],
        "parameters": f"{baseline['n_params']:,}",
        "epoch": baseline["epoch"],
        "wmape": f"{baseline_acc['wmape']:.2f}%" if baseline_acc else "unavailable",
        "basis": "measured here, full Oct–Dec 2021 replay, all 2,184 origins",
    },
    {
        "model": student["label"],
        "parameters": f"{student['n_params']:,}",
        "epoch": student["epoch"],
        "wmape": f"{paper['wmape_table5_pct']:.2f}%",
        "basis": "paper Table 5, unified protocol — not the same run as the baseline row",
    },
]
st.dataframe(rows, hide_index=True, use_container_width=True)
st.caption(
    "The three WMAPE figures are not on the same evaluation protocol — see "
    "the basis column before comparing them directly. The baseline row is "
    "the one apples-to-apples number this build can actually produce; the "
    "student figure is the paper's own headline result."
)

with st.expander("For engineers"):
    st.markdown(
        f"""
**Parameter counts**, via `sum(p.numel() for p in model.parameters())` on
each loaded checkpoint (not a `state_dict()` sum — see README §2 for why
that overcounts tied encoder/decoder layers):

| | Teacher | Baseline student | PhysDistill-EV |
|---|---|---|---|
| Parameters | {teacher['n_params']:,} | {baseline['n_params']:,} | {student['n_params']:,} |
| Epoch (best checkpoint) | {teacher['epoch']} | {baseline['epoch']} | {student['epoch']} |
| Global step | {teacher['global_step']:,} | {baseline['global_step']:,} | {student['global_step']:,} |

**KD α** = {paper['kd_alpha']:.2f} — the task/distillation loss weight for
PhysDistill-EV's training run. Not recoverable from the checkpoint itself:
the training script sets it as a plain instance attribute outside
`save_hyperparameters()`, so it's recorded here from the run record rather
than read back from the file.

**Serving path**: `physdistill_ev.ckpt` → ONNX export → `physdistill_ev_jiaxing.onnx`,
run through ONNX Runtime, single-threaded, on the System page's benchmark and
in every forecast this dashboard renders. See the System page for the live
graph input signature and a latency benchmark on this machine.
        """
    )
