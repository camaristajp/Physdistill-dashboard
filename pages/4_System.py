"""
pages/4_System.py

Runtime diagnostics: what model this is, what the graph actually expects,
how fast it runs, and how accurate each station has been historically. The
audience is whoever has to answer "is this thing working correctly," not the
operator deciding whether to curtail load.

    streamlit run app.py   (then open System from the top nav)

Routed through app.py's st.navigation — st.set_page_config() lives there
now, since it can only be called once per app.
"""

import streamlit as st

from components import ui
from core import config, loader, service

ui.inject_css()  # System page draws no plotly charts; return value unneeded here

origin = ui.origin_control()

try:
    network = service.forecast_network(origin)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not build the network forecast: {exc}")
    st.stop()

ui.station_rail(network)
ui.replay_banner()
ui.settings_panel()

manifest = service.model_manifest()
student = manifest["student"]
paper = manifest["paper_record"]

ui.page_header(
    "System",
    "Model identity, graph contract, and per-station serving latency — "
    "not an operator view.",
)

ui.tiles(
    [
        ("Model", "PhysDistill-EV", None, f"distilled student · epoch {student['epoch']}"),
        ("Parameters", f"{student['n_params']:,}", None,
         f"distillation weight {paper['kd_alpha']:.2f}"),
        ("Reduction vs teacher", f"{paper['reduction_x']}", "×",
         "paper, reported"),
        ("Validation WMAPE", f"{paper['wmape_table5_pct']:.2f}", "%",
         "paper, Table 5"),
        ("Pi 5 latency", f"{paper['pi5_latency_ms']:.2f}", "ms",
         f"{paper['desktop_latency_ms_mean']:.2f} ms on this machine"),
    ]
)

st.divider()

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("**Graph input signature**")
    signature = service.graph_signature()
    st.dataframe(
        {"input": list(signature.keys()), "shape": [str(v) for v in signature.values()]},
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        f"Serving graph: {config.ONNX_PATH.name} "
        f"({config.ONNX_PATH.stat().st_size / 1024:.0f} KB) · "
        f"single-threaded CPU, matching the paper's Pi 5 protocol."
    )

with right:
    st.markdown("**Latency benchmark**")
    st.caption("200 passes on one real window, after 10 warm-up passes.")
    if st.button("Run benchmark"):
        with st.spinner("Running 200 passes..."):
            profile = service.benchmark()
        st.session_state["system_benchmark"] = profile

    profile = st.session_state.get("system_benchmark")
    if profile:
        ui.tiles(
            [
                ("Mean", f"{profile['mean_ms']:.2f}", "ms"),
                ("Median", f"{profile['median_ms']:.2f}", "ms"),
                ("P95", f"{profile['p95_ms']:.2f}", "ms"),
                ("Std", f"{profile['std_ms']:.2f}", "ms"),
            ]
        )
        st.caption(
            f"{profile['passes']} passes on {profile['station_id']}, "
            f"origin {profile['origin_time_index']}."
        )
    else:
        st.caption("Not run yet this session.")

st.divider()

st.markdown("**Decision trace**")
st.caption("How one forecast was actually built, step by step — not a simulation, every field below is read from this run.")

trace_ids = [s.station_id for s in network["stations"]]
trace_labels = {s.station_id: s.display_name for s in network["stations"]}
trace_station_id = st.selectbox(
    "Station to trace",
    trace_ids,
    format_func=lambda sid: trace_labels[sid],
    key="system_trace_station",
    label_visibility="collapsed",
)

trace = service.decision_trace(trace_station_id, origin)
st.dataframe(
    trace,
    hide_index=True,
    use_container_width=True,
    column_config={
        "step": st.column_config.NumberColumn("Step", width="small"),
        "stage": st.column_config.TextColumn("Stage", width="small"),
        "state": st.column_config.TextColumn("State", width="small"),
        "action": st.column_config.TextColumn("Action", width="medium"),
        "reason": st.column_config.TextColumn("Reason", width="large"),
    },
)

st.divider()

st.markdown("**Per-station latency and last forecast**")
scores = loader.load_station_scores().set_index("station_id")

rows = []
for s in network["stations"]:
    score = scores.loc[s.station_id] if s.station_id in scores.index else None
    rows.append(
        {
            "station": s.display_name,
            "group": ui.GROUP_LABELS.get(s.group, s.group),
            "confidence": s.confidence_tier,
            "latency_ms": round(s.forecast.latency_ms, 3),
            "last forecast": f"{s.origin_timestamp:%d %b %Y, %H:%M}",
            "wmape_%": round(float(score["wmape"]), 2) if score is not None else None,
            "r2": round(float(score["r2"]), 3) if score is not None else None,
        }
    )

st.dataframe(rows, hide_index=True, use_container_width=True)
st.caption(
    "wmape/r2 are from station_scores.csv — the original evaluation protocol "
    "(global WMAPE ≈15.7%, not Table 5's 13.28%). Use them for relative "
    "ordering across stations, not as the headline accuracy figure."
)

inactive = loader.load_station_meta()
inactive = inactive[~inactive["active"]]
if not inactive.empty:
    st.caption(
        f"{len(inactive)} station(s) excluded above — no validation data: "
        + ", ".join(inactive["display_name"].tolist())
    )

with st.expander("What this view assumes"):
    st.markdown(
        f"""
- **Latency** in the table above is measured inline with each station's
  forecast at the sidebar's origin — one pass, not a benchmark. Use
  "Run benchmark" for the paper's 200-pass protocol on a single window.
- **Parameters** and **epoch** are read from each checkpoint's own metadata
  by `scripts/07_export_model_manifest.py`, via `sum(p.numel() for p in
  model.parameters())` — not a `state_dict()` sum, which double-counts the
  tied encoder/decoder variable-selection layers.
- **KD α, the reduction factor, Table 5 WMAPE, and the Pi 5 latency** are not
  recoverable from any checkpoint — α is a plain instance attribute the
  training script never saves, and the rest are external measurements. All
  four are the verified record from this repo's README, not recomputed here.
        """
    )
