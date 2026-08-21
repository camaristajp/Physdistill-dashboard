"""
pages/2_Stations.py

One station, one forecast, in enough detail to act on. The Overview answers
"is anything stressed"; this page answers "what does this station's forecast
actually look like, and how much should I trust it."

Interaction model: click a point on the chart to inspect that exact hour,
use the zoom presets to focus the window, use replay to watch the forecast
evolve. The station rail in the sidebar is the only station selector — there
is no separate dropdown.

    streamlit run app.py   (then open Stations from the top nav)

Routed through app.py's st.navigation — st.set_page_config() lives there
now, since it can only be called once per app.
"""

import time

import pandas as pd
import streamlit as st

from components import charts, ui
from core import config, loader, service

dark = ui.inject_css()

# ---------------------------------------------------------------------------
# Sidebar: network-wide origin (shared with Overview via the "origin" key),
# then the station rail — clickable here, since this page has no other way
# to choose a station.
# ---------------------------------------------------------------------------
network_origin = ui.origin_control()

active = loader.active_stations()
if active.empty:
    st.error("No station has usable validation data.")
    st.stop()

st.session_state.setdefault("selected_station_id", active["station_id"].iloc[0])

try:
    network = service.forecast_network(network_origin)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not build the network forecast: {exc}")
    st.stop()

station_id = ui.station_rail(
    network, selected=st.session_state["selected_station_id"], clickable=True
)
st.session_state["selected_station_id"] = station_id
ui.replay_banner()
ui.settings_panel()

meta = loader.station_index()[station_id]

# ---------------------------------------------------------------------------
# Page-specific rolling origin. Independent of the sidebar's network origin,
# so an operator can scrub one station's forecast through time without
# disturbing the network view. Starts wherever the sidebar origin points.
# ---------------------------------------------------------------------------
origins = list(loader.valid_origins(station_id))
if not origins:
    st.error(f"{meta['display_name']} has no valid forecast origin.")
    st.stop()

ui.page_header(
    meta["display_name"],
    f"{ui.GROUP_LABELS.get(meta['group'], meta['group'])} station · "
    f"click a point on the chart to inspect that hour",
    pill=meta["confidence_tier"],
)

# Streamlit forbids writing to a widget's session_state key after that widget
# has been instantiated in the same run. So the slider's position is seeded
# and advanced here, BEFORE the slider below is created — never after.
if "station_origin_idx" not in st.session_state:
    st.session_state["station_origin_idx"] = (
        origins.index(network_origin) if network_origin in origins else len(origins) // 2
    )
if st.session_state.pop("_station_replay_tick", False):
    st.session_state["station_origin_idx"] = (
        st.session_state["station_origin_idx"] + 1
    ) % len(origins)

with st.container(border=True):
    # Origin position is the primary control on this page: full width, a
    # real label, one unambiguous caption. Replay is secondary — a small
    # play toggle and a speed select, not a second slider competing for
    # the same visual weight.
    position = st.slider(
        "Forecast origin",
        min_value=0,
        max_value=len(origins) - 1,
        key="station_origin_idx",
    )
    origin = int(origins[position])

    try:
        panel_ts = loader.load_panel()
        origin_ts = panel_ts.loc[
            (panel_ts["station_id"] == station_id) & (panel_ts["Time_Index"] == origin),
            "Timestamp",
        ].iloc[0]
        st.caption(f"Origin {position + 1} of {len(origins)} · {origin_ts:%d %b %Y, %H:%M}")
    except IndexError:
        pass

    play_col, speed_col, pulse_col = st.columns([1.3, 1, 4])
    with play_col:
        playing = st.toggle("Replay", key="station_replay_on")
    with speed_col:
        speed = st.selectbox(
            "Speed", [0.5, 1, 2, 4], index=1,
            key="station_replay_speed", format_func=lambda s: f"{s}×",
            label_visibility="collapsed",
        )
    with pulse_col:
        if playing:
            st.markdown(
                f"<div style='padding-top:8px'>"
                f"<span class='pd-dot pd-dot-pulse' style='background:{ui.STATUS['good']}'></span>"
                f"<span class='pd-note' style='margin-left:6px'>playing</span></div>",
                unsafe_allow_html=True,
            )

if playing:
    time.sleep(0.6 / speed)
    st.session_state["_station_replay_tick"] = True
    st.rerun()

# ---------------------------------------------------------------------------
# Forecast for the selected station and origin.
# ---------------------------------------------------------------------------
try:
    station = service.forecast_station(station_id, origin)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not build a forecast at this origin: {exc}")
    st.stop()

accuracy = service.rolling_accuracy(station_id, origin)
idle = service.idle_rate(station_id, origin)

# Accent only where the card represents a live state: capacity turns red
# when this forecast actually breaches it; confidence turns amber/red for
# the tiers that warrant a second look. Everything else on this row is
# reference information, not an alert, so it stays neutral.
capacity_accent = ui.STATUS["critical"] if station.breaches else None
confidence_accent = {
    "moderate": ui.STATUS["warning"],
    "low": ui.STATUS["critical"],
}.get(station.confidence_tier)

ui.tiles(
    [
        ("Station type", ui.GROUP_LABELS.get(station.group, station.group)),
        (
            "Capacity",
            f"{station.capacity_kwh:,.0f}",
            "kWh",
            ui.capacity_note_short(station.capacity_source),
            capacity_accent,
        ),
        (
            "Confidence",
            ui.badge(station.confidence_tier),
            None,
            ui.CONFIDENCE_SHORT.get(station.confidence_tier, ""),
            confidence_accent,
        ),
        (
            "7-day accuracy",
            f"{accuracy['wmape']:.1f}" if accuracy else "—",
            "% WMAPE" if accuracy else None,
            "one-step ahead, last 7 days" if accuracy
            else "not enough history yet",
        ),
        (
            "Idle-hour rate",
            f"{idle * 100:.1f}" if idle is not None else "—",
            "%" if idle is not None else None,
            "trailing 7 days" if idle is not None else None,
        ),
    ]
)

# ---------------------------------------------------------------------------
# Compare toggle — baseline (no distillation) median, precomputed offline.
# ---------------------------------------------------------------------------
compare_trace = None
if service.comparison_available():
    show_compare = st.toggle(
        "Compare with the undistilled model",
        value=False,
        key="station_compare_toggle",
        help="Same 4,148-parameter architecture, trained without the "
             "teacher's soft targets. Shows what distillation bought.",
    )
    if show_compare:
        baseline = service.baseline_median(station_id, origin)
        if baseline is not None:
            compare_trace = {
                "name": "undistilled median",
                "values": baseline,
                "color": ui.COMPARE_COLOUR,
            }
        else:
            st.caption("No comparison forecast cached for this station/origin.")

# ---------------------------------------------------------------------------
# Fan chart.
# ---------------------------------------------------------------------------
history = service.station_history(station_id, origin, hours=24)
actuals = service.station_actuals(station_id, origin)
horizon_frame = station.forecast.frame

with st.container(border=True):
    header_col, zoom_col = st.columns([3, 2])
    with header_col:
        st.markdown("**24-hour forecast**")
    with zoom_col:
        zoom = st.segmented_control(
            "Zoom",
            ["24h history + forecast", "Forecast only", "Next 6h"],
            default="24h history + forecast",
            key="station_zoom",
            label_visibility="collapsed",
        )

    if zoom == "Forecast only":
        x_range = [horizon_frame["timestamp"].iloc[0], horizon_frame["timestamp"].iloc[-1]]
    elif zoom == "Next 6h":
        x_range = [horizon_frame["timestamp"].iloc[0], horizon_frame["timestamp"].iloc[5]]
    else:
        x_range = None

    chart_event = st.plotly_chart(
        charts.band_chart(
            horizon_frame,
            capacity=station.capacity_kwh,
            capacity_label=(
                "rated capacity" if station.capacity_source == "rated" else "derived capacity"
            ),
            actuals=actuals,
            history=history,
            compare=compare_trace,
            height=420,
            x_range=x_range,
            clickable=True,
            dark=dark,
        ),
        use_container_width=True,
        theme=None,
        config=charts.PLOTLY_CONFIG,
        on_select="rerun",
        selection_mode="points",
        key="station_chart",
    )
    st.caption(
        "Grey line is observed history; dotted black is what actually happened, "
        "where the replay data has it. The shaded region is the forecast — "
        "everything left of the line is measured, everything right of it is "
        "predicted. Click a point on the P50 line for the exact breakdown "
        "at that hour."
    )

    if station.forecast.rearranged:
        st.caption(
            f"Note: {station.forecast.crossing_pairs} quantile crossing(s) in the raw "
            "output were corrected by sorting before display."
        )

    # -------------------------------------------------------------------
    # Click-to-inspect: resolve the clicked point to a horizon row and show
    # its full quantile breakdown. Only the invisible click-target trace
    # carries markers, so any point in the selection is one of its hours.
    # -------------------------------------------------------------------
    points = chart_event.selection.points if chart_event else []
    if points:
        clicked_x = pd.Timestamp(points[0]["x"])
        idx = (horizon_frame["timestamp"] - clicked_x).abs().idxmin()
        row = horizon_frame.loc[idx]

        breach = next((b for b in station.breaches if b.horizon_h == row["horizon_h"]), None)
        severity = breach.severity if breach else "none"
        accent = ui.SEVERITY_COLOURS[severity]

        st.markdown(
            f"<div style='border-radius:12px;padding:12px 16px;margin-top:10px;"
            f"background:{ui.tint(accent, 0.07)};border-left:3px solid {accent}'>"
            f"<b>H+{int(row['horizon_h'])}</b> · {row['timestamp']:%a %d %b, %H:%M}"
            + (f" · <span style='color:{accent};font-weight:600'>{severity}</span>"
               f" — {breach.action}" if breach else " · no threshold crossing")
            + "</div>",
            unsafe_allow_html=True,
        )
        ui.tiles(
            [
                (f"P{int(q * 100):02d}", f"{row[f'q{q:.2f}']:,.1f}", "kWh")
                for q in config.QUANTILES
            ]
        )
        st.caption(f"H+{int(row['horizon_h'])}. Capacity: {station.capacity_kwh:,.0f} kWh.")

# ---------------------------------------------------------------------------
# This station's breaches at this origin.
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.markdown("**Threshold crossings in this horizon**")
    if not station.breaches:
        st.success("No hour in this horizon crosses capacity.")
    else:
        for breach in sorted(station.breaches, key=lambda b: b.horizon_h):
            colour = ui.SEVERITY_COLOURS[breach.severity]
            st.markdown(
                f"<div style='padding:6px 0;border-bottom:1px solid {ui.GRIDLINE}'>"
                f"<span style='color:{colour};font-weight:600'>H+{breach.horizon_h}</span>"
                f"<span style='color:{ui.INK_MUTED};font-size:0.8rem'> · {breach.timestamp:%H:%M} · "
                f"P50 {breach.p50:,.0f} / P90 {breach.p90:,.0f} vs {breach.capacity:,.0f} kWh · "
                f"{breach.action}</span></div>",
                unsafe_allow_html=True,
            )

with st.expander("What this view assumes"):
    energy = station.expected_energy
    window_start = horizon_frame["timestamp"].iloc[0]
    window_end = horizon_frame["timestamp"].iloc[-1]
    if window_start.date() == window_end.date():
        window_text = f"{window_start:%d %b %H:%M} – {window_end:%H:%M}"
    else:
        window_text = f"{window_start:%d %b %H:%M} – {window_end:%d %b %H:%M}"
    st.markdown(
        f"""
- **Origin convention**: the slider position is the last observed hour;
  the chart's right half is the forecast — {window_text}.
- **Confidence tier** ({station.confidence_tier}): {config.CONFIDENCE_TIERS.get(station.confidence_tier, '')}
- **Capacity** {ui.capacity_note(station.capacity_source).lower()}
- **Inference latency** at this origin: {station.forecast.latency_ms:.2f} ms, single-threaded CPU.
- **Expected energy over the horizon**: {energy['p50_total']:,.0f} kWh
  ({energy['p10_total']:,.0f}–{energy['p90_total']:,.0f} kWh, hourly bounds summed —
  not a proper interval on the total).
        """
    )
