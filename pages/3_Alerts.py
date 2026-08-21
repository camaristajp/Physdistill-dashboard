"""
pages/3_Alerts.py

The ranked action list. Overview says whether the network is stressed;
Stations shows one forecast in depth; this page is what an operator works
down: every hour, at every station, where the forecast crosses capacity in
this horizon, worst first.

    streamlit run app.py   (then open Alerts from the top nav)

Routed through app.py's st.navigation — st.set_page_config() lives there
now, since it can only be called once per app.
"""

import streamlit as st

from components import ui
from core import config, service

dark = ui.inject_css()

origin = ui.origin_control()

try:
    network = service.forecast_network(origin)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not build the network forecast: {exc}")
    st.stop()

ui.station_rail(network)
ui.replay_banner()
ui.settings_panel()

stations = network["stations"]
names = {s.station_id: s.display_name for s in stations}
breaches = network["breaches"].copy()
breaches["display_name"] = breaches["station_id"].map(names)

ui.page_header(
    "Alerts",
    f"{network['origin_timestamp']:%d %b %Y, %H:%M} origin · "
    f"next {config.HORIZON} hours · critical first, then by hour",
)

# ---------------------------------------------------------------------------
# Acknowledge state. Keyed on (station, absolute timestamp) rather than
# horizon_h, so an acknowledgement survives scrubbing the origin — the same
# real-world hour can re-enter the horizon window as the origin shifts, and
# an operator who already acted on it shouldn't see it flagged as new.
# ---------------------------------------------------------------------------
st.session_state.setdefault("acknowledged_alerts", set())
acknowledged = st.session_state["acknowledged_alerts"]


def ack_key(row):
    return (row.station_id, row.timestamp)


if breaches.empty:
    st.success("No station crosses its threshold in this horizon.")
    st.stop()

breaches["ack_key"] = list(map(ack_key, breaches.itertuples()))
breaches["acked"] = breaches["ack_key"].isin(acknowledged)

critical_n = int((breaches["severity"] == "critical").sum())
warning_n = int((breaches["severity"] == "warning").sum())
stations_n = breaches["station_id"].nunique()
unacked_n = int((~breaches["acked"]).sum())

ui.tiles(
    [
        ("Critical", critical_n, "hours", "expected demand (P50) over capacity",
         ui.STATUS["critical"] if critical_n else ui.INK),
        ("Warning", warning_n, "hours", "reserve requirement (P90) over capacity",
         ui.SEVERITY_COLOURS["warning"] if warning_n else ui.INK),
        ("Stations affected", stations_n, f"of {len(stations)}"),
        ("Outstanding", unacked_n, f"of {len(breaches)}", "not yet acknowledged"),
    ]
)

st.divider()

filter_col, ack_col, clear_col = st.columns([2, 2, 1])
with filter_col:
    severity_filter = st.radio(
        "Severity", ["All", "Critical", "Warning"],
        key="alerts_severity_filter", horizontal=True,
    )
with ack_col:
    show_acked = st.checkbox("Show acknowledged", key="alerts_show_acked")
with clear_col:
    if st.button("Clear acknowledgements"):
        acknowledged.clear()
        st.rerun()

view = breaches
if severity_filter != "All":
    view = view[view["severity"] == severity_filter.lower()]
if not show_acked:
    view = view[~view["acked"]]

if view.empty:
    st.info("Nothing matches this filter.")
else:
    header = st.columns([0.7, 2.4, 2.6, 1.1, 2.6, 1.2])
    for col, label in zip(
        header, ["", "station · hour", "P50 / P90 vs capacity", "headroom", "action", ""]
    ):
        col.markdown(f"<span class='pd-note'>{label}</span>", unsafe_allow_html=True)

    for row in view.itertuples():
        colour = ui.SEVERITY_COLOURS[row.severity]
        cols = st.columns([0.7, 2.4, 2.6, 1.1, 2.6, 1.2])

        with cols[0]:
            st.markdown(
                f"<span class='pd-dot' style='background:{colour};display:inline-block'></span>",
                unsafe_allow_html=True,
            )
        with cols[1]:
            weight = "400" if row.acked else "500"
            st.markdown(
                f"<span style='font-weight:{weight}'>{row.display_name}</span><br>"
                f"<span style='color:{ui.INK_MUTED};font-size:0.78rem'>"
                f"H+{row.horizon_h} · {row.timestamp:%d %b, %H:%M}</span>",
                unsafe_allow_html=True,
            )
        with cols[2]:
            st.markdown(
                f"P50 {row.p50:,.0f} · P90 {row.p90:,.0f} / cap {row.capacity:,.0f} kWh"
            )
        with cols[3]:
            headroom_colour = ui.STATUS["critical"] if row.headroom_pct < 0 else ui.SEVERITY_COLOURS["warning"]
            st.markdown(
                f"<span style='color:{headroom_colour}'>{row.headroom_pct:,.0f}%</span>",
                unsafe_allow_html=True,
            )
        with cols[4]:
            st.caption(row.action)
        with cols[5]:
            if row.acked:
                st.caption("✓ acked")
            elif st.button("Acknowledge", key=f"ack_{row.station_id}_{row.horizon_h}"):
                acknowledged.add(row.ack_key)
                st.rerun()

st.divider()

with st.expander("What this view assumes"):
    st.markdown(
        f"""
- **Severity**: critical when expected demand (P50) crosses capacity, warning
  when only the reserve requirement (P90) does.
- **Acknowledging** an alert is local to this browser session and keyed on
  station and absolute hour, not on severity — if the forecast for that hour
  later reclassifies (e.g. warning to critical) as the origin advances, it
  stays marked acknowledged rather than reappearing as new.
- {ui.capacity_note(stations[0].capacity_source)}
- Ranked critical first, then by hour — not by how far over capacity a
  station is, so the nearest deadline surfaces first within each tier.
        """
    )
