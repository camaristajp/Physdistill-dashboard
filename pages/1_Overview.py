"""
pages/1_Overview.py

Overview — the ten-second answer. Is anything on the network going to be
stressed in the next 24 hours, and where.

Routed through app.py's st.navigation(position="top") router — see app.py
for the page list and st.set_page_config(), which can only be called once
per app and so lives there now, not here.
"""

import streamlit as st

from components import charts, ui
from core import config, service

dark = ui.inject_css()

origin = ui.origin_control()

try:
    network = service.forecast_network(origin)
except Exception as exc:  # noqa: BLE001 — surface the cause, don't swallow it
    st.error(f"Could not build the network forecast: {exc}")
    st.stop()

ui.station_rail(network)
ui.replay_banner()
ui.settings_panel()

stations = network["stations"]
peak = network["peak"]
energy = network["expected_energy"]
breaches = network["breaches"]

critical_hours = int((breaches["severity"] == "critical").sum()) if len(breaches) else 0
warning_hours = int((breaches["severity"] == "warning").sum()) if len(breaches) else 0

# Network headroom is the TIGHTEST station, not the aggregate. Summing twelve
# thresholds lets spare capacity at quiet stations mask an overloaded feeder,
# which is the opposite of what this tile exists to surface.
per_station_headroom = {s.station_id: s.headroom_pct for s in stations}
tightest_id = min(per_station_headroom, key=per_station_headroom.get)
tightest = next(s for s in stations if s.station_id == tightest_id)
tightest_headroom = per_station_headroom[tightest_id]

headroom_colour = (
    ui.STATUS["critical"] if tightest_headroom < 0
    else ui.SEVERITY_COLOURS["warning"] if tightest_headroom < 10
    else ui.INK
)
risk_colour = ui.STATUS["critical"] if network["stations_at_risk"] else ui.INK

ui.page_header(
    "Jiaxing network",
    f"{network['origin_timestamp']:%d %b %Y, %H:%M} origin · "
    f"{len(stations)} stations · next {config.HORIZON} hours",
    pill="replay · 2021",
)

ui.tiles(
    [
        (
            "Predicted network peak",
            f"{peak['p50']:,.0f}",
            "kWh",
            f"H+{peak['horizon_h']} at {peak['timestamp']:%H:%M}",
        ),
        (
            "Tightest headroom",
            f"{tightest_headroom:,.0f}%",
            None,
            f"{tightest.display_name}, at P90",
            headroom_colour,
        ),
        (
            "Expected consumption",
            f"{energy['p50_total']:,.0f}",
            "kWh",
            f"{energy['p10_total']:,.0f}–{energy['p90_total']:,.0f} hourly bounds summed",
        ),
        (
            "Stations at risk",
            f"{network['stations_at_risk']}",
            f"of {len(stations)}",
            f"{critical_hours + warning_hours} breach-hours "
            f"({critical_hours} critical)",
            risk_colour,
        ),
        (
            "Inference",
            f"{network['mean_latency_ms']:.2f}",
            "ms",
            "mean per station, on CPU",
        ),
    ]
)

st.divider()

left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown("**Aggregate load · next 24 hours**")
    st.plotly_chart(
        charts.band_chart(network["aggregate"], capacity=None, height=330, dark=dark),
        use_container_width=True,
        theme=None,
        config=charts.PLOTLY_CONFIG_STATIC,
    )
    st.caption(
        "P50 is the expected curve; the band spans P10 to P90. No combined "
        "threshold is drawn — capacity is a per-station constraint, and summing "
        "it across the network would hide the stations that matter."
    )

with right:
    st.markdown("**Where the pressure is**")
    if not len(breaches):
        st.success("No station crosses its threshold in this horizon.")
    else:
        summary = (
            breaches.groupby("station_id")
            .agg(
                hours=("horizon_h", "count"),
                first_hour=("horizon_h", "min"),
                worst=("severity", lambda s: "critical" if "critical" in set(s) else "warning"),
                peak_p90=("p90", "max"),
                capacity=("capacity", "first"),
            )
            .reset_index()
        )
        names = {s.station_id: s.display_name for s in stations}
        summary["station"] = summary["station_id"].map(names)
        summary["_rank"] = summary["worst"].map({"critical": 0, "warning": 1})
        summary = summary.sort_values(["_rank", "first_hour"])

        for row in summary.itertuples():
            colour = ui.SEVERITY_COLOURS[row.worst]
            over = row.peak_p90 / row.capacity * 100 if row.capacity else float("nan")
            st.markdown(
                f"<div style='padding:7px 0;border-bottom:0.5px solid {ui.GRIDLINE}'>"
                f"<span style='color:{colour};font-weight:500;font-size:0.88rem'>"
                f"{row.station}</span>"
                f"<span style='float:right;color:{ui.INK_MUTED};font-size:0.76rem'>"
                f"{row.hours}h from H+{row.first_hour} · peak {over:.0f}%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.caption(
            "Warning when the reserve requirement (P90) crosses capacity, "
            "critical when expected demand (P50) does."
        )

st.divider()

st.markdown("**Utilisation by station and hour**")
rows = service.utilisation_matrix(network)
timestamps = network["aggregate"]["timestamp"].tolist()

st.plotly_chart(
    charts.utilisation_heatmap(rows, timestamps, dark=dark),
    use_container_width=True,
    theme=None,
    config=charts.PLOTLY_CONFIG_STATIC,
)
st.caption(
    "Each cell is forecast P90 as a share of that station's threshold. Red marks "
    "hours where the reserve requirement no longer fits. "
    + ui.capacity_note(stations[0].capacity_source)
)

with st.expander("What this view assumes"):
    st.markdown(
        f"""
Forecasts come from PhysDistill-EV, a 4,148-parameter distilled Temporal
Fusion Transformer running locally through ONNX Runtime. Each station gets an
independent 24-hour probabilistic forecast from 168 hours of history.

- **Capacity** {ui.capacity_note(stations[0].capacity_source).lower()}
- **Headroom** reports the tightest single station, not the network total.
  A network with spare capacity overall can still have an overloaded feeder.
- **Quantile ordering** is corrected by sorting before display, since pinball
  loss does not enforce monotonicity across heads.
- **Expected consumption** sums hourly P10 and P90 bounds. That is not a
  proper interval on the total, only the summed hourly bounds.
- **Data** is a replay of Jiaxing, October–December 2021.
        """
    )
