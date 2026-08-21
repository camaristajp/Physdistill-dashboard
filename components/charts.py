"""
components/charts.py

Every figure the dashboard draws. Two chart types carry the whole product: a
band chart for forecasts and a heat matrix for station-by-hour utilisation.
Keeping it to two means an operator learns to read the dashboard once.

Functions take plain arrays and a capacity number. No data fetching here.
"""

import numpy as np
import plotly.graph_objects as go

from core import config

# Two small palettes, light and dark, kept local rather than imported from
# components/ui.py so this module stays a pure function library — plain
# arrays and a capacity number in, a figure out, independently testable.
# Values mirror ui.py's chart tokens.
_PALETTES = {
    False: {  # light
        "blue": "#1d6fa5", "band_fill": "rgba(29, 111, 165, 0.16)",
        "band_fill_inner": "rgba(29, 111, 165, 0.26)", "red": "#c0392b",
        "grey": "#9ca3af", "grid": "rgba(0, 0, 0, 0.07)",
        "bg": "white", "font": "#1f2328", "actual": "#111827",
        "origin_line": "#6b7280", "origin_tint": "rgba(29, 111, 165, 0.03)",
    },
    True: {  # dark
        "blue": "#3987e5", "band_fill": "rgba(57, 135, 229, 0.22)",
        "band_fill_inner": "rgba(57, 135, 229, 0.34)", "red": "#e66767",
        "grey": "#6b6a65", "grid": "rgba(255, 255, 255, 0.09)",
        "bg": "#1a1a19", "font": "#e6e5e0", "actual": "#f5f5f3",
        "origin_line": "#93928c", "origin_tint": "rgba(57, 135, 229, 0.05)",
    },
}

# Utilisation ramp: blues below capacity, red once the threshold is crossed.
# The colour change is the alert, so it must land exactly at 1.0. One ramp
# for both themes — it reads fine on either surface since the top of the
# scale is a saturated red regardless.
UTILISATION_SCALE = [
    [0.00, "rgba(29, 111, 165, 0.10)"],
    [0.30, "rgba(29, 111, 165, 0.32)"],
    [0.55, "rgba(29, 111, 165, 0.58)"],
    [0.66, "rgba(29, 111, 165, 0.85)"],
    [0.6601, "#c0392b"],
    [1.00, "#7d1d13"],
]
UTILISATION_MAX = 1.5


# No mode bar anywhere. Seven controls an operator won't use, and the camera
# icon invites the question of what exactly it's capturing. Pan still works
# by drag (dragmode="pan" below); zoom by scroll. click-to-inspect (see
# band_chart's clickable=) doesn't depend on the mode bar being visible.
PLOTLY_CONFIG = {"displayModeBar": False}
PLOTLY_CONFIG_STATIC = {"displayModeBar": False}  # kept as an alias; same config now


def _base_layout(fig, height, y_title, dark=False):
    p = _PALETTES[dark]
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=28, b=8),
        plot_bgcolor=p["bg"],
        paper_bgcolor=p["bg"],
        font=dict(size=12, color=p["font"]),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=p["bg"], bordercolor=p["grid"],
            font=dict(size=12, color=p["font"]),
        ),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)",
        ),
        dragmode="pan",
    )
    fig.update_xaxes(
        showgrid=False, linecolor=p["grid"], ticks="outside", tickcolor=p["grid"],
        tickformat="%H:%M",  # basic hour:minute on the axis; full date stays in hover
        hoverformat="%a %d %b · %H:%M",
    )
    fig.update_yaxes(
        title_text=y_title, gridcolor=p["grid"], zeroline=False, rangemode="tozero"
    )
    return fig


def band_chart(frame, capacity=None, capacity_label="capacity",
               actuals=None, history=None, title=None, height=340,
               nested=True, compare=None, x_range=None, clickable=False,
               dark=False):
    """Forecast with quantile bands, optional capacity line and ground truth.

    frame needs 'timestamp' plus q-columns. nested=True draws P25-P75 inside
    P10-P90 so the shape of the distribution is visible, not just its span.

    compare: optional {'name': str, 'values': array-like, 'color': str} drawn
    as a dashed median line over the same x-axis, for the compare toggle.

    x_range: optional [start, end] to pre-zoom the x-axis (a range preset),
    without discarding the untrimmed data the way slicing the frame would.

    clickable: adds a generous, near-invisible hit-target layer on the P50
    line so a page can wire st.plotly_chart(..., on_select=...) to it — the
    visible line stays a thin 2.5px stroke; clicking anywhere within ~18px
    of it registers, per the hit-target-bigger-than-the-mark rule.

    dark: pass ui.inject_css()'s return value. Plotly figures don't inherit
    page CSS, so the chart's own background/grid/font need this explicitly.
    """
    p = _PALETTES[dark]
    fig = go.Figure()

    x = list(frame["timestamp"])
    q_low = frame[f"q{config.Q_LOW:.2f}"].to_numpy()
    q_high = frame[f"q{config.Q_HIGH:.2f}"].to_numpy()
    q_median = frame[f"q{config.Q_MEDIAN:.2f}"].to_numpy()

    # Every y-value that ends up on the plot, so the axis range below can be
    # asserted explicitly rather than trusted to autorange. y_values seeds
    # with 0 so a chart with only small values still floors correctly.
    y_values = [0.0]

    # History is added first, so it draws *under* everything that follows —
    # the forecast band and P50 are what this page is about. In practice the
    # two never overlap in x (history ends at the origin, the forecast
    # starts after it), so this is about not being the visually loudest
    # thing on a page whose subject is the forecast, not stacking order.
    if history is not None and len(history):
        history_y = history["Energy_kWh"].to_numpy()
        y_values.append(float(history_y.max()) if len(history_y) else 0.0)
        fig.add_trace(
            go.Scatter(
                x=list(history["Timestamp"]),
                y=history_y,
                mode="lines",
                name="observed",
                line=dict(color=p["grey"], width=1.2),
                hovertemplate="%{y:.1f} kWh<extra>observed</extra>",
            )
        )

    y_values.append(float(np.nanmax(q_high)) if len(q_high) else 0.0)

    fig.add_trace(
        go.Scatter(x=x, y=q_high, mode="lines", line=dict(width=0),
                   name="P90", hoverinfo="skip", showlegend=False)
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=q_low, mode="lines", line=dict(width=0), fill="tonexty",
            fillcolor=p["band_fill"], name="P10–P90",
            hovertemplate="%{y:.1f} kWh<extra>P10</extra>",
        )
    )

    if nested and f"q{0.75:.2f}" in frame.columns:
        fig.add_trace(
            go.Scatter(x=x, y=frame["q0.75"].to_numpy(), mode="lines",
                       line=dict(width=0), name="P75", hoverinfo="skip",
                       showlegend=False)
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=frame["q0.25"].to_numpy(), mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor=p["band_fill_inner"], name="P25–P75",
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x, y=q_median, mode="lines", name="expected (P50)",
            line=dict(color=p["blue"], width=2.5),
            hovertemplate="%{y:.1f} kWh<extra>P50</extra>",
        )
    )

    if clickable:
        # Purely a click hit-target: 18px markers, invisible, own trace so it
        # doesn't thicken the visible P50 line. hoverinfo="skip" because the
        # x-unified hover above already covers this x position.
        fig.add_trace(
            go.Scatter(
                x=x, y=q_median, mode="markers",
                marker=dict(size=18, opacity=0),
                name="_click_target", showlegend=False, hoverinfo="skip",
            )
        )

    if compare is not None and len(compare.get("values", [])):
        compare_y = list(compare["values"])
        y_values.append(float(np.nanmax(compare_y)) if compare_y else 0.0)
        fig.add_trace(
            go.Scatter(
                x=x, y=compare_y, mode="lines",
                name=compare.get("name", "compare"),
                line=dict(color=compare.get("color", p["grey"]), width=1.8, dash="dot"),
                hovertemplate="%{y:.1f} kWh<extra>" + compare.get("name", "compare") + "</extra>",
            )
        )

    if actuals is not None and len(actuals):
        actuals_y = actuals["Energy_kWh"].to_numpy()
        y_values.append(float(actuals_y.max()) if len(actuals_y) else 0.0)
        fig.add_trace(
            go.Scatter(
                x=list(actuals["Timestamp"]),
                y=actuals_y,
                mode="lines",
                name="actual",
                line=dict(color=p["actual"], width=1.6, dash="dot"),
                hovertemplate="%{y:.1f} kWh<extra>actual</extra>",
            )
        )

    if capacity and np.isfinite(capacity):
        y_values.append(float(capacity))
        fig.add_hline(
            y=capacity,
            line=dict(color=p["red"], width=1.2, dash="dash"),
            annotation_text=f"{capacity_label} {capacity:,.0f} kWh",
            annotation_position="top right",
            annotation_font=dict(color=p["red"], size=11),
        )

    # Forecast boundary — the single most important distinction on this
    # chart, and the one thing a continuous axis can't show on its own. Only
    # drawn when history is present: without it there's no boundary to mark.
    if history is not None and len(history):
        origin_x = history["Timestamp"].iloc[-1]
        fig.add_vrect(
            x0=origin_x, x1=x[-1] if x else origin_x,
            fillcolor=p["origin_tint"], line_width=0, layer="below",
        )
        fig.add_vline(
            x=origin_x,
            line=dict(color=p["origin_line"], width=1),
            annotation_text="forecast origin",
            annotation_position="top",
            annotation_font=dict(size=10, color=p["origin_line"]),
        )

    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=14), x=0, xanchor="left"))

    fig = _base_layout(fig, height, "kWh per hour", dark=dark)
    if x_range is not None:
        fig.update_xaxes(range=list(x_range))

    # Assert the floor explicitly rather than trust autorange + rangemode
    # alone — pass a real number in, not "near zero, plus whatever the
    # frontend theme layer decides to do with it.
    y_max = max(y_values) if y_values else 1.0
    fig.update_yaxes(range=[0, y_max * 1.08 if y_max > 0 else 1.0])
    return fig


def utilisation_heatmap(rows, timestamps, height=None, dark=False):
    """Station x hour utilisation. Red cells are hours over capacity.

    rows: [{display_name, values}] where values is utilisation as a fraction.
    """
    p = _PALETTES[dark]
    labels = [row["display_name"] for row in rows][::-1]
    matrix = np.vstack([row["values"] for row in rows])[::-1]

    hours = [ts.strftime("%H") if hasattr(ts, "strftime") else str(ts)
             for ts in timestamps]

    text = np.where(
        np.isfinite(matrix),
        np.round(matrix * 100).astype(object),
        "",
    )

    fig = go.Figure(
        go.Heatmap(
            z=np.clip(matrix, 0, UTILISATION_MAX),
            x=hours,
            y=labels,
            customdata=matrix,
            colorscale=UTILISATION_SCALE,
            zmin=0,
            zmax=UTILISATION_MAX,
            showscale=False,
            xgap=2,
            ygap=2,
            hovertemplate="%{y}<br>hour %{x}<br>%{customdata:.0%} of capacity<extra></extra>",
        )
    )
    del text

    fig.update_layout(
        height=height or max(240, 26 * len(labels) + 70),
        margin=dict(l=8, r=8, t=24, b=8),
        plot_bgcolor=p["bg"],
        paper_bgcolor=p["bg"],
        font=dict(size=11, color=p["font"]),
    )
    fig.update_xaxes(side="top", showgrid=False, tickfont=dict(size=10))
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11))
    return fig


def sparkline(values, colour=None, height=38, dark=False):
    """Tiny trend line for the station rail."""
    colour = colour or _PALETTES[dark]["blue"]
    fig = go.Figure(
        go.Scatter(
            y=list(values), mode="lines", line=dict(color=colour, width=1.5),
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig
