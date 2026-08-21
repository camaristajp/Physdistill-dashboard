"""
components/ui.py

Shared Streamlit pieces: the theme toggle, the origin control, the station
rail, metric tiles, confidence badges, and the CSS that gives the app its
card-based look. This is an operational surface — a grid operator reading
capacity and dispatch signals — not a consumer app, and the styling choices
follow from that: plain text over icons, colour reserved for state, no
decoration without a reason.

Tiles are hand-rolled rather than st.metric, because st.metric renders a
directional arrow on every delta and the captions here are context, not
change. An up arrow next to "at H+12" reads as a rise that was never claimed.

Theming mechanism: inject_css() reads the dark-mode choice from session
state (the actual toggle widget renders later, in settings_panel(), at the
bottom of the sidebar — session state is the source of truth regardless of
where in the script the widget itself runs) and REASSIGNS this module's
color constants (INK, SEVERITY_COLOURS, …) to the chosen theme's values
before building the stylesheet. Every other function here, and every page's
direct `ui.SEVERITY_COLOURS[...]` / `ui.INK_MUTED` reference, reads those
same names fresh each call, so they pick up the active theme automatically.
inject_css() must be the first ui call on every page, exactly as before.

Caveat this can't fix: Streamlit's own native widget chrome (buttons, the
sidebar panel, dataframes, the plotly mode bar) follows Streamlit's own
theme setting — accessible to the viewer via the app's ⋮ menu → Settings —
which app code has no supported way to read or drive. This toggle re-themes
everything this module and core/charts.py draw; it can't reach past that.
"""

import pandas as pd
import streamlit as st

from core import config, loader, service

# ---------------------------------------------------------------------------
# Palette. Status colours are reserved for state — a breach, a low-confidence
# station, a warning count — and are never used decoratively. Card accents
# are neutral by default; they only take a status colour when the card's
# value itself represents that state, and then they match the value colour.
# No inline hex belongs outside this module or charts.py.
# ---------------------------------------------------------------------------
_LIGHT = {
    "ink": "#1f2328", "ink_secondary": "#52514e", "ink_muted": "#6b7280",
    "gridline": "#e1e0d9", "border": "rgba(11,11,11,0.10)",
    "surface": "#fcfcfb", "page_plane": "#f9f9f7",
    "status": {"critical": "#c0392b", "warning": "#854f0b", "good": "#0f6e56"},
    "neutral_accent": "#d1d5db", "compare": "#6b7280", "primary": "#1d6fa5",
    "chart_bg": "#ffffff", "chart_font": "#1f2328", "chart_grid": "rgba(0, 0, 0, 0.07)",
    "band_fill": "rgba(29, 111, 165, 0.16)", "band_fill_inner": "rgba(29, 111, 165, 0.26)",
}
_DARK = {
    "ink": "#f2f1ee", "ink_secondary": "#c3c2b7", "ink_muted": "#93928c",
    "gridline": "#2c2c2a", "border": "rgba(255,255,255,0.10)",
    "surface": "#1a1a19", "page_plane": "#0d0d0d",
    "status": {"critical": "#e0574f", "warning": "#d99a3d", "good": "#3aa787"},
    "neutral_accent": "#44443f", "compare": "#93928c", "primary": "#3987e5",
    "chart_bg": "#1a1a19", "chart_font": "#e6e5e0", "chart_grid": "rgba(255, 255, 255, 0.09)",
    "band_fill": "rgba(57, 135, 229, 0.22)", "band_fill_inner": "rgba(57, 135, 229, 0.34)",
}


def tint(hex_colour, alpha=0.12):
    """Soft background wash for a badge or card accent, from a status hex."""
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _apply_theme(tokens):
    """Rebind the module's color constants to one theme's values."""
    global INK, INK_SECONDARY, INK_MUTED, GRIDLINE, BORDER, SURFACE, PAGE_PLANE
    global STATUS, NEUTRAL_ACCENT, COMPARE_COLOUR, PRIMARY, TIER_COLOURS, SEVERITY_COLOURS
    global CHART_BG, CHART_FONT, CHART_GRID, BAND_FILL, BAND_FILL_INNER

    INK, INK_SECONDARY, INK_MUTED = tokens["ink"], tokens["ink_secondary"], tokens["ink_muted"]
    GRIDLINE, BORDER = tokens["gridline"], tokens["border"]
    SURFACE, PAGE_PLANE = tokens["surface"], tokens["page_plane"]
    STATUS = tokens["status"]
    NEUTRAL_ACCENT = tokens["neutral_accent"]
    COMPARE_COLOUR = tokens["compare"]
    PRIMARY = tokens["primary"]
    CHART_BG, CHART_FONT, CHART_GRID = tokens["chart_bg"], tokens["chart_font"], tokens["chart_grid"]
    BAND_FILL, BAND_FILL_INNER = tokens["band_fill"], tokens["band_fill_inner"]

    TIER_COLOURS = {
        "high": (STATUS["good"], tint(STATUS["good"])),
        "moderate": (STATUS["warning"], tint(STATUS["warning"], 0.16)),
        "low": (STATUS["critical"], tint(STATUS["critical"], 0.12)),
        "unavailable": (INK_MUTED, tint(INK_MUTED, 0.16)),
        "unknown": (INK_MUTED, tint(INK_MUTED, 0.16)),
    }
    SEVERITY_COLOURS = {
        "critical": STATUS["critical"],
        "warning": STATUS["warning"],
        "none": INK_MUTED,
    }


_apply_theme(_LIGHT)  # sane defaults if a page ever runs without inject_css()

# Confidence tiers, shortened for a one-line tile note (~40 char budget). The
# full sentence (core/config.CONFIDENCE_TIERS) stays intact for the
# assumptions expanders — this is a display-only compression, not a second
# source of truth.
CONFIDENCE_SHORT = {
    "high": "reliable at this station",
    "moderate": "wider errors than average",
    "low": "indicative only",
    "unavailable": "no validation data",
    "unknown": "confidence not scored",
}

# Grouping reflects demand regime, which is how an operator reads these
# stations: a fleet depot and a tourist site fail in different ways. Shared
# here because both the station rail and the Stations page card use it.
GROUP_ORDER = ["schedule_driven", "retail_mixed", "corridor", "recreation"]
GROUP_LABELS = {
    "schedule_driven": "schedule-driven",
    "retail_mixed": "retail · mixed",
    "corridor": "corridor",
    "recreation": "recreation",
}


def _build_css():
    return f"""
<style>
  .stApp {{ background: {PAGE_PLANE}; }}
  /* No padding-top override on .block-container itself: with
     st.navigation(position="top"), Streamlit reserves clearance below its
     own header bar and a fixed override here fights that (round 1 learned
     this the hard way). The nav bar's own breathing room below is instead
     added via padding on the header/nav elements themselves, and the extra
     ~20px before the title lives on .pd-head's margin — additive to
     Streamlit's own clearance, not a replacement for it. */
  header[data-testid="stAppHeader"] {{
    padding-top: 14px; padding-bottom: 14px;
    border-bottom: 0.5px solid {BORDER};
  }}
  div[data-testid="stTopNav"] {{ gap: 6px; }}
  div[data-testid="stTopNav"] a, div[data-testid="stTopNav"] button {{
    padding-left: 10px; padding-right: 10px;
  }}
  .block-container {{ padding-bottom: 3rem; max-width: 1320px; }}
  section[data-testid="stSidebar"] {{ border-right: 0.5px solid {BORDER}; background: {PAGE_PLANE}; }}
  section[data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}
  hr {{ margin: 1.1rem 0; border-color: {GRIDLINE}; }}

  /* Only Streamlit's own markdown/heading text — never a blanket div/span
     rule, which would leak into native widget internals (buttons, selects,
     dataframes) that stay light-chrome regardless of this toggle and would
     go illegible if their own text turned white underneath them. */
  [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p {{
    color: {INK};
  }}
  [data-testid="stHeading"] {{ color: {INK}; }}

  @keyframes pd-fade-in {{
    from {{ opacity: 0; transform: translateY(6px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
  }}
  @keyframes pd-pulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50%      {{ opacity: 0.45; transform: scale(1.4); }}
  }}

  .pd-head {{ display:flex; align-items:center; gap:12px; margin-bottom:2px; margin-top:20px; }}
  .pd-title {{ font-size:1.6rem; font-weight:650; color:{INK}; letter-spacing:-0.01em; }}
  .pd-pill {{
    font-size:0.72rem; padding:3px 10px; border-radius:20px;
    background:{tint(STATUS["good"], 0.14)}; color:{STATUS["good"]}; font-weight:600;
  }}
  .pd-sub {{ color:{INK_SECONDARY}; font-size:0.86rem; margin-bottom:0.3rem; margin-top:2px; }}

  /* Metric card rows — one grid, not st.columns(). st.columns() is flexbox
     with default flex-shrink: even with width:100% on the card, that width
     is only the flex-basis, and when the row is tighter than the sum of
     the cards' natural content widths, siblings shrink from their own
     content/min-content size rather than from a shared equal baseline — a
     card with a longer note ends up wider than one without, which is
     exactly the bug this replaced. Grid's 1fr track has no such
     negotiation: every column gets an equal share of the row regardless of
     what's inside it. tiles() renders the whole row as one st.markdown
     call for this reason — there is no per-card Streamlit column here to
     fight with. */
  .pd-tile-row {{
    display: grid !important; gap: 0.5rem; margin: 0 0 1rem 0;
    width: 100%; align-items: stretch;
  }}

  /* min-width:0 is required, not decoration. Grid items default to
     min-width:auto, which means a track can't shrink below its content's
     intrinsic minimum width — so a card with a long unbroken value/note
     pushes its own 1fr track wider than its neighbours', which is the
     uneven-width failure mode even with minmax(0,1fr) on the row. Setting
     it here removes that floor; overflow-wrap plus the existing
     text-overflow:ellipsis rules below handle what no longer fits. */
  .pd-tile {{
    background:{SURFACE}; border-radius:12px; padding:11px 13px; min-height:96px;
    box-sizing:border-box; width:100%; min-width:0; overflow-wrap:anywhere;
    display:flex; flex-direction:column;
    border:1px solid {BORDER}; border-left:3px solid var(--pd-accent, {NEUTRAL_ACCENT});
    box-shadow: 0 1px 2px rgba(11,11,11,0.03);
    transition: transform 140ms ease, box-shadow 140ms ease;
    animation: pd-fade-in 320ms ease-out both;
  }}
  .pd-tile:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(11,11,11,0.08);
  }}
  .pd-tile-label {{
    font-size:0.72rem; color:{INK_SECONDARY}; margin-bottom:4px; font-weight:600;
    text-transform:uppercase; letter-spacing:0.03em;
  }}
  .pd-tile-value {{
    font-size:1.5rem; font-weight:650; color:{INK}; line-height:1.2;
    overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
  }}
  .pd-tile-unit {{ font-size:0.8rem; color:{INK_SECONDARY}; font-weight:500; }}
  .pd-tile-note {{
    font-size:0.74rem; color:{INK_MUTED}; margin-top:auto; padding-top:6px;
    overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
  }}

  /* Native Streamlit bordered containers, restyled as cards. */
  div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {SURFACE} !important;
    border-radius: 16px !important;
    border: 1px solid {BORDER} !important;
    box-shadow: 0 1px 3px rgba(11,11,11,0.04);
    transition: box-shadow 180ms ease;
  }}

  .pd-badge {{
    display:inline-flex; align-items:center; gap:5px; padding:2px 10px; border-radius:20px;
    font-size:0.72rem; font-weight:600;
  }}
  .pd-badge-dot {{ width:6px; height:6px; border-radius:50%; flex:none; }}
  /* Rail row: fixed three-part layout — dot flush left, name truncating
     with an ellipsis rather than wrapping, number right-aligned and muted.
     Row height target ~28px (5px 8px padding + one line of 0.78rem text). */
  .pd-rail-row {{
    display:flex; align-items:center; gap:7px; padding:5px 8px; border-radius:6px;
    font-size:0.78rem; transition: background 120ms ease;
  }}
  .pd-rail-row:hover {{ background: rgba(128,128,128,0.08); }}
  .pd-rail-row.selected {{
    background: {tint(PRIMARY, 0.10)}; border-left: 2px solid {PRIMARY};
    padding-left: 6px; /* 8px - 2px border, so content doesn't shift */
  }}
  .pd-rail-name {{
    flex:1; min-width:0; overflow:hidden; white-space:nowrap; text-overflow:ellipsis;
  }}
  .pd-rail-value {{ color:{INK_MUTED}; font-size:0.74rem; flex:none; }}
  .pd-rail-group {{
    font-size:0.68rem; color:{INK_MUTED}; margin:8px 0 2px; font-weight:600;
    text-transform:uppercase; letter-spacing:0.04em;
  }}
  .pd-rail-unit {{ font-size:0.68rem; color:{INK_MUTED}; margin-bottom:2px; }}
  .pd-dot {{ width:7px; height:7px; border-radius:50%; flex:none; }}
  .pd-dot-pulse {{ animation: pd-pulse 1.2s ease-in-out infinite; }}
  .pd-note {{ color:{INK_SECONDARY}; font-size:0.8rem; }}

  /* Rail rows rendered as buttons (Stations page, clickable=True). Base
     chrome here; per-station colour/dot/value are injected as small
     per-instance rules keyed on Streamlit's own `.st-key-<key>` class
     (documented behaviour: any widget given key=... gets that class on its
     container), since a native button can't hold custom child markup. */
  section[data-testid="stSidebar"] div[data-testid="stButton"] button {{
    background: transparent !important; border: none !important;
    box-shadow: none !important;
    text-align: left !important; justify-content: flex-start !important;
    font-weight: 400 !important; font-size: 0.78rem !important;
    padding: 5px 44px 5px 20px !important; min-height: unset !important;
    height: 28px !important; line-height: 1.4 !important;
    border-radius: 6px !important; position: relative !important;
    white-space: nowrap !important; overflow: hidden !important;
    text-overflow: ellipsis !important; display: block !important;
  }}
  section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {{
    background: rgba(128,128,128,0.08) !important;
  }}
  section[data-testid="stSidebar"] div[data-testid="stButton"] button:active {{
    transform: scale(0.99);
  }}

  button[kind="secondary"], button[kind="primary"] {{
    border-radius: 8px !important;
    transition: transform 100ms ease !important;
  }}
</style>
"""


def inject_css():
    """Applies the theme and injects the stylesheet.

    Must be the first ui call on every page. The dark-mode toggle itself
    renders later, in settings_panel() at the bottom of the sidebar — this
    only reads whatever session state already holds (False on first load),
    since the CSS has to be built before the rest of the page renders.
    Returns True if dark mode is active — pass it to charts.band_chart(...,
    dark=...) and charts.utilisation_heatmap(..., dark=...), since
    core/charts.py stays a pure function module with no session_state
    access of its own.
    """
    st.session_state.setdefault("pd_dark_mode", False)
    dark = st.session_state["pd_dark_mode"]
    _apply_theme(_DARK if dark else _LIGHT)
    st.markdown(_build_css(), unsafe_allow_html=True)
    return dark


def settings_panel():
    """Preferences, collapsed at the bottom of the sidebar — set once, not
    prime real estate. Call this last, after station_rail()/replay_banner()."""
    st.sidebar.divider()
    with st.sidebar.expander("Settings"):
        st.toggle("Dark mode", key="pd_dark_mode")


def badge(tier, dot=True):
    colour, background = TIER_COLOURS.get(tier, TIER_COLOURS["unknown"])
    dot_html = f"<span class='pd-badge-dot' style='background:{colour}'></span>" if dot else ""
    return (
        f"<span class='pd-badge' style='color:{colour};background:{background}'>"
        f"{dot_html}{tier}</span>"
    )


def page_header(title, subtitle=None, pill=None):
    pill_html = f"<span class='pd-pill'>{pill}</span>" if pill else ""
    st.markdown(
        f"<div class='pd-head'><span class='pd-title'>{title}</span>{pill_html}</div>",
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(f"<div class='pd-sub'>{subtitle}</div>", unsafe_allow_html=True)


def tiles(items):
    """Row of metric cards, rendered as ONE CSS grid — not st.columns().

    items: [(label, value, unit_or_none, note_or_none, accent_or_none)]

    A card's left accent bar is neutral grey by default. It only takes
    colour when the caller passes a 5th element — a hex from ui.STATUS —
    and then the accent matches the value colour exactly. Colour on this
    row means something happened; it is never used to tell unrelated cards
    apart, and only ever appears on a card that represents live state.

    This is the only card row in the app; every page calls this rather than
    building its own. See the .pd-tile-row CSS comment for why it can't be
    st.columns() underneath.
    """
    cards = []
    for item in items:
        label = item[0]
        value = item[1]
        unit = item[2] if len(item) > 2 else None
        note = item[3] if len(item) > 3 else None
        accent_colour = item[4] if len(item) > 4 else None
        value_colour = accent_colour or INK
        accent = accent_colour or NEUTRAL_ACCENT

        unit_html = f"<span class='pd-tile-unit'> {unit}</span>" if unit else ""
        note_html = f"<div class='pd-tile-note' title='{note}'>{note}</div>" if note else ""
        cards.append(
            f"<div class='pd-tile' style='--pd-accent:{accent}'>"
            f"<div class='pd-tile-label'>{label}</div>"
            f"<div class='pd-tile-value' style='color:{value_colour}'>{value}{unit_html}</div>"
            f"{note_html}</div>"
        )

    # minmax(0,1fr), not a bare 1fr: a bare 1fr still lets a track grow past
    # its equal share to accommodate wide content before min-width:0 (above)
    # even gets a chance to matter. The 0 floor is what actually caps it.
    st.markdown(
        f"<div class='pd-tile-row' "
        f"style='grid-template-columns:repeat({len(items)},minmax(0,1fr))'>"
        + "".join(cards) + "</div>",
        unsafe_allow_html=True,
    )


def origin_control(key="origin", label="Forecast origin"):
    """Date and hour selection, returned as a Time_Index.

    The origin is the last observed hour, so the forecast covers the 24 hours
    after it. Picking a date puts the origin at the first valid hour that day.
    """
    origins = service.common_origins()
    if not origins:
        st.sidebar.error("No origin is valid across all stations.")
        st.stop()

    panel = loader.load_panel()
    lookup = (
        panel[panel["Time_Index"].isin(origins)][["Time_Index", "Timestamp"]]
        .drop_duplicates("Time_Index")
        .set_index("Time_Index")["Timestamp"]
    )
    stamps = pd.to_datetime(pd.Series(lookup))

    st.sidebar.markdown(f"**{label}**")

    mode = st.sidebar.radio(
        "Mode",
        ["Single day", "Rolling"],
        key=f"{key}_mode",
        label_visibility="collapsed",
        horizontal=True,
    )

    if mode == "Rolling":
        position = st.sidebar.slider(
            "Scrub through origins",
            min_value=0,
            max_value=len(origins) - 1,
            value=len(origins) // 2,
            key=f"{key}_slider",
            label_visibility="collapsed",
        )
        chosen = int(origins[position])
    else:
        dates = sorted({ts.date() for ts in stamps})
        chosen_date = st.sidebar.date_input(
            "Day",
            value=dates[len(dates) // 2],
            min_value=dates[0],
            max_value=dates[-1],
            key=f"{key}_date",
            label_visibility="collapsed",
        )
        same_day = stamps[stamps.dt.date == chosen_date]
        if same_day.empty:
            target = pd.Timestamp(chosen_date)
            chosen = int((stamps - target).abs().idxmin())
        else:
            chosen = int(same_day.index[0])

    origin_ts = stamps.loc[chosen]
    horizon_start = origin_ts + pd.Timedelta(hours=1)
    horizon_end = origin_ts + pd.Timedelta(hours=config.HORIZON)
    if horizon_start.date() == horizon_end.date():
        window_text = f"Forecasting {horizon_start:%d %b %H:%M} – {horizon_end:%H:%M}"
    else:
        window_text = f"Forecasting {horizon_start:%d %b %H:%M} – {horizon_end:%d %b %H:%M}"

    st.sidebar.caption(f"{origin_ts:%d %b %Y, %H:%M}")
    st.sidebar.caption(window_text)
    return chosen


def station_rail(network, selected=None, key="rail", clickable=False):
    """Station list grouped by demand regime, with a status dot and peak kWh.

    Grouping is not decoration. A fleet depot and a tourist site fail in
    different ways, and an operator who knows that reads a weak forecast
    correctly instead of distrusting the system.

    clickable=False (default, Overview/Alerts/System): a static status list.
    clickable=True (Stations): rows are real buttons and this is the page's
    only station selector — returns the resolved station_id, which is
    `selected` unless a row was clicked this run, in which case it's the one
    that was.
    """
    st.sidebar.divider()
    st.sidebar.markdown("**Stations**")
    st.sidebar.markdown(
        "<div class='pd-rail-unit'>peak kWh, expected (P50)</div>",
        unsafe_allow_html=True,
    )

    by_group = {}
    for station in network["stations"]:
        by_group.setdefault(station.group, []).append(station)

    chosen = selected
    # Per-station CSS for the clickable (button) rows — a native button can't
    # hold a colour dot and a right-aligned value as separate child markup,
    # so those are injected as pseudo-elements on Streamlit's own
    # `.st-key-<key>` class (the documented per-widget class from key=...).
    button_rules = []

    for group in GROUP_ORDER + [g for g in by_group if g not in GROUP_ORDER]:
        members = by_group.get(group)
        if not members:
            continue
        st.sidebar.markdown(
            f"<div class='pd-rail-group'>{GROUP_LABELS.get(group, group)}</div>",
            unsafe_allow_html=True,
        )
        for station in sorted(members, key=lambda s: s.display_name):
            is_selected = station.station_id == selected
            colour = SEVERITY_COLOURS[station.worst_severity]

            if clickable:
                widget_key = f"{key}_{station.station_id}"
                name_colour = colour if station.worst_severity != "none" else INK
                selected_css = (
                    f".st-key-{widget_key} button {{ "
                    f"background:{tint(PRIMARY, 0.10)} !important; "
                    f"border-left:2px solid {PRIMARY} !important; "
                    f"padding-left:18px !important; font-weight:500 !important; }}"
                    if is_selected else ""
                )
                button_rules.append(f"""
                  .st-key-{widget_key} button {{ color:{name_colour} !important; }}
                  .st-key-{widget_key} button::before {{
                    content:''; position:absolute; left:8px; top:50%;
                    transform:translateY(-50%); width:7px; height:7px;
                    border-radius:50%; background:{colour};
                  }}
                  .st-key-{widget_key} button::after {{
                    content:'{station.peak["p50"]:,.0f}'; position:absolute;
                    right:8px; top:50%; transform:translateY(-50%);
                    color:{INK_MUTED}; font-size:0.72rem; font-weight:400;
                  }}
                  {selected_css}
                """)
                if st.sidebar.button(
                    station.display_name, key=widget_key, use_container_width=True,
                ):
                    chosen = station.station_id
            else:
                weight = "500" if is_selected else "400"
                row_class = "pd-rail-row selected" if is_selected else "pd-rail-row"
                name_colour = colour if station.worst_severity != "none" else INK
                st.sidebar.markdown(
                    f"<div class='{row_class}'>"
                    f"<span class='pd-dot' style='background:{colour}'></span>"
                    f"<span class='pd-rail-name' style='font-weight:{weight};color:{name_colour}'>"
                    f"{station.display_name}</span>"
                    f"<span class='pd-rail-value'>{station.peak['p50']:,.0f}</span></div>",
                    unsafe_allow_html=True,
                )

    if button_rules:
        st.sidebar.markdown(f"<style>{''.join(button_rules)}</style>", unsafe_allow_html=True)

    inactive = [
        row for row in loader.load_station_meta().itertuples() if not row.active
    ]
    if inactive:
        st.sidebar.markdown(
            "<div class='pd-rail-group'>no validation data</div>",
            unsafe_allow_html=True,
        )
        for row in inactive:
            st.sidebar.markdown(
                f"<div class='pd-rail-row' style='opacity:0.55'>"
                f"<span class='pd-dot' style='background:{NEUTRAL_ACCENT}'></span>"
                f"<span class='pd-rail-name'>{row.display_name}</span></div>",
                unsafe_allow_html=True,
            )

    return chosen


def capacity_note(source):
    if source == "rated":
        return "Thresholds use rated capacity."
    return (
        "Thresholds are derived from the upper tail of training demand, "
        "not a rated capacity."
    )


def capacity_note_short(source):
    """One-line variant for a tile note; capacity_note() stays full-length
    for captions and expanders, which already restate it in context."""
    return "rated capacity" if source == "rated" else "derived, not rated"


def replay_banner():
    st.sidebar.divider()
    st.sidebar.caption(
        "Replay — Jiaxing, Oct–Dec 2021. Historical data, not a live feed."
    )
