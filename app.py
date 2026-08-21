"""
app.py

Router only. Streamlit's st.navigation manages the page list and renders the
switcher — position="top" puts it in the header instead of the sidebar, so
our own sidebar (origin control, station rail, dark-mode toggle) stays free
for the utility controls it was already carrying. Actual page content lives
in pages/1_Overview.py through pages/5_About.py; each is a plain script
st.Page() points to, executed by pg.run() below.

st.set_page_config() can only be called once per app, so it lives here now —
individual pages no longer call it. That means the browser tab title is one
fixed string for the whole app rather than changing per page; a small,
accepted trade-off for centralized navigation.

    streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="PhysDistill-EV Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

overview = st.Page("pages/1_Overview.py", title="Overview", default=True)
stations = st.Page("pages/2_Stations.py", title="Stations")
alerts = st.Page("pages/3_Alerts.py", title="Alerts")
system = st.Page("pages/4_System.py", title="System")
about = st.Page("pages/5_About.py", title="About")

pg = st.navigation([overview, stations, alerts, system, about], position="top")
pg.run()
