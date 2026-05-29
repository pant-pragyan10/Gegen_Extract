import streamlit as st
from pathlib import Path

st.set_page_config(page_title="GegenExtract Dashboard")

st.title("GegenExtract — Observability Dashboard")

st.sidebar.header("Run Controls")
if st.sidebar.button("Refresh"):
    st.experimental_rerun()

st.header("Recent Experiments")
st.write("This dashboard will show experiment trajectories, prompt versions, and LLM call logs.")

st.header("Optimization Trajectory")
st.write("Placeholder for optimizer trajectory visualizations (to be implemented).")
