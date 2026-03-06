"""
app.py
SubSight: DTC Subscription Analytics & Experimentation Platform
Main landing page with project overview and navigation.
"""

import streamlit as st
from utils.kpi_bar import render_kpi_bar

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SubSight Analytics",
    page_icon="📊",
    layout="wide",
)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("📊 SubSight")
st.subheader("DTC Subscription Analytics & Experimentation Platform")

# ─────────────────────────────────────────────
# KPI bar
# ─────────────────────────────────────────────
render_kpi_bar()

# ─────────────────────────────────────────────
# Project overview
# ─────────────────────────────────────────────
st.markdown("""
SubSight is an analytics platform built for DTC subscription businesses. 
It connects directly to a **Snowflake** data warehouse and provides three core capabilities:
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🧪 A/B Test Analyzer")
    st.markdown(
        "Evaluate experiment results with statistical rigor. "
        "Compare conversion rates between control and treatment groups, "
        "calculate statistical significance, and estimate the revenue impact of rolling out a variant."
    )

with col2:
    st.markdown("### 🎯 Churn Propensity")
    st.markdown(
        "Score every customer by their likelihood to churn using a Gradient Boosting classifier. "
        "Explore SHAP-based feature importance to understand what drives churn, "
        "and identify high-risk customers for proactive retention outreach."
    )

with col3:
    st.markdown("### 👥 Customer Segments")
    st.markdown(
        "Discover natural customer groups using K-means clustering. "
        "Visualize segments in a 2D projection, compare segment-level KPIs, "
        "and drill into individual customer profiles within each segment."
    )

# ─────────────────────────────────────────────
# Tech stack
# ─────────────────────────────────────────────
st.divider()
st.markdown("#### Tech Stack")
st.markdown(
    "**Data Warehouse:** Snowflake &nbsp;|&nbsp; "
    "**App Framework:** Streamlit &nbsp;|&nbsp; "
    "**ML:** Scikit-learn, SHAP &nbsp;|&nbsp; "
    "**Visualization:** Plotly &nbsp;|&nbsp; "
    "**Language:** Python"
)

# ─────────────────────────────────────────────
# Navigation hint
# ─────────────────────────────────────────────
st.info("👈 Use the sidebar to navigate between the three analytics modules.", icon="ℹ️")