"""
utils/kpi_bar.py
Reusable KPI header bar displayed at the top of every page.
Shows core subscription business metrics pulled from Snowflake.
"""

import streamlit as st
from utils.snowflake_conn import run_query


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_kpi_data():
    """Fetch aggregate KPIs from the customers table."""
    df = run_query("""
        SELECT
            ROUND(SUM(MONTHLY_REVENUE), 2)              AS TOTAL_MRR,
            ROUND(AVG(MONTHLY_REVENUE), 2)               AS AVG_REVENUE_PER_USER,
            ROUND(AVG(SUBSCRIPTION_MONTHS), 1)           AS AVG_TENURE_MONTHS,
            COUNT(*)                                      AS TOTAL_CUSTOMERS,
            SUM(CHURNED)                                  AS CHURNED_COUNT,
            ROUND(SUM(CHURNED) / COUNT(*) * 100, 1)      AS CHURN_RATE_PCT
        FROM RAW.CUSTOMERS
    """)
    return df.iloc[0]


def render_kpi_bar():
    """Render the KPI metric cards across the top of the page."""
    kpis = _fetch_kpi_data()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Monthly Recurring Revenue",
            value=f"${kpis['TOTAL_MRR']:,.0f}",
        )
    with col2:
        st.metric(
            label="Churn Rate",
            value=f"{kpis['CHURN_RATE_PCT']}%",
        )
    with col3:
        st.metric(
            label="Avg Tenure",
            value=f"{kpis['AVG_TENURE_MONTHS']} months",
        )
    with col4:
        st.metric(
            label="Avg Revenue / User",
            value=f"${kpis['AVG_REVENUE_PER_USER']:,.2f}",
        )

    st.divider()