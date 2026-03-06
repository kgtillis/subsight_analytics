"""
utils/snowflake_conn.py
Snowflake connection helper for the Streamlit app.

Reads credentials from:
  - Streamlit secrets (st.secrets) when deployed on Streamlit Cloud
  - .env file when running locally
"""

import os
import streamlit as st
import snowflake.connector
import pandas as pd


def _get_credentials():
    """Pull Snowflake credentials from Streamlit secrets or environment."""
    try:
        # Streamlit Cloud: reads from .streamlit/secrets.toml or Streamlit Cloud secrets UI
        return {
            "account": st.secrets["snowflake"]["account"],
            "user": st.secrets["snowflake"]["user"],
            "password": st.secrets["snowflake"]["password"],
            "database": st.secrets["snowflake"]["database"],
            "warehouse": st.secrets["snowflake"]["warehouse"],
        }
    except (KeyError, FileNotFoundError):
        # Local dev: fall back to environment variables (.env loaded by dotenv)
        from dotenv import load_dotenv
        load_dotenv()
        return {
            "account": os.getenv("SNOWFLAKE_ACCOUNT"),
            "user": os.getenv("SNOWFLAKE_USER"),
            "password": os.getenv("SNOWFLAKE_PASSWORD"),
            "database": "SUBSIGHT",
            "warehouse": "WH_XS_SUBSIGHT",
        }


def get_connection():
    """Create a new Snowflake connection."""
    creds = _get_credentials()
    return snowflake.connector.connect(**creds)


@st.cache_data(ttl=600, show_spinner=False)
def run_query(query: str, params: tuple = None) -> pd.DataFrame:
    """
    Execute a SQL query and return results as a DataFrame.
    Results are cached for 10 minutes to avoid repeated Snowflake calls.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        if params:
            cur.execute(query, params)
        else:
            cur.execute(query)
        columns = [desc[0] for desc in cur.description]
        data = cur.fetchall()
        return pd.DataFrame(data, columns=columns)
    finally:
        conn.close()