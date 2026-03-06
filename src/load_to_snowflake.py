"""
load_to_snowflake.py
Loads the Telco Customer Churn dataset into Snowflake with DTC-friendly column names,
and generates a synthetic A/B experiment table.

Usage:
    1. Create a .env file in the same directory (see .env.example):
        SNOWFLAKE_ACCOUNT=your-account-identifier
        SNOWFLAKE_USER=your-username
        SNOWFLAKE_PASSWORD=your-password

    2. Place the Kaggle CSV in the same directory as this script:
        WA_Fn-UseC_-Telco-Customer-Churn.csv

    3. Run:
        pip install python-dotenv snowflake-connector-python[pandas] pandas numpy
        python load_to_snowflake.py
"""

import os
import numpy as np
import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from dotenv import load_dotenv

# Load .env file from the same directory as this script
load_dotenv()

# ─────────────────────────────────────────────
# CONFIG — reads from .env file automatically
# ─────────────────────────────────────────────
CONFIG = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "database": "SUBSIGHT",
    "warehouse": "WH_XS_SUBSIGHT",
    "csv_path": "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
}

# Validate that credentials are set
_required = ["account", "user", "password"]
_missing = [k for k in _required if not CONFIG[k]]
if _missing:
    raise EnvironmentError(
        f"Missing required credentials: {', '.join(_missing)}. "
        f"Create a .env file with SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, and SNOWFLAKE_PASSWORD."
    )

# ─────────────────────────────────────────────
# COLUMN MAPPING — Telco names → DTC-friendly
# ─────────────────────────────────────────────
COLUMN_MAP = {
    "customerID": "CUSTOMER_ID",
    "gender": "GENDER",
    "SeniorCitizen": "IS_SENIOR",
    "Partner": "HAS_PARTNER",
    "Dependents": "HAS_DEPENDENTS",
    "tenure": "SUBSCRIPTION_MONTHS",
    "PhoneService": "HAS_PHONE_SERVICE",
    "MultipleLines": "HAS_MULTIPLE_LINES",
    "InternetService": "INTERNET_SERVICE_TYPE",
    "OnlineSecurity": "HAS_ONLINE_SECURITY",
    "OnlineBackup": "HAS_ONLINE_BACKUP",
    "DeviceProtection": "HAS_DEVICE_PROTECTION",
    "TechSupport": "HAS_TECH_SUPPORT",
    "StreamingTV": "HAS_STREAMING_TV",
    "StreamingMovies": "HAS_STREAMING_MOVIES",
    "Contract": "CONTRACT_TYPE",
    "PaperlessBilling": "HAS_PAPERLESS_BILLING",
    "PaymentMethod": "PAYMENT_METHOD",
    "MonthlyCharges": "MONTHLY_REVENUE",
    "TotalCharges": "TOTAL_REVENUE",
    "Churn": "CHURNED",
}


def get_connection(with_database=False):
    """Create a Snowflake connection."""
    params = {
        "account": CONFIG["account"],
        "user": CONFIG["user"],
        "password": CONFIG["password"],
    }
    if with_database:
        params["database"] = CONFIG["database"]
        params["warehouse"] = CONFIG["warehouse"]
    return snowflake.connector.connect(**params)


def setup_infrastructure(cur):
    """Create warehouse, database, and schemas."""
    print("Setting up Snowflake infrastructure...")

    cur.execute(f"""
        CREATE WAREHOUSE IF NOT EXISTS {CONFIG['warehouse']}
        WITH WAREHOUSE_SIZE = 'XSMALL'
        AUTO_SUSPEND = 60
        AUTO_RESUME = TRUE
    """)
    cur.execute(f"USE WAREHOUSE {CONFIG['warehouse']}")

    cur.execute(f"CREATE DATABASE IF NOT EXISTS {CONFIG['database']}")
    cur.execute(f"USE DATABASE {CONFIG['database']}")

    cur.execute("CREATE SCHEMA IF NOT EXISTS RAW")
    cur.execute("CREATE SCHEMA IF NOT EXISTS ANALYTICS")

    print("  Warehouse, database, and schemas ready.")


def load_customers(cur, conn):
    """Load and transform the Telco CSV, then upload to RAW.CUSTOMERS."""
    print(f"Loading CSV from: {CONFIG['csv_path']}")

    df = pd.read_csv(CONFIG["csv_path"])
    print(f"  Read {len(df):,} rows from CSV.")

    # Rename columns to DTC-friendly names
    df = df.rename(columns=COLUMN_MAP)

    # Clean TotalCharges → TOTAL_REVENUE (has blank strings for new customers)
    df["TOTAL_REVENUE"] = pd.to_numeric(df["TOTAL_REVENUE"], errors="coerce").fillna(0.0)

    # Convert binary Yes/No columns to 1/0 for cleaner modeling
    binary_cols = [
        "HAS_PARTNER", "HAS_DEPENDENTS", "HAS_PHONE_SERVICE",
        "HAS_PAPERLESS_BILLING", "CHURNED",
    ]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(df[col])

    # IS_SENIOR is already 0/1, ensure int
    df["IS_SENIOR"] = df["IS_SENIOR"].astype(int)

    # Ensure uppercase column names (Snowflake convention)
    df.columns = [c.upper() for c in df.columns]

    cur.execute("USE SCHEMA RAW")

    # Drop if exists for idempotent reruns
    cur.execute("DROP TABLE IF EXISTS CUSTOMERS")

    # Write to Snowflake
    success, num_chunks, num_rows, _ = write_pandas(
        conn, df, "CUSTOMERS", auto_create_table=True, quote_identifiers=False
    )
    print(f"  Loaded {num_rows:,} rows into RAW.CUSTOMERS ({num_chunks} chunk(s)).")
    return df


def generate_and_load_experiments(cur, conn, customers_df):
    """Generate synthetic A/B test data and upload to RAW.EXPERIMENTS."""
    print("Generating synthetic A/B experiment data...")

    np.random.seed(42)
    n = len(customers_df)

    # Experiment: "Annual Plan Discount Offer"
    # Control sees standard pricing, Treatment sees 20% annual discount
    # Realistic conversion rates: control ~12%, treatment ~16%
    group = np.random.choice(["control", "treatment"], size=n)
    converted = np.where(
        group == "control",
        np.random.binomial(1, 0.12, size=n),
        np.random.binomial(1, 0.16, size=n),
    )

    # Revenue per conversion (annual plan value)
    annual_plan_value = 79.99 * 12  # ~$960/year
    revenue_per_user = np.where(converted == 1, annual_plan_value, 0.0)

    experiments_df = pd.DataFrame({
        "EXPERIMENT_ID": "EXP_001_ANNUAL_DISCOUNT",
        "CUSTOMER_ID": customers_df["CUSTOMER_ID"].values,
        "VARIANT": group,
        "CONVERTED": converted,
        "REVENUE": np.round(revenue_per_user, 2),
        "CONTRACT_TYPE": customers_df["CONTRACT_TYPE"].values,
        "MONTHLY_REVENUE": customers_df["MONTHLY_REVENUE"].values,
        "SUBSCRIPTION_MONTHS": customers_df["SUBSCRIPTION_MONTHS"].values,
    })

    cur.execute("USE SCHEMA RAW")
    cur.execute("DROP TABLE IF EXISTS EXPERIMENTS")

    success, num_chunks, num_rows, _ = write_pandas(
        conn, experiments_df, "EXPERIMENTS", auto_create_table=True, quote_identifiers=False
    )
    print(f"  Loaded {num_rows:,} rows into RAW.EXPERIMENTS ({num_chunks} chunk(s)).")

    # Print a quick sanity check
    control_rate = experiments_df[experiments_df["VARIANT"] == "control"]["CONVERTED"].mean()
    treatment_rate = experiments_df[experiments_df["VARIANT"] == "treatment"]["CONVERTED"].mean()
    print(f"  Sanity check — Control conversion: {control_rate:.1%}, Treatment: {treatment_rate:.1%}")


def verify_tables(cur):
    """Run quick verification queries."""
    print("\nVerification:")

    cur.execute("USE SCHEMA RAW")

    cur.execute("SELECT COUNT(*) FROM CUSTOMERS")
    count = cur.fetchone()[0]
    print(f"  RAW.CUSTOMERS: {count:,} rows")

    cur.execute("SELECT COUNT(*) FROM EXPERIMENTS")
    count = cur.fetchone()[0]
    print(f"  RAW.EXPERIMENTS: {count:,} rows")

    cur.execute("""
        SELECT
            COUNT(*) AS total_customers,
            SUM(CHURNED) AS churned_count,
            ROUND(AVG(MONTHLY_REVENUE), 2) AS avg_monthly_rev,
            ROUND(AVG(SUBSCRIPTION_MONTHS), 1) AS avg_tenure_months
        FROM CUSTOMERS
    """)
    row = cur.fetchone()
    print(f"  Customer stats — Total: {row[0]:,}, Churned: {row[1]:,}, "
          f"Avg Monthly Rev: ${row[2]}, Avg Tenure: {row[3]} months")

    cur.execute("""
        SELECT
            VARIANT,
            COUNT(*) AS n,
            SUM(CONVERTED) AS conversions,
            ROUND(AVG(CONVERTED), 4) AS conversion_rate
        FROM EXPERIMENTS
        GROUP BY VARIANT
        ORDER BY VARIANT
    """)
    print("  Experiment breakdown:")
    for row in cur.fetchall():
        print(f"    {row[0]}: n={row[1]:,}, conversions={row[2]:,}, rate={row[3]:.2%}")


def main():
    print("=" * 50)
    print("SubSight — Snowflake Data Loader")
    print("=" * 50)

    # Step 1: Set up infra (no database context yet)
    conn = get_connection(with_database=False)
    cur = conn.cursor()
    setup_infrastructure(cur)
    cur.close()
    conn.close()

    # Step 2: Reconnect with database context for data loading
    conn = get_connection(with_database=True)
    cur = conn.cursor()
    cur.execute(f"USE WAREHOUSE {CONFIG['warehouse']}")

    # Step 3: Load customer data
    customers_df = load_customers(cur, conn)

    # Step 4: Generate and load experiment data
    generate_and_load_experiments(cur, conn, customers_df)

    # Step 5: Verify
    verify_tables(cur)

    cur.close()
    conn.close()

    print("\nDone! Your Snowflake tables are ready.")
    print(f"  Database: {CONFIG['database']}")
    print(f"  Tables:   RAW.CUSTOMERS, RAW.EXPERIMENTS")
    print(f"  Warehouse: {CONFIG['warehouse']}")


if __name__ == "__main__":
    main()