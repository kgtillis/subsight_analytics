# 📊 SubSight: DTC Subscription Analytics & Experimentation Platform

A Snowflake-connected Streamlit dashboard for DTC subscription businesses, featuring A/B test analysis, churn propensity scoring, and customer segmentation.

**[Live Demo →](https://ktillis-subsight-analytics.streamlit.app/)**

---

## Business Context

Subscription-based DTC companies live and die by three questions: *Is our experiment working? Who is about to churn? What do our customer segments look like?* SubSight answers all three from a single dashboard connected to a Snowflake data warehouse.

This project simulates a subscription analytics platform using the Telco Customer Churn dataset (7,043 customers), renamed and restructured to mirror DTC subscription data. All queries run against Snowflake in real time.

---

## Key Findings

### A/B Test: Annual Plan Discount Offer
The treatment group (20% annual plan discount) converted at approximately 16% vs. 12% for control — a statistically significant ~33% relative lift (p < 0.01). If rolled out, this translates to roughly 280 incremental annual plan conversions per 7,000 users, at ~$960 revenue per conversion. **Recommendation: roll out the discount offer and monitor for long-term retention effects.**

### Churn Propensity Model
A Gradient Boosting classifier achieved an AUC-ROC of ~0.85 and F1 of ~0.60 on the churn prediction task. SHAP analysis revealed that contract type, tenure, and monthly charges are the strongest churn drivers — month-to-month customers with short tenure and high monthly bills are the highest risk cohort. **Recommendation: target High-Risk customers with retention offers before their next billing cycle.**

### Customer Segmentation
K-means clustering identified distinct customer segments ranging from high-value long-tenure loyalists to at-risk new subscribers with low spend. The highest-churn segment shows 3x the churn rate of the most stable segment despite comparable monthly revenue. **Recommendation: invest in onboarding improvements for the first 6 months of the customer lifecycle, where churn risk concentrates.**

---

## Features

### 🧪 A/B Test Analyzer
- Statistical significance testing (two-proportion z-test)
- Conversion rate comparison with 95% confidence interval error bars
- Revenue impact estimation
- Segment-level drill-down (by contract type, tenure, revenue tier)
- Sample size calculator for planning future experiments

### 🎯 Churn Propensity Scoring
- Gradient Boosting classifier with cross-validated evaluation
- ROC curve and performance metrics (AUC, F1, Precision, Recall)
- SHAP-based global feature importance
- Per-customer SHAP waterfall explanations
- Scored customer table with risk tiers (High / Medium / Low)

### 👥 Customer Segmentation
- K-means clustering with elbow and silhouette analysis
- Interactive cluster count selection
- PCA 2D projection with variance explained
- Auto-generated segment names based on cluster characteristics
- Segment-level KPI comparison (revenue, tenure, churn rate)
- Customer-level drill-down per segment
- Write-back to Snowflake (`ANALYTICS.SEGMENTS`)

---

## Architecture

```
┌─────────────────────────────────────────-─────────┐
│                 Streamlit Cloud                   │
│  ┌────────────┬──────────────┬─────────────────┐  │
│  │  A/B Test  │   Churn      │   Customer      │  │
│  │  Analyzer  │   Propensity │   Segments      │  │
│  └─────┬──────┴──────┬───────┴───────┬─────────┘  │
│        └─────────────┼───────────────┘            │
│            snowflake-connector-python             │
└──────────────────────┼────────────────────────────┘
                       │
               ┌───────┴────────┐
               │   Snowflake    │
               │                │
               │  RAW.CUSTOMERS │
               │ RAW.EXPERIMENTS│
               │   ANALYTICS.   │
               │    SEGMENTS    │
               └────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Warehouse | Snowflake (Enterprise) |
| App Framework | Streamlit |
| ML | Scikit-learn, SHAP |
| Visualization | Plotly |
| Language | Python |
| Deployment | Streamlit Cloud |

---

## Setup

### Prerequisites
- Python 3.9+
- Snowflake account (free trial works)

### Installation

```bash
git clone https://github.com/kgtillis/subsight_analytics.git
cd subsight_analytics
pip install -r src/requirements.txt
```

### Configure Credentials

Copy the environment template and fill in your Snowflake credentials:

```bash
cp src/.env.example src/.env
```

For Streamlit, also create the secrets file:

```bash
cp src/.streamlit/secrets.toml.example src/.streamlit/secrets.toml
```

### Load Data into Snowflake

```bash
cd src
python load_to_snowflake.py
```

This creates the `SUBSIGHT` database with `RAW.CUSTOMERS` and `RAW.EXPERIMENTS` tables.

### Run Locally

```bash
cd src
streamlit run app.py
```

---

## Project Structure

```
subsight_analytics/
├── src/
│   ├── .streamlit/
│   │   └── secrets.toml.example
│   ├── pages/
│   │   ├── 1_AB_Test_Analyzer.py
│   │   ├── 2_Churn_Propensity.py
│   │   └── 3_Customer_Segments.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── ab_stats.py
│   │   ├── kpi_bar.py
│   │   └── snowflake_conn.py
│   ├── .env.example
│   ├── app.py
│   ├── load_to_snowflake.py
│   └── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

Uses the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset from Kaggle (7,043 rows), with columns renamed to DTC subscription terminology. A synthetic A/B experiment table is generated during data loading to simulate a pricing experiment.

---

## Future Improvements

- Version segmentation results with timestamps for longitudinal tracking
- Add LTV estimation using survival analysis
- Integrate marketing attribution modeling
- Automated experiment monitoring with sequential testing
- Real-time data pipeline with Snowflake Streams and Tasks