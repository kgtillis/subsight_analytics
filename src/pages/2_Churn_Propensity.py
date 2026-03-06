"""
pages/2_Churn_Propensity.py
Churn propensity scoring dashboard with Gradient Boosting classifier,
SHAP feature importance, and individual customer risk profiles.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, roc_curve
)
from sklearn.preprocessing import LabelEncoder
import shap
from utils.snowflake_conn import run_query
from utils.kpi_bar import render_kpi_bar

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(page_title="Churn Propensity | SubSight", page_icon="🎯", layout="wide")
st.title("🎯 Churn Propensity Scoring")
render_kpi_bar()

# ─────────────────────────────────────────────
# Load data from Snowflake
# ─────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner="Loading customer data...")
def load_customers():
    return run_query("""
        SELECT
            CUSTOMER_ID,
            GENDER,
            IS_SENIOR,
            HAS_PARTNER,
            HAS_DEPENDENTS,
            SUBSCRIPTION_MONTHS,
            HAS_PHONE_SERVICE,
            HAS_MULTIPLE_LINES,
            INTERNET_SERVICE_TYPE,
            HAS_ONLINE_SECURITY,
            HAS_ONLINE_BACKUP,
            HAS_DEVICE_PROTECTION,
            HAS_TECH_SUPPORT,
            HAS_STREAMING_TV,
            HAS_STREAMING_MOVIES,
            CONTRACT_TYPE,
            HAS_PAPERLESS_BILLING,
            PAYMENT_METHOD,
            MONTHLY_REVENUE,
            TOTAL_REVENUE,
            CHURNED
        FROM RAW.CUSTOMERS
    """)

raw_df = load_customers()

# ─────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Preparing features...")
def prepare_features(df):
    """Encode categoricals and prepare feature matrix."""
    feature_df = df.copy()

    # Columns to label-encode
    categorical_cols = [
        "GENDER", "HAS_MULTIPLE_LINES", "INTERNET_SERVICE_TYPE",
        "HAS_ONLINE_SECURITY", "HAS_ONLINE_BACKUP", "HAS_DEVICE_PROTECTION",
        "HAS_TECH_SUPPORT", "HAS_STREAMING_TV", "HAS_STREAMING_MOVIES",
        "CONTRACT_TYPE", "PAYMENT_METHOD",
    ]

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        feature_df[col] = le.fit_transform(feature_df[col].astype(str))
        encoders[col] = le

    # Feature columns (exclude ID and target)
    feature_cols = [c for c in feature_df.columns if c not in ["CUSTOMER_ID", "CHURNED"]]

    X = feature_df[feature_cols].astype(float)
    y = feature_df["CHURNED"].astype(int)

    return X, y, feature_cols, encoders

X, y, feature_cols, encoders = prepare_features(raw_df)

# ─────────────────────────────────────────────
# Train model
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Training churn model...")
def train_model(_X, _y):
    """Train a Gradient Boosting classifier and return model + evaluation metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        _X, _y, test_size=0.2, random_state=42, stratify=_y
    )

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Cross-validated F1
    cv_f1 = cross_val_score(model, _X, _y, cv=5, scoring="f1").mean()

    # Metrics
    metrics = {
        "auc_roc": roc_auc_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "cv_f1": cv_f1,
    }

    # ROC curve data
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    return model, metrics, fpr, tpr, X_test, y_test

model, metrics, fpr, tpr, X_test, y_test = train_model(X, y)

# ─────────────────────────────────────────────
# Model performance
# ─────────────────────────────────────────────
st.markdown("### Model Performance")

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("AUC-ROC", f"{metrics['auc_roc']:.3f}")
with m2:
    st.metric("F1 Score", f"{metrics['f1']:.3f}")
with m3:
    st.metric("Precision", f"{metrics['precision']:.3f}")
with m4:
    st.metric("Recall", f"{metrics['recall']:.3f}")

st.caption(f"5-fold cross-validated F1: {metrics['cv_f1']:.3f}")

# ROC Curve
fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(
    x=fpr, y=tpr,
    mode="lines",
    name=f"Model (AUC = {metrics['auc_roc']:.3f})",
    line=dict(color="#2563EB", width=2),
))
fig_roc.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode="lines",
    name="Random Baseline",
    line=dict(color="#9CA3AF", width=1, dash="dash"),
))
fig_roc.update_layout(
    title="ROC Curve",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    height=400,
    plot_bgcolor="white",
    yaxis=dict(gridcolor="#E5E7EB"),
    xaxis=dict(gridcolor="#E5E7EB"),
)
st.plotly_chart(fig_roc, width="stretch")

# ─────────────────────────────────────────────
# SHAP feature importance
# ─────────────────────────────────────────────
st.markdown("### Feature Importance (SHAP)")

@st.cache_data(show_spinner="Computing SHAP values...")
def compute_shap(_model, _X):
    """Compute SHAP values using TreeExplainer."""
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(_X)
    return shap_values

shap_values = compute_shap(model, X)

# Mean absolute SHAP values for global importance
mean_shap = pd.DataFrame({
    "Feature": feature_cols,
    "Mean |SHAP|": np.abs(shap_values).mean(axis=0),
}).sort_values("Mean |SHAP|", ascending=True)

fig_shap = go.Figure()
fig_shap.add_trace(go.Bar(
    x=mean_shap["Mean |SHAP|"],
    y=mean_shap["Feature"],
    orientation="h",
    marker_color="#2563EB",
))
fig_shap.update_layout(
    title="Top Features Driving Churn",
    xaxis_title="Mean |SHAP Value|",
    height=500,
    margin=dict(l=200),
    plot_bgcolor="white",
    xaxis=dict(gridcolor="#E5E7EB"),
)
st.plotly_chart(fig_shap, width="stretch")

# ─────────────────────────────────────────────
# Scored customer table
# ─────────────────────────────────────────────
st.markdown("### Scored Customer List")

@st.cache_data(show_spinner="Scoring all customers...")
def score_customers(_model, _X, _raw_df):
    """Score every customer and assign risk tiers."""
    probas = _model.predict_proba(_X)[:, 1]
    scored = _raw_df[["CUSTOMER_ID", "CONTRACT_TYPE", "MONTHLY_REVENUE",
                       "SUBSCRIPTION_MONTHS", "CHURNED"]].copy()
    scored["CHURN_PROBABILITY"] = np.round(probas, 4)
    scored["RISK_TIER"] = pd.cut(
        probas,
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"],
    )
    return scored.sort_values("CHURN_PROBABILITY", ascending=False)

scored_df = score_customers(model, X, raw_df)

# Risk tier summary
st.markdown("#### Risk Distribution")
tier_col1, tier_col2, tier_col3 = st.columns(3)

tier_counts = scored_df["RISK_TIER"].value_counts()
for col, tier, color in zip(
    [tier_col1, tier_col2, tier_col3],
    ["High", "Medium", "Low"],
    ["🔴", "🟡", "🟢"],
):
    count = tier_counts.get(tier, 0)
    pct = count / len(scored_df) * 100
    with col:
        st.metric(f"{color} {tier} Risk", f"{count:,}", help=f"{pct:.1f}% of customers")

# Filter controls
filter_col1, filter_col2 = st.columns(2)
with filter_col1:
    selected_tiers = st.multiselect(
        "Filter by Risk Tier",
        options=["High", "Medium", "Low"],
        default=["High", "Medium", "Low"],
    )
with filter_col2:
    sort_order = st.selectbox("Sort by", ["Churn Probability (High to Low)", "Churn Probability (Low to High)"])

filtered_df = scored_df[scored_df["RISK_TIER"].isin(selected_tiers)]
if "Low to High" in sort_order:
    filtered_df = filtered_df.sort_values("CHURN_PROBABILITY", ascending=True)

st.dataframe(
    filtered_df,
    width="stretch",
    hide_index=True,
    column_config={
        "CHURN_PROBABILITY": st.column_config.ProgressColumn(
            "Churn Probability",
            min_value=0, max_value=1, format="%.2f",
        ),
    },
)

# ─────────────────────────────────────────────
# Individual customer SHAP waterfall
# ─────────────────────────────────────────────
st.markdown("### Individual Customer Analysis")

customer_ids = sorted(scored_df["CUSTOMER_ID"].tolist())
selected_customer = st.selectbox("Select a customer to explain", customer_ids)

if selected_customer:
    idx = raw_df[raw_df["CUSTOMER_ID"] == selected_customer].index[0]
    customer_shap = shap_values[idx]
    customer_features = X.iloc[idx]

    # Get the customer's info
    customer_info = scored_df[scored_df["CUSTOMER_ID"] == selected_customer].iloc[0]
    st.markdown(
        f"**{selected_customer}** — "
        f"Churn Probability: **{customer_info['CHURN_PROBABILITY']:.2%}** — "
        f"Risk Tier: **{customer_info['RISK_TIER']}** — "
        f"Contract: **{customer_info['CONTRACT_TYPE']}** — "
        f"Monthly Revenue: **${customer_info['MONTHLY_REVENUE']:,.2f}**"
    )

    # Build waterfall data
    waterfall_df = pd.DataFrame({
        "Feature": feature_cols,
        "SHAP Value": customer_shap,
        "Feature Value": customer_features.values,
    }).sort_values("SHAP Value", key=abs, ascending=True)

    # Show top 10 features
    top_n = waterfall_df.tail(10)

    fig_waterfall = go.Figure()
    fig_waterfall.add_trace(go.Bar(
        x=top_n["SHAP Value"],
        y=top_n["Feature"],
        orientation="h",
        marker_color=top_n["SHAP Value"].apply(
            lambda v: "#EF4444" if v > 0 else "#22C55E"
        ),
        text=top_n["SHAP Value"].apply(lambda v: f"{v:+.3f}"),
        textposition="outside",
    ))
    fig_waterfall.update_layout(
        title=f"What's Driving This Customer's Churn Score",
        xaxis_title="SHAP Value (→ increases churn risk, ← decreases)",
        height=400,
        margin=dict(l=200),
        plot_bgcolor="white",
        xaxis=dict(gridcolor="#E5E7EB", zeroline=True, zerolinecolor="#333"),
    )
    st.plotly_chart(fig_waterfall, width="stretch")

    st.caption(
        "Red bars push the prediction toward churn. "
        "Green bars push it away from churn. "
        "The length of each bar shows how much that feature influenced this specific customer's score."
    )