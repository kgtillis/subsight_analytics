"""
pages/3_Customer_Segments.py
Customer segmentation dashboard using K-means clustering,
PCA visualization, and segment-level KPI comparison.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from utils.snowflake_conn import run_query
from utils.kpi_bar import render_kpi_bar

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(page_title="Customer Segments | SubSight", page_icon="👥", layout="wide")
st.title("👥 Customer Segmentation")
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
            INTERNET_SERVICE_TYPE,
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
# Feature preparation
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Preparing features...")
def prepare_clustering_features(df):
    """Encode and scale features for clustering."""
    feature_df = df.copy()

    categorical_cols = [
        "GENDER", "INTERNET_SERVICE_TYPE", "CONTRACT_TYPE", "PAYMENT_METHOD",
    ]
    for col in categorical_cols:
        le = LabelEncoder()
        feature_df[col] = le.fit_transform(feature_df[col].astype(str))

    cluster_features = [
        "SUBSCRIPTION_MONTHS", "MONTHLY_REVENUE", "TOTAL_REVENUE",
        "IS_SENIOR", "HAS_PARTNER", "HAS_DEPENDENTS",
        "CONTRACT_TYPE", "HAS_PAPERLESS_BILLING", "INTERNET_SERVICE_TYPE",
    ]

    X = feature_df[cluster_features].astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, cluster_features

X_scaled, cluster_features = prepare_clustering_features(raw_df)

# ─────────────────────────────────────────────
# Elbow + silhouette analysis
# ─────────────────────────────────────────────
st.markdown("### Optimal Cluster Selection")

@st.cache_data(show_spinner="Evaluating cluster counts...")
def evaluate_clusters(_X_scaled):
    """Compute inertia and silhouette scores for k=2 through k=8."""
    k_range = range(2, 9)
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(_X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(_X_scaled, labels))

    return list(k_range), inertias, silhouettes

k_range, inertias, silhouettes = evaluate_clusters(X_scaled)

elbow_col, sil_col = st.columns(2)

with elbow_col:
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(
        x=k_range, y=inertias, mode="lines+markers",
        marker=dict(color="#2563EB", size=8),
        line=dict(color="#2563EB", width=2),
    ))
    fig_elbow.update_layout(
        title="Elbow Method",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Inertia",
        height=350,
        plot_bgcolor="white",
        yaxis=dict(gridcolor="#E5E7EB"),
    )
    st.plotly_chart(fig_elbow, width="stretch")

with sil_col:
    fig_sil = go.Figure()
    fig_sil.add_trace(go.Bar(
        x=k_range, y=silhouettes,
        marker_color=["#2563EB" if s == max(silhouettes) else "#93C5FD" for s in silhouettes],
    ))
    fig_sil.update_layout(
        title="Silhouette Score",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Silhouette Score",
        height=350,
        plot_bgcolor="white",
        yaxis=dict(gridcolor="#E5E7EB"),
    )
    st.plotly_chart(fig_sil, width="stretch")

# ─────────────────────────────────────────────
# Cluster selection and fitting
# ─────────────────────────────────────────────
best_k = k_range[np.argmax(silhouettes)]

selected_k = st.slider(
    "Select number of clusters",
    min_value=2, max_value=8, value=best_k,
    help=f"Best silhouette score at k={best_k}",
)

@st.cache_data(show_spinner="Clustering customers...")
def fit_clusters(_X_scaled, k):
    """Fit K-means and return labels."""
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(_X_scaled)
    return labels

cluster_labels = fit_clusters(X_scaled, selected_k)

# Add labels to raw dataframe
segmented_df = raw_df.copy()
segmented_df["CLUSTER"] = cluster_labels

# ─────────────────────────────────────────────
# Auto-generate segment names
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def name_segments(df):
    """Generate descriptive segment names based on cluster characteristics."""
    names = {}
    overall_revenue = df["MONTHLY_REVENUE"].mean()
    overall_tenure = df["SUBSCRIPTION_MONTHS"].mean()
    overall_churn = df["CHURNED"].mean()

    for cluster_id in sorted(df["CLUSTER"].unique()):
        subset = df[df["CLUSTER"] == cluster_id]
        avg_rev = subset["MONTHLY_REVENUE"].mean()
        avg_tenure = subset["SUBSCRIPTION_MONTHS"].mean()
        churn_rate = subset["CHURNED"].mean()

        # Build name from characteristics
        rev_label = "High-Value" if avg_rev > overall_revenue * 1.15 else (
            "Low-Spend" if avg_rev < overall_revenue * 0.85 else "Mid-Tier"
        )
        tenure_label = "Loyalists" if avg_tenure > overall_tenure * 1.3 else (
            "New" if avg_tenure < overall_tenure * 0.5 else "Established"
        )
        risk_label = " (At-Risk)" if churn_rate > overall_churn * 1.3 else ""

        names[cluster_id] = f"{rev_label} {tenure_label}{risk_label}"

    return names

segment_names = name_segments(segmented_df)
segmented_df["SEGMENT_NAME"] = segmented_df["CLUSTER"].map(segment_names)

# ─────────────────────────────────────────────
# PCA 2D visualization
# ─────────────────────────────────────────────
st.markdown("### Segment Visualization")

@st.cache_data(show_spinner="Computing PCA projection...")
def compute_pca(_X_scaled, labels, names_map):
    """Project to 2D via PCA for visualization."""
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(_X_scaled)
    pca_df = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "Cluster": labels,
        "Segment": [names_map[c] for c in labels],
    })
    variance = pca.explained_variance_ratio_
    return pca_df, variance

pca_df, variance = compute_pca(X_scaled, cluster_labels, segment_names)

color_sequence = px.colors.qualitative.Set2[:selected_k]

fig_pca = px.scatter(
    pca_df,
    x="PC1", y="PC2",
    color="Segment",
    color_discrete_sequence=color_sequence,
    opacity=0.6,
    hover_data={"Cluster": True},
)
fig_pca.update_layout(
    title=f"Customer Segments (PCA Projection — {variance[0]:.1%} + {variance[1]:.1%} variance explained)",
    xaxis_title=f"PC1 ({variance[0]:.1%})",
    yaxis_title=f"PC2 ({variance[1]:.1%})",
    height=500,
    plot_bgcolor="white",
    xaxis=dict(gridcolor="#E5E7EB"),
    yaxis=dict(gridcolor="#E5E7EB"),
)
fig_pca.update_traces(marker=dict(size=5))
st.plotly_chart(fig_pca, width="stretch")

# ─────────────────────────────────────────────
# Segment summary table
# ─────────────────────────────────────────────
st.markdown("### Segment Profiles")

segment_summary = (
    segmented_df.groupby("SEGMENT_NAME")
    .agg(
        CUSTOMERS=("CUSTOMER_ID", "count"),
        AVG_MONTHLY_REVENUE=("MONTHLY_REVENUE", "mean"),
        AVG_TENURE_MONTHS=("SUBSCRIPTION_MONTHS", "mean"),
        CHURN_RATE=("CHURNED", "mean"),
        TOTAL_REVENUE=("TOTAL_REVENUE", "sum"),
    )
    .reset_index()
    .sort_values("CUSTOMERS", ascending=False)
)

segment_summary["AVG_MONTHLY_REVENUE"] = segment_summary["AVG_MONTHLY_REVENUE"].round(2)
segment_summary["AVG_TENURE_MONTHS"] = segment_summary["AVG_TENURE_MONTHS"].round(1)
segment_summary["CHURN_RATE"] = segment_summary["CHURN_RATE"].round(3)
segment_summary["TOTAL_REVENUE"] = segment_summary["TOTAL_REVENUE"].round(0)

st.dataframe(
    segment_summary,
    width="stretch",
    hide_index=True,
    column_config={
        "SEGMENT_NAME": "Segment",
        "CUSTOMERS": st.column_config.NumberColumn("Customers", format="%d"),
        "AVG_MONTHLY_REVENUE": st.column_config.NumberColumn("Avg Monthly Rev", format="$%.2f"),
        "AVG_TENURE_MONTHS": st.column_config.NumberColumn("Avg Tenure (mo)", format="%.1f"),
        "CHURN_RATE": st.column_config.ProgressColumn("Churn Rate", min_value=0, max_value=1, format="%.1%%"),
        "TOTAL_REVENUE": st.column_config.NumberColumn("Total Revenue", format="$%,.0f"),
    },
)

# ─────────────────────────────────────────────
# Segment comparison charts
# ─────────────────────────────────────────────
st.markdown("### Segment Comparison")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig_rev = px.bar(
        segment_summary, x="SEGMENT_NAME", y="AVG_MONTHLY_REVENUE",
        color="SEGMENT_NAME", color_discrete_sequence=color_sequence,
        text_auto=".2f",
    )
    fig_rev.update_layout(
        title="Avg Monthly Revenue by Segment",
        xaxis_title="", yaxis_title="Monthly Revenue ($)",
        showlegend=False, height=350,
        plot_bgcolor="white", yaxis=dict(gridcolor="#E5E7EB"),
    )
    st.plotly_chart(fig_rev, width="stretch")

with chart_col2:
    fig_churn = px.bar(
        segment_summary, x="SEGMENT_NAME", y="CHURN_RATE",
        color="SEGMENT_NAME", color_discrete_sequence=color_sequence,
        text_auto=".1%",
    )
    fig_churn.update_layout(
        title="Churn Rate by Segment",
        xaxis_title="", yaxis_title="Churn Rate",
        yaxis_tickformat=".0%",
        showlegend=False, height=350,
        plot_bgcolor="white", yaxis=dict(gridcolor="#E5E7EB"),
    )
    st.plotly_chart(fig_churn, width="stretch")

# ─────────────────────────────────────────────
# Segment drill-down
# ─────────────────────────────────────────────
st.markdown("### Segment Drill-Down")

selected_segment = st.selectbox(
    "Select a segment to explore",
    sorted(segmented_df["SEGMENT_NAME"].unique()),
)

if selected_segment:
    segment_subset = segmented_df[segmented_df["SEGMENT_NAME"] == selected_segment]

    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    with info_col1:
        st.metric("Customers", f"{len(segment_subset):,}")
    with info_col2:
        st.metric("Avg Monthly Revenue", f"${segment_subset['MONTHLY_REVENUE'].mean():,.2f}")
    with info_col3:
        st.metric("Avg Tenure", f"{segment_subset['SUBSCRIPTION_MONTHS'].mean():.1f} months")
    with info_col4:
        st.metric("Churn Rate", f"{segment_subset['CHURNED'].mean():.1%}")

    # Show customer-level detail
    with st.expander(f"View all {len(segment_subset):,} customers in this segment"):
        display_cols = [
            "CUSTOMER_ID", "CONTRACT_TYPE", "MONTHLY_REVENUE",
            "SUBSCRIPTION_MONTHS", "TOTAL_REVENUE", "CHURNED",
        ]
        st.dataframe(
            segment_subset[display_cols].sort_values("MONTHLY_REVENUE", ascending=False),
            width="stretch",
            hide_index=True,
        )

# ─────────────────────────────────────────────
# Write segments back to Snowflake
# ─────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### 💾 Save Segments to Snowflake")
st.sidebar.markdown(
    "Write the current segmentation results to `ANALYTICS.SEGMENTS` "
    "for use by other teams and downstream models."
)

if st.sidebar.button("Save Segments"):
    try:
        from snowflake.connector.pandas_tools import write_pandas
        from utils.snowflake_conn import get_connection

        save_df = segmented_df[["CUSTOMER_ID", "CLUSTER", "SEGMENT_NAME"]].copy()
        save_df.columns = ["CUSTOMER_ID", "CLUSTER_ID", "SEGMENT_NAME"]

        conn = get_connection()
        cur = conn.cursor()
        cur.execute("USE SCHEMA ANALYTICS")
        cur.execute("DROP TABLE IF EXISTS SEGMENTS")

        success, num_chunks, num_rows, _ = write_pandas(
            conn, save_df, "SEGMENTS", auto_create_table=True, quote_identifiers=False
        )
        conn.close()

        st.sidebar.success(f"Saved {num_rows:,} rows to ANALYTICS.SEGMENTS")
    except Exception as e:
        st.sidebar.error(f"Error saving: {e}")