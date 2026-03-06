"""
pages/1_AB_Test_Analyzer.py
A/B Test analysis dashboard with statistical significance testing,
confidence intervals, and sample size calculator.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.snowflake_conn import run_query
from utils.kpi_bar import render_kpi_bar
from utils.ab_stats import two_proportion_ztest, required_sample_size

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(page_title="A/B Test Analyzer | SubSight", page_icon="🧪", layout="wide")
st.title("🧪 A/B Test Analyzer")
render_kpi_bar()

# ─────────────────────────────────────────────
# Load experiment data from Snowflake
# ─────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner="Loading experiment data...")
def load_experiment_data():
    return run_query("""
        SELECT
            EXPERIMENT_ID,
            VARIANT,
            CONVERTED,
            REVENUE,
            CONTRACT_TYPE,
            MONTHLY_REVENUE,
            SUBSCRIPTION_MONTHS
        FROM RAW.EXPERIMENTS
    """)

df = load_experiment_data()

# ─────────────────────────────────────────────
# Experiment selector
# ─────────────────────────────────────────────
experiments = df["EXPERIMENT_ID"].unique().tolist()
selected_exp = st.selectbox("Select Experiment", experiments)
exp_df = df[df["EXPERIMENT_ID"] == selected_exp]

# ─────────────────────────────────────────────
# Split into control / treatment
# ─────────────────────────────────────────────
control = exp_df[exp_df["VARIANT"] == "control"]
treatment = exp_df[exp_df["VARIANT"] == "treatment"]

n_control = len(control)
n_treatment = len(treatment)
conv_control = int(control["CONVERTED"].sum())
conv_treatment = int(treatment["CONVERTED"].sum())

# ─────────────────────────────────────────────
# Run statistical test
# ─────────────────────────────────────────────
results = two_proportion_ztest(conv_control, n_control, conv_treatment, n_treatment)

# ─────────────────────────────────────────────
# Results header
# ─────────────────────────────────────────────
st.markdown("### Experiment Results")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Control Conversion Rate", f"{results['rate_a']:.2%}", help=f"n = {n_control:,}")
with col2:
    st.metric("Treatment Conversion Rate", f"{results['rate_b']:.2%}", help=f"n = {n_treatment:,}")
with col3:
    st.metric(
        "Relative Lift",
        f"{results['lift']:+.1%}",
        delta=f"{results['abs_diff']:+.2%} absolute",
    )
with col4:
    if results["significant"]:
        st.metric("p-value", f"{results['p_value']:.4f}", delta="Significant", delta_color="normal")
    else:
        st.metric("p-value", f"{results['p_value']:.4f}", delta="Not Significant", delta_color="off")

# ─────────────────────────────────────────────
# Significance callout
# ─────────────────────────────────────────────
if results["significant"]:
    st.success(
        f"**Statistically significant at {results['confidence']:.0%} confidence.** "
        f"The treatment group converted at {results['rate_b']:.2%} vs. "
        f"{results['rate_a']:.2%} for control, a relative lift of {results['lift']:.1%}. "
        f"The 95% confidence interval for the absolute difference is "
        f"[{results['ci_low']:.2%}, {results['ci_high']:.2%}]."
    )
else:
    st.warning(
        f"**Not statistically significant at {results['confidence']:.0%} confidence.** "
        f"The observed difference may be due to random chance. "
        f"Consider running the experiment longer to accumulate more data."
    )

# ─────────────────────────────────────────────
# Conversion rate chart with error bars
# ─────────────────────────────────────────────
st.markdown("### Conversion Rate Comparison")

# Calculate standard errors for each group (for error bars)
se_control = (results["rate_a"] * (1 - results["rate_a"]) / n_control) ** 0.5
se_treatment = (results["rate_b"] * (1 - results["rate_b"]) / n_treatment) ** 0.5

fig = go.Figure()

fig.add_trace(go.Bar(
    x=["Control", "Treatment"],
    y=[results["rate_a"], results["rate_b"]],
    error_y=dict(
        type="data",
        array=[1.96 * se_control, 1.96 * se_treatment],
        visible=True,
        color="#333333",
        thickness=1.5,
    ),
    marker_color=["#6B7280", "#2563EB"],
    text=[f"{results['rate_a']:.2%}", f"{results['rate_b']:.2%}"],
    textposition="outside",
    textfont=dict(size=14),
))

fig.update_layout(
    yaxis_title="Conversion Rate",
    yaxis_tickformat=".1%",
    showlegend=False,
    height=400,
    margin=dict(t=40, b=40),
    plot_bgcolor="white",
    yaxis=dict(gridcolor="#E5E7EB"),
)

st.plotly_chart(fig, width="stretch")

# ─────────────────────────────────────────────
# Revenue impact estimate
# ─────────────────────────────────────────────
st.markdown("### Revenue Impact Estimate")

total_revenue_control = control["REVENUE"].sum()
total_revenue_treatment = treatment["REVENUE"].sum()
avg_revenue_per_conversion = treatment[treatment["CONVERTED"] == 1]["REVENUE"].mean()

rev_col1, rev_col2, rev_col3 = st.columns(3)

with rev_col1:
    st.metric("Control Total Revenue", f"${total_revenue_control:,.0f}")
with rev_col2:
    st.metric("Treatment Total Revenue", f"${total_revenue_treatment:,.0f}")
with rev_col3:
    rev_diff = total_revenue_treatment - total_revenue_control
    st.metric("Revenue Difference", f"${rev_diff:+,.0f}")

if results["significant"] and avg_revenue_per_conversion > 0:
    st.info(
        f"If the treatment is rolled out to all users, the expected incremental conversions "
        f"per {n_control + n_treatment:,} users would be approximately "
        f"**{int(results['abs_diff'] * (n_control + n_treatment)):,}**, "
        f"at **${avg_revenue_per_conversion:,.2f}** revenue per conversion."
    )

# ─────────────────────────────────────────────
# Detail table
# ─────────────────────────────────────────────
with st.expander("View Experiment Detail by Segment"):
    segment_col = st.selectbox(
        "Segment by",
        ["CONTRACT_TYPE", "SUBSCRIPTION_MONTHS", "MONTHLY_REVENUE"],
    )

    if segment_col == "SUBSCRIPTION_MONTHS":
        exp_df = exp_df.copy()
        exp_df["TENURE_BUCKET"] = pd.cut(
            exp_df["SUBSCRIPTION_MONTHS"],
            bins=[0, 6, 12, 24, 100],
            labels=["0-6 mo", "7-12 mo", "13-24 mo", "25+ mo"],
        )
        group_col = "TENURE_BUCKET"
    elif segment_col == "MONTHLY_REVENUE":
        exp_df = exp_df.copy()
        exp_df["REVENUE_BUCKET"] = pd.cut(
            exp_df["MONTHLY_REVENUE"],
            bins=[0, 30, 60, 90, 200],
            labels=["$0-30", "$31-60", "$61-90", "$91+"],
        )
        group_col = "REVENUE_BUCKET"
    else:
        group_col = segment_col

    segment_summary = (
        exp_df.groupby([group_col, "VARIANT"])
        .agg(
            USERS=("CONVERTED", "count"),
            CONVERSIONS=("CONVERTED", "sum"),
        )
        .reset_index()
    )
    segment_summary["CONVERSION_RATE"] = (
        segment_summary["CONVERSIONS"] / segment_summary["USERS"]
    )
    segment_summary["CONVERSION_RATE"] = segment_summary["CONVERSION_RATE"].map("{:.2%}".format)

    st.dataframe(segment_summary, width="stretch", hide_index=True)

# ─────────────────────────────────────────────
# Sample size calculator (sidebar)
# ─────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### 📐 Sample Size Calculator")
st.sidebar.markdown("Plan your next experiment:")

baseline = st.sidebar.slider(
    "Baseline conversion rate",
    min_value=0.01, max_value=0.50, value=0.12, step=0.01, format="%.2f",
)
mde = st.sidebar.slider(
    "Minimum detectable effect (absolute)",
    min_value=0.005, max_value=0.10, value=0.03, step=0.005, format="%.3f",
)
power = st.sidebar.slider(
    "Statistical power",
    min_value=0.70, max_value=0.95, value=0.80, step=0.05, format="%.2f",
)

required_n = required_sample_size(baseline, mde, power=power)
st.sidebar.metric("Required Sample Size (per group)", f"{required_n:,}")
st.sidebar.caption(
    f"To detect a {mde:.1%} absolute lift over a {baseline:.0%} baseline "
    f"at 95% confidence and {power:.0%} power."
)