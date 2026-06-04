import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Traffic Insights",
    page_icon="🔍",
    layout="wide"
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("data/Vehicle.csv")

df = load_data()

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🔍 Traffic Insights Dashboard")

st.markdown("""
Generate intelligent insights and discover hidden patterns
from vehicle traffic data.
""")

# ---------------------------------------------------
# BASIC INSIGHTS
# ---------------------------------------------------

peak_row = df.loc[df["Vehicles"].idxmax()]
low_row = df.loc[df["Vehicles"].idxmin()]

col1, col2 = st.columns(2)

with col1:
    st.success(
        f"""
        🚗 Peak Traffic

        Hour: {peak_row['Hour']}

        Vehicles: {int(peak_row['Vehicles'])}
        """
    )

with col2:
    st.info(
        f"""
        🚦 Lowest Traffic

        Hour: {low_row['Hour']}

        Vehicles: {int(low_row['Vehicles'])}
        """
    )

st.markdown("---")

# ---------------------------------------------------
# KPI CARDS
# ---------------------------------------------------

st.subheader("Traffic KPIs")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Vehicles",
    f"{int(df['Vehicles'].sum()):,}"
)

c2.metric(
    "Average Traffic",
    round(df["Vehicles"].mean(), 2)
)

c3.metric(
    "Maximum Traffic",
    int(df["Vehicles"].max())
)

c4.metric(
    "Minimum Traffic",
    int(df["Vehicles"].min())
)

# ---------------------------------------------------
# STATISTICAL INSIGHTS
# ---------------------------------------------------

st.markdown("---")

st.subheader("Statistical Insights")

stats = pd.DataFrame({
    "Metric": [
        "Mean",
        "Median",
        "Mode",
        "Standard Deviation",
        "Variance"
    ],
    "Value": [
        round(df["Vehicles"].mean(), 2),
        round(df["Vehicles"].median(), 2),
        round(df["Vehicles"].mode()[0], 2),
        round(df["Vehicles"].std(), 2),
        round(df["Vehicles"].var(), 2)
    ]
})

st.dataframe(
    stats,
    use_container_width=True
)

# ---------------------------------------------------
# TRAFFIC CATEGORIES
# ---------------------------------------------------

st.markdown("---")

st.subheader("Traffic Category Analysis")

avg_traffic = df["Vehicles"].mean()

conditions = [
    df["Vehicles"] < avg_traffic * 0.75,
    (df["Vehicles"] >= avg_traffic * 0.75)
    & (df["Vehicles"] <= avg_traffic * 1.25),
    df["Vehicles"] > avg_traffic * 1.25
]

choices = [
    "Low Traffic",
    "Moderate Traffic",
    "High Traffic"
]

df["Traffic_Category"] = np.select(
    conditions,
    choices,
    default="Moderate Traffic"
)

category_count = (
    df["Traffic_Category"]
    .value_counts()
    .reset_index()
)

category_count.columns = [
    "Category",
    "Count"
]

fig = px.pie(
    category_count,
    values="Count",
    names="Category",
    title="Traffic Category Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ---------------------------------------------------
# TOP TRAFFIC HOURS
# ---------------------------------------------------

st.markdown("---")

st.subheader("Top Traffic Hours")

top_hours = (
    df.sort_values(
        by="Vehicles",
        ascending=False
    )
    .head(10)
)

fig = px.bar(
    top_hours,
    x="Hour",
    y="Vehicles",
    title="Top 10 Traffic Hours"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ---------------------------------------------------
# LOWEST TRAFFIC HOURS
# ---------------------------------------------------

st.markdown("---")

st.subheader("Lowest Traffic Hours")

low_hours = (
    df.sort_values(
        by="Vehicles",
        ascending=True
    )
    .head(10)
)

fig = px.bar(
    low_hours,
    x="Hour",
    y="Vehicles",
    title="Lowest 10 Traffic Hours"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ---------------------------------------------------
# AUTOMATED INSIGHTS
# ---------------------------------------------------

st.markdown("---")

st.subheader("AI Generated Insights")

insights = []

if df["Vehicles"].max() > df["Vehicles"].mean() * 2:
    insights.append(
        "Peak traffic is significantly higher than average traffic."
    )

if df["Vehicles"].std() > 50:
    insights.append(
        "Traffic flow shows high variability across observations."
    )

if df["Vehicles"].median() < df["Vehicles"].mean():
    insights.append(
        "Traffic distribution is positively skewed."
    )

if len(insights) == 0:
    insights.append(
        "Traffic pattern appears relatively stable."
    )

for insight in insights:
    st.success(f"✅ {insight}")

# ---------------------------------------------------
# RECOMMENDATIONS
# ---------------------------------------------------

st.markdown("---")

st.subheader("Business Recommendations")

st.info("""
🚦 Increase monitoring during peak traffic periods.

🚗 Use historical patterns for predictive traffic planning.

📊 Investigate unusual spikes detected in traffic counts.

🏙️ Utilize insights for smart city traffic optimization.

📈 Implement advanced forecasting models for future planning.
""")

# ---------------------------------------------------
# SUMMARY REPORT
# ---------------------------------------------------

st.markdown("---")

st.subheader("Executive Summary")

st.write(f"""
The dataset contains **{len(df)} records**
with a total traffic count of
**{int(df['Vehicles'].sum()):,} vehicles**.

The average traffic volume is
**{round(df['Vehicles'].mean(),2)} vehicles**.

The highest recorded traffic was
**{int(df['Vehicles'].max())} vehicles**
at Hour **{peak_row['Hour']}**.

The lowest recorded traffic was
**{int(df['Vehicles'].min())} vehicles**
at Hour **{low_row['Hour']}**.

These insights can support transportation planning,
traffic management, and smart city initiatives.
""")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.caption(
    "Vehicle Traffic Analytics Dashboard | Insights Module"
)
