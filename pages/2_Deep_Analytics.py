import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Deep Analytics",
    page_icon="📈",
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

st.title("📈 Deep Analytics Dashboard")

st.markdown("""
Advanced analytics and visual exploration of vehicle traffic patterns.
""")

# ---------------------------------------------------
# KPI SECTION
# ---------------------------------------------------

total_vehicles = int(df["Vehicles"].sum())
avg_vehicles = round(df["Vehicles"].mean(), 2)
max_vehicles = int(df["Vehicles"].max())
min_vehicles = int(df["Vehicles"].min())

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Vehicles", f"{total_vehicles:,}")
col2.metric("Average Traffic", avg_vehicles)
col3.metric("Peak Traffic", max_vehicles)
col4.metric("Minimum Traffic", min_vehicles)

st.markdown("---")

# ---------------------------------------------------
# TABS
# ---------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Traffic Trend",
    "Distribution",
    "Outlier Analysis",
    "Moving Average",
    "Correlation"
])

# ---------------------------------------------------
# TAB 1 : TRAFFIC TREND
# ---------------------------------------------------

with tab1:

    st.subheader("Traffic Trend Analysis")

    fig = px.line(
        df,
        x="Hour",
        y="Vehicles",
        markers=True,
        title="Vehicle Traffic Trend"
    )

    fig.update_layout(
        xaxis_title="Hour",
        yaxis_title="Vehicle Count"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ---------------------------------------------------
# TAB 2 : DISTRIBUTION
# ---------------------------------------------------

with tab2:

    st.subheader("Vehicle Distribution")

    fig = px.histogram(
        df,
        x="Vehicles",
        nbins=30,
        title="Distribution of Vehicle Counts"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    fig2 = px.violin(
        df,
        y="Vehicles",
        box=True,
        points="all",
        title="Vehicle Count Density"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# ---------------------------------------------------
# TAB 3 : OUTLIER ANALYSIS
# ---------------------------------------------------

with tab3:

    st.subheader("Outlier Detection")

    Q1 = df["Vehicles"].quantile(0.25)
    Q3 = df["Vehicles"].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df["Vehicles"] < lower_bound) |
        (df["Vehicles"] > upper_bound)
    ]

    st.metric(
        "Total Outliers",
        len(outliers)
    )

    fig = px.box(
        df,
        y="Vehicles",
        title="Box Plot for Outlier Detection"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    if len(outliers) > 0:
        st.subheader("Detected Outliers")
        st.dataframe(
            outliers,
            use_container_width=True
        )

# ---------------------------------------------------
# TAB 4 : MOVING AVERAGE
# ---------------------------------------------------

with tab4:

    st.subheader("Moving Average Analysis")

    window = st.slider(
        "Select Moving Average Window",
        3,
        50,
        10
    )

    df["Moving_Average"] = (
        df["Vehicles"]
        .rolling(window=window)
        .mean()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Hour"],
            y=df["Vehicles"],
            mode="lines",
            name="Actual Traffic"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["Hour"],
            y=df["Moving_Average"],
            mode="lines",
            name=f"{window}-Point Moving Average"
        )
    )

    fig.update_layout(
        title="Moving Average Trend Analysis",
        xaxis_title="Hour",
        yaxis_title="Vehicles"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ---------------------------------------------------
# TAB 5 : CORRELATION
# ---------------------------------------------------

with tab5:

    st.subheader("Correlation Analysis")

    corr_matrix = df.corr(numeric_only=True)

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ---------------------------------------------------
# ADVANCED ANALYTICS
# ---------------------------------------------------

st.markdown("---")

st.header("📊 Advanced Statistical Analysis")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Statistics")

    stats = pd.DataFrame({
        "Metric": [
            "Mean",
            "Median",
            "Standard Deviation",
            "Variance",
            "Maximum",
            "Minimum"
        ],
        "Value": [
            round(df["Vehicles"].mean(), 2),
            round(df["Vehicles"].median(), 2),
            round(df["Vehicles"].std(), 2),
            round(df["Vehicles"].var(), 2),
            round(df["Vehicles"].max(), 2),
            round(df["Vehicles"].min(), 2)
        ]
    })

    st.dataframe(
        stats,
        use_container_width=True
    )

with col2:

    st.subheader("Percentiles")

    percentile_df = pd.DataFrame({
        "Percentile": [
            "25%",
            "50%",
            "75%",
            "90%",
            "95%"
        ],
        "Value": [
            df["Vehicles"].quantile(0.25),
            df["Vehicles"].quantile(0.50),
            df["Vehicles"].quantile(0.75),
            df["Vehicles"].quantile(0.90),
            df["Vehicles"].quantile(0.95)
        ]
    })

    st.dataframe(
        percentile_df,
        use_container_width=True
    )

# ---------------------------------------------------
# BUSINESS INSIGHTS
# ---------------------------------------------------

st.markdown("---")

st.header("💡 Analytics Insights")

st.info(
    """
    • Peak traffic periods indicate congestion zones.

    • Outlier spikes may represent special events,
      accidents, or unusual traffic conditions.

    • Moving averages help smooth short-term fluctuations.

    • Historical trends can support future traffic forecasting.

    • Analytics can help city planners optimize traffic flow.
    """
)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.caption(
    "Vehicle Traffic Analytics Dashboard | Deep Analytics Module"
)
