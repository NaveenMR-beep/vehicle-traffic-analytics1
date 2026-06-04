import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Dataset Overview",
    page_icon="📊",
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

st.title("📊 Dataset Overview")
st.markdown(
    """
    Explore the Vehicle Traffic Dataset and understand
    its structure, quality, and statistics.
    """
)

# ---------------------------------------------------
# DATASET INFO
# ---------------------------------------------------

st.subheader("Dataset Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", len(df))

with col2:
    st.metric("Total Columns", len(df.columns))

with col3:
    st.metric("Missing Values", df.isnull().sum().sum())

# ---------------------------------------------------
# DATA PREVIEW
# ---------------------------------------------------

st.subheader("Dataset Preview")

rows = st.slider(
    "Select Number of Rows",
    min_value=5,
    max_value=50,
    value=10
)

st.dataframe(
    df.head(rows),
    use_container_width=True
)

# ---------------------------------------------------
# COLUMN DETAILS
# ---------------------------------------------------

st.subheader("Column Information")

column_info = pd.DataFrame({
    "Column Name": df.columns,
    "Data Type": df.dtypes.astype(str),
    "Missing Values": df.isnull().sum().values
})

st.dataframe(
    column_info,
    use_container_width=True
)

# ---------------------------------------------------
# STATISTICAL SUMMARY
# ---------------------------------------------------

st.subheader("Statistical Summary")

st.dataframe(
    df.describe(),
    use_container_width=True
)

# ---------------------------------------------------
# MISSING VALUES
# ---------------------------------------------------

st.subheader("Missing Values Analysis")

missing_df = pd.DataFrame({
    "Column": df.columns,
    "Missing Values": df.isnull().sum().values
})

fig_missing = px.bar(
    missing_df,
    x="Column",
    y="Missing Values",
    title="Missing Values by Column"
)

st.plotly_chart(
    fig_missing,
    use_container_width=True
)

# ---------------------------------------------------
# VEHICLE DISTRIBUTION
# ---------------------------------------------------

st.subheader("Vehicle Count Distribution")

fig_hist = px.histogram(
    df,
    x="Vehicles",
    nbins=30,
    title="Vehicle Count Distribution"
)

st.plotly_chart(
    fig_hist,
    use_container_width=True
)

# ---------------------------------------------------
# DATA QUALITY REPORT
# ---------------------------------------------------

st.subheader("Data Quality Report")

duplicate_rows = df.duplicated().sum()

col1, col2 = st.columns(2)

with col1:
    st.success(
        f"✅ Missing Values: {df.isnull().sum().sum()}"
    )

with col2:
    st.warning(
        f"⚠️ Duplicate Rows: {duplicate_rows}"
    )

# ---------------------------------------------------
# DOWNLOAD DATASET
# ---------------------------------------------------

st.subheader("Download Dataset")

csv = df.to_csv(index=False)

st.download_button(
    label="📥 Download CSV",
    data=csv,
    file_name="Vehicle.csv",
    mime="text/csv"
)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.caption(
    "Vehicle Traffic Analytics Dashboard | Dataset Overview"
)
