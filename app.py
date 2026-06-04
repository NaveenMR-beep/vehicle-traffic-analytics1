import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Vehicle Traffic Analytics",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Vehicle Traffic Analytics Dashboard")

st.markdown("""
### Real-Time Traffic Analytics System

Analyze traffic volume patterns, identify peak congestion hours,
and generate actionable insights.
""")

df = pd.read_csv("data/Vehicle.csv")

col1,col2,col3,col4 = st.columns(4)

col1.metric("Total Records", len(df))
col2.metric("Total Vehicles", int(df["Vehicles"].sum()))
col3.metric("Average Traffic", round(df["Vehicles"].mean(),2))
col4.metric("Peak Traffic", int(df["Vehicles"].max()))

st.dataframe(df.head())
