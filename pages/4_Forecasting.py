import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Traffic Forecasting",
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

st.title("📈 Vehicle Traffic Forecasting")

st.markdown("""
Forecast future traffic volume using Machine Learning.

This module uses Linear Regression to estimate future
vehicle counts based on historical traffic trends.
""")

# ---------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------

X = df[["Hour"]]
y = df["Vehicles"]

# ---------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)

# ---------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------

mae = mean_absolute_error(y, predictions)
rmse = np.sqrt(mean_squared_error(y, predictions))
r2 = r2_score(y, predictions)

st.subheader("📊 Model Performance")

c1, c2, c3 = st.columns(3)

c1.metric("MAE", round(mae, 2))
c2.metric("RMSE", round(rmse, 2))
c3.metric("R² Score", round(r2, 4))

st.markdown("---")

# ---------------------------------------------------
# ACTUAL VS PREDICTED
# ---------------------------------------------------

st.subheader("Actual vs Predicted Traffic")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df["Hour"],
        y=y,
        mode="lines",
        name="Actual Traffic"
    )
)

fig.add_trace(
    go.Scatter(
        x=df["Hour"],
        y=predictions,
        mode="lines",
        name="Predicted Traffic"
    )
)

fig.update_layout(
    title="Actual vs Predicted Traffic",
    xaxis_title="Hour",
    yaxis_title="Vehicle Count"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ---------------------------------------------------
# FUTURE FORECAST
# ---------------------------------------------------

st.markdown("---")

st.subheader("🔮 Future Traffic Forecast")

future_hour = st.slider(
    "Select Future Hour",
    min_value=int(df["Hour"].min()),
    max_value=int(df["Hour"].max()) + 200,
    value=int(df["Hour"].max()) + 10
)

future_prediction = model.predict(
    [[future_hour]]
)[0]

st.metric(
    "Predicted Vehicle Count",
    int(future_prediction)
)

# ---------------------------------------------------
# FORECAST VISUALIZATION
# ---------------------------------------------------

future_df = pd.DataFrame({
    "Hour": [future_hour],
    "Vehicles": [future_prediction]
})

fig = px.scatter(
    df,
    x="Hour",
    y="Vehicles",
    title="Future Traffic Forecast"
)

fig.add_scatter(
    x=future_df["Hour"],
    y=future_df["Vehicles"],
    mode="markers",
    name="Forecast"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ---------------------------------------------------
# MULTI-HOUR FORECAST
# ---------------------------------------------------

st.markdown("---")

st.subheader("📅 Multi-Hour Forecast")

num_hours = st.slider(
    "Forecast Next N Hours",
    5,
    50,
    20
)

future_hours = np.arange(
    int(df["Hour"].max()) + 1,
    int(df["Hour"].max()) + num_hours + 1
)

future_preds = model.predict(
    future_hours.reshape(-1, 1)
)

forecast_df = pd.DataFrame({
    "Hour": future_hours,
    "Predicted_Vehicles": future_preds
})

st.dataframe(
    forecast_df,
    use_container_width=True
)

fig = px.line(
    forecast_df,
    x="Hour",
    y="Predicted_Vehicles",
    markers=True,
    title="Future Traffic Forecast Trend"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ---------------------------------------------------
# FORECAST INSIGHTS
# ---------------------------------------------------

st.markdown("---")

st.subheader("💡 Forecast Insights")

avg_future = forecast_df[
    "Predicted_Vehicles"
].mean()

max_future = forecast_df[
    "Predicted_Vehicles"
].max()

min_future = forecast_df[
    "Predicted_Vehicles"
].min()

col1, col2, col3 = st.columns(3)

col1.metric(
    "Average Forecast",
    round(avg_future, 2)
)

col2.metric(
    "Maximum Forecast",
    round(max_future, 2)
)

col3.metric(
    "Minimum Forecast",
    round(min_future, 2)
)

# ---------------------------------------------------
# FORECAST TABLE DOWNLOAD
# ---------------------------------------------------

st.markdown("---")

st.subheader("📥 Download Forecast Results")

csv = forecast_df.to_csv(index=False)

st.download_button(
    label="Download Forecast CSV",
    data=csv,
    file_name="traffic_forecast.csv",
    mime="text/csv"
)

# ---------------------------------------------------
# RECOMMENDATIONS
# ---------------------------------------------------

st.markdown("---")

st.subheader("🚦 Recommendations")

st.info("""
• Monitor peak forecast periods carefully.

• Allocate resources during expected high traffic hours.

• Use advanced forecasting models such as:
    - ARIMA
    - Prophet
    - XGBoost
    - LSTM

• Continuously retrain the model as new data becomes available.

• Combine forecasting with real-time traffic monitoring.
""")

# ---------------------------------------------------
# FUTURE ENHANCEMENTS
# ---------------------------------------------------

with st.expander("Future Enhancements"):

    st.write("""
    ✔ ARIMA Time-Series Forecasting

    ✔ Facebook Prophet Forecasting

    ✔ LSTM Deep Learning Models

    ✔ Real-Time Traffic Prediction

    ✔ Smart City Analytics Integration

    ✔ Interactive Traffic Alerts

    ✔ Congestion Prediction System
    """)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.caption(
    "Vehicle Traffic Analytics Dashboard | Forecasting Module"
)
