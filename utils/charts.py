import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# ---------------------------------------------------
# TRAFFIC TREND CHART
# ---------------------------------------------------

def traffic_trend_chart(df):
    """
    Line chart showing vehicle traffic trend.
    """

    fig = px.line(
        df,
        x="Hour",
        y="Vehicles",
        markers=True,
        title="Vehicle Traffic Trend"
    )

    fig.update_layout(
        xaxis_title="Hour",
        yaxis_title="Vehicle Count",
        template="plotly_white"
    )

    return fig


# ---------------------------------------------------
# HISTOGRAM
# ---------------------------------------------------

def vehicle_distribution_chart(df):
    """
    Histogram of vehicle counts.
    """

    fig = px.histogram(
        df,
        x="Vehicles",
        nbins=30,
        title="Vehicle Count Distribution"
    )

    fig.update_layout(
        template="plotly_white"
    )

    return fig


# ---------------------------------------------------
# BOX PLOT
# ---------------------------------------------------

def outlier_boxplot(df):
    """
    Box plot for outlier detection.
    """

    fig = px.box(
        df,
        y="Vehicles",
        title="Outlier Detection"
    )

    fig.update_layout(
        template="plotly_white"
    )

    return fig


# ---------------------------------------------------
# MOVING AVERAGE CHART
# ---------------------------------------------------

def moving_average_chart(df, window=10):
    """
    Moving Average Analysis.
    """

    temp_df = df.copy()

    temp_df["Moving_Average"] = (
        temp_df["Vehicles"]
        .rolling(window=window)
        .mean()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=temp_df["Hour"],
            y=temp_df["Vehicles"],
            mode="lines",
            name="Actual Traffic"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=temp_df["Hour"],
            y=temp_df["Moving_Average"],
            mode="lines",
            name=f"{window}-Point Moving Average"
        )
    )

    fig.update_layout(
        title="Moving Average Analysis",
        xaxis_title="Hour",
        yaxis_title="Vehicles",
        template="plotly_white"
    )

    return fig


# ---------------------------------------------------
# PIE CHART
# ---------------------------------------------------

def traffic_category_pie(category_df):
    """
    Pie chart for traffic categories.
    """

    fig = px.pie(
        category_df,
        values="Count",
        names="Category",
        title="Traffic Category Distribution"
    )

    return fig


# ---------------------------------------------------
# BAR CHART
# ---------------------------------------------------

def top_traffic_chart(df):
    """
    Top traffic hours chart.
    """

    fig = px.bar(
        df,
        x="Hour",
        y="Vehicles",
        title="Top Traffic Hours"
    )

    fig.update_layout(
        template="plotly_white"
    )

    return fig


# ---------------------------------------------------
# LOW TRAFFIC CHART
# ---------------------------------------------------

def low_traffic_chart(df):
    """
    Lowest traffic hours chart.
    """

    fig = px.bar(
        df,
        x="Hour",
        y="Vehicles",
        title="Lowest Traffic Hours"
    )

    fig.update_layout(
        template="plotly_white"
    )

    return fig


# ---------------------------------------------------
# CORRELATION HEATMAP
# ---------------------------------------------------

def correlation_heatmap(df):
    """
    Correlation heatmap.
    """

    corr_matrix = df.corr(
        numeric_only=True
    )

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        title="Correlation Heatmap",
        aspect="auto"
    )

    return fig


# ---------------------------------------------------
# ACTUAL VS PREDICTED
# ---------------------------------------------------

def actual_vs_predicted_chart(
    actual_hours,
    actual_values,
    predicted_values
):
    """
    Actual vs Predicted Traffic.
    """

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=actual_hours,
            y=actual_values,
            mode="lines",
            name="Actual"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=actual_hours,
            y=predicted_values,
            mode="lines",
            name="Predicted"
        )
    )

    fig.update_layout(
        title="Actual vs Predicted Traffic",
        xaxis_title="Hour",
        yaxis_title="Vehicles",
        template="plotly_white"
    )

    return fig


# ---------------------------------------------------
# FORECAST CHART
# ---------------------------------------------------

def forecast_chart(forecast_df):
    """
    Future traffic forecast chart.
    """

    fig = px.line(
        forecast_df,
        x="Hour",
        y="Predicted_Vehicles",
        markers=True,
        title="Future Traffic Forecast"
    )

    fig.update_layout(
        xaxis_title="Future Hour",
        yaxis_title="Predicted Vehicles",
        template="plotly_white"
    )

    return fig


# ---------------------------------------------------
# SCATTER CHART
# ---------------------------------------------------

def traffic_scatter_chart(df):
    """
    Scatter chart of traffic data.
    """

    fig = px.scatter(
        df,
        x="Hour",
        y="Vehicles",
        title="Traffic Scatter Plot"
    )

    fig.update_layout(
        template="plotly_white"
    )

    return fig


# ---------------------------------------------------
# KPI HELPER
# ---------------------------------------------------

def get_kpis(df):
    """
    Generate dashboard KPIs.
    """

    return {
        "total_records": len(df),
        "total_vehicles": int(df["Vehicles"].sum()),
        "avg_vehicles": round(
            df["Vehicles"].mean(), 2
        ),
        "max_vehicles": int(
            df["Vehicles"].max()
        ),
        "min_vehicles": int(
            df["Vehicles"].min()
        )
    }
