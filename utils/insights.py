import pandas as pd
import numpy as np

# ---------------------------------------------------
# PEAK TRAFFIC
# ---------------------------------------------------

def get_peak_traffic(df):
    """
    Returns the row containing the highest traffic.
    """

    return df.loc[df["Vehicles"].idxmax()]


# ---------------------------------------------------
# LOWEST TRAFFIC
# ---------------------------------------------------

def get_low_traffic(df):
    """
    Returns the row containing the lowest traffic.
    """

    return df.loc[df["Vehicles"].idxmin()]


# ---------------------------------------------------
# TRAFFIC METRICS
# ---------------------------------------------------

def get_traffic_metrics(df):
    """
    Returns key traffic statistics.
    """

    metrics = {
        "Total Vehicles": int(df["Vehicles"].sum()),
        "Average Traffic": round(df["Vehicles"].mean(), 2),
        "Maximum Traffic": int(df["Vehicles"].max()),
        "Minimum Traffic": int(df["Vehicles"].min()),
        "Median Traffic": round(df["Vehicles"].median(), 2),
        "Standard Deviation": round(df["Vehicles"].std(), 2),
        "Variance": round(df["Vehicles"].var(), 2)
    }

    return metrics


# ---------------------------------------------------
# TRAFFIC CATEGORY ANALYSIS
# ---------------------------------------------------

def classify_traffic(df):
    """
    Categorize traffic into Low, Moderate and High.
    """

    avg_traffic = df["Vehicles"].mean()

    conditions = [
        df["Vehicles"] < avg_traffic * 0.75,
        (
            (df["Vehicles"] >= avg_traffic * 0.75)
            &
            (df["Vehicles"] <= avg_traffic * 1.25)
        ),
        df["Vehicles"] > avg_traffic * 1.25
    ]

    categories = [
        "Low Traffic",
        "Moderate Traffic",
        "High Traffic"
    ]

    result_df = df.copy()

    result_df["Traffic_Category"] = np.select(
        conditions,
        categories,
        default="Moderate Traffic"
    )

    return result_df


# ---------------------------------------------------
# CATEGORY COUNTS
# ---------------------------------------------------

def get_category_counts(df):
    """
    Returns traffic category counts.
    """

    category_df = (
        df["Traffic_Category"]
        .value_counts()
        .reset_index()
    )

    category_df.columns = [
        "Category",
        "Count"
    ]

    return category_df


# ---------------------------------------------------
# TOP TRAFFIC HOURS
# ---------------------------------------------------

def get_top_traffic_hours(df, n=10):
    """
    Returns top traffic hours.
    """

    return (
        df.sort_values(
            by="Vehicles",
            ascending=False
        )
        .head(n)
    )


# ---------------------------------------------------
# LOWEST TRAFFIC HOURS
# ---------------------------------------------------

def get_low_traffic_hours(df, n=10):
    """
    Returns lowest traffic hours.
    """

    return (
        df.sort_values(
            by="Vehicles",
            ascending=True
        )
        .head(n)
    )


# ---------------------------------------------------
# OUTLIER DETECTION
# ---------------------------------------------------

def detect_outliers(df):
    """
    Detect outliers using IQR method.
    """

    q1 = df["Vehicles"].quantile(0.25)
    q3 = df["Vehicles"].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    outliers = df[
        (df["Vehicles"] < lower_bound)
        |
        (df["Vehicles"] > upper_bound)
    ]

    return outliers


# ---------------------------------------------------
# GENERATE AUTOMATED INSIGHTS
# ---------------------------------------------------

def generate_insights(df):
    """
    Generate automatic business insights.
    """

    insights = []

    mean_val = df["Vehicles"].mean()
    median_val = df["Vehicles"].median()
    std_val = df["Vehicles"].std()
    max_val = df["Vehicles"].max()

    if max_val > mean_val * 2:
        insights.append(
            "Peak traffic is significantly higher than average traffic."
        )

    if std_val > (mean_val * 0.5):
        insights.append(
            "Traffic flow shows high variability."
        )

    if median_val < mean_val:
        insights.append(
            "Traffic data is positively skewed."
        )

    if len(insights) == 0:
        insights.append(
            "Traffic pattern appears stable across observations."
        )

    return insights


# ---------------------------------------------------
# EXECUTIVE SUMMARY
# ---------------------------------------------------

def generate_summary(df):
    """
    Generates an executive summary.
    """

    peak = get_peak_traffic(df)
    low = get_low_traffic(df)

    summary = f"""
Dataset contains {len(df)} traffic records.

Total vehicle count: {int(df['Vehicles'].sum()):,}

Average traffic volume: {round(df['Vehicles'].mean(),2)}

Peak traffic:
Hour {peak['Hour']} with {int(peak['Vehicles'])} vehicles.

Lowest traffic:
Hour {low['Hour']} with {int(low['Vehicles'])} vehicles.

The data can support transportation planning,
traffic forecasting, and smart city analytics.
"""

    return summary


# ---------------------------------------------------
# STATISTICAL SUMMARY
# ---------------------------------------------------

def get_statistics_table(df):
    """
    Returns a statistics dataframe.
    """

    stats = pd.DataFrame({
        "Metric": [
            "Mean",
            "Median",
            "Mode",
            "Standard Deviation",
            "Variance",
            "Maximum",
            "Minimum"
        ],
        "Value": [
            round(df["Vehicles"].mean(), 2),
            round(df["Vehicles"].median(), 2),
            round(df["Vehicles"].mode()[0], 2),
            round(df["Vehicles"].std(), 2),
            round(df["Vehicles"].var(), 2),
            round(df["Vehicles"].max(), 2),
            round(df["Vehicles"].min(), 2)
        ]
    })

    return stats
