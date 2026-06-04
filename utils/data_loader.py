import pandas as pd
import streamlit as st

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

@st.cache_data
def load_data(file_path="data/Vehicle.csv"):
    """
    Load the vehicle traffic dataset.

    Parameters
    ----------
    file_path : str
        Path to CSV file.

    Returns
    -------
    DataFrame
        Loaded dataset.
    """

    try:
        df = pd.read_csv(file_path)

        # Remove duplicate rows
        df = df.drop_duplicates()

        # Handle missing values
        df = df.dropna()

        return df

    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return pd.DataFrame()

    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()


# ---------------------------------------------------
# DATASET SUMMARY
# ---------------------------------------------------

def get_dataset_summary(df):
    """
    Generate dataset summary statistics.
    """

    summary = {
        "Total Records": len(df),
        "Total Columns": len(df.columns),
        "Missing Values": df.isnull().sum().sum(),
        "Duplicate Rows": df.duplicated().sum()
    }

    return summary


# ---------------------------------------------------
# COLUMN INFORMATION
# ---------------------------------------------------

def get_column_info(df):
    """
    Return column details.
    """

    info = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum().values
    })

    return info


# ---------------------------------------------------
# STATISTICAL SUMMARY
# ---------------------------------------------------

def get_statistics(df):
    """
    Return statistical summary.
    """

    return df.describe()


# ---------------------------------------------------
# VEHICLE METRICS
# ---------------------------------------------------

def get_vehicle_metrics(df):
    """
    Calculate key traffic metrics.
    """

    metrics = {
        "Total Vehicles": int(df["Vehicles"].sum()),
        "Average Traffic": round(df["Vehicles"].mean(), 2),
        "Maximum Traffic": int(df["Vehicles"].max()),
        "Minimum Traffic": int(df["Vehicles"].min()),
        "Median Traffic": round(df["Vehicles"].median(), 2),
        "Standard Deviation": round(df["Vehicles"].std(), 2)
    }

    return metrics


# ---------------------------------------------------
# PEAK TRAFFIC
# ---------------------------------------------------

def get_peak_traffic(df):
    """
    Return peak traffic row.
    """

    return df.loc[df["Vehicles"].idxmax()]


# ---------------------------------------------------
# LOW TRAFFIC
# ---------------------------------------------------

def get_low_traffic(df):
    """
    Return lowest traffic row.
    """

    return df.loc[df["Vehicles"].idxmin()]


# ---------------------------------------------------
# MOVING AVERAGE
# ---------------------------------------------------

def calculate_moving_average(df, window=10):
    """
    Calculate moving average.
    """

    temp_df = df.copy()

    temp_df["Moving_Average"] = (
        temp_df["Vehicles"]
        .rolling(window=window)
        .mean()
    )

    return temp_df


# ---------------------------------------------------
# OUTLIER DETECTION
# ---------------------------------------------------

def detect_outliers(df):
    """
    Detect outliers using IQR method.
    """

    Q1 = df["Vehicles"].quantile(0.25)
    Q3 = df["Vehicles"].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df["Vehicles"] < lower) |
        (df["Vehicles"] > upper)
    ]

    return outliers


# ---------------------------------------------------
# FORECAST DATA PREPARATION
# ---------------------------------------------------

def prepare_forecasting_data(df):
    """
    Prepare X and y for ML forecasting.
    """

    X = df[["Hour"]]
    y = df["Vehicles"]

    return X, y
