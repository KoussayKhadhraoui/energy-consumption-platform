
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data.loader import load_energy_data
from src.data.preprocessing import preprocess_energy_data
from src.data.feature_engineering import (
    create_time_features,
    create_lag_features,
    create_rolling_features,
    create_target,
)
from src.models.evaluation import evaluate_model, get_predictions


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "household_power_consumption.txt"
)

MODEL_PATH = PROJECT_ROOT / "models" / "energy_model.pkl"

MODEL_FEATURES = [
    "current_power",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "power_lag_2",
    "power_lag_3",
    "power_lag_6",
    "power_lag_12",
    "power_lag_24",
    "power_lag_48",
    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_mean_24",
]

st.set_page_config(
    page_title="Energy Consumption Platform",
    page_icon="⚡",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("⚡ Energy Consumption Prediction Platform")

st.markdown(
    """
    Analyse historical household energy consumption and predict
    average active power for the next hour using Machine Learning.
    """
)

st.divider()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found at:\n{MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found at:\n{DATA_PATH}"
        )

    data = load_energy_data(DATA_PATH)
    data = preprocess_energy_data(data)

    return data


# ============================================================
# FEATURE ENGINEERING
# ============================================================

@st.cache_data
def prepare_features(df):
    data = create_time_features(df)
    data = create_lag_features(data)
    data = create_rolling_features(data)
    data = create_target(data)
    data = data.dropna()

    return data


# ============================================================
# LOAD PROJECT
# ============================================================

try:
    model = load_model()
    df = load_data()
    features_df = prepare_features(df)

except FileNotFoundError as error:
    st.error("⚠️ A required file could not be found.")

    st.markdown(
        f"""
        **Details:** `{error}`

        To fix this, check that the following files exist:

        - Dataset: `data/raw/household_power_consumption.txt`
        - Trained model: `models/energy_model.pkl`

        If the dataset is missing, download it from the
        [official UCI website](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption),
        extract the archive and place the text file in `data/raw/`.
        """
    )

    st.stop()

except Exception as error:
    st.error("An unexpected error occurred while loading the project.")

    with st.expander("Technical details"):
        st.exception(error)

    st.stop()


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("⚡ Energy Platform")
st.sidebar.markdown("Navigate between the application pages.")

page = st.sidebar.radio(
    "Choose a page:",
    [
        "🏠 Overview",
        "📊 Analysis",
        "🤖 Prediction",
        "🔎 Model",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Energy Consumption Prediction Platform")
st.sidebar.caption("Python • Pandas • Scikit-learn • Streamlit")


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.header("🏠 Overview")

    st.write(
        "General overview of the household energy consumption dataset."
    )

    st.divider()

    # KPIs
    average_power = df["Global_active_power"].mean()
    max_power = df["Global_active_power"].max()
    min_power = df["Global_active_power"].min()
    total_records = len(df)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Average Power", f"{average_power:.2f} kW")
    col2.metric("Maximum Power", f"{max_power:.2f} kW")
    col3.metric("Minimum Power", f"{min_power:.2f} kW")
    col4.metric("Records", f"{total_records:,}")

    st.divider()

    # Consumption over time
    st.subheader("📈 Energy Consumption Over Time")

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        df.index,
        df["Global_active_power"],
        linewidth=0.8,
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Average Active Power (kW)")
    ax.set_title("Energy Consumption Over Time")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Latest records
    st.subheader("🕐 Latest Consumption")

    latest_data = df.tail(24).copy()

    st.dataframe(
        latest_data[["Global_active_power"]],
        width="stretch",
    )


# ============================================================
# ANALYSIS
# ============================================================

elif page == "📊 Analysis":

    st.header("📊 Consumption Analysis")

    st.write(
        "Analyse energy consumption according to hour, day of week and month."
    )

    st.divider()

    # Hourly analysis
    st.subheader("🕐 Average Consumption by Hour")

    hourly = df.groupby(
        df.index.hour
    )["Global_active_power"].mean()

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        hourly.index,
        hourly.values,
        marker="o",
    )

    ax.set_xlabel("Hour")
    ax.set_ylabel("Average Power (kW)")
    ax.set_title("Average Consumption by Hour")
    ax.set_xticks(range(24))
    ax.grid(alpha=0.2)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # Day-of-week analysis
    st.subheader("📅 Average Consumption by Day of Week")

    daily = df.groupby(
        df.index.dayofweek
    )["Global_active_power"].mean()

    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    daily.index = [days[i] for i in daily.index]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(daily.index, daily.values)

    ax.set_xlabel("Day")
    ax.set_ylabel("Average Power (kW)")
    ax.set_title("Average Consumption by Day of Week")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # Monthly analysis
    st.subheader("📅 Average Consumption by Month")

    monthly = df.groupby(
        df.index.month
    )["Global_active_power"].mean()

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        monthly.index,
        monthly.values,
        marker="o",
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Average Power (kW)")
    ax.set_title("Average Consumption by Month")
    ax.set_xticks(range(1, 13))
    ax.grid(alpha=0.2)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# PREDICTION
# ============================================================

elif page == "🤖 Prediction":

    st.header("🤖 Next-Hour Energy Prediction")

    st.write(
        """
        Enter the current energy consumption and select a date and hour.
        The model predicts average active power for the next hour.
        """
    )

    st.info(
        "The lag and rolling features are taken from the historical "
        "dataset. The current power value is supplied by you."
    )

    st.divider()

    st.subheader("⚙️ Prediction Inputs")

    latest_features = features_df.iloc[-1]

    col1, col2, col3 = st.columns(3)

    with col1:
        default_power = float(latest_features["current_power"])

        current_power = st.number_input(
            "Current Power (kW)",
            min_value=0.0,
            max_value=10.0,
            value=max(0.0, min(10.0, default_power)),
            step=0.1,
        )

    with col2:
        selected_date = st.date_input(
            "Date",
            value=features_df.index[-1].date(),
        )

    with col3:
        selected_hour = st.slider(
            "Current Hour",
            min_value=0,
            max_value=23,
            value=int(latest_features["hour"]),
        )

    st.divider()

    predict_button = st.button(
        "⚡ Predict Next Hour",
        type="primary",
        width="stretch",
    )

    if predict_button:

        selected_datetime = (
            pd.Timestamp(selected_date)
            + pd.Timedelta(hours=selected_hour)
        )

        historical_data = features_df[
            features_df.index <= selected_datetime
        ]

        if historical_data.empty:

            st.error(
                "No historical data is available for the selected date. "
                "Choose a date that exists in the dataset."
            )

        else:

            latest = historical_data.iloc[-1]

            # Time features
            hour = selected_hour
            day_of_week = selected_datetime.dayofweek
            month = selected_datetime.month
            is_weekend = int(day_of_week >= 5)

            # Model input
            input_data = pd.DataFrame(
                [
                    {
                        "current_power": current_power,
                        "hour": hour,
                        "day_of_week": day_of_week,
                        "month": month,
                        "is_weekend": is_weekend,
                        "power_lag_2": latest["power_lag_2"],
                        "power_lag_3": latest["power_lag_3"],
                        "power_lag_6": latest["power_lag_6"],
                        "power_lag_12": latest["power_lag_12"],
                        "power_lag_24": latest["power_lag_24"],
                        "power_lag_48": latest["power_lag_48"],
                        "rolling_mean_3": latest["rolling_mean_3"],
                        "rolling_mean_6": latest["rolling_mean_6"],
                        "rolling_mean_24": latest["rolling_mean_24"],
                    }
                ],
                columns=MODEL_FEATURES,
            )

            try:
                prediction = float(model.predict(input_data)[0])
                prediction = max(0.0, prediction)

                st.divider()
                st.subheader("⚡ Prediction Result")

                col1, col2 = st.columns(2)

                col1.metric(
                    "Current Power",
                    f"{current_power:.2f} kW",
                )

                col2.metric(
                    "Predicted Next Hour",
                    f"{prediction:.2f} kW",
                )

                st.success(
                    f"Predicted average active power for the next hour: "
                    f"**{prediction:.2f} kW**"
                )

                with st.expander("🔎 Show Model Input"):
                    st.dataframe(input_data, width="stretch")

            except Exception as error:
                st.error("Prediction failed.")

                with st.expander("Technical details"):
                    st.exception(error)


# ============================================================
# MODEL
# ============================================================

elif page == "🔎 Model":

    st.header("🔎 Machine Learning Model")

    st.write(
        """
        The application uses a Random Forest Regressor to predict
        next-hour energy consumption.
        """
    )

    st.divider()

    # Model performance
    st.subheader("📊 Model Performance")

    try:
        metrics = evaluate_model(model, features_df)

        col1, col2, col3 = st.columns(3)

        col1.metric("MAE", f"{metrics['MAE']:.4f} kW")
        col2.metric("RMSE", f"{metrics['RMSE']:.4f} kW")
        col3.metric("R²", f"{metrics['R2']:.4f}")

        st.caption(
            "Metrics are calculated on the chronological test set."
        )

    except Exception as error:
        st.error("Could not calculate model metrics.")

        with st.expander("Technical details"):
            st.exception(error)

    st.divider()

    # Feature importance
    st.subheader("🎯 Feature Importance")

    if hasattr(model, "feature_importances_"):

        importances = model.feature_importances_

        if len(importances) == len(MODEL_FEATURES):

            importance = pd.DataFrame(
                {
                    "Feature": MODEL_FEATURES,
                    "Importance": importances,
                }
            ).sort_values(
                "Importance",
                ascending=False,
            )

            st.dataframe(
                importance,
                width="stretch",
            )

            fig, ax = plt.subplots(figsize=(10, 7))

            ax.barh(
                importance["Feature"],
                importance["Importance"],
            )

            ax.set_xlabel("Importance")
            ax.set_title("Random Forest Feature Importance")
            ax.invert_yaxis()

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        else:
            st.warning(
                "The number of feature importances does not match "
                "the expected model features."
            )

    else:
        st.warning(
            "Feature importance is not available for this model."
        )

    st.divider()

    # Actual vs predicted
    st.subheader("📈 Actual vs Predicted")

    try:
        results = get_predictions(model, features_df)

        if results.empty:
            st.warning("No prediction results are available.")

        else:
            n_samples = min(200, len(results))

            fig, ax = plt.subplots(figsize=(14, 6))

            ax.plot(
                results["Actual"].iloc[:n_samples].values,
                label="Actual",
            )

            ax.plot(
                results["Predicted"].iloc[:n_samples].values,
                label="Predicted",
            )

            ax.set_xlabel("Test Samples")
            ax.set_ylabel("Average Active Power (kW)")
            ax.set_title("Actual vs Predicted Energy Consumption")
            ax.legend()
            ax.grid(alpha=0.2)

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                f"Showing the first {n_samples} samples of the test set."
            )

            with st.expander("🔎 Show Prediction Results"):
                st.dataframe(
                    results.head(50),
                    width="stretch",
                )

    except Exception as error:
        st.error("Could not generate prediction results.")

        with st.expander("Technical details"):
            st.exception(error)
