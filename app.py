import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from src.data.loader import load_energy_data
from src.data.preprocessing import preprocess_energy_data

from src.data.feature_engineering import (
    create_time_features,
    create_lag_features,
    create_rolling_features,
    create_target
)

from src.models.evaluation import (
    evaluate_model,
    get_predictions
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Energy Consumption Platform",
    page_icon="⚡",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("⚡ Energy Consumption Prediction Platform")

st.markdown(
    """
    **Energy Consumption Prediction Platform**

    Analyse historical energy consumption and predict
    the average active power for the next hour using
    Machine Learning.
    """
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "models/energy_model.pkl"
    )

    return model


# ============================================================
# LOAD RAW DATA
# ============================================================

@st.cache_data
def load_data():

    file_path = (
        "data/raw/household_power_consumption.txt"
    )

    df = load_energy_data(
        file_path
    )

    df = preprocess_energy_data(
        df
    )

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

@st.cache_data
def prepare_features(df):

    data = create_time_features(
        df
    )

    data = create_lag_features(
        data
    )

    data = create_rolling_features(
        data
    )

    data = create_target(
        data
    )

    data = data.dropna()

    return data


# ============================================================
# LOAD PROJECT DATA
# ============================================================

try:

    model = load_model()

    df = load_data()

    features_df = prepare_features(
        df
    )

except Exception as e:

    st.error(
        f"Error while loading the project: {e}"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚡ Energy Platform")

st.sidebar.markdown(
    "Navigation"
)

page = st.sidebar.radio(
    "Choose a page:",
    [
        "🏠 Overview",
        "📊 Analysis",
        "🤖 Prediction",
        "🔎 Model"
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.header(
        "🏠 Overview"
    )

    st.write(
        "General overview of the household energy "
        "consumption dataset."
    )

    st.divider()

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    average_power = (
        df["Global_active_power"].mean()
    )

    max_power = (
        df["Global_active_power"].max()
    )

    min_power = (
        df["Global_active_power"].min()
    )

    total_records = len(df)

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Average Power",
            f"{average_power:.2f} kW"
        )

    with col2:

        st.metric(
            "Maximum Power",
            f"{max_power:.2f} kW"
        )

    with col3:

        st.metric(
            "Minimum Power",
            f"{min_power:.2f} kW"
        )

    with col4:

        st.metric(
            "Records",
            f"{total_records:,}"
        )

    st.divider()

    # --------------------------------------------------------
    # Consumption over time
    # --------------------------------------------------------

    st.subheader(
        "📈 Energy Consumption Over Time"
    )

    fig, ax = plt.subplots(
        figsize=(14, 5)
    )

    ax.plot(
        df.index,
        df["Global_active_power"]
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Average Active Power (kW)"
    )

    ax.set_title(
        "Energy Consumption Over Time"
    )

    fig.tight_layout()

    st.pyplot(
        fig
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Latest data
    # --------------------------------------------------------

    st.subheader(
        "🕐 Latest Consumption"
    )

    latest_data = df.tail(
        24
    ).copy()

    st.dataframe(
        latest_data[
            ["Global_active_power"]
        ],
        use_container_width=True
    )


# ============================================================
# ANALYSIS
# ============================================================

elif page == "📊 Analysis":

    st.header(
        "📊 Consumption Analysis"
    )

    st.write(
        "Analyse energy consumption according to "
        "time, day and month."
    )

    st.divider()

    # ========================================================
    # HOURLY ANALYSIS
    # ========================================================

    st.subheader(
        "🕐 Average Consumption by Hour"
    )

    hourly = (
        df.groupby(
            df.index.hour
        )["Global_active_power"]
        .mean()
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        hourly.index,
        hourly.values,
        marker="o"
    )

    ax.set_xlabel(
        "Hour"
    )

    ax.set_ylabel(
        "Average Power (kW)"
    )

    ax.set_title(
        "Average Consumption by Hour"
    )

    ax.set_xticks(
        range(24)
    )

    fig.tight_layout()

    st.pyplot(
        fig
    )

    plt.close(fig)

    st.divider()

    # ========================================================
    # DAY ANALYSIS
    # ========================================================

    st.subheader(
        "📅 Average Consumption by Day of Week"
    )

    daily = (
        df.groupby(
            df.index.dayofweek
        )["Global_active_power"]
        .mean()
    )

    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    daily.index = [
        days[i]
        for i in daily.index
    ]

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.bar(
        daily.index,
        daily.values
    )

    ax.set_xlabel(
        "Day"
    )

    ax.set_ylabel(
        "Average Power (kW)"
    )

    ax.set_title(
        "Average Consumption by Day of Week"
    )

    ax.tick_params(
        axis="x",
        rotation=30
    )

    fig.tight_layout()

    st.pyplot(
        fig
    )

    plt.close(fig)

    st.divider()

    # ========================================================
    # MONTHLY ANALYSIS
    # ========================================================

    st.subheader(
        "📅 Average Consumption by Month"
    )

    monthly = (
        df.groupby(
            df.index.month
        )["Global_active_power"]
        .mean()
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        monthly.index,
        monthly.values,
        marker="o"
    )

    ax.set_xlabel(
        "Month"
    )

    ax.set_ylabel(
        "Average Power (kW)"
    )

    ax.set_title(
        "Average Consumption by Month"
    )

    ax.set_xticks(
        range(1, 13)
    )

    fig.tight_layout()

    st.pyplot(
        fig
    )

    plt.close(fig)


# ============================================================
# PREDICTION
# ============================================================

elif page == "🤖 Prediction":

    st.header(
        "🤖 Next-Hour Energy Prediction"
    )

    st.write(
        """
        Enter the current energy consumption and select
        the date and hour. The model will predict the
        average active power for the next hour.
        """
    )

    st.divider()

    # ========================================================
    # INPUTS
    # ========================================================

    st.subheader(
        "⚙️ Prediction Inputs"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        default_power = float(
            features_df.iloc[-1][
                "current_power"
            ]
        )

        current_power = st.number_input(
            "Current Power (kW)",
            min_value=0.0,
            max_value=10.0,
            value=default_power,
            step=0.1
        )

    with col2:

        selected_date = st.date_input(
            "Date",
            value=features_df.index[-1].date()
        )

    with col3:

        selected_hour = st.slider(
            "Current Hour",
            min_value=0,
            max_value=23,
            value=int(
                features_df.iloc[-1][
                    "hour"
                ]
            )
        )

    st.divider()

    predict_button = st.button(
        "⚡ Predict Next Hour",
        use_container_width=True
    )

    # ========================================================
    # MAKE PREDICTION
    # ========================================================

    if predict_button:

        selected_datetime = (
            pd.Timestamp(
                selected_date
            )
            + pd.Timedelta(
                hours=selected_hour
            )
        )

        historical_data = features_df[
            features_df.index <= selected_datetime
        ]

        if len(historical_data) == 0:

            st.error(
                "No historical data is available "
                "for the selected date."
            )

        else:

            latest = historical_data.iloc[-1]

            # ------------------------------------------------
            # Time features
            # ------------------------------------------------

            hour = selected_hour

            day_of_week = (
                selected_datetime.dayofweek
            )

            month = (
                selected_datetime.month
            )

            is_weekend = int(
                day_of_week >= 5
            )

            # ------------------------------------------------
            # Lag features
            # ------------------------------------------------

            power_lag_2 = (
                latest["power_lag_2"]
            )

            power_lag_3 = (
                latest["power_lag_3"]
            )

            power_lag_6 = (
                latest["power_lag_6"]
            )

            power_lag_12 = (
                latest["power_lag_12"]
            )

            power_lag_24 = (
                latest["power_lag_24"]
            )

            power_lag_48 = (
                latest["power_lag_48"]
            )

            # ------------------------------------------------
            # Rolling features
            # ------------------------------------------------

            rolling_mean_3 = (
                latest["rolling_mean_3"]
            )

            rolling_mean_6 = (
                latest["rolling_mean_6"]
            )

            rolling_mean_24 = (
                latest["rolling_mean_24"]
            )

            # ------------------------------------------------
            # Model input
            # ------------------------------------------------

            input_data = pd.DataFrame({

                "current_power": [
                    current_power
                ],

                "hour": [
                    hour
                ],

                "day_of_week": [
                    day_of_week
                ],

                "month": [
                    month
                ],

                "is_weekend": [
                    is_weekend
                ],

                "power_lag_2": [
                    power_lag_2
                ],

                "power_lag_3": [
                    power_lag_3
                ],

                "power_lag_6": [
                    power_lag_6
                ],

                "power_lag_12": [
                    power_lag_12
                ],

                "power_lag_24": [
                    power_lag_24
                ],

                "power_lag_48": [
                    power_lag_48
                ],

                "rolling_mean_3": [
                    rolling_mean_3
                ],

                "rolling_mean_6": [
                    rolling_mean_6
                ],

                "rolling_mean_24": [
                    rolling_mean_24
                ]
            })

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            prediction = model.predict(
                input_data
            )[0]

            prediction = max(
                0,
                prediction
            )

            # ------------------------------------------------
            # Result
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "⚡ Prediction Result"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Current Power",
                    f"{current_power:.2f} kW"
                )

            with col2:

                st.metric(
                    "Predicted Next Hour",
                    f"{prediction:.2f} kW"
                )

            st.success(
                f"Predicted average active power "
                f"for the next hour: "
                f"**{prediction:.2f} kW**"
            )

            # ------------------------------------------------
            # Model input
            # ------------------------------------------------

            with st.expander(
                "🔎 Show Model Input"
            ):

                st.dataframe(
                    input_data,
                    use_container_width=True
                )


# ============================================================
# MODEL
# ============================================================

elif page == "🔎 Model":

    st.header(
        "🔎 Machine Learning Model"
    )

    st.write(
        """
        Random Forest Regressor used to predict
        next-hour energy consumption.
        """
    )

    st.divider()

    # ========================================================
    # AUTOMATIC MODEL EVALUATION
    # ========================================================

    st.subheader(
        "📊 Model Performance"
    )

    try:

        metrics = evaluate_model(
            model,
            features_df
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "MAE",
                f"{metrics['MAE']:.4f} kW"
            )

        with col2:

            st.metric(
                "RMSE",
                f"{metrics['RMSE']:.4f} kW"
            )

        with col3:

            st.metric(
                "R²",
                f"{metrics['R2']:.4f}"
            )

        st.caption(
            "Metrics are calculated automatically "
            "on the chronological test set."
        )

    except Exception as e:

        st.error(
            f"Could not calculate model metrics: {e}"
        )

    st.divider()

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    st.subheader(
        "🎯 Feature Importance"
    )

    if hasattr(
        model,
        "feature_importances_"
    ):

        feature_names = [
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
            "rolling_mean_24"
        ]

        importance = pd.DataFrame({

            "Feature": feature_names,

            "Importance":
                model.feature_importances_
        })

        importance = importance.sort_values(
            "Importance",
            ascending=False
        )

        st.dataframe(
            importance,
            use_container_width=True
        )

        # ----------------------------------------------------
        # Feature importance chart
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        ax.barh(
            importance["Feature"],
            importance["Importance"]
        )

        ax.set_xlabel(
            "Importance"
        )

        ax.set_title(
            "Random Forest Feature Importance"
        )

        ax.invert_yaxis()

        fig.tight_layout()

        st.pyplot(
            fig
        )

        plt.close(fig)

    else:

        st.warning(
            "Feature importance is not available "
            "for this model."
        )

    st.divider()

    # ========================================================
    # ACTUAL VS PREDICTED
    # ========================================================

    st.subheader(
        "📈 Actual vs Predicted"
    )

    try:

        results = get_predictions(
            model,
            features_df
        )

        n_samples = min(
            200,
            len(results)
        )

        fig, ax = plt.subplots(
            figsize=(14, 6)
        )

        ax.plot(
            results[
                "Actual"
            ].iloc[:n_samples].values,
            label="Actual"
        )

        ax.plot(
            results[
                "Predicted"
            ].iloc[:n_samples].values,
            label="Predicted"
        )

        ax.set_xlabel(
            "Test Samples"
        )

        ax.set_ylabel(
            "Average Active Power (kW)"
        )

        ax.set_title(
            "Actual vs Predicted Energy Consumption"
        )

        ax.legend()

        fig.tight_layout()

        st.pyplot(
            fig
        )

        plt.close(fig)

        st.caption(
            f"Showing the first {n_samples} "
            "samples of the test set."
        )

        # ----------------------------------------------------
        # Prediction table
        # ----------------------------------------------------

        with st.expander(
            "🔎 Show Prediction Results"
        ):

            st.dataframe(
                results.head(50),
                use_container_width=True
            )

    except Exception as e:

        st.error(
            f"Could not generate predictions: {e}"
        )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "Energy Consumption Prediction Platform"
)

st.sidebar.caption(
    "Python • Pandas • Scikit-learn • Streamlit"
)