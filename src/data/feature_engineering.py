import pandas as pd


def create_time_features(df):
    df = df.copy()

    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    return df


def create_lag_features(df):
    df = df.copy()

    # Current known consumption
    df["current_power"] = df["Global_active_power"]

    # Previous consumption
    df["power_lag_2"] = df["Global_active_power"].shift(2)
    df["power_lag_3"] = df["Global_active_power"].shift(3)
    df["power_lag_6"] = df["Global_active_power"].shift(6)
    df["power_lag_12"] = df["Global_active_power"].shift(12)
    df["power_lag_24"] = df["Global_active_power"].shift(24)
    df["power_lag_48"] = df["Global_active_power"].shift(48)

    return df


def create_rolling_features(df):
    df = df.copy()

    df["rolling_mean_3"] = (
        df["Global_active_power"]
        .rolling(window=3)
        .mean()
    )

    df["rolling_mean_6"] = (
        df["Global_active_power"]
        .rolling(window=6)
        .mean()
    )

    df["rolling_mean_24"] = (
        df["Global_active_power"]
        .rolling(window=24)
        .mean()
    )

    return df


def create_target(df):
    df = df.copy()

    # Predict the next hour
    df["target"] = df["Global_active_power"].shift(-1)

    return df