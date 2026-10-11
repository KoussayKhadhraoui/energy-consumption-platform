import pandas as pd


def get_basic_statistics(df):
    """
    Return basic statistics for the energy consumption data.
    """
    return df.describe()


def get_consumption_statistics(df):
    """
    Return statistics specifically for global active power.
    """
    consumption = df["Global_active_power"]

    return {
        "mean": consumption.mean(),
        "min": consumption.min(),
        "max": consumption.max(),
        "median": consumption.median()
    }