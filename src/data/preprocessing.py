import pandas as pd


def preprocess_energy_data(df):
    """
    Clean and transform the raw energy consumption dataset.
    """

    # 1. Create a datetime column
    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        dayfirst=True,
        errors="coerce"
    )

    # 2. Remove original date/time columns
    df = df.drop(columns=["Date", "Time"])

    # 3. Convert measurements to numeric
    numeric_columns = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # 4. Sort by datetime
    df = df.sort_values("Datetime")

    # 5. Set datetime as index
    df = df.set_index("Datetime")

    # 6. Handle missing values
    df = df.interpolate(method="time")

    # 7. Resample from minute-level to hourly data
    df_hourly = df.resample("1h").mean()

    # 8. Remove remaining missing values
    df_hourly = df_hourly.dropna()

    return df_hourly