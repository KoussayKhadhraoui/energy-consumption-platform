from src.data.loader import load_energy_data
from src.data.preprocessing import preprocess_energy_data
from src.data.feature_engineering import (
    create_time_features,
    create_lag_features,
    create_target
)


file_path = "data/raw/household_power_consumption.txt"

print("Loading data...")

df = load_energy_data(file_path)

print("Preprocessing...")

df_hourly = preprocess_energy_data(df)

print("Creating time features...")

df_features = create_time_features(df_hourly)

print("Creating lag features...")

df_features = create_lag_features(df_features)

print("Creating target...")

df_features = create_target(df_features)

# Remove rows with missing lag/target values
df_features = df_features.dropna()

print("\nFinal shape:", df_features.shape)

print("\nML columns:")

print(
    df_features[
        [
            "Global_active_power",
            "hour",
            "day_of_week",
            "month",
            "is_weekend",
            "power_lag_1",
            "power_lag_2",
            "power_lag_3",
            "power_lag_24",
            "target"
        ]
    ].head(10)
)

print("\nMissing values:")
print(df_features.isnull().sum())