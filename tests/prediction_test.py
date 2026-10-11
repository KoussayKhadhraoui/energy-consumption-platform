import joblib
import pandas as pd

from src.data.loader import load_energy_data
from src.data.preprocessing import preprocess_energy_data

from src.data.feature_engineering import (
    create_time_features,
    create_lag_features,
    create_rolling_features,
    create_target
)


print("Loading data...")

df = load_energy_data(
    "data/raw/household_power_consumption.txt"
)

print("Preprocessing...")

df = preprocess_energy_data(df)

print("Creating features...")

df = create_time_features(df)
df = create_lag_features(df)
df = create_rolling_features(df)
df = create_target(df)

df = df.dropna()


# Same features used during training
features = [
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


print("\nChecking features...")

print("Number of features:", len(features))

print("\nFeatures:")

for i, feature in enumerate(features, 1):
    print(f"{i}. {feature}")


# Prepare input
latest = df.iloc[-1]

X = pd.DataFrame(
    [[latest[feature] for feature in features]],
    columns=features
)


print("\nInput shape:")
print(X.shape)

print("\nInput data:")
print(X)


# Load model
print("\nLoading model...")

model = joblib.load(
    "models/energy_model.pkl"
)


# Prediction
print("\nMaking prediction...")

prediction = model.predict(X)[0]

print(
    f"\nPredicted next-hour consumption: "
    f"{prediction:.4f} kW"
)


print("\nPrediction test completed successfully!")