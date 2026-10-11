import joblib

from src.data.loader import load_energy_data
from src.data.preprocessing import preprocess_energy_data

from src.data.feature_engineering import (
    create_time_features,
    create_lag_features,
    create_rolling_features,
    create_target
)

from src.models.evaluation import evaluate_model


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


print("Loading model...")

model = joblib.load(
    "models/energy_model.pkl"
)


print("\nEvaluating model...")

metrics = evaluate_model(
    model,
    df
)


print("\nModel Performance:")

print(
    f"MAE:  {metrics['MAE']:.4f}"
)

print(
    f"RMSE: {metrics['RMSE']:.4f}"
)

print(
    f"R2:   {metrics['R2']:.4f}"
)


print("\nEvaluation completed successfully!")