
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from src.data.loader import load_energy_data
from src.data.preprocessing import preprocess_energy_data

from src.data.feature_engineering import (
    create_time_features,
    create_lag_features,
    create_rolling_features,
    create_target
)

from src.models.train import (
    prepare_ml_data,
    train_model,
    evaluate_model,
    get_feature_importance
)


# 1. Define project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "household_power_consumption.txt"
MODEL_PATH = PROJECT_ROOT / "models" / "energy_model.pkl"

# Make sure the models directory exists
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


# 2. Load data
print("Loading energy consumption data...")
df = load_energy_data(str(DATA_PATH))

print("Raw data shape:", df.shape)


# 3. Preprocess data
print("\nPreprocessing data...")
df_hourly = preprocess_energy_data(df)

print("Hourly data shape:", df_hourly.shape)


# 4. Feature engineering
print("\nCreating time features...")
df_features = create_time_features(df_hourly)

print("Creating lag features...")
df_features = create_lag_features(df_features)

print("Creating rolling features...")
df_features = create_rolling_features(df_features)

print("Creating target variable...")
df_features = create_target(df_features)

# Remove rows containing NaN values created by lagging,
# rolling windows, or target shifting
df_features = df_features.dropna()

print("Final dataset shape:", df_features.shape)


# 5. Prepare features and target
print("\nPreparing machine learning data...")
X, y = prepare_ml_data(df_features)

print("Features shape:", X.shape)
print("Target shape:", y.shape)


# 6. Split data chronologically
# Avoid randomly mixing past and future observations
split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# 7. Train model
print("\nTraining Random Forest model...")
model = train_model(X_train, y_train)

print("Model training completed.")


# 8. Evaluate model
print("\nEvaluating model...")
metrics = evaluate_model(model, X_test, y_test)

print("\nModel Performance")
print("-" * 35)
print(f"MAE:  {metrics['MAE']:.4f} kW")
print(f"RMSE: {metrics['RMSE']:.4f} kW")
print(f"R2:   {metrics['R2']:.4f}")
print("-" * 35)


# 9. Calculate feature importance
print("\nFeature Importance:")

# Corrected call: this function accepts only the model
importance = get_feature_importance(model)

print(importance.to_string(index=False))


# 10. Save the trained model
joblib.dump(model, MODEL_PATH)

print(f"\nModel saved successfully to: {MODEL_PATH}")


# 11. Generate predictions on the test set
print("\nGenerating predictions...")
predictions = model.predict(X_test)

print("Number of predictions:", len(predictions))


# 12. Plot actual vs predicted values
plot_size = min(200, len(y_test))

plt.figure(figsize=(14, 6))

plt.plot(
    y_test.iloc[:plot_size].to_numpy(),
    label="Actual",
    linewidth=2
)

plt.plot(
    predictions[:plot_size],
    label="Predicted",
    linewidth=2
)

plt.title("Actual vs Predicted Average Active Power")
plt.xlabel("Test Samples")
plt.ylabel("Average Active Power (kW)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

print("\nDisplaying prediction plot...")
plt.show()

print("\nMachine learning test completed successfully!")
