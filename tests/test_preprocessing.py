from src.data.loader import load_energy_data
from src.data.preprocessing import preprocess_energy_data


file_path = "data/raw/household_power_consumption.txt"

print("Loading dataset...")

df = load_energy_data(file_path)

print("Raw shape:", df.shape)

print("\nPreprocessing...")

df_hourly = preprocess_energy_data(df)

print("\nProcessed shape:", df_hourly.shape)

print("\nColumns:")
print(df_hourly.columns.tolist())

print("\nFirst 5 rows:")
print(df_hourly.head())

print("\nMissing values:")
print(df_hourly.isnull().sum())