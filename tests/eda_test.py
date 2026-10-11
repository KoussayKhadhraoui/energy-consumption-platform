from src.data.loader import load_energy_data
from src.data.preprocessing import preprocess_energy_data
from src.visualization.charts import plot_consumption_over_time


file_path = "data/raw/household_power_consumption.txt"

print("Loading data...")

df = load_energy_data(file_path)

print("Preprocessing...")

df_hourly = preprocess_energy_data(df)

print("Plotting...")

plot_consumption_over_time(df_hourly)