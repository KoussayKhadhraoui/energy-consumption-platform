import pandas as pd


def load_energy_data(file_path):
    df = pd.read_csv(
        file_path,
        sep=";",
        na_values=["?"],
        low_memory=False
    )

    return df


if __name__ == "__main__":
    file_path = "data/raw/household_power_consumption.txt"

    df = load_energy_data(file_path)

    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())