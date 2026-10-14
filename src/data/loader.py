from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = (
    PROJECT_ROOT / "data" / "raw" / "household_power_consumption.txt"
)


def load_energy_data(file_path=DEFAULT_DATA_PATH):
    file_path = Path(file_path)

    if not file_path.is_absolute():
        file_path = PROJECT_ROOT / file_path

    if not file_path.is_file():
        raise FileNotFoundError(
            "\nDataset not found!\n"
            f"Expected location: {file_path}\n\n"
            "Please download the Individual Household Electric Power "
            "Consumption dataset from:\n"
            "https://archive.ics.uci.edu/dataset/235/"
            "individual+household+electric+power+consumption\n\n"
            "Extract the archive and place "
            "'household_power_consumption.txt' inside:\n"
            "data/raw/"
        )

    df = pd.read_csv(
        file_path,
        sep=";",
        na_values=["?"],
        low_memory=False
    )

    return df


if __name__ == "__main__":
    df = load_energy_data()

    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())