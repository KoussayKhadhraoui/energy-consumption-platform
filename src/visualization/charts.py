import matplotlib.pyplot as plt


def plot_consumption_over_time(df):
    plt.figure(figsize=(14, 6))

    plt.plot(
        df.index,
        df["Global_active_power"]
    )

    plt.title("Energy Consumption Over Time")
    plt.xlabel("Time")
    plt.ylabel("Global Active Power")

    plt.tight_layout()
    plt.show()