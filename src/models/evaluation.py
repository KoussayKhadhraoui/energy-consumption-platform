import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


MODEL_FEATURES = [
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


def evaluate_model(model, df):
    """
    Evaluate the trained model using
    a chronological 80/20 train-test split.
    """

    X = df[MODEL_FEATURES]
    y = df["target"]

    split_index = int(len(X) * 0.8)

    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    return metrics


def get_predictions(model, df):
    """
    Return actual and predicted values
    from the test set.
    """

    X = df[MODEL_FEATURES]
    y = df["target"]

    split_index = int(len(X) * 0.8)

    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    predictions = model.predict(X_test)

    result = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": predictions
    })

    return result