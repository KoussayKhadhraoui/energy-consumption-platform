import pandas as pd

from sklearn.ensemble import RandomForestRegressor
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


def prepare_ml_data(df):
    """
    Prepare features and target for Machine Learning.
    """

    X = df[MODEL_FEATURES]
    y = df["target"]

    return X, y


def train_model(X_train, y_train):
    """
    Train the Random Forest regression model.
    """

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using MAE, RMSE and R2.
    """

    predictions = model.predict(
        X_test
    )

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

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def get_feature_importance(model):
    """
    Return feature importance sorted by importance.
    """

    importance = pd.DataFrame({
        "feature": MODEL_FEATURES,
        "importance": model.feature_importances_
    })

    return importance.sort_values(
        by="importance",
        ascending=False
    )