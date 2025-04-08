import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    f1_score,
)


def get_r2_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Calculate the R^2 score between true and predicted values.

    Parameters:
    -----------
    y_true : pd.Series
        True target values.
    y_pred : pd.Series
        Predicted target values.

    Returns:
    --------
    float
        R^2 score.
    """

    if len(y_true) != len(y_pred):
        raise ValueError("Length of true and predicted values must be the same.")

    return r2_score(y_true, y_pred)


def get_rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Calculate the Root Mean Squared Error (RMSE) between true and predicted values.

    Parameters:
    -----------
    y_true : pd.Series
        True target values.
    y_pred : pd.Series
        Predicted target values.

    Returns:
    --------
    float
        RMSE value.
    """

    if len(y_true) != len(y_pred):
        raise ValueError("Length of true and predicted values must be the same.")

    return root_mean_squared_error(y_true, y_pred)


def get_mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Calculate the Mean Absolute Error (MAE) between true and predicted values.

    Parameters:
    -----------
    y_true : pd.Series
        True target values.
    y_pred : pd.Series
        Predicted target values.

    Returns:
    --------
    float
        MAE value.
    """

    if len(y_true) != len(y_pred):
        raise ValueError("Length of true and predicted values must be the same.")

    return mean_absolute_error(y_true, y_pred)


def get_f1_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Calculate the F1 score between true and predicted values.

    Parameters:
    -----------
    y_true : pd.Series
        True target values.
    y_pred : pd.Series
        Predicted target values.

    Returns:
    --------
    float
        F1 score.
    """

    if len(y_true) != len(y_pred):
        raise ValueError("Length of true and predicted values must be the same.")

    return f1_score(y_true, y_pred, average="weighted")


def get_regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Calculate various metrics between true and predicted values.

    Parameters:
    -----------
    y_true : pd.Series
        True target values.
    y_pred : pd.Series
        Predicted target values.

    Returns:
    --------
    dict
        Dictionary containing RMSE, MAE, R^2, and F1 scores.
    """

    return {
        "rmse": get_rmse(y_true, y_pred),
        "mae": get_mae(y_true, y_pred),
        "r2": get_r2_score(y_true, y_pred),
    }
