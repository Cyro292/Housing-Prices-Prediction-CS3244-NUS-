import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Tuple, Optional, Union, Dict, Any
from sklearn.model_selection import train_test_split


def load_resale_data(
    file_paths: Union[str, List[str]],
    include_features: Optional[List[str]] = None,
    exclude_features: Optional[List[str]] = None,
    price_column: str = "resale_price",
    month_column: str = "month",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load resale flat price data from CSV files and split into features and target dataframes.

    Parameters:
    -----------
    file_paths : str or List[str]
        Path(s) to the CSV file(s) containing resale flat price data.
    include_features : List[str], optional
        List of feature columns to include. If None, all columns except price_column are included.
    exclude_features : List[str], optional
        List of feature columns to exclude. Applied after include_features.
    price_column : str, default='resale_price'
        Name of the column containing the resale price (target variable).
    month_column : str, default='month'
        Name of the column containing the date information (e.g. 1990-01).
    verbose : bool, default=True
        Whether to print information during loading.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        X: Features dataframe
        y: Target dataframe (resale prices)
    """
    # Convert single file path to list
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    # Load and concatenate all dataframes
    dfs = []
    for file_path in file_paths:
        if verbose:
            print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        dfs.append(df)

    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    if verbose:
        print(f"Combined dataset shape: {combined_df.shape}")

    # Handle feature selection
    if include_features is None:
        # By default, use all columns except the price column
        feature_cols = [col for col in combined_df.columns if col != price_column]
    else:
        feature_cols = include_features

    # Apply exclusions
    if exclude_features is not None:
        feature_cols = [col for col in feature_cols if col not in exclude_features]

    # Ensure that all selected feature columns exist in the dataframe
    for col in feature_cols:
        if col not in combined_df.columns:
            raise ValueError(f"Feature column '{col}' not found in the dataset")

    # Check if price column exists
    if price_column not in combined_df.columns:
        raise ValueError(f"Price column '{price_column}' not found in the dataset")

    # Convert month column to datetime if it exists
    if month_column in combined_df.columns:
        combined_df[month_column] = pd.to_datetime(combined_df[month_column])

    # Create feature and target dataframes
    X = combined_df[feature_cols].copy()
    y = combined_df[price_column].copy()

    if verbose:
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Selected features: {', '.join(feature_cols)}")

    return X, y


def load_all_resale_data(
    include_features: Optional[List[str]] = None,
    exclude_features: Optional[List[str]] = None,
    price_column: str = "resale_price",
    month_column: str = "month",
    verbose: bool = True,
):
    """
    Load resale flat price data from CSV files and split into features and target dataframes.

    Parameters:
    -----------
    include_features : List[str], optional
        List of feature columns to include. If None, all columns except price_column are included.
    exclude_features : List[str], optional
        List of feature columns to exclude. Applied after include_features.
    price_column : str, default='resale_price'
        Name of the column containing the resale price (target variable).
    date_column : str, default='month'
        Name of the column containing the date information (e.g. 1990-01).
    verbose : bool, default=True
        Whether to print information during loading.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        X: Features dataframe
        y: Target dataframe (resale prices)
    """

    data = [
        "Resale Flat Prices (Based on Approval Date), 1990 - 1999.csv",
        "Resale Flat Prices (Based on Approval Date), 2000 - Feb 2012.csv",
        "Resale Flat Prices (Based on Registration Date), From Mar 2012 to Dec 2014.csv",
        "Resale Flat Prices (Based on Registration Date), From Jan 2015 to Dec 2016.csv",
        "Resale flat prices based on registration date from Jan-2017 onwards.csv",
    ]

    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    data_files = [os.path.join(data_dir, file) for file in data]

    return load_resale_data(
        file_paths=data_files,
        include_features=include_features,
        exclude_features=exclude_features,
        price_column=price_column,
        month_column=month_column,
        verbose=verbose,
    )


def get_cleaned_data(
    X: pd.DataFrame = None,
    y: pd.Series = None,
    include_features=None,
    exclude_features=None,
):
    """
    Clean the dataset by removing rows with missing values and perform data normalization.

    Parameters:
    -----------
    X : pd.DataFrame
        Feature dataframe.
    y : pd.Series
        Target series.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series]
        Cleaned feature dataframe and target series.
    """

    # If include_features or exclude_features are provided, filter the columns
    if include_features is not None:
        X = X[include_features]
    if exclude_features is not None:
        X = X.drop(columns=exclude_features, errors="ignore")

    # Step 1: Drop rows with any missing values
    original_shape = X.shape
    if y is not None:
        # Combine X and y to ensure we drop the same rows from both
        combined = pd.concat([X, y.rename("target")], axis=1)
        combined = combined.dropna()
        X = combined.drop("target", axis=1)
        y = combined["target"]
    else:
        X = X.dropna()

    if original_shape[0] > X.shape[0]:
        print(f"Dropped {original_shape[0] - X.shape[0]} rows with missing values.")

    # Step 2: Process categorical features

    # Convert storey_range to numeric (take the average of the range)
    if "storey_range" in X.columns:
        X["storey_range"] = X["storey_range"].apply(
            lambda x: (
                sum(int(i) for i in x.replace("TO", "").split() if i.isdigit()) / 2
                if isinstance(x, str)
                else x
            )
        )

    # One-hot encode town and flat_type if they exist
    categorical_cols = ["town", "flat_type", "flat_model"]
    for col in categorical_cols:
        if col in X.columns:
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
            X = pd.concat([X, dummies], axis=1)
            X = X.drop(col, axis=1)

    # Step 3: Process time-related features

    # Calculate flat age based on lease_commence_date if it exists
    if "lease_commence_date" in X.columns and "month" in X.columns:
        # Extract the year from the month column
        X["transaction_year"] = pd.to_datetime(X["month"]).dt.year

        # Calculate flat age at transaction
        X["flat_age"] = X["transaction_year"] - X["lease_commence_date"]

        # Drop the original columns as we've created a derived feature
        X = X.drop(["lease_commence_date", "month"], axis=1)

    if "street_name" in X.columns:
        X = X.drop("street_name", axis=1)

    if "storey_range" in X.columns:
        # convert storey_range to numeric (take the average of the range)
        X["storey_range"] = X["storey_range"].apply(
            lambda x: (
                sum(int(i) for i in x.replace("TO", "").split() if i.isdigit()) / 2
                if isinstance(x, str)
                else x
            )
        )
        
        

    # Step 4: Handle non-numeric columns that shouldn't be one-hot encoded

    # Drop street_name as it's too granular for modeling

    # Step 5: Normalize/scale numeric features
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        # Skip the target column if it's in X
        if col == "resale_price":
            continue

        # Normalize using min-max scaling
        min_val = X[col].min()
        max_val = X[col].max()
        if max_val > min_val:  # Avoid division by zero
            X[col] = (X[col] - min_val) / (max_val - min_val)

    return X, y


def get_train_split_data(
    X: pd.DataFrame = None,
    y: pd.Series = None,
    train_size: float = 0.5,
    random_state: Optional[int] = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing sets.

    Parameters:
    -----------
    X : pd.DataFrame
        Feature dataframe.
    y : pd.Series
        Target series.
    train_size : float, default=0.8
        Proportion of the dataset to include in the training set.
    random_state : int, optional
        Random seed for reproducibility.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train: Training feature dataframe
        X_test: Testing feature dataframe
        y_train: Training target series
        y_test: Testing target series
    """

    return train_test_split(X, y, train_size=train_size, random_state=random_state)
